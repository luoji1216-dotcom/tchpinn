from __future__ import annotations

import csv
import json
import math
import sys
from dataclasses import replace
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from scipy.special import hankel1

ROOT = Path(__file__).resolve().parents[1]
CORE_ROOT = ROOT / "square_target"
if str(CORE_ROOT) not in sys.path:
    sys.path.insert(0, str(CORE_ROOT))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from austria_optimized_common import build_target  # noqa: E402
from pinn_pixel_inverse_core import (  # noqa: E402
    DoubleBranchPINN,
    TargetSpec,
    TrainConfig,
    load_fem_table,
    parse_direction_from_name,
    resolve_device,
    resolve_dtype,
    target_mask,
)


HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data_1GHz"
B53000_CHECKPOINT = (
    HERE
    / "results_5_3_3_austria_1GHz_direct_lineB_detach52000_detach_to60000"
    / "checkpoint_adam_053000.pt"
)
DIRECTIONS = ("+x", "-x", "+y", "-y")


class EpsilonOverride(torch.nn.Module):
    def __init__(self, base: DoubleBranchPINN, epsilon_fn: Callable[[torch.Tensor], torch.Tensor]) -> None:
        super().__init__()
        self.base = base
        self.epsilon_fn = epsilon_fn

    def epsilon(self, xy: torch.Tensor) -> torch.Tensor:
        return self.epsilon_fn(xy)

    def scattered_field(self, xy: torch.Tensor, directions: torch.Tensor) -> torch.Tensor:
        return self.base.scattered_field(xy, directions)


def checkpoint_config(path: Path, *, frequency_hz: float) -> TrainConfig:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    raw = checkpoint.get("config", {}) if isinstance(checkpoint, dict) else {}
    valid = set(TrainConfig.__dataclass_fields__.keys())
    kwargs = {key: value for key, value in raw.items() if key in valid}
    config = TrainConfig(**kwargs)
    config.frequency_hz = frequency_hz
    config.device = "auto"
    config.dtype = "float32"
    config.max_points_per_direction = 0
    config.integral_grid_size = 36
    config.incident_phase_sign = 1.0
    config.observation_imag_sign = -1.0
    return config


def load_model(path: Path, target: TargetSpec, device: torch.device, dtype: torch.dtype) -> DoubleBranchPINN:
    config = checkpoint_config(path, frequency_hz=1.0e9)
    model = DoubleBranchPINN(config, target).to(device=device, dtype=dtype)
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    state = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    model.load_state_dict(state, strict=True)
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return model


def true_epsilon_fn(target: TargetSpec, dtype: torch.dtype) -> Callable[[torch.Tensor], torch.Tensor]:
    def fn(xy: torch.Tensor) -> torch.Tensor:
        xy_np = xy.detach().cpu().numpy()
        mask = target_mask(target, xy_np[:, 0], xy_np[:, 1])
        values = np.full((xy_np.shape[0], 1), target.eps_background, dtype=np.float64)
        values[mask, 0] = target.eps_object
        return torch.as_tensor(values, dtype=dtype, device=xy.device)

    return fn


def model_epsilon_fn(model: DoubleBranchPINN) -> Callable[[torch.Tensor], torch.Tensor]:
    def fn(xy: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return model.epsilon(xy)

    return fn


def load_raw(config: TrainConfig):
    xy_all = []
    obs_scattered_all = []
    dirs_all = []
    labels_all = []
    amps: dict[str, complex] = {}
    fit_rows = []
    for label in DIRECTIONS:
        path = DATA_DIR / f"{label}.txt"
        parsed, direction = parse_direction_from_name(path)
        xy, obs_total = load_fem_table(path, imag_sign=config.observation_imag_sign)
        direction_arr = np.asarray(direction, dtype=np.float64)
        phase = np.exp(1j * config.k0 * (xy @ direction_arr))
        amp = np.vdot(phase, obs_total) / np.vdot(phase, phase)
        fitted_incident = amp * phase
        obs_scattered = obs_total - fitted_incident
        amps[label] = complex(amp)
        xy_all.append(xy)
        obs_scattered_all.append(obs_scattered)
        dirs_all.append(np.tile(direction_arr, (xy.shape[0], 1)))
        labels_all.append(np.full(xy.shape[0], label, dtype=object))
        fit_rows.append(
            {
                "direction": label,
                "parsed": parsed,
                "fit_amp_real": float(amp.real),
                "fit_amp_imag": float(amp.imag),
                "fit_amp_abs": float(abs(amp)),
                "total_fit_rel_l2": float(np.linalg.norm(obs_total - fitted_incident) / np.linalg.norm(obs_total)),
                "total_amp_mean": float(np.mean(np.abs(obs_total))),
                "obs_scattered_amp_mean": float(np.mean(np.abs(obs_scattered))),
            }
        )
    return (
        np.vstack(xy_all),
        np.concatenate(obs_scattered_all),
        np.vstack(dirs_all),
        np.concatenate(labels_all),
        amps,
        fit_rows,
    )


def integral_grid_and_green(xy_obs: np.ndarray, target: TargetSpec, config: TrainConfig, device, dtype):
    n_grid = int(config.integral_grid_size)
    half = float(target.roi_half_width)
    dx = 2.0 * half / n_grid
    coords = np.linspace(-half + 0.5 * dx, half - 0.5 * dx, n_grid)
    xx, yy = np.meshgrid(coords, coords)
    quad_xy_np = np.column_stack((xx.reshape(-1), yy.reshape(-1))).astype(np.float64)
    delta = xy_obs[:, None, :] - quad_xy_np[None, :, :]
    distance = np.linalg.norm(delta, axis=2).clip(min=1e-9)
    green = 0.25j * hankel1(0, config.k0 * distance)
    return (
        torch.as_tensor(quad_xy_np, dtype=dtype, device=device),
        torch.as_tensor(green.real, dtype=dtype, device=device),
        torch.as_tensor(green.imag, dtype=dtype, device=device),
        float(dx * dx),
    )


def torch_incident(xy: torch.Tensor, dirs: torch.Tensor, k0: float, amps: torch.Tensor):
    phase = k0 * torch.sum(xy * dirs, dim=1, keepdim=True)
    cos_p = torch.cos(phase)
    sin_p = torch.sin(phase)
    amp_re = amps[:, 0:1]
    amp_im = amps[:, 1:2]
    return amp_re * cos_p - amp_im * sin_p, amp_re * sin_p + amp_im * cos_p


def predict(
    model,
    target: TargetSpec,
    config: TrainConfig,
    xy_obs: np.ndarray,
    dirs_np: np.ndarray,
    labels_np: np.ndarray,
    amps_by_label: dict[str, complex],
    *,
    quad_xy: torch.Tensor,
    green_re: torch.Tensor,
    green_im: torch.Tensor,
    area_weight: float,
    device,
    dtype,
) -> np.ndarray:
    directions_t = torch.as_tensor(dirs_np, dtype=dtype, device=device)
    pred = torch.zeros((xy_obs.shape[0], 2), dtype=dtype, device=device)
    with torch.no_grad():
        eps = model.epsilon(quad_xy)
        contrast = eps - float(target.eps_background)
        k2_area = (config.k0**2) * area_weight
        for label in DIRECTIONS:
            first = np.flatnonzero(labels_np == label)[0]
            direction = directions_t[first : first + 1]
            mask_np = labels_np == label
            mask = torch.as_tensor(mask_np, dtype=torch.bool, device=device)
            quad_dirs = direction.expand(quad_xy.shape[0], 2)
            amp = amps_by_label[label]
            quad_amps = torch.as_tensor([[amp.real, amp.imag]], dtype=dtype, device=device).expand(
                quad_xy.shape[0], 2
            )
            inc_re, inc_im = torch_incident(quad_xy, quad_dirs, config.k0, quad_amps)
            scattered_quad = model.scattered_field(quad_xy, quad_dirs).detach()
            total_re = inc_re + scattered_quad[:, 0:1]
            total_im = inc_im + scattered_quad[:, 1:2]
            source_re = (k2_area * contrast * total_re).squeeze(1)
            source_im = (k2_area * contrast * total_im).squeeze(1)
            gre = green_re[mask]
            gim = green_im[mask]
            pred_re = torch.matmul(gre, source_re) - torch.matmul(gim, source_im)
            pred_im = torch.matmul(gre, source_im) + torch.matmul(gim, source_re)
            pred[mask, 0] = pred_re
            pred[mask, 1] = pred_im
    out = pred.detach().cpu().numpy()
    return out[:, 0].astype(np.float64) + 1j * out[:, 1].astype(np.float64)


def stats(pred: np.ndarray, obs: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    diff = pred - obs
    corrected = np.zeros_like(pred)
    losses = []
    for label in DIRECTIONS:
        idx = labels == label
        p = pred[idx]
        o = obs[idx]
        denom = np.vdot(p, p)
        alpha = 0.0 + 0.0j if abs(denom) < 1e-20 else np.vdot(p, o) / denom
        corrected[idx] = alpha * p
        d = diff[idx]
        losses.append(float(np.mean(d.real**2 + d.imag**2)))
    phase_diff = np.angle(np.exp(1j * (np.angle(pred) - np.angle(obs))))
    return {
        "raw_integral_loss": float(np.mean(losses)),
        "alpha_corrected_rel_l2": float(np.linalg.norm(corrected - obs) / np.linalg.norm(obs)),
        "pred_amp_mean": float(np.mean(np.abs(pred))),
        "obs_amp_mean": float(np.mean(np.abs(obs))),
        "phase_mae_rad": float(np.mean(np.abs(phase_diff))),
    }


def scale_target(target: TargetSpec, scale: float) -> TargetSpec:
    return replace(
        target,
        circle_radius=target.circle_radius * scale,
        circle_center_y=target.circle_center_y * scale,
        circle_center_offset_x=target.circle_center_offset_x * scale,
        ring_center_y=target.ring_center_y * scale,
        ring_inner_radius=target.ring_inner_radius * scale,
        ring_outer_radius=target.ring_outer_radius * scale,
    )


def shift_target_y(target: TargetSpec, shift: float) -> TargetSpec:
    return replace(
        target,
        circle_center_y=target.circle_center_y + shift,
        ring_center_y=target.ring_center_y + shift,
    )


def geometry_variants(default: TargetSpec) -> list[tuple[str, str, float | str, TargetSpec]]:
    variants: list[tuple[str, str, float | str, TargetSpec]] = [
        ("true_default", "baseline", "default", default)
    ]
    for value in (2.5, 2.8, 3.0, 3.2, 3.5):
        variants.append((f"eps_object_{value:g}", "eps_object", value, replace(default, eps_object=value)))
    for value in (0.9, 0.95, 1.0, 1.05, 1.1):
        variants.append((f"global_scale_{value:g}", "global_scale", value, scale_target(default, value)))
    for value in (0.08, 0.09, 0.10, 0.11, 0.12):
        variants.append((f"upper_circle_radius_{value:g}", "upper_circle_radius", value, replace(default, circle_radius=value)))
    for value in (0.25, 0.275, 0.30):
        variants.append((f"ring_outer_radius_{value:g}", "ring_outer_radius", value, replace(default, ring_outer_radius=value)))
    for value in (0.15, 0.175, 0.20):
        variants.append((f"ring_inner_radius_{value:g}", "ring_inner_radius", value, replace(default, ring_inner_radius=value)))
    for value in (-0.03, 0.0, 0.03):
        variants.append((f"y_shift_{value:+g}", "y_shift", value, shift_target_y(default, value)))
    return variants


def sensitivity(rows: list[dict[str, object]], metric: str) -> list[dict[str, object]]:
    out = []
    for parameter in sorted({str(row["parameter"]) for row in rows if row["source"] == "true_variant"}):
        if parameter == "baseline":
            continue
        group = [row for row in rows if row["source"] == "true_variant" and row["parameter"] == parameter]
        values = [float(row[metric]) for row in group]
        best = min(group, key=lambda row: float(row[metric]))
        out.append(
            {
                "parameter": parameter,
                "metric": metric,
                "range": max(values) - min(values),
                "best_variant": best["variant"],
                "best_value": best[metric],
            }
        )
    return sorted(out, key=lambda row: float(row["range"]), reverse=True)


def main() -> None:
    default_target = build_target(0.0)
    config = checkpoint_config(B53000_CHECKPOINT, frequency_hz=1.0e9)
    device = resolve_device(config.device)
    dtype = resolve_dtype(config.dtype)

    print(f"frequency_hz={config.frequency_hz}")
    print(f"k0={config.k0}")
    print(f"data_dir={DATA_DIR.resolve()}")
    print(f"b53000_checkpoint={B53000_CHECKPOINT}")
    print("obs_scattered=Fieldz_total - fitted_incident")
    print("incident_phase=exp(+i k d dot r)")
    print("operator=G(+i/4 H0), source(+k0^2), contrast(epsilon-1)")
    print("mode=diagnostic_only_no_training")

    xy, obs_scattered, dirs, labels, amps, fit_rows = load_raw(config)
    quad_xy, green_re, green_im, area_weight = integral_grid_and_green(xy, default_target, config, device, dtype)

    b_model = load_model(B53000_CHECKPOINT, default_target, device, dtype)
    rows: list[dict[str, object]] = []
    for name, parameter, value, target in geometry_variants(default_target):
        model = EpsilonOverride(b_model, true_epsilon_fn(target, dtype))
        pred = predict(
            model,
            target,
            config,
            xy,
            dirs,
            labels,
            amps,
            quad_xy=quad_xy,
            green_re=green_re,
            green_im=green_im,
            area_weight=area_weight,
            device=device,
            dtype=dtype,
        )
        rows.append(
            {
                "source": "true_variant",
                "variant": name,
                "parameter": parameter,
                "value": value,
                **stats(pred, obs_scattered, labels),
            }
        )

    b_pred = predict(
        EpsilonOverride(b_model, model_epsilon_fn(b_model)),
        default_target,
        config,
        xy,
        dirs,
        labels,
        amps,
        quad_xy=quad_xy,
        green_re=green_re,
        green_im=green_im,
        area_weight=area_weight,
        device=device,
        dtype=dtype,
    )
    rows.append(
        {
            "source": "B53000_epsilon",
            "variant": "direct_B53000",
            "parameter": "checkpoint",
            "value": str(B53000_CHECKPOINT),
            **stats(b_pred, obs_scattered, labels),
        }
    )

    default_row = next(row for row in rows if row["variant"] == "true_default")
    b_row = next(row for row in rows if row["variant"] == "direct_B53000")
    true_rows = [row for row in rows if row["source"] == "true_variant"]
    best_true_raw = min(true_rows, key=lambda row: float(row["raw_integral_loss"]))
    best_true_alpha = min(true_rows, key=lambda row: float(row["alpha_corrected_rel_l2"]))
    sens_raw = sensitivity(rows, "raw_integral_loss")
    sens_alpha = sensitivity(rows, "alpha_corrected_rel_l2")

    print("\nfitted_incident_table")
    print(",".join(fit_rows[0].keys()))
    for row in fit_rows:
        print(",".join(f"{v:.8e}" if isinstance(v, float) else str(v) for v in row.values()))

    print("\ngeometry_sweep_table")
    fieldnames = list(rows[0].keys())
    print(",".join(fieldnames))
    for row in rows:
        print(",".join(f"{v:.8e}" if isinstance(v, float) else str(v) for v in row.values()))

    print("\nsensitivity_by_raw_integral_loss")
    print("parameter,range,best_variant,best_value")
    for row in sens_raw:
        print(f"{row['parameter']},{row['range']:.8e},{row['best_variant']},{float(row['best_value']):.8e}")

    print("\nsensitivity_by_alpha_corrected_rel_l2")
    print("parameter,range,best_variant,best_value")
    for row in sens_alpha:
        print(f"{row['parameter']},{row['range']:.8e},{row['best_variant']},{float(row['best_value']):.8e}")

    clearly_better_threshold = 0.95 * float(default_row["raw_integral_loss"])
    clearly_better = [
        row for row in true_rows if float(row["raw_integral_loss"]) < clearly_better_threshold
    ]
    beats_b53000_raw = [
        row for row in true_rows if float(row["raw_integral_loss"]) < float(b_row["raw_integral_loss"])
    ]
    beats_b53000_alpha = [
        row for row in true_rows if float(row["alpha_corrected_rel_l2"]) < float(b_row["alpha_corrected_rel_l2"])
    ]

    summary = {
        "default_true": default_row,
        "best_true_by_raw_integral_loss": best_true_raw,
        "best_true_by_alpha_corrected_rel_l2": best_true_alpha,
        "B53000_epsilon": b_row,
        "most_sensitive_raw_integral_loss": sens_raw[0] if sens_raw else None,
        "most_sensitive_alpha_corrected_rel_l2": sens_alpha[0] if sens_alpha else None,
        "variant_raw_loss_at_least_5pct_better_than_default": clearly_better,
        "any_variant_raw_better_than_B53000": bool(beats_b53000_raw),
        "any_variant_alpha_better_than_B53000": bool(beats_b53000_alpha),
    }
    print("\nsummary")
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    output_csv = HERE / "diagnose_1ghz_geometry_sweep.csv"
    try:
        with output_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nwrote={output_csv}")
    except PermissionError as exc:
        print(f"\ncsv_write_skipped={output_csv} reason={exc}")


if __name__ == "__main__":
    main()
