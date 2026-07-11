from __future__ import annotations

import csv
import json
import math
import sys
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
AUSTRIA_03_BEST = (
    HERE
    / "results_5_3_3_austria_0_3GHz_opt_epsfreq12_eps128_12k"
    / "checkpoint_adam_012000.pt"
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
    return config


def load_model(path: Path, target, device: torch.device, dtype: torch.dtype, *, frequency_hz: float) -> DoubleBranchPINN:
    config = checkpoint_config(path, frequency_hz=frequency_hz)
    model = DoubleBranchPINN(config, target).to(device=device, dtype=dtype)
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    state = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    model.load_state_dict(state, strict=True)
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return model


def true_epsilon_fn(target, dtype: torch.dtype) -> Callable[[torch.Tensor], torch.Tensor]:
    def fn(xy: torch.Tensor) -> torch.Tensor:
        xy_np = xy.detach().cpu().numpy()
        mask = target_mask(target, xy_np[:, 0], xy_np[:, 1])
        values = np.full((xy_np.shape[0], 1), target.eps_background, dtype=np.float64)
        values[mask, 0] = target.eps_object
        return torch.as_tensor(values, dtype=dtype, device=xy.device)

    return fn


def background_epsilon_fn(target) -> Callable[[torch.Tensor], torch.Tensor]:
    def fn(xy: torch.Tensor) -> torch.Tensor:
        return torch.full((xy.shape[0], 1), float(target.eps_background), dtype=xy.dtype, device=xy.device)

    return fn


def model_epsilon_fn(model: DoubleBranchPINN) -> Callable[[torch.Tensor], torch.Tensor]:
    def fn(xy: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return model.epsilon(xy)

    return fn


def load_raw(config: TrainConfig, phase_sign: float):
    xy_all = []
    obs_all = []
    scattered_all = []
    dirs_all = []
    labels_all = []
    amps = {}
    fit_rows = []
    for label in DIRECTIONS:
        path = DATA_DIR / f"{label}.txt"
        parsed, direction = parse_direction_from_name(path)
        xy, obs = load_fem_table(path, imag_sign=config.observation_imag_sign)
        phase = np.exp(1j * phase_sign * config.k0 * (xy @ np.asarray(direction, dtype=np.float64)))
        amp = np.vdot(phase, obs) / np.vdot(phase, phase)
        fitted = amp * phase
        scattered = obs - fitted
        amps[label] = complex(amp)
        xy_all.append(xy)
        obs_all.append(obs)
        scattered_all.append(scattered)
        dirs_all.append(np.tile(np.asarray(direction, dtype=np.float64), (xy.shape[0], 1)))
        labels_all.append(np.full(xy.shape[0], label, dtype=object))
        fit_rows.append(
            {
                "incident_phase": "+" if phase_sign > 0 else "-",
                "direction": label,
                "parsed": parsed,
                "fit_amp_real": float(amp.real),
                "fit_amp_imag": float(amp.imag),
                "fit_amp_abs": float(abs(amp)),
                "obs_fit_rel_l2": float(np.linalg.norm(obs - fitted) / np.linalg.norm(obs)),
                "obs_amp_mean": float(np.mean(np.abs(obs))),
                "scattered_amp_mean": float(np.mean(np.abs(scattered))),
            }
        )
    xy_cat = np.vstack(xy_all)
    obs_cat = np.concatenate(obs_all)
    scattered_cat = np.concatenate(scattered_all)
    dirs_cat = np.vstack(dirs_all)
    labels_cat = np.concatenate(labels_all)
    return xy_cat, obs_cat, scattered_cat, dirs_cat, labels_cat, amps, fit_rows


def integral_grid_and_green(xy_obs: np.ndarray, target, config: TrainConfig, green_sign: float, device, dtype):
    n_grid = int(config.integral_grid_size)
    half = float(target.roi_half_width)
    dx = 2.0 * half / n_grid
    coords = np.linspace(-half + 0.5 * dx, half - 0.5 * dx, n_grid)
    xx, yy = np.meshgrid(coords, coords)
    quad_xy_np = np.column_stack((xx.reshape(-1), yy.reshape(-1))).astype(np.float64)
    delta = xy_obs[:, None, :] - quad_xy_np[None, :, :]
    distance = np.linalg.norm(delta, axis=2).clip(min=1e-9)
    green = green_sign * 0.25j * hankel1(0, config.k0 * distance)
    return (
        torch.as_tensor(quad_xy_np, dtype=dtype, device=device),
        torch.as_tensor(green.real, dtype=dtype, device=device),
        torch.as_tensor(green.imag, dtype=dtype, device=device),
        float(dx * dx),
    )


def torch_incident(xy: torch.Tensor, dirs: torch.Tensor, k0: float, amps: torch.Tensor, phase_sign: float):
    phase = phase_sign * k0 * torch.sum(xy * dirs, dim=1, keepdim=True)
    cos_p = torch.cos(phase)
    sin_p = torch.sin(phase)
    amp_re = amps[:, 0:1]
    amp_im = amps[:, 1:2]
    return amp_re * cos_p - amp_im * sin_p, amp_re * sin_p + amp_im * cos_p


def predict(
    model,
    target,
    config: TrainConfig,
    xy_obs: np.ndarray,
    dirs_np: np.ndarray,
    labels_np: np.ndarray,
    amps_by_label: dict[str, complex],
    *,
    phase_sign: float,
    green_sign: float,
    source_sign: float,
    contrast_mode: str,
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
        if contrast_mode == "eps_minus_1":
            contrast = eps - float(target.eps_background)
        elif contrast_mode == "one_minus_eps":
            contrast = float(target.eps_background) - eps
        else:
            raise ValueError(contrast_mode)
        k2_area = source_sign * (config.k0**2) * area_weight
        for dir_idx, label in enumerate(DIRECTIONS):
            direction = directions_t[np.flatnonzero(labels_np == label)[0] : np.flatnonzero(labels_np == label)[0] + 1]
            mask_np = labels_np == label
            mask = torch.as_tensor(mask_np, dtype=torch.bool, device=device)
            quad_dirs = direction.expand(quad_xy.shape[0], 2)
            amp = amps_by_label[label]
            quad_amps = torch.as_tensor([[amp.real, amp.imag]], dtype=dtype, device=device).expand(quad_xy.shape[0], 2)
            inc_re, inc_im = torch_incident(quad_xy, quad_dirs, config.k0, quad_amps, phase_sign)
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


def metrics(pred: np.ndarray, obs: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    diff = pred - obs
    corrected = np.zeros_like(pred)
    for label in DIRECTIONS:
        idx = labels == label
        p = pred[idx]
        o = obs[idx]
        denom = np.vdot(p, p)
        alpha = 0.0 + 0.0j if abs(denom) < 1e-20 else np.vdot(p, o) / denom
        corrected[idx] = alpha * p
    phase_diff = np.angle(np.exp(1j * (np.angle(pred) - np.angle(obs))))
    losses = []
    for label in DIRECTIONS:
        idx = labels == label
        d = diff[idx]
        losses.append(float(np.mean(d.real**2 + d.imag**2)))
    return {
        "raw_integral_loss": float(np.mean(losses)),
        "alpha_corrected_complex_rel_l2": float(np.linalg.norm(corrected - obs) / np.linalg.norm(obs)),
        "pred_amp_mean": float(np.mean(np.abs(pred))),
        "obs_scattered_amp_mean": float(np.mean(np.abs(obs))),
        "phase_mae": float(np.mean(np.abs(phase_diff))),
    }


def main() -> None:
    target = build_target(0.0)
    config = checkpoint_config(B53000_CHECKPOINT, frequency_hz=1.0e9)
    device = resolve_device(config.device)
    dtype = resolve_dtype(config.dtype)
    print(f"frequency_hz={config.frequency_hz}", flush=True)
    print(f"k0={config.k0}", flush=True)
    print(f"data_dir={DATA_DIR.resolve()}", flush=True)
    print(f"b53000_checkpoint={B53000_CHECKPOINT}", flush=True)
    print("true_epsilon_use=diagnostic_only", flush=True)

    b_model = load_model(B53000_CHECKPOINT, target, device, dtype, frequency_hz=1.0e9)
    eps_models: dict[str, EpsilonOverride] = {
        "true": EpsilonOverride(b_model, true_epsilon_fn(target, dtype)),
        "background": EpsilonOverride(b_model, background_epsilon_fn(target)),
        "direct_B53000": EpsilonOverride(b_model, model_epsilon_fn(b_model)),
    }
    if AUSTRIA_03_BEST.exists():
        try:
            model_03 = load_model(AUSTRIA_03_BEST, target, device, dtype, frequency_hz=0.3e9)
            eps_models["austria_0_3GHz_best"] = EpsilonOverride(b_model, model_epsilon_fn(model_03))
            print(f"austria_0_3GHz_best_checkpoint={AUSTRIA_03_BEST}", flush=True)
        except Exception as exc:
            print(f"austria_0_3GHz_best_skipped={AUSTRIA_03_BEST} reason={exc}", flush=True)

    rows = []
    fit_rows = []
    for phase_sign in (1.0, -1.0):
        xy, _obs_raw, obs_scattered, dirs, labels, amps, phase_fit_rows = load_raw(config, phase_sign)
        fit_rows.extend(phase_fit_rows)
        for green_sign in (1.0, -1.0):
            quad_xy, green_re, green_im, area_weight = integral_grid_and_green(
                xy, target, config, green_sign, device, dtype
            )
            for source_sign in (1.0, -1.0):
                for contrast_mode in ("eps_minus_1", "one_minus_eps"):
                    convention = {
                        "incident_phase": "+" if phase_sign > 0 else "-",
                        "green_sign": "+i/4" if green_sign > 0 else "-i/4",
                        "source_sign": "+k0^2" if source_sign > 0 else "-k0^2",
                        "contrast": "epsilon-1" if contrast_mode == "eps_minus_1" else "1-epsilon",
                    }
                    for eps_name, model in eps_models.items():
                        pred = predict(
                            model,
                            target,
                            config,
                            xy,
                            dirs,
                            labels,
                            amps,
                            phase_sign=phase_sign,
                            green_sign=green_sign,
                            source_sign=source_sign,
                            contrast_mode=contrast_mode,
                            quad_xy=quad_xy,
                            green_re=green_re,
                            green_im=green_im,
                            area_weight=area_weight,
                            device=device,
                            dtype=dtype,
                        )
                        rows.append({"epsilon": eps_name, **convention, **metrics(pred, obs_scattered, labels)})

    print("\nfitted_incident_table")
    print(",".join(fit_rows[0].keys()))
    for row in fit_rows:
        print(",".join(f"{v:.8e}" if isinstance(v, float) else str(v) for v in row.values()))

    print("\nconvention_table")
    print(",".join(rows[0].keys()))
    for row in rows:
        print(",".join(f"{v:.8e}" if isinstance(v, float) else str(v) for v in row.values()))

    print("\ntop10_by_raw_integral_loss")
    ranked = sorted(rows, key=lambda row: (row["raw_integral_loss"], row["alpha_corrected_complex_rel_l2"]))
    print(",".join(ranked[0].keys()))
    for row in ranked[:10]:
        print(",".join(f"{v:.8e}" if isinstance(v, float) else str(v) for v in row.values()))

    print("\ntrue_vs_B53000_by_convention")
    any_true_better = False
    convention_keys = ("incident_phase", "green_sign", "source_sign", "contrast")
    for phase in ("+", "-"):
        for green in ("+i/4", "-i/4"):
            for source in ("+k0^2", "-k0^2"):
                for contrast in ("epsilon-1", "1-epsilon"):
                    def match(row):
                        return (
                            row["incident_phase"] == phase
                            and row["green_sign"] == green
                            and row["source_sign"] == source
                            and row["contrast"] == contrast
                        )

                    true_row = next(row for row in rows if row["epsilon"] == "true" and match(row))
                    b_row = next(row for row in rows if row["epsilon"] == "direct_B53000" and match(row))
                    raw_better = true_row["raw_integral_loss"] < b_row["raw_integral_loss"]
                    alpha_better = (
                        true_row["alpha_corrected_complex_rel_l2"]
                        < b_row["alpha_corrected_complex_rel_l2"]
                    )
                    any_true_better = any_true_better or raw_better or alpha_better
                    print(
                        f"phase={phase},green={green},source={source},contrast={contrast}: "
                        f"raw_true_better={raw_better} alpha_true_better={alpha_better} "
                        f"raw_true={true_row['raw_integral_loss']:.8e} raw_B53000={b_row['raw_integral_loss']:.8e} "
                        f"alpha_true={true_row['alpha_corrected_complex_rel_l2']:.8e} "
                        f"alpha_B53000={b_row['alpha_corrected_complex_rel_l2']:.8e}"
                    )

    true_rows = [row for row in rows if row["epsilon"] == "true"]
    best_true = min(true_rows, key=lambda row: (row["raw_integral_loss"], row["alpha_corrected_complex_rel_l2"]))
    print("\nbest_true_convention")
    print(json.dumps(best_true, indent=2, ensure_ascii=False))
    print(f"any_convention_true_better_than_B53000={any_true_better}")

    recommended = min(
        [row for row in rows if row["epsilon"] in ("true", "direct_B53000")],
        key=lambda row: (row["raw_integral_loss"], row["alpha_corrected_complex_rel_l2"]),
    )
    print("\nrecommended_by_lowest_true_or_B53000_raw_loss")
    print(json.dumps(recommended, indent=2, ensure_ascii=False))

    output_csv = HERE / "diagnose_1ghz_operator_convention_sweep.csv"
    try:
        with output_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nwrote={output_csv}")
    except PermissionError as exc:
        print(f"\ncsv_write_skipped={output_csv} reason={exc}")


if __name__ == "__main__":
    main()
