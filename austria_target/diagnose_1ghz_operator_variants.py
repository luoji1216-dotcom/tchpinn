from __future__ import annotations

import csv
import json
import math
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Callable

import numpy as np
import torch

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
    evaluate_epsilon_reconstruction,
    incident_field_torch,
    load_observations,
    make_integral_tensors,
    pde_residual_loss,
    resolve_device,
    resolve_dtype,
    set_global_seed,
    target_mask,
    to_observation_tensors,
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

    def total_field(self, xy: torch.Tensor, directions: torch.Tensor, amplitudes: torch.Tensor) -> torch.Tensor:
        return self.base.total_field(xy, directions, amplitudes)


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
        return model.epsilon(xy)

    return fn


def config_from_checkpoint(path: Path, *, frequency_hz: float | None = None) -> TrainConfig:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    raw = checkpoint.get("config", {}) if isinstance(checkpoint, dict) else {}
    valid = set(TrainConfig.__dataclass_fields__.keys())
    kwargs = {key: value for key, value in raw.items() if key in valid}
    config = TrainConfig(**kwargs)
    if frequency_hz is not None:
        config.frequency_hz = frequency_hz
    config.device = "auto"
    config.dtype = "float32"
    config.epochs_adam = 0
    config.weight_integral_data = 1.0
    config.max_points_per_direction = 836
    config.data_batch_per_direction = 836
    config.integral_grid_size = 36
    config.plot_grid_size = 220
    return config


def load_model_from_checkpoint(path: Path, target, device: torch.device, dtype: torch.dtype, *, frequency_hz: float) -> DoubleBranchPINN:
    config = config_from_checkpoint(path, frequency_hz=frequency_hz)
    model = DoubleBranchPINN(config, target).to(device=device, dtype=dtype)
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    state = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def pde_batch(obs_tensors, n_pde: int, radius: float, device: torch.device, dtype: torch.dtype):
    theta = torch.linspace(0.0, 2.0 * math.pi, n_pde + 1, dtype=dtype, device=device)[:-1]
    r = radius * torch.sqrt(torch.linspace(1.0 / n_pde, 1.0, n_pde, dtype=dtype, device=device))
    xy = torch.stack((r * torch.cos(theta), r * torch.sin(theta)), dim=1)
    labels = torch.arange(n_pde, device=device) % obs_tensors.unique_directions.shape[0]
    dirs = obs_tensors.unique_directions[labels]
    amps = obs_tensors.unique_amplitudes[labels]
    return xy, dirs, amps


def observed_complex(obs) -> np.ndarray:
    return obs.scattered.astype(np.complex128)


def operator_prediction(
    model,
    obs,
    integral,
    obs_tensors,
    target,
    config: TrainConfig,
    *,
    total_mode: str,
    detach_field: bool,
    device: torch.device,
    dtype: torch.dtype,
) -> np.ndarray:
    with torch.no_grad():
        eps = model.epsilon(integral.quad_xy)
        contrast = eps - float(target.eps_background)
        k2_area = (config.k0**2) * integral.area_weight
        pred = torch.zeros((obs.xy.shape[0], 2), dtype=dtype, device=device)
        for dir_idx in range(obs_tensors.unique_directions.shape[0]):
            direction = obs_tensors.unique_directions[dir_idx : dir_idx + 1]
            mask = torch.all(obs_tensors.directions == direction, dim=1)
            quad_dirs = direction.expand(integral.quad_xy.shape[0], 2)
            quad_amps = obs_tensors.unique_amplitudes[dir_idx : dir_idx + 1].expand(
                integral.quad_xy.shape[0], 2
            )
            inc_re, inc_im = incident_field_torch(
                integral.quad_xy,
                quad_dirs,
                config.k0,
                quad_amps,
                config.incident_phase_sign,
            )
            scattered_quad = model.scattered_field(integral.quad_xy, quad_dirs)
            if detach_field:
                scattered_quad = scattered_quad.detach()

            if total_mode == "current":
                if config.integral_internal_field_mode == "incident_only":
                    total_re, total_im = inc_re, inc_im
                else:
                    current_scattered = scattered_quad.detach() if config.integral_internal_field_mode == "detach_field" else scattered_quad
                    total_re = inc_re + current_scattered[:, 0:1]
                    total_im = inc_im + current_scattered[:, 1:2]
            elif total_mode == "incident_only":
                total_re, total_im = inc_re, inc_im
            elif total_mode == "incident_plus_scattered":
                total_re = inc_re + scattered_quad[:, 0:1]
                total_im = inc_im + scattered_quad[:, 1:2]
            elif total_mode == "scattered_only":
                total_re = scattered_quad[:, 0:1]
                total_im = scattered_quad[:, 1:2]
            else:
                raise ValueError(f"Unknown total_mode: {total_mode}")

            source_re = (k2_area * contrast * total_re).squeeze(1)
            source_im = (k2_area * contrast * total_im).squeeze(1)
            green_re = integral.green_re[mask]
            green_im = integral.green_im[mask]
            pred_re = torch.matmul(green_re, source_re) - torch.matmul(green_im, source_im)
            pred_im = torch.matmul(green_re, source_im) + torch.matmul(green_im, source_re)
            pred[mask, 0] = pred_re
            pred[mask, 1] = pred_im
    arr = pred.detach().cpu().numpy()
    return arr[:, 0].astype(np.float64) + 1j * arr[:, 1].astype(np.float64)


def raw_integral_loss(pred: np.ndarray, obs_values: np.ndarray, labels: np.ndarray) -> float:
    losses = []
    diff = pred - obs_values
    for label in DIRECTIONS:
        idx = labels == label
        losses.append(float(np.mean(diff[idx].real**2 + diff[idx].imag**2)))
    return float(np.mean(losses))


def alpha_corrected_rel_l2(pred: np.ndarray, obs_values: np.ndarray, labels: np.ndarray) -> tuple[float, dict[str, complex]]:
    corrected = np.zeros_like(pred)
    alphas: dict[str, complex] = {}
    for label in DIRECTIONS:
        idx = labels == label
        p = pred[idx]
        o = obs_values[idx]
        denom = np.vdot(p, p)
        alpha = 0.0 + 0.0j if abs(denom) < 1e-20 else np.vdot(p, o) / denom
        alphas[label] = complex(alpha)
        corrected[idx] = alpha * p
    return float(np.linalg.norm(corrected - obs_values) / np.linalg.norm(obs_values)), alphas


def stats(pred: np.ndarray, obs_values: np.ndarray, labels: np.ndarray) -> dict[str, float | str]:
    phase_diff = np.angle(np.exp(1j * (np.angle(pred) - np.angle(obs_values))))
    alpha_rel, alphas = alpha_corrected_rel_l2(pred, obs_values, labels)
    alpha_text = ";".join(
        f"{label}:{abs(alpha):.6g}@{math.degrees(math.atan2(alpha.imag, alpha.real)):.3f}deg"
        for label, alpha in alphas.items()
    )
    return {
        "raw_integral_loss": raw_integral_loss(pred, obs_values, labels),
        "alpha_corrected_complex_rel_l2": alpha_rel,
        "amplitude_rel_l2": float(np.linalg.norm(np.abs(pred) - np.abs(obs_values)) / np.linalg.norm(np.abs(obs_values))),
        "phase_mae_rad": float(np.mean(np.abs(phase_diff))),
        "pred_amp_mean": float(np.mean(np.abs(pred))),
        "obs_amp_mean": float(np.mean(np.abs(obs_values))),
        "alpha_by_direction": alpha_text,
    }


def main() -> None:
    set_global_seed(20260430)
    target = build_target(0.0)
    config = config_from_checkpoint(B53000_CHECKPOINT, frequency_hz=1.0e9)
    device = resolve_device(config.device)
    dtype = resolve_dtype(config.dtype)

    print(f"frequency_hz={config.frequency_hz}")
    print(f"k0={config.k0}")
    print(f"data_dir={DATA_DIR.resolve()}")
    print(f"b53000_checkpoint={B53000_CHECKPOINT}")
    print(f"current_integral_internal_field_mode={config.integral_internal_field_mode}")
    print("true_epsilon_use=diagnostic_only")

    obs = load_observations(DATA_DIR, config, direction_labels=DIRECTIONS)
    obs_tensors = to_observation_tensors(obs, device=device, dtype=dtype)
    integral = make_integral_tensors(obs, target, config, device=device, dtype=dtype)
    if integral is None:
        raise RuntimeError("integral tensors were not created")
    observed = observed_complex(obs)

    b53000_model = load_model_from_checkpoint(B53000_CHECKPOINT, target, device, dtype, frequency_hz=1.0e9)
    epsilon_sources: dict[str, EpsilonOverride] = {
        "true": EpsilonOverride(b53000_model, true_epsilon_fn(target, dtype)),
        "background": EpsilonOverride(b53000_model, background_epsilon_fn(target)),
        "direct_B53000": EpsilonOverride(b53000_model, model_epsilon_fn(b53000_model)),
    }

    if AUSTRIA_03_BEST.exists():
        try:
            model_03 = load_model_from_checkpoint(AUSTRIA_03_BEST, target, device, dtype, frequency_hz=0.3e9)
            epsilon_sources["austria_0_3GHz_best"] = EpsilonOverride(b53000_model, model_epsilon_fn(model_03))
            print(f"austria_0_3GHz_best_checkpoint={AUSTRIA_03_BEST}")
        except Exception as exc:
            print(f"austria_0_3GHz_best_skipped={AUSTRIA_03_BEST} reason={exc}")
    else:
        print(f"austria_0_3GHz_best_missing={AUSTRIA_03_BEST}")

    pde_xy, pde_dirs, pde_amps = pde_batch(obs_tensors, config.n_pde, obs.receiver_radius * 0.98, device, dtype)
    pde_by_source: dict[str, float] = {}
    eps_metrics_by_source: dict[str, dict[str, float]] = {}
    for name, model in epsilon_sources.items():
        with torch.enable_grad():
            pde_by_source[name] = float(pde_residual_loss(model, pde_xy, pde_dirs, pde_amps, config).detach().cpu())
        _, _, _, _, _, eps_metrics = evaluate_epsilon_reconstruction(model, target, config, device, dtype)
        eps_metrics_by_source[name] = eps_metrics

    variants = [
        ("A_current", "current", False),
        ("B_incident_only", "incident_only", False),
        ("C_incident_plus_learned_scattered", "incident_plus_scattered", False),
        ("D_learned_scattered_only", "scattered_only", False),
        ("E_detach_incident_plus_learned_scattered", "incident_plus_scattered", True),
        ("F_detach_learned_scattered_only", "scattered_only", True),
    ]

    rows = []
    for eps_name, model in epsilon_sources.items():
        for variant_name, mode, detach in variants:
            pred = operator_prediction(
                model,
                obs,
                integral,
                obs_tensors,
                target,
                config,
                total_mode=mode,
                detach_field=detach,
                device=device,
                dtype=dtype,
            )
            row = {
                "epsilon": eps_name,
                "operator_variant": variant_name,
                "pde_loss": pde_by_source[eps_name],
                **stats(pred, observed, obs.labels),
                **{f"epsilon_{key}": value for key, value in eps_metrics_by_source[eps_name].items()},
            }
            rows.append(row)

    output_csv = HERE / "diagnose_1ghz_operator_variants.csv"
    wrote_csv = False
    try:
        with output_csv.open("w", newline="", encoding="utf-8") as f:
            fieldnames = list(rows[0].keys())
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        wrote_csv = True
    except PermissionError as exc:
        print(f"csv_write_skipped={output_csv} reason={exc}")

    print("\noperator_variant_table")
    print(
        "epsilon,operator_variant,raw_integral_loss,alpha_corrected_complex_rel_l2,"
        "amplitude_rel_l2,phase_mae_rad,pred_amp_mean,obs_amp_mean,pde_loss"
    )
    for row in rows:
        print(
            f"{row['epsilon']},{row['operator_variant']},"
            f"{row['raw_integral_loss']:.8e},"
            f"{row['alpha_corrected_complex_rel_l2']:.8e},"
            f"{row['amplitude_rel_l2']:.8e},"
            f"{row['phase_mae_rad']:.8e},"
            f"{row['pred_amp_mean']:.8e},"
            f"{row['obs_amp_mean']:.8e},"
            f"{row['pde_loss']:.8e}"
        )

    print("\ntrue_vs_B53000")
    for variant_name, _, _ in variants:
        true_row = next(row for row in rows if row["epsilon"] == "true" and row["operator_variant"] == variant_name)
        b_row = next(row for row in rows if row["epsilon"] == "direct_B53000" and row["operator_variant"] == variant_name)
        print(
            f"{variant_name}: "
            f"raw_true_better={true_row['raw_integral_loss'] < b_row['raw_integral_loss']} "
            f"alpha_true_better={true_row['alpha_corrected_complex_rel_l2'] < b_row['alpha_corrected_complex_rel_l2']} "
            f"raw_true={true_row['raw_integral_loss']:.8e} raw_B53000={b_row['raw_integral_loss']:.8e} "
            f"alpha_true={true_row['alpha_corrected_complex_rel_l2']:.8e} "
            f"alpha_B53000={b_row['alpha_corrected_complex_rel_l2']:.8e}"
        )

    if wrote_csv:
        print(f"\nwrote={output_csv}")
    summary = {
        "b53000_checkpoint": str(B53000_CHECKPOINT),
        "austria_0_3GHz_best_used": str(AUSTRIA_03_BEST) if "austria_0_3GHz_best" in epsilon_sources else None,
        "config": asdict(config),
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
