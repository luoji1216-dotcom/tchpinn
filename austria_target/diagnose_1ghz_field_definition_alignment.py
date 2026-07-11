from __future__ import annotations

import csv
import json
import math
import sys
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
    incident_field_torch,
    load_fem_table,
    make_integral_tensors,
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


class ObservationLike:
    def __init__(self, xy, target_field, labels, directions, receiver_radius, direction_labels, incident_amplitudes):
        self.xy = xy
        self.scattered = target_field
        self.labels = labels
        self.directions = directions
        self.receiver_radius = receiver_radius
        self.direction_labels = direction_labels
        self.incident_amplitudes = incident_amplitudes


class ObservationTensorsLike:
    def __init__(self, obs: ObservationLike, device: torch.device, dtype: torch.dtype):
        self.xy = torch.as_tensor(obs.xy, dtype=dtype, device=device)
        self.target = torch.as_tensor(
            np.column_stack((obs.scattered.real, obs.scattered.imag)),
            dtype=dtype,
            device=device,
        )
        self.directions = torch.as_tensor(obs.directions, dtype=dtype, device=device)
        self.indices_by_label = {}
        unique_dirs = []
        unique_amps = []
        for label in obs.direction_labels:
            idx = np.flatnonzero(obs.labels == label)
            self.indices_by_label[label] = torch.as_tensor(idx, dtype=torch.long, device=device)
            unique_dirs.append(obs.directions[idx[0]])
            amp = obs.incident_amplitudes[label]
            unique_amps.append([amp.real, amp.imag])
        self.unique_directions = torch.as_tensor(np.vstack(unique_dirs), dtype=dtype, device=device)
        self.unique_amplitudes = torch.as_tensor(np.vstack(unique_amps), dtype=dtype, device=device)
        self.receiver_radius = obs.receiver_radius


class EpsilonOverride(torch.nn.Module):
    def __init__(self, base: DoubleBranchPINN, epsilon_fn: Callable[[torch.Tensor], torch.Tensor]) -> None:
        super().__init__()
        self.base = base
        self.epsilon_fn = epsilon_fn

    def epsilon(self, xy: torch.Tensor) -> torch.Tensor:
        return self.epsilon_fn(xy)

    def scattered_field(self, xy: torch.Tensor, directions: torch.Tensor) -> torch.Tensor:
        return self.base.scattered_field(xy, directions)


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


def checkpoint_config(path: Path) -> TrainConfig:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    raw = checkpoint.get("config", {}) if isinstance(checkpoint, dict) else {}
    valid = set(TrainConfig.__dataclass_fields__.keys())
    kwargs = {key: value for key, value in raw.items() if key in valid}
    config = TrainConfig(**kwargs)
    config.frequency_hz = 1.0e9
    config.device = "auto"
    config.dtype = "float32"
    config.weight_integral_data = 1.0
    config.max_points_per_direction = 0
    config.data_batch_per_direction = 0
    config.integral_grid_size = 36
    config.integral_internal_field_mode = "detach_field"
    return config


def load_b53000(target, device: torch.device, dtype: torch.dtype, config: TrainConfig) -> DoubleBranchPINN:
    model = DoubleBranchPINN(config, target).to(device=device, dtype=dtype)
    checkpoint = torch.load(B53000_CHECKPOINT, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    return model


def fit_incident(xy: np.ndarray, obs: np.ndarray, direction, k0: float) -> tuple[complex, np.ndarray, float]:
    phase = np.exp(1j * k0 * (xy @ np.asarray(direction, dtype=np.float64)))
    amp = np.vdot(phase, obs) / np.vdot(phase, phase)
    fitted = amp * phase
    rel = float(np.linalg.norm(obs - fitted) / np.linalg.norm(obs))
    return complex(amp), fitted, rel


def load_raw_observations(config: TrainConfig):
    xy_all = []
    obs_all = []
    dirs_all = []
    labels_all = []
    fitted_all = []
    amps = {}
    rows = []
    for label in DIRECTIONS:
        path = DATA_DIR / f"{label}.txt"
        parsed, direction = parse_direction_from_name(path)
        xy, obs = load_fem_table(path, imag_sign=config.observation_imag_sign)
        amp, fitted, rel = fit_incident(xy, obs, direction, config.k0)
        residual = obs - fitted
        amps[label] = amp
        xy_all.append(xy)
        obs_all.append(obs)
        fitted_all.append(fitted)
        dirs_all.append(np.tile(np.asarray(direction, dtype=np.float64), (xy.shape[0], 1)))
        labels_all.append(np.full(xy.shape[0], label, dtype=object))
        residual_mean = float(np.mean(np.abs(residual)))
        obs_mean = float(np.mean(np.abs(obs)))
        likeness = "total_like" if rel < 0.75 and residual_mean < obs_mean else "scattered_like"
        rows.append(
            {
                "direction": label,
                "parsed": parsed,
                "fit_amp_real": amp.real,
                "fit_amp_imag": amp.imag,
                "obs_amp_mean": obs_mean,
                "obs_amp_max": float(np.max(np.abs(obs))),
                "fitted_incident_amp_mean": float(np.mean(np.abs(fitted))),
                "fitted_incident_amp_max": float(np.max(np.abs(fitted))),
                "obs_vs_fitted_incident_rel_l2": rel,
                "obs_minus_fitted_incident_amp_mean": residual_mean,
                "obs_minus_fitted_incident_amp_max": float(np.max(np.abs(residual))),
                "likeness": likeness,
            }
        )
    xy_cat = np.vstack(xy_all)
    obs_cat = np.concatenate(obs_all)
    fitted_cat = np.concatenate(fitted_all)
    dirs_cat = np.vstack(dirs_all)
    labels_cat = np.concatenate(labels_all)
    receiver_radius = float(np.median(np.linalg.norm(xy_cat, axis=1)))
    return xy_cat, obs_cat, fitted_cat, dirs_cat, labels_cat, receiver_radius, amps, rows


def make_obs_like(xy, field, labels, directions, receiver_radius, amps) -> ObservationLike:
    return ObservationLike(
        xy=xy,
        target_field=field,
        labels=labels,
        directions=directions,
        receiver_radius=receiver_radius,
        direction_labels=list(DIRECTIONS),
        incident_amplitudes=amps,
    )


def operator_prediction(model, obs, integral, obs_tensors, target, config: TrainConfig, device, dtype) -> np.ndarray:
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
            scattered_quad = model.scattered_field(integral.quad_xy, quad_dirs).detach()
            total_re = inc_re + scattered_quad[:, 0:1]
            total_im = inc_im + scattered_quad[:, 1:2]
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


def alpha_corrected_rel_l2(pred: np.ndarray, obs_values: np.ndarray, labels: np.ndarray) -> float:
    corrected = np.zeros_like(pred)
    for label in DIRECTIONS:
        idx = labels == label
        p = pred[idx]
        o = obs_values[idx]
        denom = np.vdot(p, p)
        alpha = 0.0 + 0.0j if abs(denom) < 1e-20 else np.vdot(p, o) / denom
        corrected[idx] = alpha * p
    return float(np.linalg.norm(corrected - obs_values) / np.linalg.norm(obs_values))


def compare_epsilons(target, config, device, dtype, xy, obs_field, labels, dirs, radius, amps, b_model):
    obs_like = make_obs_like(xy, obs_field, labels, dirs, radius, amps)
    obs_tensors = ObservationTensorsLike(obs_like, device, dtype)
    integral = make_integral_tensors(obs_like, target, config, device=device, dtype=dtype)
    if integral is None:
        raise RuntimeError("integral tensors were not created")
    eps_models = {
        "true": EpsilonOverride(b_model, true_epsilon_fn(target, dtype)),
        "background": EpsilonOverride(b_model, background_epsilon_fn(target)),
        "direct_B53000": EpsilonOverride(b_model, lambda xy_t: b_model.epsilon(xy_t)),
    }
    rows = []
    for name, model in eps_models.items():
        pred = operator_prediction(model, obs_like, integral, obs_tensors, target, config, device, dtype)
        rows.append(
            {
                "epsilon": name,
                "raw_integral_loss": raw_integral_loss(pred, obs_field, labels),
                "alpha_corrected_rel_l2": alpha_corrected_rel_l2(pred, obs_field, labels),
                "pred_amp_mean": float(np.mean(np.abs(pred))),
                "obs_amp_mean": float(np.mean(np.abs(obs_field))),
            }
        )
    return rows


def print_table(title: str, rows: list[dict]) -> None:
    print(f"\n{title}")
    if not rows:
        return
    print(",".join(rows[0].keys()))
    for row in rows:
        parts = []
        for key, value in row.items():
            if isinstance(value, float):
                parts.append(f"{value:.8e}")
            else:
                parts.append(str(value))
        print(",".join(parts))


def main() -> None:
    target = build_target(0.0)
    config = checkpoint_config(B53000_CHECKPOINT)
    device = resolve_device(config.device)
    dtype = resolve_dtype(config.dtype)
    print(f"frequency_hz={config.frequency_hz}")
    print(f"k0={config.k0}")
    print(f"data_dir={DATA_DIR.resolve()}")
    print(f"b53000_checkpoint={B53000_CHECKPOINT}")
    print("true_epsilon_use=diagnostic_only")

    xy, obs_raw, fitted_incident, dirs, labels, radius, fitted_amps, definition_rows = load_raw_observations(config)
    obs_scattered = obs_raw - fitted_incident
    b_model = load_b53000(target, device, dtype, config)

    raw_rows = compare_epsilons(target, config, device, dtype, xy, obs_raw, labels, dirs, radius, fitted_amps, b_model)
    scattered_rows = compare_epsilons(
        target,
        config,
        device,
        dtype,
        xy,
        obs_scattered,
        labels,
        dirs,
        radius,
        fitted_amps,
        b_model,
    )

    print_table("field_definition_table", definition_rows)
    print_table("operator_sort_obs_raw", raw_rows)
    print_table("operator_sort_obs_minus_fitted_incident", scattered_rows)

    print("\ntrue_vs_B53000")
    for label, rows in (("obs_raw", raw_rows), ("obs_scattered", scattered_rows)):
        true_row = next(row for row in rows if row["epsilon"] == "true")
        b_row = next(row for row in rows if row["epsilon"] == "direct_B53000")
        print(
            f"{label}: "
            f"raw_true_better={true_row['raw_integral_loss'] < b_row['raw_integral_loss']} "
            f"alpha_true_better={true_row['alpha_corrected_rel_l2'] < b_row['alpha_corrected_rel_l2']} "
            f"raw_true={true_row['raw_integral_loss']:.8e} raw_B53000={b_row['raw_integral_loss']:.8e} "
            f"alpha_true={true_row['alpha_corrected_rel_l2']:.8e} "
            f"alpha_B53000={b_row['alpha_corrected_rel_l2']:.8e}"
        )

    output_csv = HERE / "diagnose_1ghz_field_definition_alignment.csv"
    try:
        with output_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["section_json", json.dumps({"field_definition": definition_rows}, ensure_ascii=False)])
            writer.writerow(["section_json", json.dumps({"operator_sort_obs_raw": raw_rows}, ensure_ascii=False)])
            writer.writerow(["section_json", json.dumps({"operator_sort_obs_scattered": scattered_rows}, ensure_ascii=False)])
        print(f"\nwrote={output_csv}")
    except PermissionError as exc:
        print(f"\ncsv_write_skipped={output_csv} reason={exc}")


if __name__ == "__main__":
    main()
