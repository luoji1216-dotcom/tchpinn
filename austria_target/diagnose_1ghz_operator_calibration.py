from __future__ import annotations

import itertools
import math
import sys
from pathlib import Path

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
    incident_field_numpy,
    load_fem_table,
    parse_direction_from_name,
    target_mask,
)


DATA_DIR = Path(__file__).resolve().parent / "data_1GHz"
CHECKPOINT = (
    Path(__file__).resolve().parent
    / "results_5_3_3_austria_1GHz_integral_direct_prune_C_l1_1e1"
    / "checkpoint_adam_009000.pt"
)
DIRECTIONS = ("+x", "-x", "+y", "-y")


def load_case(*, imag_sign: float, phase_sign: float, amplitude: float, k0: float):
    xy_all = []
    obs_all = []
    dirs_all = []
    labels_all = []
    for filename in ("+x.txt", "-x.txt", "+y.txt", "-y.txt"):
        path = DATA_DIR / filename
        label, direction = parse_direction_from_name(path)
        xy, total = load_fem_table(path, imag_sign=imag_sign)
        keep = np.linspace(0, xy.shape[0] - 1, 836).round().astype(int)
        xy = xy[keep]
        total = total[keep]
        inc = incident_field_numpy(xy, direction, k0, complex(amplitude, 0.0), phase_sign)
        xy_all.append(xy)
        obs_all.append(total - inc)
        dirs_all.append(np.tile(np.asarray(direction, dtype=np.float64), (xy.shape[0], 1)))
        labels_all.append(np.full(xy.shape[0], label, dtype=object))
    return (
        np.vstack(xy_all),
        np.concatenate(obs_all),
        np.vstack(dirs_all),
        np.concatenate(labels_all),
    )


def quad_grid(target, n_grid: int = 36):
    half = target.roi_half_width
    dx = 2.0 * half / n_grid
    coords = np.linspace(-half + 0.5 * dx, half - 0.5 * dx, n_grid)
    xx, yy = np.meshgrid(coords, coords)
    xy = np.column_stack((xx.reshape(-1), yy.reshape(-1)))
    return xy, dx * dx


def epsilon_values(name: str, target, quad_xy: np.ndarray, model: DoubleBranchPINN | None = None) -> np.ndarray:
    if name == "background":
        return np.full(quad_xy.shape[0], target.eps_background, dtype=np.float64)
    if name == "true":
        eps = np.full(quad_xy.shape[0], target.eps_background, dtype=np.float64)
        mask = target_mask(target, quad_xy[:, 0], quad_xy[:, 1])
        eps[mask] = target.eps_object
        return eps
    if name == "C9000":
        if model is None:
            raise ValueError("C9000 epsilon requires model")
        with torch.no_grad():
            tensor = torch.as_tensor(quad_xy, dtype=torch.float32)
            return model.epsilon(tensor).detach().cpu().numpy().reshape(-1).astype(np.float64)
    raise ValueError(name)


def model_scattered_on_quad(model: DoubleBranchPINN, quad_xy: np.ndarray, direction: tuple[float, float]) -> np.ndarray:
    with torch.no_grad():
        xy_t = torch.as_tensor(quad_xy, dtype=torch.float32)
        dirs_t = torch.as_tensor(np.tile(np.asarray(direction, dtype=np.float64), (quad_xy.shape[0], 1)), dtype=torch.float32)
        pred = model.scattered_field(xy_t, dirs_t).detach().cpu().numpy()
    return pred[:, 0] + 1j * pred[:, 1]


def predict_integral(
    *,
    mode: str,
    variant: str,
    obs_xy: np.ndarray,
    obs_dirs: np.ndarray,
    labels: np.ndarray,
    target,
    quad_xy: np.ndarray,
    area_weight: float,
    k0: float,
    phase_sign: float,
    amplitude: float,
    green_sign: float,
    source_sign: float,
    model: DoubleBranchPINN,
) -> np.ndarray:
    eps = epsilon_values(variant, target, quad_xy, model)
    contrast = eps - target.eps_background
    pred = np.zeros(obs_xy.shape[0], dtype=np.complex128)
    coeff = green_sign * 0.25j
    source_coeff = source_sign * (k0**2) * area_weight
    for label in DIRECTIONS:
        direction = parse_direction_from_name(DATA_DIR / f"{label}.txt")[1]
        mask = labels == label
        inc_quad = incident_field_numpy(quad_xy, direction, k0, complex(amplitude, 0.0), phase_sign)
        if mode == "born":
            total_quad = inc_quad
        elif mode == "volume_c9000_field":
            total_quad = inc_quad + model_scattered_on_quad(model, quad_xy, direction)
        else:
            raise ValueError(mode)
        source = source_coeff * contrast * total_quad
        distance = np.linalg.norm(obs_xy[mask, None, :] - quad_xy[None, :, :], axis=2).clip(min=1e-9)
        green = coeff * hankel1(0, k0 * distance)
        pred[mask] = green @ source
    return pred


def fit_alpha(pred: np.ndarray, obs: np.ndarray) -> complex:
    denom = np.vdot(pred, pred)
    if abs(denom) < 1e-30:
        return 0.0 + 0.0j
    return np.vdot(pred, obs) / denom


def metrics(pred: np.ndarray, obs: np.ndarray) -> dict[str, float]:
    raw_rel = float(np.linalg.norm(pred - obs) / np.linalg.norm(obs))
    alpha = fit_alpha(pred, obs)
    corrected = alpha * pred
    corr_rel = float(np.linalg.norm(corrected - obs) / np.linalg.norm(obs))
    phase_diff = np.angle(np.exp(1j * (np.angle(corrected) - np.angle(obs))))
    return {
        "raw_rel": raw_rel,
        "corr_rel": corr_rel,
        "alpha_abs": float(abs(alpha)),
        "alpha_phase": float(np.angle(alpha)),
        "phase_mae": float(np.mean(np.abs(phase_diff))),
        "pred_amp_mean": float(np.mean(np.abs(pred))),
        "obs_amp_mean": float(np.mean(np.abs(obs))),
    }


def main() -> None:
    target = build_target(0.0)
    k0 = 2.0 * math.pi * 1.0e9 / 3.0e8
    quad_xy, area_weight = quad_grid(target)
    config = TrainConfig(
        frequency_hz=1.0e9,
        incident_amplitude=0.1,
        eps_min=1.0,
        eps_max=4.0,
        eps_initial=1.5,
        field_hidden_layers=5,
        field_hidden_units=88,
        eps_hidden_layers=5,
        eps_hidden_units=96,
        fourier_bands=5,
        fourier_max_frequency=8.0,
    )
    model = DoubleBranchPINN(config, target)
    checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    print(f"data_dir={DATA_DIR.resolve()}")
    print(f"k0={k0}")
    print("mode,variant,obs_sign,phase_sign,green,source,amp,raw_rel,corr_rel,alpha_abs,alpha_phase,phase_mae,pred_amp_mean,obs_amp_mean,true_corr_better_than_background")
    rows = []
    for mode in ("born", "volume_c9000_field"):
        for obs_sign, phase_sign, green_sign, source_sign, amplitude in itertools.product(
            (1.0, -1.0),
            (1.0, -1.0),
            (1.0, -1.0),
            (1.0, -1.0),
            (0.1, 1.0),
        ):
            obs_xy, obs_scattered, obs_dirs, labels = load_case(
                imag_sign=obs_sign,
                phase_sign=phase_sign,
                amplitude=amplitude,
                k0=k0,
            )
            combo_metrics = {}
            for variant in ("true", "background", "C9000"):
                pred = predict_integral(
                    mode=mode,
                    variant=variant,
                    obs_xy=obs_xy,
                    obs_dirs=obs_dirs,
                    labels=labels,
                    target=target,
                    quad_xy=quad_xy,
                    area_weight=area_weight,
                    k0=k0,
                    phase_sign=phase_sign,
                    amplitude=amplitude,
                    green_sign=green_sign,
                    source_sign=source_sign,
                    model=model,
                )
                combo_metrics[variant] = metrics(pred, obs_scattered)
            true_better = combo_metrics["true"]["corr_rel"] < combo_metrics["background"]["corr_rel"]
            for variant in ("true", "background", "C9000"):
                m = combo_metrics[variant]
                row = (
                    mode,
                    variant,
                    obs_sign,
                    phase_sign,
                    "i/4" if green_sign > 0 else "-i/4",
                    "+k0^2" if source_sign > 0 else "-k0^2",
                    amplitude,
                    m["raw_rel"],
                    m["corr_rel"],
                    m["alpha_abs"],
                    m["alpha_phase"],
                    m["phase_mae"],
                    m["pred_amp_mean"],
                    m["obs_amp_mean"],
                    true_better,
                )
                rows.append(row)
                print(",".join(str(x) for x in row))

    for mode in ("born", "volume_c9000_field"):
        true_rows = [r for r in rows if r[0] == mode and r[1] == "true"]
        best_true = min(true_rows, key=lambda r: r[8])
        c9000_rows = [r for r in rows if r[0] == mode and r[1] == "C9000"]
        best_c9000 = min(c9000_rows, key=lambda r: r[8])
        bg_rows = [r for r in rows if r[0] == mode and r[1] == "background"]
        best_bg = min(bg_rows, key=lambda r: r[8])
        print(
            f"BEST,{mode},true_corr={best_true[8]:.8e},true_raw={best_true[7]:.8e},"
            f"true_alpha_abs={best_true[9]:.8e},true_alpha_phase={best_true[10]:.8e},"
            f"obs_sign={best_true[2]},phase_sign={best_true[3]},green={best_true[4]},"
            f"source={best_true[5]},amp={best_true[6]},"
            f"best_bg_corr={best_bg[8]:.8e},best_c9000_corr={best_c9000[8]:.8e}"
        )


if __name__ == "__main__":
    main()
