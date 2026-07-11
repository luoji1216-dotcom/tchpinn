from __future__ import annotations

import sys
from dataclasses import replace
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
    data_loss,
    load_observations,
    make_integral_tensors,
    pde_residual_loss,
    resolve_device,
    resolve_dtype,
    target_mask,
    to_observation_tensors,
    volume_integral_data_loss,
)


ROOT_DIR = Path(__file__).resolve().parent
DIRECTIONS = ("+x", "-x", "+y", "-y")
CASES = [
    {
        "name": "0.3GHz",
        "frequency_hz": 0.3e9,
        "data_dir": ROOT_DIR,
        "checkpoint": ROOT_DIR
        / "results_5_3_3_austria_0_3GHz_opt_v4_30000"
        / "checkpoint_adam_012000.pt",
    },
    {
        "name": "1GHz",
        "frequency_hz": 1.0e9,
        "data_dir": ROOT_DIR / "data_1GHz",
        "checkpoint": ROOT_DIR
        / "results_5_3_3_austria_1GHz_integral_direct_prune_C_l1_1e1"
        / "checkpoint_adam_009000.pt",
    },
]


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


def trained_epsilon_fn(model: DoubleBranchPINN) -> Callable[[torch.Tensor], torch.Tensor]:
    def fn(xy: torch.Tensor) -> torch.Tensor:
        return model.epsilon(xy)

    return fn


def torch_incident(xy, dirs, k0, amps, phase_sign):
    phase = phase_sign * k0 * torch.sum(xy * dirs, dim=1, keepdim=True)
    cos_p = torch.cos(phase)
    sin_p = torch.sin(phase)
    amp_re = amps[:, 0:1]
    amp_im = amps[:, 1:2]
    return amp_re * cos_p - amp_im * sin_p, amp_re * sin_p + amp_im * cos_p


def integral_prediction_numpy(model, obs, integral, obs_tensors, target, config, device, dtype) -> np.ndarray:
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
            scattered_quad = model.scattered_field(integral.quad_xy, quad_dirs)
            inc_re, inc_im = torch_incident(
                integral.quad_xy,
                quad_dirs,
                config.k0,
                quad_amps,
                config.incident_phase_sign,
            )
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
    return arr[:, 0] + 1j * arr[:, 1]


def fit_alpha(pred: np.ndarray, obs: np.ndarray) -> complex:
    denom = np.vdot(pred, pred)
    if abs(denom) < 1e-30:
        return 0.0 + 0.0j
    return np.vdot(pred, obs) / denom


def complex_stats(pred: np.ndarray, obs: np.ndarray) -> dict[str, float]:
    alpha = fit_alpha(pred, obs)
    corrected = alpha * pred
    phase_diff = np.angle(np.exp(1j * (np.angle(corrected) - np.angle(obs))))
    return {
        "corr_rel": float(np.linalg.norm(corrected - obs) / np.linalg.norm(obs)),
        "alpha_abs": float(abs(alpha)),
        "alpha_phase": float(np.angle(alpha)),
        "phase_mae": float(np.mean(np.abs(phase_diff))),
        "pred_amp_mean": float(np.mean(np.abs(pred))),
        "obs_amp_mean": float(np.mean(np.abs(obs))),
    }


def pde_batch(obs_tensors, n_pde: int, radius: float, device: torch.device, dtype: torch.dtype):
    theta = torch.linspace(0.0, 2.0 * np.pi, n_pde + 1, dtype=dtype, device=device)[:-1]
    r = radius * torch.sqrt(torch.linspace(1.0 / n_pde, 1.0, n_pde, dtype=dtype, device=device))
    xy = torch.stack((r * torch.cos(theta), r * torch.sin(theta)), dim=1)
    labels = torch.arange(n_pde, device=device) % obs_tensors.unique_directions.shape[0]
    return xy, obs_tensors.unique_directions[labels], obs_tensors.unique_amplitudes[labels]


def make_config(frequency_hz: float) -> TrainConfig:
    return TrainConfig(
        frequency_hz=frequency_hz,
        incident_amplitude=0.1,
        incident_phase_sign=1.0,
        observation_imag_sign=-1.0,
        estimate_incident_amplitude=False,
        eps_min=1.0,
        eps_max=4.0,
        eps_initial=1.5,
        epochs_adam=0,
        learning_rate=8.0e-4,
        device="auto",
        dtype="float32",
        max_points_per_direction=836,
        data_batch_per_direction=836,
        n_pde=1280,
        n_boundary=320,
        n_tv_grid=40,
        integral_grid_size=36,
        plot_grid_size=220,
        checkpoint_every=0,
        weight_data=0.0,
        weight_pde=0.02,
        weight_boundary=0.02,
        weight_integral_data=1.0,
        weight_tv=0.006,
        field_hidden_layers=5,
        field_hidden_units=88,
        eps_hidden_layers=5,
        eps_hidden_units=96,
        fourier_bands=5,
        fourier_max_frequency=8.0,
        random_seed=20260430,
        noise_level=0.0,
    )


def main() -> None:
    target = build_target(0.0)
    print("freq,variant,raw_integral_loss,alpha_corr_rel_l2,alpha_abs,alpha_phase,phase_mae,pred_amp_mean,obs_amp_mean,pde_loss,field_data_loss,checkpoint")
    for case in CASES:
        config = make_config(case["frequency_hz"])
        device = resolve_device(config.device)
        dtype = resolve_dtype(config.dtype)
        obs = load_observations(case["data_dir"], config, direction_labels=DIRECTIONS)
        obs_tensors = to_observation_tensors(obs, device=device, dtype=dtype)
        integral = make_integral_tensors(obs, target, config, device=device, dtype=dtype)
        if integral is None:
            raise RuntimeError("integral tensors were not created")
        model = DoubleBranchPINN(config, target).to(device=device, dtype=dtype)
        checkpoint = torch.load(case["checkpoint"], map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        model.eval()
        pde_xy, pde_dirs, pde_amps = pde_batch(obs_tensors, config.n_pde, obs.receiver_radius * 0.98, device, dtype)
        all_indices = torch.arange(obs.xy.shape[0], dtype=torch.long, device=device)
        variants = {
            "true": EpsilonOverride(model, true_epsilon_fn(target, dtype)),
            "background": EpsilonOverride(model, background_epsilon_fn(target)),
            "trained_best": EpsilonOverride(model, trained_epsilon_fn(model)),
        }
        for name, variant in variants.items():
            variant.eval()
            with torch.no_grad():
                raw = volume_integral_data_loss(
                    variant,
                    all_indices,
                    obs_tensors.directions,
                    obs_tensors.target,
                    integral,
                    obs_tensors,
                    target,
                    config,
                )
                field_data = data_loss(
                    variant,
                    obs_tensors.xy,
                    obs_tensors.directions,
                    obs_tensors.target,
                    robust=config.robust_data_weighting,
                )
                pred = integral_prediction_numpy(variant, obs, integral, obs_tensors, target, config, device, dtype)
                stats = complex_stats(pred, obs.scattered)
            with torch.enable_grad():
                pde = pde_residual_loss(variant, pde_xy, pde_dirs, pde_amps, config)
            print(
                f"{case['name']},{name},{float(raw):.8e},{stats['corr_rel']:.8e},"
                f"{stats['alpha_abs']:.8e},{stats['alpha_phase']:.8e},"
                f"{stats['phase_mae']:.8e},{stats['pred_amp_mean']:.8e},"
                f"{stats['obs_amp_mean']:.8e},{float(pde.detach()):.8e},"
                f"{float(field_data):.8e},{case['checkpoint']}"
            )


if __name__ == "__main__":
    main()
