from __future__ import annotations

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
    data_loss,
    incident_field_numpy,
    load_observations,
    make_integral_tensors,
    parse_direction_from_name,
    pde_residual_loss,
    resolve_device,
    resolve_dtype,
    set_global_seed,
    target_mask,
    to_observation_tensors,
    volume_integral_data_loss,
)


DATA_DIR = Path(__file__).resolve().parent / "data_1GHz"
CHECKPOINT = (
    Path(__file__).resolve().parent
    / "results_5_3_3_austria_1GHz_integral_direct_prune_C_l1_1e1"
    / "checkpoint_adam_009000.pt"
)
DIRECTIONS = ("+x", "-x", "+y", "-y")


class EpsilonOverride(torch.nn.Module):
    def __init__(
        self,
        base: DoubleBranchPINN,
        epsilon_fn: Callable[[torch.Tensor], torch.Tensor],
    ) -> None:
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


def c9000_epsilon_fn(model: DoubleBranchPINN) -> Callable[[torch.Tensor], torch.Tensor]:
    def fn(xy: torch.Tensor) -> torch.Tensor:
        return model.epsilon(xy)

    return fn


def pde_batch(obs_tensors, n_pde: int, radius: float, device: torch.device, dtype: torch.dtype):
    theta = torch.linspace(0.0, 2.0 * math.pi, n_pde + 1, dtype=dtype, device=device)[:-1]
    r = radius * torch.sqrt(torch.linspace(1.0 / n_pde, 1.0, n_pde, dtype=dtype, device=device))
    xy = torch.stack((r * torch.cos(theta), r * torch.sin(theta)), dim=1)
    labels = torch.arange(n_pde, device=device) % obs_tensors.unique_directions.shape[0]
    dirs = obs_tensors.unique_directions[labels]
    amps = obs_tensors.unique_amplitudes[labels]
    return xy, dirs, amps


def complex_stats(pred: np.ndarray, obs: np.ndarray) -> dict[str, float]:
    diff = pred - obs
    phase_diff = np.angle(np.exp(1j * (np.angle(pred) - np.angle(obs))))
    amp_obs = np.abs(obs)
    amp_pred = np.abs(pred)
    return {
        "amp_rel_l2": float(np.linalg.norm(amp_pred - amp_obs) / np.linalg.norm(amp_obs)),
        "complex_rel_l2": float(np.linalg.norm(diff) / np.linalg.norm(obs)),
        "phase_mae_rad": float(np.mean(np.abs(phase_diff))),
        "pred_amp_mean": float(np.mean(amp_pred)),
        "obs_amp_mean": float(np.mean(amp_obs)),
    }


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
            inc_re, inc_im = torch_incident(integral.quad_xy, quad_dirs, config.k0, quad_amps, config.incident_phase_sign)
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


def born_integral_prediction_numpy(model, obs, integral, obs_tensors, target, config, device, dtype) -> np.ndarray:
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
            inc_re, inc_im = torch_incident(
                integral.quad_xy,
                quad_dirs,
                config.k0,
                quad_amps,
                config.incident_phase_sign,
            )
            source_re = (k2_area * contrast * inc_re).squeeze(1)
            source_im = (k2_area * contrast * inc_im).squeeze(1)
            green_re = integral.green_re[mask]
            green_im = integral.green_im[mask]
            pred_re = torch.matmul(green_re, source_re) - torch.matmul(green_im, source_im)
            pred_im = torch.matmul(green_re, source_im) + torch.matmul(green_im, source_re)
            pred[mask, 0] = pred_re
            pred[mask, 1] = pred_im
    arr = pred.detach().cpu().numpy()
    return arr[:, 0] + 1j * arr[:, 1]


def born_integral_loss(model, obs, integral, obs_tensors, target, config, device, dtype) -> float:
    pred = born_integral_prediction_numpy(model, obs, integral, obs_tensors, target, config, device, dtype)
    diff = np.column_stack(((pred - obs.scattered).real, (pred - obs.scattered).imag))
    labels = obs.labels
    losses = []
    for label in DIRECTIONS:
        losses.append(float(np.mean(np.sum(diff[labels == label] ** 2, axis=1))))
    return float(np.mean(losses))


def torch_incident(xy, dirs, k0, amps, phase_sign):
    phase = phase_sign * k0 * torch.sum(xy * dirs, dim=1, keepdim=True)
    cos_p = torch.cos(phase)
    sin_p = torch.sin(phase)
    amp_re = amps[:, 0:1]
    amp_im = amps[:, 1:2]
    return amp_re * cos_p - amp_im * sin_p, amp_re * sin_p + amp_im * cos_p


def adapt_field_branch(
    *,
    label: str,
    epsilon_fn: Callable[[torch.Tensor], torch.Tensor],
    target,
    config: TrainConfig,
    obs_tensors,
    integral,
    pde_xy,
    pde_dirs,
    pde_amps,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[EpsilonOverride, dict[str, float]]:
    set_global_seed(config.random_seed)
    base = DoubleBranchPINN(config, target).to(device=device, dtype=dtype)
    for param in base.epsilon_branch.parameters():
        param.requires_grad_(False)
    model = EpsilonOverride(base, epsilon_fn).to(device=device, dtype=dtype)
    optimizer = torch.optim.Adam(base.field_branch.parameters(), lr=4.0e-4)
    all_indices = torch.arange(obs_tensors.xy.shape[0], dtype=torch.long, device=device)
    for step in range(1, 1001):
        optimizer.zero_grad(set_to_none=True)
        loss_data = data_loss(
            model,
            obs_tensors.xy,
            obs_tensors.directions,
            obs_tensors.target,
            robust=config.robust_data_weighting,
        )
        loss_integral = volume_integral_data_loss(
            model,
            all_indices,
            obs_tensors.directions,
            obs_tensors.target,
            integral,
            obs_tensors,
            target,
            config,
        )
        loss_pde = pde_residual_loss(model, pde_xy, pde_dirs, pde_amps, config)
        total = loss_data + 1000.0 * loss_integral + config.weight_pde * loss_pde
        total.backward()
        torch.nn.utils.clip_grad_norm_(base.field_branch.parameters(), config.gradient_clip_norm)
        optimizer.step()
        if step in (1, 250, 500, 750, 1000):
            print(
                f"freeze_adapt {label} step={step} total={float(total.detach()):.8e} "
                f"data={float(loss_data.detach()):.8e} integral={float(loss_integral.detach()):.8e} "
                f"pde={float(loss_pde.detach()):.8e}",
                flush=True,
            )
    with torch.no_grad():
        final_data = data_loss(
            model,
            obs_tensors.xy,
            obs_tensors.directions,
            obs_tensors.target,
            robust=config.robust_data_weighting,
        )
        final_integral = volume_integral_data_loss(
            model,
            all_indices,
            obs_tensors.directions,
            obs_tensors.target,
            integral,
            obs_tensors,
            target,
            config,
        )
    with torch.enable_grad():
        final_pde = pde_residual_loss(model, pde_xy, pde_dirs, pde_amps, config)
    return model, {
        "data": float(final_data),
        "integral": float(final_integral),
        "pde": float(final_pde.detach()),
    }


def main() -> None:
    target = build_target(0.0)
    config = TrainConfig(
        frequency_hz=1.0e9,
        incident_amplitude=0.1,
        incident_phase_sign=1.0,
        observation_imag_sign=-1.0,
        estimate_incident_amplitude=False,
        eps_min=1.0,
        eps_max=4.0,
        eps_initial=1.5,
        epochs_adam=0,
        learning_rate=1.0e-4,
        device="auto",
        dtype="float32",
        max_points_per_direction=836,
        data_batch_per_direction=836,
        n_pde=1280,
        n_boundary=320,
        n_tv_grid=40,
        integral_grid_size=36,
        plot_grid_size=220,
        log_every=100,
        checkpoint_every=0,
        weight_data=0.0,
        weight_pde=0.02,
        weight_boundary=0.02,
        weight_integral_data=1.0,
        weight_tv=0.006,
        weight_contrast_l1=0.0,
        field_hidden_layers=5,
        field_hidden_units=88,
        eps_hidden_layers=5,
        eps_hidden_units=96,
        fourier_bands=5,
        fourier_max_frequency=8.0,
        random_seed=20260430,
        noise_level=0.0,
    )
    device = resolve_device(config.device)
    dtype = resolve_dtype(config.dtype)

    print(f"frequency_hz={config.frequency_hz}")
    print(f"k0={config.k0}")
    print(f"data_dir={DATA_DIR.resolve()}")
    print(f"directions={','.join(DIRECTIONS)}")
    print(f"observation_imag_sign={config.observation_imag_sign}")
    print(f"incident_phase_sign={config.incident_phase_sign}")
    print(f"checkpoint_field_branch={CHECKPOINT}")

    obs = load_observations(DATA_DIR, config, direction_labels=DIRECTIONS)
    obs_tensors = to_observation_tensors(obs, device=device, dtype=dtype)
    integral = make_integral_tensors(obs, target, config, device=device, dtype=dtype)
    if integral is None:
        raise RuntimeError("integral tensors were not created")

    model = DoubleBranchPINN(config, target).to(device=device, dtype=dtype)
    checkpoint = torch.load(CHECKPOINT, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    all_indices = torch.arange(obs.xy.shape[0], dtype=torch.long, device=device)
    pde_xy, pde_dirs, pde_amps = pde_batch(obs_tensors, config.n_pde, obs.receiver_radius * 0.98, device, dtype)

    variants = {
        "true": EpsilonOverride(model, true_epsilon_fn(target, dtype)),
        "background": EpsilonOverride(model, background_epsilon_fn(target)),
        "C9000": EpsilonOverride(model, c9000_epsilon_fn(model)),
    }

    observed_scattered = obs.scattered
    print("\nDirection check:")
    for label in DIRECTIONS:
        path = DATA_DIR / f"{label}.txt"
        parsed_label, direction = parse_direction_from_name(path)
        idx = np.flatnonzero(obs.labels == label)
        total = obs.total[idx]
        inc = incident_field_numpy(obs.xy[idx], direction, config.k0, obs.incident_amplitudes[label], config.incident_phase_sign)
        scattered = total - inc
        print(
            f"{label}: parsed={parsed_label}, direction={direction}, "
            f"mean|total|={np.abs(total).mean():.6e}, "
            f"mean|incident|={np.abs(inc).mean():.6e}, "
            f"mean|scattered|={np.abs(scattered).mean():.6e}, "
            f"incident_amp={obs.incident_amplitudes[label].real:+.6f}{obs.incident_amplitudes[label].imag:+.6f}j"
        )

    print("\nLoss table:")
    print("variant,integral_loss,field_data_loss,pde_loss,amp_rel_l2,complex_rel_l2,phase_mae_rad,pred_amp_mean,obs_amp_mean")
    for name, variant in variants.items():
        variant.eval()
        with torch.enable_grad():
            pde_loss_value = pde_residual_loss(variant, pde_xy, pde_dirs, pde_amps, config)
        with torch.no_grad():
            integral_loss_value = volume_integral_data_loss(
                variant,
                all_indices,
                obs_tensors.directions,
                obs_tensors.target,
                integral,
                obs_tensors,
                target,
                config,
            )
            data_loss_value = data_loss(
                variant,
                obs_tensors.xy,
                obs_tensors.directions,
                obs_tensors.target,
                robust=config.robust_data_weighting,
            )
            pred = integral_prediction_numpy(variant, obs, integral, obs_tensors, target, config, device, dtype)
            stats = complex_stats(pred, observed_scattered)
        print(
            f"{name},{float(integral_loss_value):.8e},{float(data_loss_value):.8e},"
            f"{float(pde_loss_value.detach()):.8e},{stats['amp_rel_l2']:.8e},"
            f"{stats['complex_rel_l2']:.8e},{stats['phase_mae_rad']:.8e},"
            f"{stats['pred_amp_mean']:.8e},{stats['obs_amp_mean']:.8e}"
            )

    print("\nBorn integral table:")
    print("variant,born_integral_loss,complex_rel_l2,phase_mae_rad,pred_amp_mean,obs_amp_mean")
    for name, variant in variants.items():
        pred = born_integral_prediction_numpy(variant, obs, integral, obs_tensors, target, config, device, dtype)
        stats = complex_stats(pred, observed_scattered)
        print(
            f"{name},{born_integral_loss(variant, obs, integral, obs_tensors, target, config, device, dtype):.8e},"
            f"{stats['complex_rel_l2']:.8e},{stats['phase_mae_rad']:.8e},"
            f"{stats['pred_amp_mean']:.8e},{stats['obs_amp_mean']:.8e}"
        )

    print("\nPer-direction integral prediction errors:")
    for name, variant in variants.items():
        pred = integral_prediction_numpy(variant, obs, integral, obs_tensors, target, config, device, dtype)
        for label in DIRECTIONS:
            idx = np.flatnonzero(obs.labels == label)
            stats = complex_stats(pred[idx], observed_scattered[idx])
            print(
                f"{name},{label},amp_rel_l2={stats['amp_rel_l2']:.8e},"
                f"complex_rel_l2={stats['complex_rel_l2']:.8e},"
                f"phase_mae_rad={stats['phase_mae_rad']:.8e},"
                f"pred_amp_mean={stats['pred_amp_mean']:.8e},"
                f"obs_amp_mean={stats['obs_amp_mean']:.8e}"
            )

    print("\nFreeze-epsilon field adaptation:")
    print("fixed_epsilon,data_loss,integral_loss,pde_loss,complex_rel_l2,phase_mae_rad,pred_amp_mean,obs_amp_mean")
    for name, eps_fn in {
        "true": true_epsilon_fn(target, dtype),
        "C9000": c9000_epsilon_fn(model),
    }.items():
        adapted_model, losses = adapt_field_branch(
            label=name,
            epsilon_fn=eps_fn,
            target=target,
            config=config,
            obs_tensors=obs_tensors,
            integral=integral,
            pde_xy=pde_xy,
            pde_dirs=pde_dirs,
            pde_amps=pde_amps,
            device=device,
            dtype=dtype,
        )
        pred = integral_prediction_numpy(adapted_model, obs, integral, obs_tensors, target, config, device, dtype)
        stats = complex_stats(pred, observed_scattered)
        print(
            f"{name},{losses['data']:.8e},{losses['integral']:.8e},{losses['pde']:.8e},"
            f"{stats['complex_rel_l2']:.8e},{stats['phase_mae_rad']:.8e},"
            f"{stats['pred_amp_mean']:.8e},{stats['obs_amp_mean']:.8e}"
        )


if __name__ == "__main__":
    main()
