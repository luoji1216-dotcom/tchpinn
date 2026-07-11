from __future__ import annotations

import argparse
import csv
import json
import random
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
from torch import nn

from pinn_pixel_inverse_core import (
    DoubleBranchPINN,
    TargetSpec,
    TrainConfig,
    compute_loss_terms,
    data_loss,
    epsilon_metrics,
    load_observations,
    make_integral_tensors,
    reconstruct_epsilon,
    sample_boundary_points,
    sample_collocation_points,
    sample_direction_batch,
    sample_directions,
    sample_observation_batch,
    sommerfeld_boundary_loss,
    pde_residual_loss,
    to_observation_tensors,
    true_epsilon_grid,
    volume_integral_data_loss,
)


class FixedMaskEpsilonModel(nn.Module):
    def __init__(
        self,
        base: DoubleBranchPINN,
        mask: np.ndarray,
        target: TargetSpec,
        amplitude: float,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        self.field_branch = base.field_branch
        self.register_buffer("mask", torch.as_tensor(mask.astype(np.float32), dtype=dtype, device=device))
        self.roi_half_width = float(target.roi_half_width)
        self.eps_background = float(target.eps_background)
        self.amplitude = float(amplitude)
        self.n_grid = int(mask.shape[0])

    def scattered_field(self, xy: torch.Tensor, directions: torch.Tensor) -> torch.Tensor:
        return self.field_branch(xy, directions)

    def epsilon(self, xy: torch.Tensor) -> torch.Tensor:
        scale = self.n_grid - 1
        u = (xy[:, 0] + self.roi_half_width) / (2.0 * self.roi_half_width) * scale
        v = (xy[:, 1] + self.roi_half_width) / (2.0 * self.roi_half_width) * scale
        ix = torch.clamp(torch.round(u).long(), 0, self.n_grid - 1)
        iy = torch.clamp(torch.round(v).long(), 0, self.n_grid - 1)
        m = self.mask[iy, ix].reshape(-1, 1)
        return torch.ones_like(m) * self.eps_background + self.amplitude * m


class FixedCheckpointEpsilonModel(nn.Module):
    def __init__(self, base: DoubleBranchPINN) -> None:
        super().__init__()
        self.field_branch = base.field_branch
        self.epsilon_branch = base.epsilon_branch
        for param in self.epsilon_branch.parameters():
            param.requires_grad_(False)

    def scattered_field(self, xy: torch.Tensor, directions: torch.Tensor) -> torch.Tensor:
        return self.field_branch(xy, directions)

    def epsilon(self, xy: torch.Tensor) -> torch.Tensor:
        return self.epsilon_branch(xy)


def load_run_config(checkpoint_path: Path) -> Tuple[TrainConfig, TargetSpec, Tuple[str, ...]]:
    with (checkpoint_path.parent / "run_config.json").open("r", encoding="utf-8") as f:
        payload = json.load(f)
    config = TrainConfig(**payload["config"])
    target = TargetSpec(**payload["target"])
    labels = tuple(payload.get("direction_labels", ["+x", "-x"]))
    return config, target, labels


def make_model(
    checkpoint_path: Path,
    config: TrainConfig,
    target: TargetSpec,
    device: torch.device,
    dtype: torch.dtype,
) -> DoubleBranchPINN:
    model = DoubleBranchPINN(config, target).to(device=device, dtype=dtype)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    model.load_state_dict(state, strict=not config.learn_integral_alpha)
    return model


def full_losses(
    model: nn.Module,
    obs_tensors,
    integral_tensors,
    target: TargetSpec,
    config: TrainConfig,
    device: torch.device,
    dtype: torch.dtype,
) -> Dict[str, float]:
    data_indices = torch.arange(obs_tensors.xy.shape[0], device=device)
    radius = float(obs_tensors.receiver_radius)
    pde_xy = sample_collocation_points(config.n_pde, target, radius, device, dtype)
    pde_dirs, pde_amps = sample_direction_batch(
        obs_tensors.unique_directions, obs_tensors.unique_amplitudes, config.n_pde
    )
    bc_xy = sample_boundary_points(config.n_boundary, radius, device, dtype)
    bc_dirs = sample_directions(obs_tensors.unique_directions, config.n_boundary)
    losses = {
        "data": data_loss(model, obs_tensors.xy, obs_tensors.directions, obs_tensors.target, config.robust_data_weighting),
        "integral_data": volume_integral_data_loss(
            model,
            data_indices,
            obs_tensors.directions,
            obs_tensors.target,
            integral_tensors,
            obs_tensors,
            target,
            config,
        ),
        "pde": pde_residual_loss(model, pde_xy, pde_dirs, pde_amps, config),
        "boundary": sommerfeld_boundary_loss(model, bc_xy, bc_dirs, config),
    }
    total = (
        config.weight_data * losses["data"]
        + config.weight_integral_data * losses["integral_data"]
        + config.weight_pde * losses["pde"]
        + config.weight_boundary * losses["boundary"]
    )
    losses["total"] = total
    return {key: float(value.detach().cpu()) for key, value in losses.items()}


def run_case(
    *,
    name: str,
    fixed_model: nn.Module,
    config: TrainConfig,
    target: TargetSpec,
    obs_tensors,
    integral_tensors,
    output_dir: Path,
    epochs: int,
    log_every: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Dict[str, float]:
    for param in fixed_model.parameters():
        param.requires_grad_(False)
    for param in fixed_model.field_branch.parameters():
        param.requires_grad_(True)

    optimizer = torch.optim.Adam(fixed_model.field_branch.parameters(), lr=config.learning_rate)
    history = []
    fixed_model.train()
    radius = float(obs_tensors.receiver_radius)

    initial = full_losses(fixed_model, obs_tensors, integral_tensors, target, config, device, dtype)
    print(f"[{name}] initial {initial}", flush=True)

    for step in range(1, epochs + 1):
        optimizer.zero_grad(set_to_none=True)
        data_indices, data_xy, data_dirs, _data_amps, data_target = sample_observation_batch(
            obs_tensors, config.data_batch_per_direction
        )
        pde_xy = sample_collocation_points(config.n_pde, target, radius, device, dtype)
        pde_dirs, pde_amps = sample_direction_batch(
            obs_tensors.unique_directions, obs_tensors.unique_amplitudes, config.n_pde
        )
        bc_xy = sample_boundary_points(config.n_boundary, radius, device, dtype)
        bc_dirs = sample_directions(obs_tensors.unique_directions, config.n_boundary)
        losses = compute_loss_terms(
            fixed_model,
            data_xy=data_xy,
            data_dirs=data_dirs,
            data_target=data_target,
            data_indices=data_indices,
            pde_xy=pde_xy,
            pde_dirs=pde_dirs,
            pde_amps=pde_amps,
            bc_xy=bc_xy,
            bc_dirs=bc_dirs,
            integral_tensors=integral_tensors,
            obs_tensors=obs_tensors,
            target=target,
            config=config,
            device=device,
            dtype=dtype,
        )
        total = losses["total"]
        total.backward()
        if config.gradient_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(fixed_model.field_branch.parameters(), config.gradient_clip_norm)
        optimizer.step()

        if step == 1 or step % log_every == 0 or step == epochs:
            row = {
                "step": step,
                "total": float(total.detach().cpu()),
                "data": float(losses["data"].detach().cpu()),
                "integral_data": float(losses["integral_data"].detach().cpu()),
                "pde": float(losses["pde"].detach().cpu()),
                "boundary": float(losses["boundary"].detach().cpu()),
            }
            history.append(row)
            print(f"[{name}] step {step}: {row}", flush=True)

    final = full_losses(fixed_model, obs_tensors, integral_tensors, target, config, device, dtype)
    print(f"[{name}] final {final}", flush=True)

    with (output_dir / f"{name}_history.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["step", "total", "data", "integral_data", "pde", "boundary"])
        writer.writeheader()
        writer.writerows(history)
    torch.save({"model": fixed_model.state_dict(), "config": asdict(config), "target": asdict(target)}, output_dir / f"{name}_field_refit.pt")

    return {
        **{f"initial_{key}": value for key, value in initial.items()},
        **{f"final_{key}": value for key, value in final.items()},
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Square 0.3GHz fixed-epsilon field refit diagnostic.")
    parser.add_argument(
        "--checkpoint",
        default="square_target/results_5_3_2_square_0_3GHz_current_tv_plus_lep_0p001_14000/checkpoint_adam_014000.pt",
    )
    parser.add_argument("--threshold", type=float, default=2.32)
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--log-every", type=int, default=500)
    parser.add_argument(
        "--output-dir",
        default="square_target/results_5_3_2_square_0_3GHz_fixed_epsilon_field_refit",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    checkpoint_path = (root / args.checkpoint).resolve() if not Path(args.checkpoint).is_absolute() else Path(args.checkpoint)
    output_dir = (root / args.output_dir).resolve() if not Path(args.output_dir).is_absolute() else Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config, target, labels = load_run_config(checkpoint_path)
    config.loss_preset = "current"
    config.weight_tv = 0.0
    config.weight_contrast_l1 = 0.0
    config.weight_edge_preserving = 0.0
    config.binary_push_weight = 0.0
    config.epsilon_binary_sharpen_weight = 0.0
    config.phase_binary_weight = 0.0
    config.pseudo_separation_weight = 0.0
    config.epsilon_prior_weight = 0.0
    config.background_anchor_weight = 0.0

    dtype = torch.float32 if config.dtype == "float32" else torch.float64
    device = torch.device("cuda" if torch.cuda.is_available() and config.device != "cpu" else "cpu")
    random.seed(config.random_seed)
    np.random.seed(config.random_seed)
    torch.manual_seed(config.random_seed)

    obs = load_observations(root / "square_target", config, direction_labels=labels)
    obs_tensors = to_observation_tensors(obs, device, dtype)
    integral_tensors = make_integral_tensors(obs, target, config, device, dtype)
    if integral_tensors is None:
        raise RuntimeError("integral_data loss is required for this diagnostic")

    base_for_mask = make_model(checkpoint_path, config, target, device, dtype)
    _, _, checkpoint_eps = reconstruct_epsilon(base_for_mask, target, config.plot_grid_size, device, dtype)
    _, _, truth = true_epsilon_grid(target, config.plot_grid_size)
    mask = checkpoint_eps >= args.threshold
    fixed_a = np.where(mask, target.eps_background + 2.5, target.eps_background)
    fixed_b = np.where(mask, target.eps_background + 3.0, target.eps_background)
    metrics = {
        "checkpoint_original": epsilon_metrics(checkpoint_eps, truth, target),
        "A_eps3p5_fixed_mask": epsilon_metrics(fixed_a, truth, target),
        "B_eps4p0_fixed_mask": epsilon_metrics(fixed_b, truth, target),
    }
    np.save(output_dir / "threshold_mask.npy", mask.astype(np.uint8))

    cases = {
        "A_eps3p5": lambda: FixedMaskEpsilonModel(
            make_model(checkpoint_path, config, target, device, dtype), mask, target, 2.5, device, dtype
        ),
        "B_eps4p0": lambda: FixedMaskEpsilonModel(
            make_model(checkpoint_path, config, target, device, dtype), mask, target, 3.0, device, dtype
        ),
        "C_checkpoint_eps": lambda: FixedCheckpointEpsilonModel(
            make_model(checkpoint_path, config, target, device, dtype)
        ),
    }

    summary: Dict[str, Dict[str, float]] = {}
    for name, factory in cases.items():
        summary[name] = run_case(
            name=name,
            fixed_model=factory().to(device=device, dtype=dtype),
            config=config,
            target=target,
            obs_tensors=obs_tensors,
            integral_tensors=integral_tensors,
            output_dir=output_dir,
            epochs=args.epochs,
            log_every=args.log_every,
            device=device,
            dtype=dtype,
        )

    payload = {
        "checkpoint": str(checkpoint_path),
        "threshold": args.threshold,
        "mask_object_pixels": int(mask.sum()),
        "epochs": args.epochs,
        "config": asdict(config),
        "epsilon_metrics": metrics,
        "results": summary,
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
