from __future__ import annotations

import csv
import argparse
import json
import math
import sys
import time
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
    data_loss,
    incident_field_torch,
    load_observations,
    make_integral_tensors,
    pde_residual_loss,
    resolve_device,
    resolve_dtype,
    sample_collocation_points,
    sample_direction_batch,
    sample_observation_batch,
    set_global_seed,
    target_mask,
    to_observation_tensors,
    volume_integral_data_loss,
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
EPOCHS = 6000
EVAL_EVERY = 500


class FixedEpsilonModel(torch.nn.Module):
    def __init__(
        self,
        config: TrainConfig,
        target,
        epsilon_fn: Callable[[torch.Tensor], torch.Tensor],
    ) -> None:
        super().__init__()
        self.base = DoubleBranchPINN(config, target)
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


def checkpoint_config(path: Path, *, frequency_hz: float | None = None) -> TrainConfig:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    raw = checkpoint.get("config", {}) if isinstance(checkpoint, dict) else {}
    valid = set(TrainConfig.__dataclass_fields__.keys())
    kwargs = {key: value for key, value in raw.items() if key in valid}
    config = TrainConfig(**kwargs)
    if frequency_hz is not None:
        config.frequency_hz = frequency_hz
    return config


def load_checkpoint_model(path: Path, target, device: torch.device, dtype: torch.dtype, *, frequency_hz: float) -> DoubleBranchPINN:
    config = checkpoint_config(path, frequency_hz=frequency_hz)
    model = DoubleBranchPINN(config, target).to(device=device, dtype=dtype)
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    state = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    model.load_state_dict(state, strict=True)
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return model


def checkpoint_epsilon_fn(source: DoubleBranchPINN) -> Callable[[torch.Tensor], torch.Tensor]:
    def fn(xy: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return source.epsilon(xy)

    return fn


def make_train_config(mode: str) -> TrainConfig:
    base = checkpoint_config(B53000_CHECKPOINT, frequency_hz=1.0e9)
    base.device = "auto"
    base.dtype = "float32"
    base.epochs_adam = EPOCHS
    base.max_points_per_direction = 836
    base.data_batch_per_direction = 256
    base.n_pde = 1280
    base.n_boundary = 0
    base.integral_grid_size = 36
    base.weight_data = 120.0
    base.weight_integral_data = 50.0
    base.weight_pde = 0.006
    base.weight_boundary = 0.0
    base.weight_tv = 0.0
    base.weight_contrast_l1 = 0.0
    base.integral_internal_field_mode = mode
    base.learning_rate = 4.0e-4
    base.random_seed = 20260430
    return base


def operator_prediction(
    model,
    obs,
    integral,
    obs_tensors,
    target,
    config: TrainConfig,
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
            if config.integral_internal_field_mode == "detach_field":
                scattered_quad = scattered_quad.detach()
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


def full_eval(model, obs, obs_tensors, integral, target, config, device, dtype) -> dict[str, float]:
    all_indices = torch.arange(obs.xy.shape[0], dtype=torch.long, device=device)
    with torch.no_grad():
        data = data_loss(model, obs_tensors.xy, obs_tensors.directions, obs_tensors.target, robust=False)
        integral_loss = volume_integral_data_loss(
            model,
            all_indices,
            obs_tensors.directions,
            obs_tensors.target,
            integral,
            obs_tensors,
            target,
            config,
        )
    pde_xy, pde_dirs, pde_amps = fixed_pde_batch(obs_tensors, config.n_pde, obs.receiver_radius * 0.98, device, dtype)
    with torch.enable_grad():
        pde = pde_residual_loss(model, pde_xy, pde_dirs, pde_amps, config)
    pred = operator_prediction(model, obs, integral, obs_tensors, target, config, device, dtype)
    observed = obs.scattered.astype(np.complex128)
    return {
        "data_loss": float(data.detach().cpu()),
        "integral_loss": float(integral_loss.detach().cpu()),
        "pde_loss": float(pde.detach().cpu()),
        "alpha_corrected_rel_l2": alpha_corrected_rel_l2(pred, observed, obs.labels),
    }


def fixed_pde_batch(obs_tensors, n_pde: int, radius: float, device: torch.device, dtype: torch.dtype):
    theta = torch.linspace(0.0, 2.0 * math.pi, n_pde + 1, dtype=dtype, device=device)[:-1]
    r = radius * torch.sqrt(torch.linspace(1.0 / n_pde, 1.0, n_pde, dtype=dtype, device=device))
    xy = torch.stack((r * torch.cos(theta), r * torch.sin(theta)), dim=1)
    labels = torch.arange(n_pde, device=device) % obs_tensors.unique_directions.shape[0]
    return xy, obs_tensors.unique_directions[labels], obs_tensors.unique_amplitudes[labels]


def train_one(
    *,
    epsilon_name: str,
    epsilon_fn: Callable[[torch.Tensor], torch.Tensor],
    mode: str,
    target,
    obs,
    obs_tensors,
    integral,
    config: TrainConfig,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, float | str]:
    set_global_seed(config.random_seed)
    model = FixedEpsilonModel(config, target, epsilon_fn).to(device=device, dtype=dtype)
    for param in model.base.epsilon_branch.parameters():
        param.requires_grad_(False)
    optimizer = torch.optim.Adam(model.base.field_branch.parameters(), lr=config.learning_rate)
    radius = obs.receiver_radius * 0.98
    best = None
    start = time.time()

    for step in range(1, config.epochs_adam + 1):
        optimizer.zero_grad(set_to_none=True)
        data_indices, data_xy, data_dirs, _data_amps, data_target = sample_observation_batch(
            obs_tensors, config.data_batch_per_direction
        )
        pde_xy = sample_collocation_points(config.n_pde, target, radius, device, dtype)
        pde_dirs, pde_amps = sample_direction_batch(
            obs_tensors.unique_directions, obs_tensors.unique_amplitudes, config.n_pde
        )
        loss_data = data_loss(model, data_xy, data_dirs, data_target, robust=config.robust_data_weighting)
        loss_integral = volume_integral_data_loss(
            model,
            data_indices,
            data_dirs,
            data_target,
            integral,
            obs_tensors,
            target,
            config,
        )
        loss_pde = pde_residual_loss(model, pde_xy, pde_dirs, pde_amps, config)
        total = (
            config.weight_data * loss_data
            + config.weight_integral_data * loss_integral
            + config.weight_pde * loss_pde
        )
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.base.field_branch.parameters(), config.gradient_clip_norm)
        optimizer.step()

        if step == 1 or step % EVAL_EVERY == 0 or step == config.epochs_adam:
            metrics = full_eval(model, obs, obs_tensors, integral, target, config, device, dtype)
            if best is None or metrics["data_loss"] < best["best_data_loss"]:
                best = {
                    "best_step": step,
                    "best_data_loss": metrics["data_loss"],
                    "best_integral_loss": metrics["integral_loss"],
                    "best_pde_loss": metrics["pde_loss"],
                    "best_alpha_corrected_rel_l2": metrics["alpha_corrected_rel_l2"],
                }
            print(
                f"{mode},{epsilon_name},step={step},"
                f"data={metrics['data_loss']:.8e},"
                f"integral={metrics['integral_loss']:.8e},"
                f"pde={metrics['pde_loss']:.8e},"
                f"alpha_rel={metrics['alpha_corrected_rel_l2']:.8e}",
                flush=True,
            )

    final = full_eval(model, obs, obs_tensors, integral, target, config, device, dtype)
    assert best is not None
    return {
        "mode": mode,
        "epsilon": epsilon_name,
        **best,
        "final_data_loss": final["data_loss"],
        "final_integral_loss": final["integral_loss"],
        "final_pde_loss": final["pde_loss"],
        "final_alpha_corrected_rel_l2": final["alpha_corrected_rel_l2"],
        "elapsed_s": time.time() - start,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Refit field branch with fixed Austria 1GHz epsilon candidates.")
    parser.add_argument("--mode", choices=["coupled", "detach_field"], default=None)
    parser.add_argument("--epsilon", default=None, help="Run only one epsilon source name.")
    args = parser.parse_args()

    target = build_target(0.0)
    probe_config = make_train_config("coupled")
    device = resolve_device(probe_config.device)
    dtype = resolve_dtype(probe_config.dtype)

    print(f"frequency_hz={probe_config.frequency_hz}")
    print(f"k0={probe_config.k0}")
    print(f"data_dir={DATA_DIR.resolve()}")
    print(f"epochs_per_case={EPOCHS}")
    print("true_epsilon_use=diagnostic_fixed_candidate_only")
    print("epsilon_training=disabled")

    obs = load_observations(DATA_DIR, probe_config, direction_labels=DIRECTIONS)
    obs_tensors = to_observation_tensors(obs, device=device, dtype=dtype)
    integral = make_integral_tensors(obs, target, probe_config, device=device, dtype=dtype)
    if integral is None:
        raise RuntimeError("integral tensors were not created")

    b53000_model = load_checkpoint_model(B53000_CHECKPOINT, target, device, dtype, frequency_hz=1.0e9)
    epsilon_sources: dict[str, Callable[[torch.Tensor], torch.Tensor]] = {
        "true": true_epsilon_fn(target, dtype),
        "background": background_epsilon_fn(target),
        "direct_B53000": checkpoint_epsilon_fn(b53000_model),
    }
    if AUSTRIA_03_BEST.exists():
        model_03 = load_checkpoint_model(AUSTRIA_03_BEST, target, device, dtype, frequency_hz=0.3e9)
        epsilon_sources["austria_0_3GHz_best"] = checkpoint_epsilon_fn(model_03)
        print(f"austria_0_3GHz_best_checkpoint={AUSTRIA_03_BEST}")
    else:
        print(f"austria_0_3GHz_best_missing={AUSTRIA_03_BEST}")

    rows = []
    progress_jsonl = HERE / "diagnose_1ghz_fixed_epsilon_field_refit_progress.jsonl"
    modes = (args.mode,) if args.mode else ("coupled", "detach_field")
    for mode in modes:
        config = make_train_config(mode)
        for epsilon_name, epsilon_fn in epsilon_sources.items():
            if args.epsilon and epsilon_name != args.epsilon:
                continue
            row = train_one(
                epsilon_name=epsilon_name,
                epsilon_fn=epsilon_fn,
                mode=mode,
                target=target,
                obs=obs,
                obs_tensors=obs_tensors,
                integral=integral,
                config=config,
                device=device,
                dtype=dtype,
            )
            rows.append(row)
            try:
                with progress_jsonl.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
            except PermissionError as exc:
                print(f"progress_write_skipped={progress_jsonl} reason={exc}")

    output_csv = HERE / "diagnose_1ghz_fixed_epsilon_field_refit.csv"
    wrote_csv = False
    try:
        with output_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        wrote_csv = True
    except PermissionError as exc:
        print(f"csv_write_skipped={output_csv} reason={exc}")

    print("\nsummary_table")
    print(
        "mode,epsilon,best_step,best_data_loss,best_integral_loss,best_pde_loss,"
        "best_alpha_corrected_rel_l2,final_data_loss,final_integral_loss,final_pde_loss,"
        "final_alpha_corrected_rel_l2,elapsed_s"
    )
    for row in rows:
        print(
            f"{row['mode']},{row['epsilon']},{row['best_step']},"
            f"{row['best_data_loss']:.8e},{row['best_integral_loss']:.8e},"
            f"{row['best_pde_loss']:.8e},{row['best_alpha_corrected_rel_l2']:.8e},"
            f"{row['final_data_loss']:.8e},{row['final_integral_loss']:.8e},"
            f"{row['final_pde_loss']:.8e},{row['final_alpha_corrected_rel_l2']:.8e},"
            f"{row['elapsed_s']:.2f}"
        )

    print("\ntrue_vs_B53000")
    for mode in modes:
        true_row = next((row for row in rows if row["mode"] == mode and row["epsilon"] == "true"), None)
        b_row = next((row for row in rows if row["mode"] == mode and row["epsilon"] == "direct_B53000"), None)
        if true_row is None or b_row is None:
            print(f"{mode}: skipped true_vs_B53000; need both true and direct_B53000 in this run")
            continue
        print(
            f"{mode}: "
            f"best_data_true_better={true_row['best_data_loss'] < b_row['best_data_loss']} "
            f"final_data_true_better={true_row['final_data_loss'] < b_row['final_data_loss']} "
            f"integral_true_better={true_row['final_integral_loss'] < b_row['final_integral_loss']} "
            f"pde_true_better={true_row['final_pde_loss'] < b_row['final_pde_loss']} "
            f"alpha_true_better={true_row['final_alpha_corrected_rel_l2'] < b_row['final_alpha_corrected_rel_l2']}"
        )

    if wrote_csv:
        print(f"\nwrote={output_csv}")
    print(json.dumps({"config": asdict(make_train_config("coupled"))}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
