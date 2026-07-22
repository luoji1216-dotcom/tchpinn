from __future__ import annotations

"""Real Fresnel dual-branch exact-LS inversion with smooth rank-2 measurement error.

The data term remains owned by the epsilon branch through an exact LS solve:

    Es_pred = Es_LS(epsilon) + B(theta, beta)

The field branch is still supervised only by LS-generated fields and is never
fit directly to measured receiver data.
"""

import csv
import json
import shutil
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

import run_fresnel_1ghz_m12_dual_branch_ls_synthetic_closure as dual
import run_fresnel_1ghz_m12_exp_iwt_epsilon_only_exact_ls_closure as exact
import run_fresnel_1ghz_m12_exp_iwt_synthetic_closure as closure
from square_target import pinn_pixel_inverse_core as core


HERE = Path(__file__).resolve().parent
OUTPUT_DIR = HERE / "results_fresnel_1GHz_m12_dual_branch_ls_real_30train6val_lowrankB_rank2_K2"
SEED = 20260722
STEPS = 1000
WARMUP_STEPS = 200
SAVE_EVERY = 100
VIEWS_PER_STEP = 6
FIELD_WEIGHT = 10.0
B_WEIGHT = 1.0e-2
RANK = 2
FOURIER_ORDER = 2
EPS = 1.0e-12
VALIDATION_VIEW_IDS = (5, 11, 17, 23, 29, 35)


class SmoothLowRankMeasurementError(torch.nn.Module):
    """B(theta,beta)=sum_r u_r(theta) v_r(beta), complex Fourier K=2."""

    def __init__(self, rank: int = RANK, order: int = FOURIER_ORDER) -> None:
        super().__init__()
        self.rank = rank
        self.order = order
        self.modes = torch.arange(-order, order + 1, dtype=torch.float32)
        n_modes = 2 * order + 1
        self.u_real = torch.nn.Parameter(torch.zeros(rank, n_modes))
        self.u_imag = torch.nn.Parameter(torch.zeros(rank, n_modes))
        self.v_real = torch.nn.Parameter(torch.zeros(rank, n_modes))
        self.v_imag = torch.nn.Parameter(torch.zeros(rank, n_modes))
        with torch.no_grad():
            center = order
            for r in range(rank):
                self.v_real[r, center] = 1.0
                if r == 1 and center + 1 < n_modes:
                    self.v_real[r, center + 1] = 0.25

    def _basis(self, angles: torch.Tensor) -> torch.Tensor:
        modes = self.modes.to(device=angles.device, dtype=angles.real.dtype)
        return torch.exp(1j * angles[:, None].to(modes.dtype) * modes[None, :])

    def forward(self, theta: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
        theta_basis = self._basis(theta)
        beta_basis = self._basis(beta)
        u_coeff = torch.complex(self.u_real, self.u_imag).to(theta_basis.dtype)
        v_coeff = torch.complex(self.v_real, self.v_imag).to(beta_basis.dtype)
        u_values = theta_basis @ u_coeff.transpose(0, 1)
        v_values = beta_basis @ v_coeff.transpose(0, 1)
        return torch.sum(u_values * v_values, dim=1)


def wrap_to_pi(angle: np.ndarray) -> np.ndarray:
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def measurement_angles(measurement: dict[str, np.ndarray], device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    source_angle = -measurement["target_rotation"]
    receiver_angle = measurement["receiver_angles_raw"] - measurement["target_rotation"]
    beta = wrap_to_pi(receiver_angle - source_angle)
    return (
        torch.as_tensor(source_angle, dtype=dtype, device=device),
        torch.as_tensor(beta, dtype=dtype, device=device),
    )


def selected_receiver_error_with_b(
    ls_receiver: torch.Tensor,
    b_model: SmoothLowRankMeasurementError,
    observed: torch.Tensor,
    indices: dict[int, np.ndarray],
    view_ids: torch.Tensor,
    theta_rows: torch.Tensor,
    beta_rows: torch.Tensor,
    use_b: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    prediction_parts: list[torch.Tensor] = []
    prediction_no_b_parts: list[torch.Tensor] = []
    observation_parts: list[torch.Tensor] = []
    b_parts: list[torch.Tensor] = []
    for view_id in view_ids.detach().cpu().tolist():
        rows = torch.as_tensor(indices[int(view_id) + 1], dtype=torch.long, device=ls_receiver.device)
        base_prediction = ls_receiver[rows, int(view_id)]
        if use_b:
            b_value = b_model(theta_rows[rows], beta_rows[rows])
        else:
            b_value = torch.zeros_like(base_prediction)
        prediction_no_b_parts.append(base_prediction)
        prediction_parts.append(base_prediction + b_value)
        observation_parts.append(observed[rows])
        b_parts.append(b_value)
    prediction = torch.cat(prediction_parts)
    prediction_no_b = torch.cat(prediction_no_b_parts)
    observation = torch.cat(observation_parts)
    b_values = torch.cat(b_parts)
    return (
        dual.normalized_error(prediction, observation),
        dual.normalized_error(prediction_no_b, observation),
        torch.mean(torch.abs(b_values).square()),
    )


def b_norm_ratio(
    b_model: SmoothLowRankMeasurementError,
    observed: torch.Tensor,
    indices: dict[int, np.ndarray],
    view_ids: torch.Tensor,
    theta_rows: torch.Tensor,
    beta_rows: torch.Tensor,
) -> float:
    b_parts: list[torch.Tensor] = []
    obs_parts: list[torch.Tensor] = []
    with torch.no_grad():
        for view_id in view_ids.detach().cpu().tolist():
            rows = torch.as_tensor(indices[int(view_id) + 1], dtype=torch.long, device=observed.device)
            b_parts.append(b_model(theta_rows[rows], beta_rows[rows]))
            obs_parts.append(observed[rows])
        b_value = torch.cat(b_parts)
        obs_value = torch.cat(obs_parts)
        return float(torch.linalg.norm(b_value).cpu() / (torch.linalg.norm(obs_value).cpu() + EPS))


def train_view_ids_for_step(step: int, train_ids: torch.Tensor) -> torch.Tensor:
    groups = train_ids.reshape(-1, VIEWS_PER_STEP)
    return groups[(step - 1) % groups.shape[0]]


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def looks_like_saturated_islands(epsilon_map: np.ndarray) -> bool:
    saturated_fraction = float(np.mean(epsilon_map > 3.8))
    active_fraction = float(np.mean(epsilon_map > 1.5))
    return saturated_fraction > 0.01 and active_fraction < 0.08


def main() -> None:
    if OUTPUT_DIR.exists():
        raise FileExistsError(f"Refusing to overwrite existing result directory: {OUTPUT_DIR}")
    config = closure.make_config()
    target = core.TargetSpec(name="Fresnel real dual LS low-rank B", kind="circle", eps_background=1.0, eps_object=4.0, roi_half_width=0.09)
    device, dtype = core.resolve_device(config.device), core.resolve_dtype(config.dtype)
    measurement, indices, directions_np, fits, fit_metadata = closure.make_geometry_and_m12(config)
    receiver_xy_np = closure.base.mapped_receiver_xy(measurement)
    receiver_xy = torch.as_tensor(receiver_xy_np, dtype=dtype, device=device)
    observed = torch.as_tensor(measurement["scattered"], dtype=torch.complex64, device=device)
    theta_rows, beta_rows = measurement_angles(measurement, device, dtype)

    class Geometry:
        xy = receiver_xy_np

    integral = core.make_integral_tensors(Geometry(), target, config, device, dtype)
    if integral is None:
        raise RuntimeError("Low-rank B inversion requires static integral tensors.")
    direct_ls = core.make_direct_ls_tensors(integral, config, device, dtype)
    closure.base.replace_with_exp_iwt_green(direct_ls, receiver_xy_np, config.k0)
    incident = closure.M12IncidentProvider(fits, integral.quad_xy, device, direct_ls.domain_green.dtype)
    if observed.dtype != direct_ls.domain_green.dtype:
        observed = observed.to(direct_ls.domain_green.dtype)
    directions_by_view = torch.as_tensor(np.stack([directions_np[indices[view][0]] for view in range(1, 37)]), dtype=dtype, device=device)
    validation_ids = torch.as_tensor(VALIDATION_VIEW_IDS, dtype=torch.long, device=device)
    validation_set = set(VALIDATION_VIEW_IDS)
    train_ids = torch.as_tensor([view for view in range(36) if view not in validation_set], dtype=torch.long, device=device)

    core.set_global_seed(SEED)
    model = dual.LSSupervisedDualBranch(config, target).to(device=device)
    b_model = SmoothLowRankMeasurementError().to(device=device)
    epsilon_optimizer = torch.optim.Adam(model.epsilon_branch.parameters(), lr=config.learning_rate)
    field_optimizer = torch.optim.Adam(model.field_branch.parameters(), lr=config.learning_rate)
    b_optimizer = torch.optim.Adam(b_model.parameters(), lr=config.learning_rate)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=False)
    image_dir = OUTPUT_DIR / "epsilon_images"
    image_dir.mkdir()
    (OUTPUT_DIR / "config.json").write_text(json.dumps({
        "config": asdict(config),
        "view_split": {
            "train_views_1based": [int(view) + 1 for view in train_ids.cpu().tolist()],
            "validation_views_1based": [int(view) + 1 for view in validation_ids.cpu().tolist()],
        },
        "convention": {
            "geometry": "R(-theta)",
            "theta_for_B": "source_angle=-target_rotation",
            "beta_for_B": "wrap((receiver_angle_raw-target_rotation)-source_angle)",
            "incident": "M12IncidentProvider only",
            "time": "exp(+i omega t)",
            "green": "-i/4 H0^(2)(k0*distance)",
            "chi": "epsilon-1",
        },
        "measurement_error_branch": {
            "definition": "B(theta,beta)=sum_{r=1}^2 u_r(theta)*v_r(beta)",
            "rank": RANK,
            "fourier_order_u": FOURIER_ORDER,
            "fourier_order_v": FOURIER_ORDER,
            "warmup_steps_with_B_zero": WARMUP_STEPS,
            "regularization": "1e-2*mean(|B|^2) on the current train minibatch",
        },
        "loss": {
            "receiver": "normalized_error(Es_LS(epsilon)+B, Es_measured)",
            "receiver_no_B_diagnostic": "normalized_error(Es_LS(epsilon), Es_measured)",
            "field_supervision": "10*inside_LS_supervision+10*receiver_LS_supervision",
            "forbidden": "error(Es_field_receiver, Es_measured)",
        },
        "checkpoint_selection": "minimum validation receiver loss with B; no GT, RE, SSIM, material, or geometry prior",
        "incident_fit_metadata": fit_metadata,
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    history: list[dict[str, Any]] = []
    best_validation = float("inf")
    stopped_reason: str | None = None
    started = time.time()
    for step in range(1, STEPS + 1):
        use_b = step > WARMUP_STEPS
        model.train()
        b_model.train()
        epsilon_optimizer.zero_grad(set_to_none=True)
        field_optimizer.zero_grad(set_to_none=True)
        b_optimizer.zero_grad(set_to_none=True)
        _epsilon, ls_inside, ls_receiver = dual.exact_ls_fields(model, direct_ls, incident, config)
        field_views = train_view_ids_for_step(step, train_ids)
        receiver_loss, _receiver_loss_no_b, b_power = selected_receiver_error_with_b(
            ls_receiver, b_model, observed, indices, field_views, theta_rows, beta_rows, use_b
        )
        inside_loss, field_receiver_loss = dual.field_losses(model, direct_ls, ls_inside, ls_receiver, receiver_xy, indices, directions_by_view, field_views)
        total_loss = receiver_loss + (B_WEIGHT * b_power if use_b else 0.0) + FIELD_WEIGHT * inside_loss + FIELD_WEIGHT * field_receiver_loss
        if not bool(torch.isfinite(total_loss)):
            raise FloatingPointError(f"Non-finite low-rank B objective at step {step}.")
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.epsilon_branch.parameters(), config.gradient_clip_norm)
        torch.nn.utils.clip_grad_norm_(model.field_branch.parameters(), config.gradient_clip_norm)
        if use_b:
            torch.nn.utils.clip_grad_norm_(b_model.parameters(), config.gradient_clip_norm)
        epsilon_optimizer.step()
        field_optimizer.step()
        if use_b:
            b_optimizer.step()

        if step % SAVE_EVERY:
            continue
        model.eval()
        b_model.eval()
        with torch.no_grad():
            _epsilon, full_inside, full_receiver = dual.exact_ls_fields(model, direct_ls, incident, config)
            train_receiver, train_receiver_no_b, train_b_power = selected_receiver_error_with_b(
                full_receiver, b_model, observed, indices, train_ids, theta_rows, beta_rows, step > WARMUP_STEPS
            )
            validation_receiver, validation_receiver_no_b, validation_b_power = selected_receiver_error_with_b(
                full_receiver, b_model, observed, indices, validation_ids, theta_rows, beta_rows, step > WARMUP_STEPS
            )
            xx, yy, epsilon_map = exact.epsilon_image(model, target, device, dtype)
        full_inside_loss, full_field_receiver_loss = dual.field_losses(model, direct_ls, full_inside, full_receiver, receiver_xy, indices, directions_by_view, train_ids)
        total_value = float((train_receiver + B_WEIGHT * train_b_power + FIELD_WEIGHT * full_inside_loss + FIELD_WEIGHT * full_field_receiver_loss).detach().cpu())
        validation_value = float(validation_receiver.cpu())
        b_ratio = b_norm_ratio(b_model, observed, indices, train_ids, theta_rows, beta_rows) if step > WARMUP_STEPS else 0.0
        exact.save_epsilon(epsilon_map, image_dir / f"epsilon_step_{step:04d}.png", f"Real dual LS low-rank B, step {step}")
        checkpoint = OUTPUT_DIR / f"checkpoint_step_{step:04d}.pt"
        torch.save({
            "model": model.state_dict(),
            "b_model": b_model.state_dict(),
            "epsilon_optimizer": epsilon_optimizer.state_dict(),
            "field_optimizer": field_optimizer.state_dict(),
            "b_optimizer": b_optimizer.state_dict(),
            "step": step,
            "config": asdict(config),
        }, checkpoint)
        if validation_value < best_validation:
            best_validation = validation_value
            shutil.copy2(checkpoint, OUTPUT_DIR / "best_checkpoint.pt")
        row = {
            "step": step,
            "B_enabled": bool(step > WARMUP_STEPS),
            "receiver_train_with_B": float(train_receiver.cpu()),
            "receiver_train_no_B": float(train_receiver_no_b.cpu()),
            "receiver_validation_with_B": validation_value,
            "receiver_validation_no_B": float(validation_receiver_no_b.cpu()),
            "B_power_train": float(train_b_power.cpu()),
            "B_power_validation": float(validation_b_power.cpu()),
            "B_over_Es_train": b_ratio,
            "L_field_inside_train": float(full_inside_loss.detach().cpu()),
            "L_field_receiver_train": float(full_field_receiver_loss.detach().cpu()),
            "total_train": total_value,
            "epsilon_min": float(epsilon_map.min()),
            "epsilon_max": float(epsilon_map.max()),
            "epsilon_mean": float(epsilon_map.mean()),
            "epsilon_std": float(epsilon_map.std()),
            "elapsed_s": time.time() - started,
        }
        history.append(row)
        write_csv(history, OUTPUT_DIR / "history.csv")
        print(
            f"step={step:4d} Lrx(B/noB)={row['receiver_train_with_B']:.3e}/{row['receiver_train_no_B']:.3e} "
            f"Lval(B/noB)={validation_value:.3e}/{row['receiver_validation_no_B']:.3e} "
            f"B/Es={b_ratio:.3f} epsmax/std={row['epsilon_max']:.3f}/{row['epsilon_std']:.3f}",
            flush=True,
        )
        if step > WARMUP_STEPS and b_ratio > 0.7:
            stopped_reason = f"Stopped at step {step}: B/Es={b_ratio:.3f} exceeded 0.7."
            break
        if step > WARMUP_STEPS and looks_like_saturated_islands(epsilon_map):
            stopped_reason = f"Stopped at step {step}: epsilon map remained saturated-island dominated."
            break

    final_step = history[-1]["step"] if history else 0
    torch.save({
        "model": model.state_dict(),
        "b_model": b_model.state_dict(),
        "epsilon_optimizer": epsilon_optimizer.state_dict(),
        "field_optimizer": field_optimizer.state_dict(),
        "b_optimizer": b_optimizer.state_dict(),
        "step": final_step,
        "config": asdict(config),
    }, OUTPUT_DIR / "final_checkpoint.pt")
    selected = torch.load(OUTPUT_DIR / "best_checkpoint.pt", map_location=device, weights_only=False)
    model.load_state_dict(selected["model"])
    b_model.load_state_dict(selected["b_model"])
    model.eval()
    b_model.eval()
    xx, yy, epsilon_map = exact.epsilon_image(model, target, device, dtype)
    exact.save_epsilon(epsilon_map, OUTPUT_DIR / "best_continuous_reconstruction.png", f"Real dual LS low-rank B best validation, step {selected['step']}")
    summary = {
        "steps_completed": int(final_step),
        "best_checkpoint_step": int(selected["step"]),
        "best_validation_receiver_loss_with_B": best_validation,
        "stopped_reason": stopped_reason,
        "best_epsilon": {
            "min": float(epsilon_map.min()),
            "max": float(epsilon_map.max()),
            "mean": float(epsilon_map.mean()),
            "std": float(epsilon_map.std()),
        },
        "selection": "minimum validation receiver loss with B only; no GT/RE/SSIM",
        "elapsed_s": time.time() - started,
    }
    (OUTPUT_DIR / "metrics.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
