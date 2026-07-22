from __future__ import annotations

"""Fresh 32x32 direct-grid CSI with an alternating low-rank additive nuisance.

No ground truth or oracle residual SVD is used.  The nuisance is reconstructed
only from the current CSI data residual after every 20 Adam updates.
"""

import csv
import json
import math
import shutil
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import audit_fresnel_1ghz_rminus_m12_incident_ratio_calibration as base
from square_target import pinn_pixel_inverse_core as core


HERE = Path(__file__).resolve().parent
OUTPUT_DIR = HERE / "results_fresnel_1GHz_rminus_m12_csi_lowrank_nuisance_fresh900"
SEED = 20260430
GRID_SIZE = 32
STEPS = 900
VIEWS_PER_STEP = 6
NUISANCE_UPDATE_EVERY = 20
NUISANCE_SHRINK = 0.25
MAX_RANK = 2
SAVE_EVERY = 100
EPS = 1.0e-12


def make_config() -> core.TrainConfig:
    return core.TrainConfig(
        frequency_hz=1.0e9,
        eps_min=1.0,
        eps_max=4.0,
        eps_initial=1.5,
        epsilon_constant_init=True,
        eps_hidden_layers=5,
        eps_hidden_units=96,
        fourier_bands=5,
        fourier_max_frequency=8.0,
        epsilon_fourier_encoding="axis",
        epsilon_fourier_random_seed=SEED,
        integral_grid_size=GRID_SIZE,
        integral_sampling_mode="static",
        integral_internal_field_mode="coupled",
        weight_integral_data=1.0,
        weight_pde=0.0,
        weight_tv=0.0,
        weight_field_integral_consistency=0.0,
        weight_edge_preserving=0.0,
        learning_rate=1.0e-3,
        gradient_clip_norm=1.0,
        random_seed=SEED,
        dtype="float32",
        device="auto",
    )


def make_observations(config: core.TrainConfig) -> tuple[core.ObservationSet, dict[str, Any]]:
    measurement = base.load_measurement()
    indices = base.view_indices(measurement["views"])
    xy = base.mapped_receiver_xy(measurement)
    views = measurement["views"]
    source_angle = -measurement["target_rotation"]
    directions = np.column_stack((-np.cos(source_angle), -np.sin(source_angle))).astype(np.float64)
    labels = np.asarray([f"view_{view:02d}" for view in views], dtype=object)
    direction_labels = [f"view_{view:02d}" for view in range(1, 37)]
    fits: dict[str, Any] = {}
    metadata: dict[str, Any] = {}
    for view, label in enumerate(direction_labels, start=1):
        rows = indices[view]
        fit = core.fit_fourier_bessel_incident(xy[rows], measurement["incident"][rows], config.k0, base.M12_ORDER, base.M12_RIDGE)
        fits[label] = fit
        metadata[label] = {"fit_metrics": fit.fit_metrics, "coefficient_count": int(fit.coefficients.size)}
    return core.ObservationSet(
        xy=xy,
        total=measurement["total"],
        scattered=measurement["scattered"],
        directions=directions,
        labels=labels,
        receiver_radius=base.RHO_M,
        direction_labels=direction_labels,
        incident_amplitudes={label: 0.0j for label in direction_labels},
        incident_fits=fits,
    ), metadata


class DirectGridCSIWithNuisance(torch.nn.Module):
    """Square epsilon MLP, one direct W grid per view, and B=U V^H."""

    def __init__(self, config: core.TrainConfig, target: core.TargetSpec, w_initial: np.ndarray) -> None:
        super().__init__()
        n_cells = GRID_SIZE * GRID_SIZE
        if w_initial.shape != (n_cells, 36):
            raise ValueError(f"Expected W shape {(n_cells, 36)}, got {w_initial.shape}.")
        self.epsilon_branch = core.EpsilonBranch(config, target)
        initial_two = np.stack((w_initial.T.real, w_initial.T.imag), axis=-1).astype(np.float32, copy=False)
        self.w_grid = torch.nn.Parameter(torch.as_tensor(initial_two))
        self.register_buffer("u_factor", torch.zeros((36, MAX_RANK), dtype=torch.complex64))
        self.register_buffer("v_factor", torch.zeros((49, MAX_RANK), dtype=torch.complex64))
        self.register_buffer("active_rank", torch.zeros((), dtype=torch.int64))

    def epsilon(self, xy: torch.Tensor) -> torch.Tensor:
        return self.epsilon_branch(xy)

    def w(self, view_ids: torch.Tensor) -> torch.Tensor:
        values = self.w_grid[view_ids]
        return torch.complex(values[..., 0], values[..., 1]).transpose(0, 1)

    def nuisance(self, view_ids: torch.Tensor) -> torch.Tensor:
        # [selected views, 49], exactly B=U @ V^H.
        return (self.u_factor @ self.v_factor.conj().transpose(0, 1))[view_ids]

    @torch.no_grad()
    def set_nuisance_from_residual(self, residual: torch.Tensor, rank: int) -> dict[str, float]:
        """Set B to shrink * rank-r truncated SVD of a current data residual."""
        if residual.shape != (36, 49):
            raise ValueError(f"Expected residual shape (36,49), got {tuple(residual.shape)}.")
        u, singular, vh = torch.linalg.svd(residual, full_matrices=False)
        self.u_factor.zero_(); self.v_factor.zero_()
        root_singular = torch.sqrt(NUISANCE_SHRINK * singular[:rank])
        self.u_factor[:, :rank].copy_(u[:, :rank] * root_singular.unsqueeze(0))
        self.v_factor[:, :rank].copy_(vh[:rank].conj().transpose(0, 1) * root_singular.unsqueeze(0))
        self.active_rank.fill_(rank)
        nuisance = self.nuisance(torch.arange(36, device=residual.device))
        return {
            "rank": float(rank),
            "residual_singular_1": float(singular[0].cpu()),
            "residual_singular_rank": float(singular[rank - 1].cpu()),
            "nuisance_norm": float(torch.linalg.vector_norm(nuisance).cpu()),
        }


def view_ids_for_step(step: int, device: torch.device) -> torch.Tensor:
    group = (step - 1) % (36 // VIEWS_PER_STEP)
    return torch.arange(group * VIEWS_PER_STEP, (group + 1) * VIEWS_PER_STEP, device=device)


def complex_from_two(values: torch.Tensor) -> torch.Tensor:
    return torch.complex(values[:, 0], values[:, 1])


def backprop_w_initialization(direct_ls: Any, tensors: core.ObservationTensors, obs: core.ObservationSet, config: core.TrainConfig) -> tuple[np.ndarray, list[dict[str, float]]]:
    integral = direct_ls.integral
    receiver_green = integral.green_re.detach().cpu().numpy() + 1j * integral.green_im.detach().cpu().numpy()
    scale = float(config.k0**2 * integral.area_weight)
    columns: list[np.ndarray] = []
    rows_out: list[dict[str, float]] = []
    for view_id, label in enumerate(obs.direction_labels):
        rows = tensors.indices_by_label[label].detach().cpu().numpy()
        operator = scale * receiver_green[rows]
        observation = obs.scattered[rows]
        adjoint = operator.conj().T @ observation
        response = operator @ adjoint
        alpha = np.vdot(response, observation) / np.vdot(response, response)
        columns.append(np.asarray(alpha * adjoint, dtype=np.complex64))
        rows_out.append({"view": view_id + 1, "alpha_real": float(alpha.real), "alpha_imag": float(alpha.imag), "alpha_abs": float(abs(alpha)), "alpha_phase_rad": float(np.angle(alpha))})
    return np.column_stack(columns), rows_out


def physical_prediction_matrix(model: DirectGridCSIWithNuisance, direct_ls: Any, tensors: core.ObservationTensors, obs: core.ObservationSet, config: core.TrainConfig) -> tuple[torch.Tensor, torch.Tensor]:
    """Return uncorrected Gs(w) and Es observations as 36x49 matrices."""
    all_ids = torch.arange(36, device=direct_ls.domain_green.device)
    w = model.w(all_ids)
    scale = float(config.k0**2 * direct_ls.integral.area_weight)
    receiver_green = torch.complex(direct_ls.integral.green_re, direct_ls.integral.green_im).to(direct_ls.domain_green.dtype)
    predicted = torch.empty((36, 49), dtype=direct_ls.domain_green.dtype, device=all_ids.device)
    measured = torch.empty_like(predicted)
    for view_id, label in enumerate(obs.direction_labels):
        rows = tensors.indices_by_label[label]
        if rows.numel() != 49:
            raise ValueError(f"{label} does not contain 49 receiver samples.")
        predicted[view_id] = scale * (receiver_green[rows] @ w[:, view_id])
        measured[view_id] = complex_from_two(tensors.target[rows])
    return predicted, measured


def csi_losses(model: DirectGridCSIWithNuisance, direct_ls: Any, tensors: core.ObservationTensors, obs: core.ObservationSet, target: core.TargetSpec, config: core.TrainConfig, incident_cache: torch.Tensor, view_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    integral = direct_ls.integral
    w = model.w(view_ids)
    chi = (model.epsilon(integral.quad_xy) - float(target.eps_background)).to(direct_ls.domain_green.dtype)
    scale = float(config.k0**2 * integral.area_weight)
    total_field = incident_cache[:, view_ids] + scale * (direct_ls.domain_green @ w)
    state_residual = w - chi * total_field
    state_loss = torch.sum(torch.abs(state_residual).square()) / (torch.sum(torch.abs(w).square()) + EPS)
    receiver_green = torch.complex(integral.green_re, integral.green_im).to(direct_ls.domain_green.dtype)
    predictions: list[torch.Tensor] = []
    measured: list[torch.Tensor] = []
    nuisance = model.nuisance(view_ids)
    for local_index, view_id in enumerate(view_ids.detach().cpu().tolist()):
        label = obs.direction_labels[int(view_id)]
        rows = tensors.indices_by_label[label]
        predictions.append(scale * (receiver_green[rows] @ w[:, local_index]) + nuisance[local_index])
        measured.append(complex_from_two(tensors.target[rows]))
    predicted_flat = torch.cat(predictions)
    measured_flat = torch.cat(measured)
    data_loss = torch.sum(torch.abs(predicted_flat - measured_flat).square()) / (torch.sum(torch.abs(measured_flat).square()) + EPS)
    return data_loss, state_loss


def epsilon_image(model: DirectGridCSIWithNuisance, target: core.TargetSpec, device: torch.device, dtype: torch.dtype) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    axis = np.linspace(-target.roi_half_width, target.roi_half_width, 256, dtype=np.float32)
    xx, yy = np.meshgrid(axis, axis)
    xy = torch.as_tensor(np.column_stack((xx.reshape(-1), yy.reshape(-1))), dtype=dtype, device=device)
    with torch.no_grad():
        epsilon = model.epsilon(xy).reshape(xx.shape).cpu().numpy()
    return xx, yy, epsilon


def epsilon_stats(epsilon: np.ndarray) -> dict[str, float]:
    return {"epsilon_min": float(epsilon.min()), "epsilon_max": float(epsilon.max()), "epsilon_mean": float(epsilon.mean()), "epsilon_std": float(epsilon.std())}


def save_epsilon(epsilon: np.ndarray, path: Path, title: str) -> None:
    figure, axis = plt.subplots(figsize=(5.8, 5.0))
    image = axis.imshow(epsilon, extent=[-0.09, 0.09, -0.09, 0.09], origin="lower", cmap="jet", vmin=1.0, vmax=4.0, interpolation="bilinear")
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel("x (m)"); axis.set_ylabel("y (m)"); axis.set_title(title)
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04, label="Relative Permittivity")
    figure.tight_layout(); figure.savefig(path, dpi=190, bbox_inches="tight"); plt.close(figure)


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def stage_for_step(step: int) -> tuple[str, int]:
    if step <= 300:
        return "stage1_B0", 0
    if step <= 600:
        return "stage2_rank1", 1
    return "stage3_rank2", 2


def main() -> None:
    if OUTPUT_DIR.exists():
        raise FileExistsError(f"Refusing to overwrite existing result directory: {OUTPUT_DIR}")
    config = make_config()
    target = core.TargetSpec(name="Fresnel CSI no-geometry target", kind="circle", eps_background=1.0, eps_object=4.0, roi_half_width=0.09)
    device, dtype = core.resolve_device(config.device), core.resolve_dtype(config.dtype)
    obs, fit_metadata = make_observations(config)
    tensors = core.to_observation_tensors(obs, device=device, dtype=dtype)
    integral = core.make_integral_tensors(obs, target, config, device, dtype)
    if integral is None:
        raise RuntimeError("CSI requires static integral tensors.")
    direct_ls = core.make_direct_ls_tensors(integral, config, device, dtype)
    base.replace_with_exp_iwt_green(direct_ls, obs.xy, config.k0)
    incident_cache = torch.as_tensor(np.stack([obs.incident_fits[label].evaluate(integral.quad_xy.detach().cpu().numpy()) for label in obs.direction_labels], axis=1), dtype=direct_ls.domain_green.dtype, device=device)
    w_initial, alpha_rows = backprop_w_initialization(direct_ls, tensors, obs, config)
    core.set_global_seed(SEED)
    # The epsilon branch is float32 by configuration; do not pass a real dtype
    # to Module.to here because B's U/V factors are complex buffers.
    model = DirectGridCSIWithNuisance(config, target, w_initial).to(device=device)
    with torch.no_grad():
        written_w = model.w(torch.arange(36, device=device)).detach().cpu().numpy()
        initial_b_norm = float(torch.linalg.vector_norm(model.nuisance(torch.arange(36, device=device))).cpu())
    initialization_error = float(np.linalg.norm(written_w - w_initial) / max(np.linalg.norm(w_initial), 1.0e-15))
    if initialization_error != 0.0 or initial_b_norm != 0.0:
        raise AssertionError(f"Fresh direct CSI initialization failed: W error={initialization_error}, B norm={initial_b_norm}.")
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=False)
    image_dir = OUTPUT_DIR / "epsilon_images"; image_dir.mkdir()
    write_csv(alpha_rows, OUTPUT_DIR / "backprop_alpha.csv")
    (OUTPUT_DIR / "config.json").write_text(json.dumps({
        "config": asdict(config),
        "geometry": "R(-theta): source=-theta; receiver=raw_receiver_angle-theta",
        "incident": {"model": "per-view M=12 Fourier-Bessel", "ridge": base.M12_RIDGE, "fit_input": "measured empty incident only"},
        "initialization": {"epsilon": "fresh Square epsilon MLP, constant eps_initial=1.5", "W": "alpha_v*Gs^H*Es_v", "W_relative_error": initialization_error, "B": "U@V^H=0; no oracle SVD initialization"},
        "loss": {"data": "||Gs(W)+B-Es||^2/||Es||^2", "state": "||W-chi*(Einc+Gd(W))||^2/||W||^2", "PDE": 0, "FIC": 0, "TV": 0, "LEP": 0},
        "nuisance_schedule": {"steps_1_300": "B=0", "steps_301_600": "rank=1", "steps_601_900": "rank=2", "update_every_adam_steps": NUISANCE_UPDATE_EVERY, "update": "B=0.25*truncated_svd(Es-Gs(W))", "oracle_or_GT": "not used"},
        "selection": "minimum full-36-view L_data+L_state only; no GT/RE/SSIM",
        "incident_fit_metadata": fit_metadata,
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    history: list[dict[str, Any]] = []
    nuisance_updates: list[dict[str, Any]] = []
    best_total = float("inf")
    previous_b_ratio = 0.0
    stopped_early = False
    started = time.time()
    for step in range(1, STEPS + 1):
        stage, rank = stage_for_step(step)
        update_rank = 0
        if step == 301:
            update_rank = 1
        elif step == 601:
            update_rank = 2
        elif 301 < step <= 600 and (step - 301) % NUISANCE_UPDATE_EVERY == 0:
            update_rank = 1
        elif step > 601 and (step - 601) % NUISANCE_UPDATE_EVERY == 0:
            update_rank = 2
        if update_rank:
            # W and epsilon are fixed for this closed-form nuisance update.
            model.eval()
            with torch.no_grad():
                predicted, measured = physical_prediction_matrix(model, direct_ls, tensors, obs, config)
                update = model.set_nuisance_from_residual(measured - predicted, update_rank)
                nuisance_ratio = update["nuisance_norm"] / max(float(torch.linalg.vector_norm(measured).cpu()), EPS)
            nuisance_updates.append({"step": step, "stage": stage, "shrink": NUISANCE_SHRINK, "B_over_Es": nuisance_ratio, **update})
        model.train(); optimizer.zero_grad(set_to_none=True)
        batch_ids = view_ids_for_step(step, device)
        batch_data, batch_state = csi_losses(model, direct_ls, tensors, obs, target, config, incident_cache, batch_ids)
        batch_total = batch_data + batch_state
        if not bool(torch.isfinite(batch_total)):
            raise FloatingPointError(f"Non-finite CSI objective at step {step}.")
        batch_total.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_norm); optimizer.step()

        if step % SAVE_EVERY:
            continue
        model.eval()
        with torch.no_grad():
            all_ids = torch.arange(36, device=device)
            full_data, full_state = csi_losses(model, direct_ls, tensors, obs, target, config, incident_cache, all_ids)
            _predicted, full_measured = physical_prediction_matrix(model, direct_ls, tensors, obs, config)
            b_ratio = float(torch.linalg.vector_norm(model.nuisance(all_ids)).cpu() / max(float(torch.linalg.vector_norm(full_measured).cpu()), EPS))
            xx, yy, epsilon = epsilon_image(model, target, device, dtype)
        total = float((full_data + full_state).cpu())
        stats = epsilon_stats(epsilon)
        save_epsilon(epsilon, image_dir / f"epsilon_step_{step:04d}.png", f"CSI low-rank nuisance, {stage}, step {step}")
        checkpoint = OUTPUT_DIR / f"checkpoint_step_{step:04d}.pt"
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "step": step, "stage": stage, "config": asdict(config), "nuisance_rank": int(model.active_rank.cpu()), "nuisance_shrink": NUISANCE_SHRINK}, checkpoint)
        if total < best_total:
            best_total = total; shutil.copy2(checkpoint, OUTPUT_DIR / "best_checkpoint.pt")
        row = {"step": step, "stage": stage, "active_B_rank": int(model.active_rank.cpu()), "batch_views": ",".join(str(int(value) + 1) for value in batch_ids.cpu()), "batch_L_data": float(batch_data.detach().cpu()), "batch_L_state": float(batch_state.detach().cpu()), "L_data": float(full_data.cpu()), "L_state": float(full_state.cpu()), "physical_total": total, "B_over_Es": b_ratio, "elapsed_s": time.time() - started, **stats}
        history.append(row); write_csv(history, OUTPUT_DIR / "history.csv"); write_csv(nuisance_updates, OUTPUT_DIR / "nuisance_updates.csv")
        print(f"step={step:3d} {stage} Ldata/Lstate={row['L_data']:.4e}/{row['L_state']:.4e} B/Es={b_ratio:.4f} epsmax/std={stats['epsilon_max']:.3f}/{stats['epsilon_std']:.4f}", flush=True)
        if step > 300 and b_ratio > previous_b_ratio + 0.05 and stats["epsilon_max"] < 1.2 and stats["epsilon_std"] < 0.01:
            stopped_early = True
            print("Stopping: B grew while epsilon collapsed to background.", flush=True)
            break
        previous_b_ratio = b_ratio

    completed_steps = step
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "step": completed_steps, "config": asdict(config), "nuisance_rank": int(model.active_rank.cpu()), "nuisance_shrink": NUISANCE_SHRINK}, OUTPUT_DIR / "final_checkpoint.pt")
    selected = torch.load(OUTPUT_DIR / "best_checkpoint.pt", map_location=device, weights_only=False)
    model.load_state_dict(selected["model"]); model.eval()
    xx, yy, epsilon = epsilon_image(model, target, device, dtype)
    save_epsilon(epsilon, OUTPUT_DIR / "best_continuous_reconstruction.png", f"CSI low-rank nuisance best, step {selected['step']}")
    summary = {"steps_completed": completed_steps, "stopped_early": stopped_early, "best_checkpoint_step": int(selected["step"]), "best_physical_total": best_total, "best_epsilon": epsilon_stats(epsilon), "selection": "minimum full-36-view L_data+L_state only; no GT/RE/SSIM", "elapsed_s": time.time() - started}
    (OUTPUT_DIR / "metrics.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
