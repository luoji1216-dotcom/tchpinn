from __future__ import annotations

"""Exact-LS dual branch with fixed incident-residual subspace projection.

Q is built only from empty-target incident mismatch:

    R_inc = Eincident_measured - Eincident_M12

No scattered data, ground truth, or trainable measurement-error coefficients
are used to define Q.  The receiver loss is evaluated in the orthogonal
complement of span(Q).
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
OUTPUT_DIR = HERE / "results_fresnel_1GHz_m12_dual_branch_ls_projected_incident_residual_v2"
SEED = 20260722
STEPS = 1000
SAVE_EVERY = 100
VIEWS_PER_STEP = 6
FIELD_WEIGHT = 10.0
MAX_PROJECTION_RANK = 2
TARGET_EXPLAINED_ENERGY = 0.90
SYNTHETIC_RE_REFERENCE = 0.1284487
SYNTHETIC_RE_MAX_ALLOWED = 0.160
VALIDATION_VIEW_IDS = (5, 11, 17, 23, 29, 35)
EPS = 1.0e-12


def build_incident_residual_basis(
    measurement: dict[str, np.ndarray],
    indices: dict[int, np.ndarray],
    fits: list[Any],
    receiver_xy_np: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    residual = np.zeros((36, 49), dtype=np.complex128)
    fit_errors: list[float] = []
    for view in range(1, 37):
        rows = indices[view]
        fitted = fits[view - 1].evaluate(receiver_xy_np[rows])
        measured = measurement["incident"][rows]
        residual[view - 1, :] = measured - fitted
        fit_errors.append(float(np.linalg.norm(measured - fitted) / (np.linalg.norm(measured) + EPS)))
    u, singular, vh = np.linalg.svd(residual, full_matrices=False)
    energy = singular**2
    cumulative = np.cumsum(energy) / (np.sum(energy) + EPS)
    rank_needed = int(np.searchsorted(cumulative, TARGET_EXPLAINED_ENERGY) + 1)
    rank = min(MAX_PROJECTION_RANK, rank_needed)
    q_columns: list[np.ndarray] = []
    for mode in range(rank):
        basis_matrix = u[:, mode:mode + 1] @ vh[mode:mode + 1, :]
        q = basis_matrix.reshape(-1)
        q_columns.append(q / (np.linalg.norm(q) + EPS))
    q_full = np.stack(q_columns, axis=1) if q_columns else np.zeros((36 * 49, 0), dtype=np.complex128)
    metadata = {
        "definition": "R_inc(view,receiver)=Eincident_measured-Eincident_M12, SVD over the 36x49 complex matrix.",
        "rank_needed_for_90_percent": rank_needed,
        "rank_used": rank,
        "rank_cap": MAX_PROJECTION_RANK,
        "explained_energy_used": float(cumulative[rank - 1]) if rank else 0.0,
        "cumulative_energy_first_5": [float(x) for x in cumulative[:5]],
        "singular_values_first_5": [float(x) for x in singular[:5]],
        "mean_m12_incident_receiver_relative_error": float(np.mean(fit_errors)),
        "max_m12_incident_receiver_relative_error": float(np.max(fit_errors)),
    }
    return q_full, metadata


def selected_vector(
    matrix: torch.Tensor,
    vector_observed: torch.Tensor,
    indices: dict[int, np.ndarray],
    view_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, np.ndarray]:
    prediction_parts: list[torch.Tensor] = []
    observation_parts: list[torch.Tensor] = []
    flat_positions: list[np.ndarray] = []
    for view_id in view_ids.detach().cpu().tolist():
        rows_np = indices[int(view_id) + 1]
        rows = torch.as_tensor(rows_np, dtype=torch.long, device=matrix.device)
        prediction_parts.append(matrix[rows, int(view_id)])
        if vector_observed.ndim == 2:
            observation_parts.append(vector_observed[rows, int(view_id)])
        else:
            observation_parts.append(vector_observed[rows])
        flat_positions.append(int(view_id) * 49 + np.arange(rows_np.size))
    return torch.cat(prediction_parts), torch.cat(observation_parts), np.concatenate(flat_positions)


def project_orthogonal(values: torch.Tensor, q_selected: torch.Tensor) -> torch.Tensor:
    if q_selected.numel() == 0:
        return values
    gram = q_selected.conj().transpose(0, 1) @ q_selected
    rhs = q_selected.conj().transpose(0, 1) @ values
    eye = torch.eye(gram.shape[0], dtype=gram.dtype, device=gram.device)
    coeff = torch.linalg.solve(gram + 1.0e-8 * eye, rhs)
    return values - q_selected @ coeff


def receiver_losses_projected(
    ls_receiver: torch.Tensor,
    observed: torch.Tensor,
    q_full: torch.Tensor,
    indices: dict[int, np.ndarray],
    view_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    prediction, observation, flat_positions = selected_vector(ls_receiver, observed, indices, view_ids)
    raw_loss = dual.normalized_error(prediction, observation)
    q_selected = q_full[torch.as_tensor(flat_positions, dtype=torch.long, device=ls_receiver.device), :]
    projected_residual = project_orthogonal(prediction - observation, q_selected)
    projected_observation = project_orthogonal(observation, q_selected)
    projected_loss = torch.sum(torch.abs(projected_residual).square()) / (torch.sum(torch.abs(projected_observation).square()) + EPS)
    return projected_loss, raw_loss


def train_view_ids_for_step(step: int, train_ids: torch.Tensor) -> torch.Tensor:
    groups = train_ids.reshape(-1, VIEWS_PER_STEP)
    return groups[(step - 1) % groups.shape[0]]


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def saturated_island_warning(epsilon_map: np.ndarray) -> bool:
    high_fraction = float(np.mean(epsilon_map > 3.8))
    active_fraction = float(np.mean(epsilon_map > 1.5))
    return high_fraction > 0.01 and active_fraction < 0.08


def prepare_common() -> dict[str, Any]:
    config = closure.make_config()
    target = core.TargetSpec(name="Fresnel projected exact LS", kind="circle", eps_background=1.0, eps_object=4.0, roi_half_width=0.09)
    device, dtype = core.resolve_device(config.device), core.resolve_dtype(config.dtype)
    measurement, indices, directions_np, fits, fit_metadata = closure.make_geometry_and_m12(config)
    receiver_xy_np = closure.base.mapped_receiver_xy(measurement)
    receiver_xy = torch.as_tensor(receiver_xy_np, dtype=dtype, device=device)

    class Geometry:
        xy = receiver_xy_np

    integral = core.make_integral_tensors(Geometry(), target, config, device, dtype)
    if integral is None:
        raise RuntimeError("Projected exact-LS training requires static integral tensors.")
    direct_ls = core.make_direct_ls_tensors(integral, config, device, dtype)
    closure.base.replace_with_exp_iwt_green(direct_ls, receiver_xy_np, config.k0)
    incident = closure.M12IncidentProvider(fits, integral.quad_xy, device, direct_ls.domain_green.dtype)
    directions_by_view = torch.as_tensor(np.stack([directions_np[indices[view][0]] for view in range(1, 37)]), dtype=dtype, device=device)
    q_np, projection_metadata = build_incident_residual_basis(measurement, indices, fits, receiver_xy_np)
    q_full = torch.as_tensor(q_np, dtype=direct_ls.domain_green.dtype, device=device)
    observed_real = torch.as_tensor(measurement["scattered"], dtype=direct_ls.domain_green.dtype, device=device)
    validation_ids = torch.as_tensor(VALIDATION_VIEW_IDS, dtype=torch.long, device=device)
    validation_set = set(VALIDATION_VIEW_IDS)
    train_ids = torch.as_tensor([view for view in range(36) if view not in validation_set], dtype=torch.long, device=device)
    return {
        "config": config,
        "target": target,
        "device": device,
        "dtype": dtype,
        "measurement": measurement,
        "indices": indices,
        "fits": fits,
        "fit_metadata": fit_metadata,
        "receiver_xy": receiver_xy,
        "receiver_xy_np": receiver_xy_np,
        "direct_ls": direct_ls,
        "incident": incident,
        "directions_by_view": directions_by_view,
        "q_full": q_full,
        "projection_metadata": projection_metadata,
        "observed_real": observed_real,
        "train_ids": train_ids,
        "validation_ids": validation_ids,
    }


def train_one(
    common: dict[str, Any],
    observed: torch.Tensor,
    run_dir: Path,
    selection_view_ids: torch.Tensor,
    train_ids: torch.Tensor,
    validation_ids: torch.Tensor | None,
    synthetic: bool,
) -> dict[str, Any]:
    config: core.TrainConfig = common["config"]
    target: core.TargetSpec = common["target"]
    device: torch.device = common["device"]
    dtype: torch.dtype = common["dtype"]
    direct_ls = common["direct_ls"]
    incident = common["incident"]
    indices = common["indices"]
    receiver_xy = common["receiver_xy"]
    directions_by_view = common["directions_by_view"]
    q_full = common["q_full"]

    core.set_global_seed(SEED)
    model = dual.LSSupervisedDualBranch(config, target).to(device=device)
    epsilon_optimizer = torch.optim.Adam(model.epsilon_branch.parameters(), lr=config.learning_rate)
    field_optimizer = torch.optim.Adam(model.field_branch.parameters(), lr=config.learning_rate)
    image_dir = run_dir / "epsilon_images"
    image_dir.mkdir(parents=True, exist_ok=False)
    history: list[dict[str, Any]] = []
    best_selection = float("inf")
    stopped_reason: str | None = None
    started = time.time()
    for step in range(1, STEPS + 1):
        model.train()
        epsilon_optimizer.zero_grad(set_to_none=True)
        field_optimizer.zero_grad(set_to_none=True)
        _epsilon, ls_inside, ls_receiver = dual.exact_ls_fields(model, direct_ls, incident, config)
        batch_views = train_view_ids_for_step(step, train_ids)
        projected_loss, _raw_loss = receiver_losses_projected(ls_receiver, observed, q_full, indices, batch_views)
        inside_loss, field_receiver_loss = dual.field_losses(model, direct_ls, ls_inside, ls_receiver, receiver_xy, indices, directions_by_view, batch_views)
        total_loss = projected_loss + FIELD_WEIGHT * inside_loss + FIELD_WEIGHT * field_receiver_loss
        if not bool(torch.isfinite(total_loss)):
            raise FloatingPointError(f"Non-finite projected objective at step {step}.")
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.epsilon_branch.parameters(), config.gradient_clip_norm)
        torch.nn.utils.clip_grad_norm_(model.field_branch.parameters(), config.gradient_clip_norm)
        epsilon_optimizer.step()
        field_optimizer.step()
        if step % SAVE_EVERY:
            continue

        model.eval()
        with torch.no_grad():
            _epsilon, full_inside, full_receiver = dual.exact_ls_fields(model, direct_ls, incident, config)
            train_projected, train_raw = receiver_losses_projected(full_receiver, observed, q_full, indices, train_ids)
            selection_projected, selection_raw = receiver_losses_projected(full_receiver, observed, q_full, indices, selection_view_ids)
            if validation_ids is not None:
                validation_projected, validation_raw = receiver_losses_projected(full_receiver, observed, q_full, indices, validation_ids)
            else:
                validation_projected, validation_raw = selection_projected, selection_raw
            xx, yy, epsilon_map = exact.epsilon_image(model, target, device, dtype)
        full_inside_loss, full_field_receiver_loss = dual.field_losses(model, direct_ls, full_inside, full_receiver, receiver_xy, indices, directions_by_view, train_ids)
        total_value = float((train_projected + FIELD_WEIGHT * full_inside_loss + FIELD_WEIGHT * full_field_receiver_loss).detach().cpu())
        selection_value = float(selection_projected.cpu())
        exact.save_epsilon(epsilon_map, image_dir / f"epsilon_step_{step:04d}.png", f"Projected exact LS {'synthetic' if synthetic else 'real'}, step {step}")
        checkpoint = run_dir / f"checkpoint_step_{step:04d}.pt"
        torch.save({
            "model": model.state_dict(),
            "epsilon_optimizer": epsilon_optimizer.state_dict(),
            "field_optimizer": field_optimizer.state_dict(),
            "step": step,
            "config": asdict(config),
            "projection_metadata": common["projection_metadata"],
        }, checkpoint)
        if selection_value < best_selection:
            best_selection = selection_value
            shutil.copy2(checkpoint, run_dir / "best_checkpoint.pt")
        row: dict[str, Any] = {
            "step": step,
            "receiver_train_projected": float(train_projected.cpu()),
            "receiver_train_raw": float(train_raw.cpu()),
            "receiver_selection_projected": selection_value,
            "receiver_selection_raw": float(selection_raw.cpu()),
            "receiver_validation_projected": float(validation_projected.cpu()),
            "receiver_validation_raw": float(validation_raw.cpu()),
            "L_field_inside_train": float(full_inside_loss.detach().cpu()),
            "L_field_receiver_train": float(full_field_receiver_loss.detach().cpu()),
            "total_train": total_value,
            "epsilon_min": float(epsilon_map.min()),
            "epsilon_max": float(epsilon_map.max()),
            "epsilon_mean": float(epsilon_map.mean()),
            "epsilon_std": float(epsilon_map.std()),
            "elapsed_s": time.time() - started,
        }
        if synthetic:
            row.update(exact.offline_metrics(epsilon_map, xx, yy))
        history.append(row)
        write_csv(history, run_dir / "history.csv")
        if synthetic:
            print(
                f"synthetic step={step:4d} Lproj/raw={row['receiver_selection_projected']:.3e}/{row['receiver_selection_raw']:.3e} "
                f"epsmax={row['epsilon_max']:.3f} RE/SSIM={row['continuous_RE']:.4f}/{row['continuous_SSIM']:.4f}",
                flush=True,
            )
        else:
            print(
                f"real step={step:4d} Lval proj/raw={row['receiver_validation_projected']:.3e}/{row['receiver_validation_raw']:.3e} "
                f"epsmax/std={row['epsilon_max']:.3f}/{row['epsilon_std']:.3f}",
                flush=True,
            )
        if not synthetic and step >= 300 and saturated_island_warning(epsilon_map):
            stopped_reason = f"Stopped at step {step}: epsilon map remained saturated-island dominated."
            break

    final_step = history[-1]["step"] if history else 0
    torch.save({
        "model": model.state_dict(),
        "epsilon_optimizer": epsilon_optimizer.state_dict(),
        "field_optimizer": field_optimizer.state_dict(),
        "step": final_step,
        "config": asdict(config),
        "projection_metadata": common["projection_metadata"],
    }, run_dir / "final_checkpoint.pt")
    selected = torch.load(run_dir / "best_checkpoint.pt", map_location=device, weights_only=False)
    model.load_state_dict(selected["model"])
    model.eval()
    xx, yy, epsilon_map = exact.epsilon_image(model, target, device, dtype)
    exact.save_epsilon(epsilon_map, run_dir / "best_continuous_reconstruction.png", f"Projected exact LS best, step {selected['step']}")
    summary: dict[str, Any] = {
        "steps_completed": int(final_step),
        "best_checkpoint_step": int(selected["step"]),
        "best_selection_projected_receiver_loss": best_selection,
        "stopped_reason": stopped_reason,
        "best_epsilon": {
            "min": float(epsilon_map.min()),
            "max": float(epsilon_map.max()),
            "mean": float(epsilon_map.mean()),
            "std": float(epsilon_map.std()),
        },
        "projection_metadata": common["projection_metadata"],
        "selection": "synthetic uses projected full-view receiver loss; real uses projected validation receiver loss. No GT/RE/SSIM selection.",
        "elapsed_s": time.time() - started,
    }
    if synthetic:
        summary["offline_synthetic_evaluation_only"] = exact.offline_metrics(epsilon_map, xx, yy)
    (run_dir / "metrics.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def main() -> None:
    if OUTPUT_DIR.exists():
        raise FileExistsError(f"Refusing to overwrite existing result directory: {OUTPUT_DIR}")
    common = prepare_common()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=False)
    (OUTPUT_DIR / "config.json").write_text(json.dumps({
        "config": asdict(common["config"]),
        "projection_metadata": common["projection_metadata"],
        "projection_source": "empty-target incident data only: Eincident_measured-Eincident_M12",
        "loss": "||(I-QQH)(Es_LS(epsilon)-Es_observed)||^2 / ||(I-QQH)Es_observed||^2 + field LS supervision",
        "forbidden": "No trainable B, no Es residual SVD basis, no GT/material/geometry prior, no field-vs-measured receiver loss.",
        "view_split": {
            "train_views_1based": [int(view) + 1 for view in common["train_ids"].cpu().tolist()],
            "validation_views_1based": [int(view) + 1 for view in common["validation_ids"].cpu().tolist()],
        },
        "incident_fit_metadata": common["fit_metadata"],
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"projection_metadata": common["projection_metadata"]}, indent=2, ensure_ascii=False), flush=True)

    synthetic_dir = OUTPUT_DIR / "synthetic_closure"
    synthetic_observed = closure.synthesize_receiver_data(common["direct_ls"], common["incident"], common["config"])
    all_views = torch.arange(36, dtype=torch.long, device=common["device"])
    synthetic_summary = train_one(
        common=common,
        observed=synthetic_observed,
        run_dir=synthetic_dir,
        selection_view_ids=all_views,
        train_ids=all_views,
        validation_ids=None,
        synthetic=True,
    )
    synthetic_re = float(synthetic_summary["offline_synthetic_evaluation_only"]["continuous_RE"])
    if synthetic_re > SYNTHETIC_RE_MAX_ALLOWED:
        summary = {
            "status": "synthetic_failed",
            "synthetic_RE": synthetic_re,
            "reference_RE": SYNTHETIC_RE_REFERENCE,
            "max_allowed_RE": SYNTHETIC_RE_MAX_ALLOWED,
            "real_training_started": False,
            "projection_metadata": common["projection_metadata"],
        }
        (OUTPUT_DIR / "metrics.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
        return

    real_dir = OUTPUT_DIR / "real_30train6val"
    real_summary = train_one(
        common=common,
        observed=common["observed_real"],
        run_dir=real_dir,
        selection_view_ids=common["validation_ids"],
        train_ids=common["train_ids"],
        validation_ids=common["validation_ids"],
        synthetic=False,
    )
    summary = {
        "status": "complete",
        "synthetic_summary": synthetic_summary,
        "real_summary": real_summary,
        "projection_metadata": common["projection_metadata"],
    }
    (OUTPUT_DIR / "metrics.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
