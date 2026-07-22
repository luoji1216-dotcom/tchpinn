from __future__ import annotations

"""Real Fresnel dual-branch exact-LS inversion with a 30/6 view split."""

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
OUTPUT_DIR = HERE / "results_fresnel_1GHz_m12_dual_branch_ls_real_30train6val"
SEED = 20260722
STEPS = 1000
SAVE_EVERY = 100
VIEWS_PER_STEP = 6
FIELD_WEIGHT = 10.0
EPS = 1.0e-12
# Every sixth view is held out, covering the full 360-degree target rotation.
VALIDATION_VIEW_IDS = (5, 11, 17, 23, 29, 35)  # zero based: views 6,12,...,36


def selected_receiver_error(ls_receiver: torch.Tensor, observed: torch.Tensor, indices: dict[int, np.ndarray], view_ids: torch.Tensor) -> torch.Tensor:
    prediction_parts: list[torch.Tensor] = []
    observation_parts: list[torch.Tensor] = []
    for view_id in view_ids.detach().cpu().tolist():
        rows = torch.as_tensor(indices[int(view_id) + 1], dtype=torch.long, device=ls_receiver.device)
        prediction_parts.append(ls_receiver[rows, int(view_id)])
        observation_parts.append(observed[rows])
    return dual.normalized_error(torch.cat(prediction_parts), torch.cat(observation_parts))


def train_view_ids_for_step(step: int, train_ids: torch.Tensor) -> torch.Tensor:
    groups = train_ids.reshape(-1, VIEWS_PER_STEP)
    return groups[(step - 1) % groups.shape[0]]


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)


def main() -> None:
    if OUTPUT_DIR.exists():
        raise FileExistsError(f"Refusing to overwrite existing result directory: {OUTPUT_DIR}")
    config = closure.make_config()
    target = core.TargetSpec(name="Fresnel real dual LS no-geometry", kind="circle", eps_background=1.0, eps_object=4.0, roi_half_width=0.09)
    device, dtype = core.resolve_device(config.device), core.resolve_dtype(config.dtype)
    measurement, indices, directions_np, fits, fit_metadata = closure.make_geometry_and_m12(config)
    receiver_xy_np = closure.base.mapped_receiver_xy(measurement)
    receiver_xy = torch.as_tensor(receiver_xy_np, dtype=dtype, device=device)
    observed = torch.as_tensor(measurement["scattered"], dtype=complex(0).dtype if False else torch.complex64, device=device)

    class Geometry:
        xy = receiver_xy_np

    integral = core.make_integral_tensors(Geometry(), target, config, device, dtype)
    if integral is None:
        raise RuntimeError("Real dual-branch inversion requires static integral tensors.")
    direct_ls = core.make_direct_ls_tensors(integral, config, device, dtype)
    closure.base.replace_with_exp_iwt_green(direct_ls, receiver_xy_np, config.k0)
    incident = closure.M12IncidentProvider(fits, integral.quad_xy, device, direct_ls.domain_green.dtype)
    if observed.dtype != direct_ls.domain_green.dtype:
        observed = observed.to(direct_ls.domain_green.dtype)
    directions_by_view = torch.as_tensor(np.stack([directions_np[indices[view][0]] for view in range(1, 37)]), dtype=dtype, device=device)
    validation_ids = torch.as_tensor(VALIDATION_VIEW_IDS, dtype=torch.long, device=device)
    validation_set = set(VALIDATION_VIEW_IDS)
    train_ids = torch.as_tensor([view for view in range(36) if view not in validation_set], dtype=torch.long, device=device)
    if train_ids.numel() != 30 or validation_ids.numel() != 6:
        raise AssertionError("Expected a 30/6 view split.")

    core.set_global_seed(SEED)
    model = dual.LSSupervisedDualBranch(config, target).to(device=device)
    epsilon_optimizer = torch.optim.Adam(model.epsilon_branch.parameters(), lr=config.learning_rate)
    field_optimizer = torch.optim.Adam(model.field_branch.parameters(), lr=config.learning_rate)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=False)
    image_dir = OUTPUT_DIR / "epsilon_images"; image_dir.mkdir()
    (OUTPUT_DIR / "config.json").write_text(json.dumps({
        "config": asdict(config),
        "view_split": {"train_views_1based": [int(view) + 1 for view in train_ids.cpu().tolist()], "validation_views_1based": [int(view) + 1 for view in validation_ids.cpu().tolist()]},
        "convention": {"geometry": "R(-theta)", "incident": "M12IncidentProvider only", "time": "exp(+i omega t)", "green": "-i/4 H0^(2)(k0*distance)", "chi": "epsilon-1"},
        "loss": {"L_data_train": "normalized_error(Es_LS_receiver[30 train views], Es_measured[30 train views])", "L_field_inside": "error(Es_field_inside, detach(Es_LS_inside))", "L_field_receiver": "error(Es_field_receiver, detach(Es_LS_receiver))", "total_train": "L_data_train+10*L_field_inside+10*L_field_receiver", "forbidden": "error(Es_field_receiver, Es_measured)"},
        "checkpoint_selection": "minimum validation L_data exact-LS only; no GT, RE, SSIM, material, or geometry prior",
        "incident_fit_metadata": fit_metadata,
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    history: list[dict[str, Any]] = []
    best_validation = float("inf")
    started = time.time()
    for step in range(1, STEPS + 1):
        model.train(); epsilon_optimizer.zero_grad(set_to_none=True); field_optimizer.zero_grad(set_to_none=True)
        _epsilon, ls_inside, ls_receiver = dual.exact_ls_fields(model, direct_ls, incident, config)
        data_train = selected_receiver_error(ls_receiver, observed, indices, train_ids)
        field_views = train_view_ids_for_step(step, train_ids)
        inside_loss, field_receiver_loss = dual.field_losses(model, direct_ls, ls_inside, ls_receiver, receiver_xy, indices, directions_by_view, field_views)
        total_loss = data_train + FIELD_WEIGHT * inside_loss + FIELD_WEIGHT * field_receiver_loss
        if not bool(torch.isfinite(total_loss)):
            raise FloatingPointError(f"Non-finite real dual-branch objective at step {step}.")
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.epsilon_branch.parameters(), config.gradient_clip_norm)
        torch.nn.utils.clip_grad_norm_(model.field_branch.parameters(), config.gradient_clip_norm)
        epsilon_optimizer.step(); field_optimizer.step()
        if step % SAVE_EVERY:
            continue
        model.eval()
        with torch.no_grad():
            _epsilon, full_inside, full_receiver = dual.exact_ls_fields(model, direct_ls, incident, config)
            full_train_data = selected_receiver_error(full_receiver, observed, indices, train_ids)
            validation_data = selected_receiver_error(full_receiver, observed, indices, validation_ids)
            xx, yy, epsilon_map = exact.epsilon_image(model, target, device, dtype)
        full_inside_loss, full_field_receiver_loss = dual.field_losses(model, direct_ls, full_inside, full_receiver, receiver_xy, indices, directions_by_view, train_ids)
        total_value = float((full_train_data + FIELD_WEIGHT * full_inside_loss + FIELD_WEIGHT * full_field_receiver_loss).detach().cpu())
        validation_value = float(validation_data.cpu())
        exact.save_epsilon(epsilon_map, image_dir / f"epsilon_step_{step:04d}.png", f"Real dual branch LS, step {step}")
        checkpoint = OUTPUT_DIR / f"checkpoint_step_{step:04d}.pt"
        torch.save({"model": model.state_dict(), "epsilon_optimizer": epsilon_optimizer.state_dict(), "field_optimizer": field_optimizer.state_dict(), "step": step, "config": asdict(config)}, checkpoint)
        if validation_value < best_validation:
            best_validation = validation_value; shutil.copy2(checkpoint, OUTPUT_DIR / "best_checkpoint.pt")
        row = {"step": step, "L_data_train": float(full_train_data.cpu()), "L_data_validation": validation_value, "L_field_inside_train": float(full_inside_loss.detach().cpu()), "L_field_receiver_train": float(full_field_receiver_loss.detach().cpu()), "total_train": total_value, "epsilon_min": float(epsilon_map.min()), "epsilon_max": float(epsilon_map.max()), "epsilon_mean": float(epsilon_map.mean()), "epsilon_std": float(epsilon_map.std()), "elapsed_s": time.time() - started}
        history.append(row); write_csv(history, OUTPUT_DIR / "history.csv")
        print(f"step={step:4d} Ltrain/Lval={row['L_data_train']:.3e}/{validation_value:.3e} Lfield={row['L_field_inside_train']:.3e}/{row['L_field_receiver_train']:.3e} epsmax={row['epsilon_max']:.3f}", flush=True)

    torch.save({"model": model.state_dict(), "epsilon_optimizer": epsilon_optimizer.state_dict(), "field_optimizer": field_optimizer.state_dict(), "step": STEPS, "config": asdict(config)}, OUTPUT_DIR / "final_checkpoint.pt")
    selected = torch.load(OUTPUT_DIR / "best_checkpoint.pt", map_location=device, weights_only=False)
    model.load_state_dict(selected["model"]); model.eval()
    xx, yy, epsilon_map = exact.epsilon_image(model, target, device, dtype)
    exact.save_epsilon(epsilon_map, OUTPUT_DIR / "best_continuous_reconstruction.png", f"Real dual branch LS best validation, step {selected['step']}")
    summary = {"steps_completed": STEPS, "best_checkpoint_step": int(selected["step"]), "best_validation_L_data": best_validation, "best_epsilon": {"min": float(epsilon_map.min()), "max": float(epsilon_map.max()), "mean": float(epsilon_map.mean()), "std": float(epsilon_map.std())}, "selection": "minimum validation exact-LS L_data only; no GT/RE/SSIM", "elapsed_s": time.time() - started}
    (OUTPUT_DIR / "metrics.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
