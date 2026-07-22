from __future__ import annotations

"""Dual-branch synthetic closure with epsilon-owned exact-LS data loss.

The field branch is supervised only by fields generated from the current
epsilon LS solve.  It never receives a loss against measured/synthetic data.
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

import run_fresnel_1ghz_m12_exp_iwt_epsilon_only_exact_ls_closure as exact
import run_fresnel_1ghz_m12_exp_iwt_synthetic_closure as closure
from square_target import pinn_pixel_inverse_core as core


HERE = Path(__file__).resolve().parent
OUTPUT_DIR = HERE / "results_fresnel_1GHz_m12_dual_branch_ls_synthetic_closure1000"
SEED = 20260722
STEPS = 1000
SAVE_EVERY = 100
VIEWS_PER_STEP = 6
FIELD_WEIGHT = 10.0
EPS = 1.0e-12


class LSSupervisedDualBranch(torch.nn.Module):
    def __init__(self, config: core.TrainConfig, target: core.TargetSpec) -> None:
        super().__init__()
        # Construct epsilon first to preserve the exact-LS epsilon initialization
        # stream; field weights are independent and never alter epsilon loss.
        self.epsilon_branch = core.EpsilonBranch(config, target)
        epsilon_last = self.epsilon_branch.mlp.net[-1]
        if not isinstance(epsilon_last, torch.nn.Linear):
            raise TypeError("Expected a Linear epsilon final layer.")
        torch.nn.init.normal_(epsilon_last.weight, mean=0.0, std=1.0e-3)
        self.field_branch = core.FieldBranch(config, target.roi_half_width)

    def epsilon(self, xy: torch.Tensor) -> torch.Tensor:
        return self.epsilon_branch(xy)

    def scattered(self, xy: torch.Tensor, directions: torch.Tensor) -> torch.Tensor:
        values = self.field_branch(xy, directions)
        return torch.complex(values[:, 0], values[:, 1])


def exact_ls_fields(model: LSSupervisedDualBranch, direct_ls: Any, incident: closure.M12IncidentProvider, config: core.TrainConfig) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    integral = direct_ls.integral
    epsilon = model.epsilon(integral.quad_xy).reshape(-1, 1)
    chi = (epsilon - 1.0).to(direct_ls.domain_green.dtype)
    scale = float(config.k0**2 * integral.area_weight)
    n_cells = epsilon.numel()
    system = torch.eye(n_cells, dtype=direct_ls.domain_green.dtype, device=epsilon.device) - scale * direct_ls.domain_green * chi.reshape(1, -1)
    total = torch.linalg.solve(system, incident.quad)
    scattered_inside = total - incident.quad
    receiver_green = torch.complex(integral.green_re, integral.green_im).to(direct_ls.domain_green.dtype)
    scattered_receiver = scale * (receiver_green @ (chi * total))
    return epsilon, scattered_inside, scattered_receiver


def normalized_error(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.sum(torch.abs(prediction - target).square()) / (torch.sum(torch.abs(target).square()) + EPS)


def view_ids_for_step(step: int, device: torch.device) -> torch.Tensor:
    group = (step - 1) % (36 // VIEWS_PER_STEP)
    return torch.arange(group * VIEWS_PER_STEP, (group + 1) * VIEWS_PER_STEP, device=device)


def field_losses(model: LSSupervisedDualBranch, direct_ls: Any, ls_inside: torch.Tensor, ls_receiver: torch.Tensor, receiver_xy: torch.Tensor, indices: dict[int, np.ndarray], directions_by_view: torch.Tensor, view_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Detach LS targets so field fitting cannot alter epsilon or bypass L_data."""
    quad_xy = direct_ls.integral.quad_xy
    n_cells = quad_xy.shape[0]
    n_views = int(view_ids.numel())
    directions = directions_by_view[view_ids]
    expanded_xy = quad_xy.repeat(n_views, 1)
    expanded_dirs = directions[:, None, :].expand(-1, n_cells, -1).reshape(-1, 2)
    predicted_inside = model.scattered(expanded_xy, expanded_dirs).reshape(n_views, n_cells).transpose(0, 1)
    target_inside = ls_inside[:, view_ids].detach()
    loss_inside = normalized_error(predicted_inside, target_inside)
    receiver_predictions: list[torch.Tensor] = []
    receiver_targets: list[torch.Tensor] = []
    for view_id in view_ids.detach().cpu().tolist():
        rows = torch.as_tensor(indices[int(view_id) + 1], dtype=torch.long, device=quad_xy.device)
        xy = receiver_xy[rows]
        directions_view = directions_by_view[int(view_id)].expand(rows.numel(), 2)
        receiver_predictions.append(model.scattered(xy, directions_view))
        receiver_targets.append(ls_receiver[rows, int(view_id)].detach())
    loss_receiver = normalized_error(torch.cat(receiver_predictions), torch.cat(receiver_targets))
    return loss_inside, loss_receiver


def epsilon_image(model: LSSupervisedDualBranch, target: core.TargetSpec, device: torch.device, dtype: torch.dtype) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return exact.epsilon_image(model, target, device, dtype)


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)


def main() -> None:
    if OUTPUT_DIR.exists():
        raise FileExistsError(f"Refusing to overwrite existing result directory: {OUTPUT_DIR}")
    config = closure.make_config()
    target = core.TargetSpec(name="Fresnel dual LS synthetic closure", kind="circle", eps_background=1.0, eps_object=4.0, roi_half_width=0.09)
    device, dtype = core.resolve_device(config.device), core.resolve_dtype(config.dtype)
    measurement, indices, directions_np, fits, fit_metadata = closure.make_geometry_and_m12(config)
    receiver_xy_np = closure.base.mapped_receiver_xy(measurement)
    receiver_xy = torch.as_tensor(receiver_xy_np, dtype=dtype, device=device)

    class Geometry:
        xy = receiver_xy_np

    integral = core.make_integral_tensors(Geometry(), target, config, device, dtype)
    if integral is None:
        raise RuntimeError("Dual-branch closure requires static integral tensors.")
    direct_ls = core.make_direct_ls_tensors(integral, config, device, dtype)
    closure.base.replace_with_exp_iwt_green(direct_ls, receiver_xy_np, config.k0)
    incident = closure.M12IncidentProvider(fits, integral.quad_xy, device, direct_ls.domain_green.dtype)
    synthetic_observation = closure.synthesize_receiver_data(direct_ls, incident, config)
    directions_by_view = torch.as_tensor(np.stack([directions_np[indices[view][0]] for view in range(1, 37)]), dtype=dtype, device=device)
    core.set_global_seed(SEED)
    model = LSSupervisedDualBranch(config, target).to(device=device)
    epsilon_optimizer = torch.optim.Adam(model.epsilon_branch.parameters(), lr=config.learning_rate)
    field_optimizer = torch.optim.Adam(model.field_branch.parameters(), lr=config.learning_rate)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=False)
    image_dir = OUTPUT_DIR / "epsilon_images"; image_dir.mkdir()
    np.save(OUTPUT_DIR / "synthetic_receiver_scattered_m12.npy", synthetic_observation.detach().cpu().numpy())
    (OUTPUT_DIR / "config.json").write_text(json.dumps({
        "config": asdict(config),
        "convention": {"geometry": "R(-theta)", "incident": "M12IncidentProvider only", "time": "exp(+i omega t)", "green": "-i/4 H0^(2)(k0*distance)", "chi": "epsilon-1"},
        "loss": {"L_data": "normalized_error(Es_LS_receiver, Es_measured)", "L_field_inside": "error(Es_field_inside, detach(Etotal_LS-Einc))", "L_field_receiver": "error(Es_field_receiver, detach(Es_LS_receiver))", "total": "L_data+10*L_field_inside+10*L_field_receiver", "forbidden": "error(Es_field_receiver, Es_measured)"},
        "selection": "minimum synthetic L_data only; no GT/RE/SSIM",
        "synthetic_generation_only": {"center_m": list(closure.SYNTHETIC_CENTER), "radius_m": closure.SYNTHETIC_RADIUS, "epsilon": closure.SYNTHETIC_EPSILON, "not_used_in_optimization": True},
        "incident_fit_metadata": fit_metadata,
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    history: list[dict[str, Any]] = []
    best_data = float("inf")
    started = time.time()
    for step in range(1, STEPS + 1):
        model.train(); epsilon_optimizer.zero_grad(set_to_none=True); field_optimizer.zero_grad(set_to_none=True)
        epsilon, ls_inside, ls_receiver = exact_ls_fields(model, direct_ls, incident, config)
        data_loss = normalized_error(ls_receiver, synthetic_observation)
        batch_views = view_ids_for_step(step, device)
        inside_loss, field_receiver_loss = field_losses(model, direct_ls, ls_inside, ls_receiver, receiver_xy, indices, directions_by_view, batch_views)
        total_loss = data_loss + FIELD_WEIGHT * inside_loss + FIELD_WEIGHT * field_receiver_loss
        if not bool(torch.isfinite(total_loss)):
            raise FloatingPointError(f"Non-finite dual-branch objective at step {step}.")
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.epsilon_branch.parameters(), config.gradient_clip_norm)
        torch.nn.utils.clip_grad_norm_(model.field_branch.parameters(), config.gradient_clip_norm)
        epsilon_optimizer.step(); field_optimizer.step()
        if step % SAVE_EVERY:
            continue
        model.eval()
        with torch.no_grad():
            epsilon, full_inside, full_receiver = exact_ls_fields(model, direct_ls, incident, config)
            full_data = normalized_error(full_receiver, synthetic_observation)
            xx, yy, epsilon_map = epsilon_image(model, target, device, dtype)
        # Field targets are detached by definition; no epsilon gradient is needed here.
        full_inside_loss, full_field_receiver_loss = field_losses(model, direct_ls, full_inside, full_receiver, receiver_xy, indices, directions_by_view, torch.arange(36, device=device))
        total_value = float((full_data + FIELD_WEIGHT * full_inside_loss + FIELD_WEIGHT * full_field_receiver_loss).detach().cpu())
        data_value = float(full_data.cpu())
        offline = exact.offline_metrics(epsilon_map, xx, yy)
        exact.save_epsilon(epsilon_map, image_dir / f"epsilon_step_{step:04d}.png", f"Dual branch LS synthetic, step {step}")
        checkpoint = OUTPUT_DIR / f"checkpoint_step_{step:04d}.pt"
        torch.save({"model": model.state_dict(), "epsilon_optimizer": epsilon_optimizer.state_dict(), "field_optimizer": field_optimizer.state_dict(), "step": step, "config": asdict(config)}, checkpoint)
        if data_value < best_data:
            best_data = data_value; shutil.copy2(checkpoint, OUTPUT_DIR / "best_checkpoint.pt")
        row = {"step": step, "L_data": data_value, "L_field_inside": float(full_inside_loss.detach().cpu()), "L_field_receiver": float(full_field_receiver_loss.detach().cpu()), "total": total_value, "epsilon_min": float(epsilon_map.min()), "epsilon_max": float(epsilon_map.max()), "epsilon_mean": float(epsilon_map.mean()), "epsilon_std": float(epsilon_map.std()), "elapsed_s": time.time() - started, **offline}
        history.append(row); write_csv(history, OUTPUT_DIR / "history.csv")
        print(f"step={step:4d} Ldata/Lfield_in/Lfield_rx={row['L_data']:.3e}/{row['L_field_inside']:.3e}/{row['L_field_receiver']:.3e} epsmax={row['epsilon_max']:.3f} RE/SSIM={offline['continuous_RE']:.4f}/{offline['continuous_SSIM']:.4f}", flush=True)

    torch.save({"model": model.state_dict(), "epsilon_optimizer": epsilon_optimizer.state_dict(), "field_optimizer": field_optimizer.state_dict(), "step": STEPS, "config": asdict(config)}, OUTPUT_DIR / "final_checkpoint.pt")
    selected = torch.load(OUTPUT_DIR / "best_checkpoint.pt", map_location=device, weights_only=False)
    model.load_state_dict(selected["model"]); model.eval()
    xx, yy, epsilon_map = epsilon_image(model, target, device, dtype)
    exact.save_epsilon(epsilon_map, OUTPUT_DIR / "best_continuous_reconstruction.png", f"Dual branch LS best, step {selected['step']}")
    summary = {"steps_completed": STEPS, "best_checkpoint_step": int(selected["step"]), "best_L_data": best_data, "offline_synthetic_evaluation_only": exact.offline_metrics(epsilon_map, xx, yy), "elapsed_s": time.time() - started, "real_fresnel_training_started": False}
    (OUTPUT_DIR / "metrics.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
