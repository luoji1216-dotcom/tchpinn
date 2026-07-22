from __future__ import annotations

"""Synthetic closure test for the exp(+i omega t) Fresnel physics chain.

The same M=12 incident provider, Green tensors, receiver integral, and
interior state equation are used both to synthesize data and invert epsilon.
No measured scattered field, GT mask, or geometric prior enters optimization.
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
OUTPUT_DIR = HERE / "results_fresnel_1GHz_m12_exp_iwt_synthetic_closure500"
SEED = 20260722
GRID_SIZE = 32
STEPS = 500
VIEWS_PER_STEP = 6
SAVE_EVERY = 100
BOUNDARY_POINTS = 64
BOUNDARY_WEIGHT = 0.02
EPS = 1.0e-12
SYNTHETIC_CENTER = (0.0, -0.03)
SYNTHETIC_RADIUS = 0.015
SYNTHETIC_EPSILON = 3.0


def make_config() -> core.TrainConfig:
    return core.TrainConfig(
        frequency_hz=1.0e9,
        eps_min=1.0,
        eps_max=4.0,
        eps_initial=1.5,
        epsilon_constant_init=True,
        field_hidden_layers=5,
        field_hidden_units=80,
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
        learning_rate=1.0e-3,
        gradient_clip_norm=1.0,
        random_seed=SEED,
        dtype="float32",
        device="auto",
    )


def make_geometry_and_m12(config: core.TrainConfig) -> tuple[dict[str, np.ndarray], dict[int, np.ndarray], np.ndarray, list[Any], dict[str, Any]]:
    measurement = base.load_measurement()
    indices = base.view_indices(measurement["views"])
    receiver_xy = base.mapped_receiver_xy(measurement)
    source_angle = -measurement["target_rotation"]
    directions = np.column_stack((-np.cos(source_angle), -np.sin(source_angle))).astype(np.float64)
    fits: list[Any] = []
    metadata: dict[str, Any] = {}
    for view in range(1, 37):
        rows = indices[view]
        fit = core.fit_fourier_bessel_incident(receiver_xy[rows], measurement["incident"][rows], config.k0, base.M12_ORDER, base.M12_RIDGE)
        fits.append(fit)
        metadata[f"view_{view:02d}"] = {"fit_metrics": fit.fit_metrics, "coefficient_count": int(fit.coefficients.size)}
    return measurement, indices, directions, fits, metadata


class M12IncidentProvider:
    """The only incident-field implementation used in synthetic generation and loss."""

    def __init__(self, fits: list[Any], quad_xy: torch.Tensor, device: torch.device, complex_dtype: torch.dtype) -> None:
        values = np.stack([fit.evaluate(quad_xy.detach().cpu().numpy()) for fit in fits], axis=1)
        self.quad = torch.as_tensor(values, dtype=complex_dtype, device=device)

    def at_quadrature(self, view_ids: torch.Tensor) -> torch.Tensor:
        return self.quad[:, view_ids]


class IntegralFieldPINN(torch.nn.Module):
    """Es_inside MLP plus epsilon MLP; no plane-wave incident path exists here."""

    def __init__(self, config: core.TrainConfig, target: core.TargetSpec) -> None:
        super().__init__()
        self.field_branch = core.FieldBranch(config, target.roi_half_width)
        self.epsilon_branch = core.EpsilonBranch(config, target)
        # Keep the requested mean epsilon initialization in the bias, while
        # breaking exact spatial symmetry with a very small final-layer weight.
        final_layer = self.epsilon_branch.mlp.net[-1]
        if not isinstance(final_layer, torch.nn.Linear):
            raise TypeError("Expected a Linear epsilon final layer.")
        torch.nn.init.normal_(final_layer.weight, mean=0.0, std=1.0e-3)

    def epsilon(self, xy: torch.Tensor) -> torch.Tensor:
        return self.epsilon_branch(xy)

    def inside_scattered(self, xy: torch.Tensor, directions: torch.Tensor) -> torch.Tensor:
        values = self.field_branch(xy, directions)
        return torch.complex(values[:, 0], values[:, 1])


def true_epsilon_on_grid(quad_xy: torch.Tensor) -> torch.Tensor:
    radius_sq = (quad_xy[:, 0] - SYNTHETIC_CENTER[0]).square() + (quad_xy[:, 1] - SYNTHETIC_CENTER[1]).square()
    return torch.where(radius_sq <= SYNTHETIC_RADIUS**2, torch.full_like(radius_sq, SYNTHETIC_EPSILON), torch.ones_like(radius_sq))


def synthesize_receiver_data(direct_ls: Any, incident: M12IncidentProvider, config: core.TrainConfig) -> torch.Tensor:
    """Multi-RHS LS solve with exactly the inversion operator and Green convention."""
    integral = direct_ls.integral
    epsilon = true_epsilon_on_grid(integral.quad_xy)
    chi = (epsilon - 1.0).to(direct_ls.domain_green.dtype).reshape(-1, 1)
    scale = float(config.k0**2 * integral.area_weight)
    n_cells = epsilon.numel()
    system = torch.eye(n_cells, dtype=direct_ls.domain_green.dtype, device=epsilon.device) - scale * direct_ls.domain_green * chi.reshape(1, -1)
    internal_total = torch.linalg.solve(system, incident.quad)
    source = chi * internal_total
    receiver_green = torch.complex(integral.green_re, integral.green_im).to(direct_ls.domain_green.dtype)
    return scale * (receiver_green @ source)


def view_ids_for_step(step: int, device: torch.device) -> torch.Tensor:
    group = (step - 1) % (36 // VIEWS_PER_STEP)
    return torch.arange(group * VIEWS_PER_STEP, (group + 1) * VIEWS_PER_STEP, device=device)


def receiver_and_state_losses(
    model: IntegralFieldPINN,
    direct_ls: Any,
    incident: M12IncidentProvider,
    receiver_data: torch.Tensor,
    indices: dict[int, np.ndarray],
    directions_by_view: torch.Tensor,
    config: core.TrainConfig,
    view_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    integral = direct_ls.integral
    n_cells = integral.quad_xy.shape[0]
    n_views = int(view_ids.numel())
    directions = directions_by_view[view_ids]
    expanded_xy = integral.quad_xy.repeat(n_views, 1)
    expanded_directions = directions[:, None, :].expand(-1, n_cells, -1).reshape(-1, 2)
    es_inside = model.inside_scattered(expanded_xy, expanded_directions).reshape(n_views, n_cells).transpose(0, 1)
    epsilon = model.epsilon(integral.quad_xy).reshape(n_cells, 1)
    chi = (epsilon - 1.0).to(direct_ls.domain_green.dtype)
    total_field = incident.at_quadrature(view_ids) + es_inside
    source = chi * total_field
    scale = float(config.k0**2 * integral.area_weight)
    predicted_inside = scale * (direct_ls.domain_green @ source)
    state_loss = torch.sum(torch.abs(es_inside - predicted_inside).square()) / (torch.sum(torch.abs(es_inside).square()) + EPS)
    receiver_green = torch.complex(integral.green_re, integral.green_im).to(direct_ls.domain_green.dtype)
    prediction_parts: list[torch.Tensor] = []
    observation_parts: list[torch.Tensor] = []
    for local_index, view_id in enumerate(view_ids.detach().cpu().tolist()):
        rows = torch.as_tensor(indices[int(view_id) + 1], dtype=torch.long, device=view_ids.device)
        prediction_parts.append(scale * (receiver_green[rows] @ source[:, local_index]))
        observation_parts.append(receiver_data[rows, int(view_id)])
    predicted = torch.cat(prediction_parts)
    observed = torch.cat(observation_parts)
    receiver_loss = torch.sum(torch.abs(predicted - observed).square()) / (torch.sum(torch.abs(observed).square()) + EPS)
    return receiver_loss, state_loss


def sommerfeld_loss(model: IntegralFieldPINN, target: core.TargetSpec, directions_by_view: torch.Tensor, view_ids: torch.Tensor, config: core.TrainConfig, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """exp(+iwt) scattered-field condition: d_r Es + i*k0*Es = 0."""
    theta = torch.linspace(0.0, 2.0 * math.pi, BOUNDARY_POINTS + 1, device=device, dtype=dtype)[:-1]
    xy = torch.stack((target.roi_half_width * torch.cos(theta), target.roi_half_width * torch.sin(theta)), dim=1).requires_grad_(True)
    normal = xy / target.roi_half_width
    losses: list[torch.Tensor] = []
    for view_id in view_ids.detach().cpu().tolist():
        directions = directions_by_view[int(view_id)].expand(BOUNDARY_POINTS, 2)
        field = model.field_branch(xy, directions)
        grad_re = torch.autograd.grad(field[:, 0].sum(), xy, create_graph=True)[0]
        grad_im = torch.autograd.grad(field[:, 1].sum(), xy, create_graph=True)[0]
        radial_re = torch.sum(grad_re * normal, dim=1)
        radial_im = torch.sum(grad_im * normal, dim=1)
        # (d_r Re - k Im) + i(d_r Im + k Re) = 0.
        residual_re = radial_re - config.k0 * field[:, 1]
        residual_im = radial_im + config.k0 * field[:, 0]
        denominator = config.k0**2 * torch.mean(field.square()) + EPS
        losses.append(torch.mean(residual_re.square() + residual_im.square()) / denominator)
    return torch.stack(losses).mean()


def epsilon_gradient_norm(loss: torch.Tensor, model: IntegralFieldPINN) -> float:
    gradients = torch.autograd.grad(loss, tuple(model.epsilon_branch.parameters()), retain_graph=True, allow_unused=True)
    norm_sq = sum(float(torch.sum(gradient.detach().square()).cpu()) for gradient in gradients if gradient is not None)
    return math.sqrt(norm_sq)


def epsilon_image(model: IntegralFieldPINN, target: core.TargetSpec, device: torch.device, dtype: torch.dtype) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    axis = np.linspace(-target.roi_half_width, target.roi_half_width, 256, dtype=np.float32)
    xx, yy = np.meshgrid(axis, axis)
    xy = torch.as_tensor(np.column_stack((xx.reshape(-1), yy.reshape(-1))), dtype=dtype, device=device)
    with torch.no_grad():
        epsilon = model.epsilon(xy).reshape(xx.shape).cpu().numpy()
    return xx, yy, epsilon


def epsilon_stats(epsilon: np.ndarray) -> dict[str, float]:
    return {"epsilon_min": float(epsilon.min()), "epsilon_max": float(epsilon.max()), "epsilon_mean": float(epsilon.mean()), "epsilon_std": float(epsilon.std())}


def save_epsilon(epsilon: np.ndarray, path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(5.8, 5.0))
    image = ax.imshow(epsilon, extent=[-0.09, 0.09, -0.09, 0.09], origin="lower", cmap="jet", vmin=1.0, vmax=4.0, interpolation="bilinear")
    ax.set_aspect("equal"); ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)"); ax.set_title(title)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="Relative Permittivity")
    fig.tight_layout(); fig.savefig(path, dpi=200, bbox_inches="tight"); plt.close(fig)


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)


def offline_synthetic_metrics(epsilon: np.ndarray, xx: np.ndarray, yy: np.ndarray) -> dict[str, float]:
    truth = np.where((xx - SYNTHETIC_CENTER[0]) ** 2 + (yy - SYNTHETIC_CENTER[1]) ** 2 <= SYNTHETIC_RADIUS**2, SYNTHETIC_EPSILON, 1.0)
    mask = truth > 1.0
    return {
        "continuous_RE": float(np.linalg.norm(epsilon - truth) / np.linalg.norm(truth)),
        "target_mean": float(epsilon[mask].mean()),
        "target_max": float(epsilon[mask].max()),
        "background_mean": float(epsilon[~mask].mean()),
        "background_std": float(epsilon[~mask].std()),
    }


def main() -> None:
    if OUTPUT_DIR.exists():
        raise FileExistsError(f"Refusing to overwrite existing result directory: {OUTPUT_DIR}")
    config = make_config()
    target = core.TargetSpec(name="Fresnel synthetic closure", kind="circle", eps_background=1.0, eps_object=4.0, roi_half_width=0.09)
    device, dtype = core.resolve_device(config.device), core.resolve_dtype(config.dtype)
    measurement, indices, directions_np, fits, fit_metadata = make_geometry_and_m12(config)
    class Geometry:
        xy = base.mapped_receiver_xy(measurement)
    integral = core.make_integral_tensors(Geometry(), target, config, device, dtype)
    if integral is None:
        raise RuntimeError("Closure test requires static integral tensors.")
    direct_ls = core.make_direct_ls_tensors(integral, config, device, dtype)
    base.replace_with_exp_iwt_green(direct_ls, Geometry.xy, config.k0)
    incident = M12IncidentProvider(fits, integral.quad_xy, device, direct_ls.domain_green.dtype)
    receiver_data = synthesize_receiver_data(direct_ls, incident, config)
    directions_by_view = torch.as_tensor(
        np.stack([directions_np[indices[view][0]] for view in range(1, 37)]), dtype=dtype, device=device
    )
    if directions_by_view.shape != (36, 2):
        raise AssertionError("Expected one direction for each of 36 views.")
    core.set_global_seed(SEED)
    model = IntegralFieldPINN(config, target).to(device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=False)
    image_dir = OUTPUT_DIR / "epsilon_images"; image_dir.mkdir()
    np.save(OUTPUT_DIR / "synthetic_receiver_scattered_m12.npy", receiver_data.detach().cpu().numpy())
    (OUTPUT_DIR / "config.json").write_text(json.dumps({
        "config": asdict(config),
        "convention": {"time": "exp(+i omega t)", "incident": "M12IncidentProvider only", "green": "-i/4 H0^(2)(k0*distance)", "sommerfeld": "d_r Es + i*k0*Es = 0", "chi": "epsilon-1"},
        "loss": {"receiver": "||k0^2 integral G_R chi(Einc+Es_inside)-Es_measured||^2/||Es_measured||^2", "state": "||Es_inside-k0^2 integral G_D chi(Einc+Es_inside)||^2/||Es_inside||^2", "sommerfeld_weight": BOUNDARY_WEIGHT},
        "synthetic_generation_only": {"circle_center_m": list(SYNTHETIC_CENTER), "radius_m": SYNTHETIC_RADIUS, "epsilon": SYNTHETIC_EPSILON, "note": "not supplied to optimizer or checkpoint selection"},
        "epsilon_initialization": {"mean": 1.5, "final_weight_std": 1.0e-3, "bias": "constant logit for eps_initial"},
        "incident_fit_metadata": fit_metadata,
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    initial_views = view_ids_for_step(1, device)
    receiver_loss, state_loss = receiver_and_state_losses(model, direct_ls, incident, receiver_data, indices, directions_by_view, config, initial_views)
    receiver_grad = epsilon_gradient_norm(receiver_loss, model)
    state_grad = epsilon_gradient_norm(state_loss, model)
    if not (math.isfinite(receiver_grad) and math.isfinite(state_grad) and receiver_grad > 0.0 and state_grad > 0.0):
        raise AssertionError(f"Expected nonzero epsilon gradients: receiver={receiver_grad}, state={state_grad}")
    gradient_check = {"L_receiver_epsilon_grad_norm": receiver_grad, "L_state_epsilon_grad_norm": state_grad}
    (OUTPUT_DIR / "gradient_check.json").write_text(json.dumps(gradient_check, indent=2), encoding="utf-8")

    history: list[dict[str, Any]] = []
    best_total = float("inf")
    started = time.time()
    for step in range(1, STEPS + 1):
        model.train(); optimizer.zero_grad(set_to_none=True)
        view_ids = view_ids_for_step(step, device)
        receiver_loss, state_loss = receiver_and_state_losses(model, direct_ls, incident, receiver_data, indices, directions_by_view, config, view_ids)
        boundary_loss = sommerfeld_loss(model, target, directions_by_view, view_ids, config, device, dtype)
        total_loss = receiver_loss + state_loss + BOUNDARY_WEIGHT * boundary_loss
        if not bool(torch.isfinite(total_loss)):
            raise FloatingPointError(f"Non-finite closure objective at step {step}.")
        total_loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_norm); optimizer.step()
        if step % SAVE_EVERY:
            continue
        model.eval()
        all_views = torch.arange(36, device=device)
        # Boundary loss requires autograd, so evaluate it outside no_grad.
        with torch.no_grad():
            full_receiver, full_state = receiver_and_state_losses(model, direct_ls, incident, receiver_data, indices, directions_by_view, config, all_views)
            xx, yy, epsilon = epsilon_image(model, target, device, dtype)
        full_boundary = sommerfeld_loss(model, target, directions_by_view, all_views, config, device, dtype)
        total = float((full_receiver + full_state + BOUNDARY_WEIGHT * full_boundary).detach().cpu())
        stats = epsilon_stats(epsilon)
        save_epsilon(epsilon, image_dir / f"epsilon_step_{step:04d}.png", f"Synthetic exp(+iwt) closure, step {step}")
        checkpoint = OUTPUT_DIR / f"checkpoint_step_{step:04d}.pt"
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "step": step, "config": asdict(config)}, checkpoint)
        if total < best_total:
            best_total = total; shutil.copy2(checkpoint, OUTPUT_DIR / "best_checkpoint.pt")
        row = {"step": step, "batch_views": ",".join(str(int(value) + 1) for value in view_ids.cpu()), "L_receiver": float(full_receiver.cpu()), "L_state": float(full_state.cpu()), "L_sommerfeld": float(full_boundary.detach().cpu()), "physical_total": total, "elapsed_s": time.time() - started, **stats}
        history.append(row); write_csv(history, OUTPUT_DIR / "history.csv")
        print(f"step={step:3d} Lrecv/Lstate/Lbc={row['L_receiver']:.4e}/{row['L_state']:.4e}/{row['L_sommerfeld']:.4e} epsmax/std={stats['epsilon_max']:.3f}/{stats['epsilon_std']:.4f}", flush=True)

    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "step": STEPS, "config": asdict(config)}, OUTPUT_DIR / "final_checkpoint.pt")
    selected = torch.load(OUTPUT_DIR / "best_checkpoint.pt", map_location=device, weights_only=False)
    model.load_state_dict(selected["model"]); model.eval()
    xx, yy, epsilon = epsilon_image(model, target, device, dtype)
    save_epsilon(epsilon, OUTPUT_DIR / "best_continuous_reconstruction.png", f"Synthetic closure best, step {selected['step']}")
    summary = {"steps_completed": STEPS, "best_checkpoint_step": int(selected["step"]), "best_physical_total": best_total, "gradient_check": gradient_check, "best_epsilon": epsilon_stats(epsilon), "offline_synthetic_evaluation_only": offline_synthetic_metrics(epsilon, xx, yy), "elapsed_s": time.time() - started, "real_fresnel_training_started": False}
    (OUTPUT_DIR / "metrics.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
