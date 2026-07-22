from __future__ import annotations

"""R(-theta) Fresnel 1 GHz field/epsilon integral-consistency experiment.

This keeps the author's FieldBranch and EpsilonBranch.  The only new physics
connection is the contrast-source integral consistency:
    w = k0^2 (epsilon - 1) (Einc + Es)
    Es_grid ~= G_D w, Es_receiver ~= G_R w.
"""

import csv
import json
import math
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

import matplotlib
import numpy as np
import torch
from scipy.special import hankel2

matplotlib.use("Agg")
import matplotlib.pyplot as plt


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
AUTHOR_CORE_DIR = REPO_ROOT / "代码汇总" / "实验数据"
if str(AUTHOR_CORE_DIR) not in sys.path:
    sys.path.insert(0, str(AUTHOR_CORE_DIR))

from pinn_pixel_inverse_core import DoubleBranchPINN, TargetSpec, TrainConfig, set_global_seed  # noqa: E402


DATA_FILE = HERE / "fresnel_2001" / "dielTM_dec8f.exp"
OUTPUT_DIR = HERE / "results_fresnel_1GHz_rminus_contrast_integral_32x32"
FREQUENCY_HZ = 1.0e9
ROI_HALF_WIDTH_M = 0.09
RECEIVER_RADIUS_M = 0.760
GRID_SIZE = 32
STEPS = 500
VIEWS_PER_STEP = 6
SAVE_EVERY = 100
SEED = 20260430
EPS_INITIAL = 1.5
LEARNING_RATE = 8.0e-4
WEIGHT_DATA = 200.0


def complex_from_two(values: torch.Tensor) -> torch.Tensor:
    return torch.complex(values[:, 0], values[:, 1])


def two_from_complex(values: np.ndarray) -> np.ndarray:
    return np.column_stack((values.real, values.imag)).astype(np.float32)


def rminus_geometry(views: np.ndarray, receivers: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    theta = np.deg2rad((views.astype(np.float64) - 1.0) * 10.0)
    receiver_angle = np.deg2rad((receivers.astype(np.float64) - 1.0) * 5.0) - theta
    source_angle = -theta
    source_xy = RECEIVER_RADIUS_M * np.column_stack((np.cos(source_angle), np.sin(source_angle)))
    receiver_xy = RECEIVER_RADIUS_M * np.column_stack((np.cos(receiver_angle), np.sin(receiver_angle)))
    directions = -source_xy / RECEIVER_RADIUS_M
    return receiver_xy.astype(np.float64), directions.astype(np.float64)


def load_measurement() -> dict[str, np.ndarray]:
    raw = np.loadtxt(DATA_FILE, comments="#")
    selected = raw[raw[:, 2].astype(int) == 1]
    views = selected[:, 0].astype(int)
    receivers = selected[:, 1].astype(int)
    total = selected[:, 3] + 1j * selected[:, 4]
    incident = selected[:, 5] + 1j * selected[:, 6]
    scattered = total - incident
    xy, directions = rminus_geometry(views, receivers)
    if selected.shape != (36 * 49, 7):
        raise ValueError(f"Expected 36x49 rows at 1 GHz, got {selected.shape}.")
    if any(np.count_nonzero(views == view) != 49 for view in range(1, 37)):
        raise ValueError("Each view must have exactly 49 receivers.")
    return {
        "views": views,
        "receivers": receivers,
        "xy": xy,
        "directions": directions,
        "incident": incident.astype(np.complex128),
        "scattered": scattered.astype(np.complex128),
    }


def make_grid(k0: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    dx = 2.0 * ROI_HALF_WIDTH_M / GRID_SIZE
    axis = np.linspace(-ROI_HALF_WIDTH_M + 0.5 * dx, ROI_HALF_WIDTH_M - 0.5 * dx, GRID_SIZE)
    xx, yy = np.meshgrid(axis, axis, indexing="xy")
    xy = np.column_stack((xx.reshape(-1), yy.reshape(-1))).astype(np.float64)
    delta = xy[:, None, :] - xy[None, :, :]
    distance = np.linalg.norm(delta, axis=2)
    green = -0.25j * hankel2(0, k0 * np.maximum(distance, 1.0e-12)) * (dx * dx)

    # Disk-equivalent cell integration avoids the point-Green singularity.
    radius = math.sqrt((dx * dx) / math.pi)
    radial_integral = radius / k0 * hankel2(1, k0 * radius) - 2j / (math.pi * k0 * k0)
    np.fill_diagonal(green, -0.25j * 2.0 * math.pi * radial_integral)
    return axis, xx, yy, xy, green


def plane_wave_incident_cache(
    measurement: dict[str, np.ndarray],
    grid_xy: np.ndarray,
    k0: float,
) -> tuple[np.ndarray, dict[str, dict[str, float]]]:
    cache = np.empty((grid_xy.shape[0], 36), dtype=np.complex128)
    metadata: dict[str, dict[str, float]] = {}
    for view in range(1, 37):
        rows = np.flatnonzero(measurement["views"] == view)
        direction = measurement["directions"][rows[0]]
        receiver_basis = np.exp(-1j * k0 * (measurement["xy"][rows] @ direction))
        amplitude = np.vdot(receiver_basis, measurement["incident"][rows]) / np.vdot(receiver_basis, receiver_basis)
        cache[:, view - 1] = amplitude * np.exp(-1j * k0 * (grid_xy @ direction))
        metadata[f"view_{view:02d}"] = {
            "amplitude_real": float(amplitude.real),
            "amplitude_imag": float(amplitude.imag),
        }
    return cache, metadata


def robust_data_loss(predicted: torch.Tensor, observed: torch.Tensor) -> torch.Tensor:
    diff = predicted - observed
    magnitude = torch.sqrt(torch.sum(diff.detach().square(), dim=1) + 1.0e-12)
    scale = torch.quantile(magnitude, 0.75).clamp_min(1.0e-6)
    weights = 1.0 / (1.0 + (magnitude / scale).square())
    weights = weights / weights.mean().clamp_min(1.0e-6)
    return torch.mean(weights * torch.sum(diff.square(), dim=1))


def epsilon_gradient_norm(loss: torch.Tensor, parameters: Iterable[torch.nn.Parameter]) -> float:
    gradients = torch.autograd.grad(loss, tuple(parameters), retain_graph=True, allow_unused=True)
    squared = [torch.sum(gradient.detach().square()) for gradient in gradients if gradient is not None]
    if not squared:
        return 0.0
    return float(torch.sqrt(torch.stack(squared).sum()).cpu())


def batch_view_ids(step: int, device: torch.device) -> torch.Tensor:
    start = ((step - 1) * VIEWS_PER_STEP) % 36
    return torch.as_tensor([(start + offset) % 36 for offset in range(VIEWS_PER_STEP)], device=device)


def normalized_complex_mse(predicted: torch.Tensor, observed: torch.Tensor) -> torch.Tensor:
    return torch.sum(torch.abs(predicted - observed).square()) / torch.sum(torch.abs(observed).square()).clamp_min(1.0e-12)


def consistency_loss(
    model: DoubleBranchPINN,
    grid_xy: torch.Tensor,
    domain_green: torch.Tensor,
    receiver_green: torch.Tensor,
    receiver_indices: torch.Tensor,
    view_ids: torch.Tensor,
    directions_by_view: torch.Tensor,
    incident_grid: torch.Tensor,
    scattered_observed: torch.Tensor,
) -> torch.Tensor:
    n_cells = grid_xy.shape[0]
    directions = directions_by_view[view_ids]
    n_views = directions.shape[0]
    expanded_xy = grid_xy.repeat(n_views, 1)
    expanded_directions = directions[:, None, :].expand(-1, n_cells, -1).reshape(-1, 2)
    scattered_grid = complex_from_two(model.scattered_field(expanded_xy, expanded_directions)).reshape(n_views, n_cells).transpose(0, 1)
    epsilon = model.epsilon(grid_xy).reshape(n_cells, 1)
    total_grid = incident_grid[:, view_ids] + scattered_grid
    contrast_source = (FREQUENCY_HZ / 3.0e8 * 2.0 * math.pi) ** 2 * (epsilon - 1.0).to(total_grid.dtype) * total_grid
    integral_grid = domain_green @ contrast_source

    state_terms = []
    receiver_terms = []
    for column, view_id in enumerate(view_ids.tolist()):
        state_terms.append(normalized_complex_mse(integral_grid[:, column], scattered_grid[:, column]))
        rows = receiver_indices[view_id]
        predicted_receiver = receiver_green[rows] @ contrast_source[:, column]
        receiver_terms.append(normalized_complex_mse(predicted_receiver, scattered_observed[rows]))
    return 0.5 * (torch.stack(state_terms).mean() + torch.stack(receiver_terms).mean())


def epsilon_image(model: DoubleBranchPINN, device: torch.device) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    axis = np.linspace(-ROI_HALF_WIDTH_M, ROI_HALF_WIDTH_M, 256, dtype=np.float32)
    xx, yy = np.meshgrid(axis, axis, indexing="xy")
    xy = torch.as_tensor(np.column_stack((xx.reshape(-1), yy.reshape(-1))), dtype=torch.float32, device=device)
    with torch.no_grad():
        epsilon = model.epsilon(xy).reshape(xx.shape).cpu().numpy()
    return xx, yy, epsilon


def save_epsilon_image(xx: np.ndarray, yy: np.ndarray, epsilon: np.ndarray, path: Path, step: int) -> None:
    figure, axis = plt.subplots(figsize=(6, 5), constrained_layout=True)
    image = axis.imshow(
        epsilon,
        origin="lower",
        extent=(float(xx.min()), float(xx.max()), float(yy.min()), float(yy.max())),
        vmin=1.0,
        vmax=4.0,
        cmap="viridis",
        interpolation="bilinear",
        aspect="equal",
    )
    axis.set_xlabel("x (m)")
    axis.set_ylabel("y (m)")
    axis.set_title(f"R(-theta) contrast-integral PINN, step {step}")
    figure.colorbar(image, ax=axis, label="Relative permittivity")
    figure.savefig(path, dpi=180)
    plt.close(figure)


def evaluate(
    model: DoubleBranchPINN,
    measurement: dict[str, np.ndarray],
    tensors: dict[str, torch.Tensor],
) -> dict[str, float | np.ndarray]:
    model.eval()
    all_views = torch.arange(36, device=tensors["grid_xy"].device)
    with torch.no_grad():
        directions = tensors["directions_by_view"]
        predicted = model.scattered_field(tensors["receiver_xy"], directions[tensors["view_ids_by_record"]])
        data = robust_data_loss(predicted, tensors["scattered_two"])
        consistency = consistency_loss(
            model,
            tensors["grid_xy"],
            tensors["domain_green"],
            tensors["receiver_green"],
            tensors["receiver_indices"],
            all_views,
            tensors["directions_by_view"],
            tensors["incident_grid"],
            tensors["scattered_complex"],
        )
        xx, yy, epsilon = epsilon_image(model, tensors["grid_xy"].device)
    contrast = np.maximum(epsilon - 1.0, 0.0)
    contrast_sum = float(contrast.sum())
    return {
        "data_loss": float(data.cpu()),
        "consistency_loss": float(consistency.cpu()),
        "epsilon_min": float(epsilon.min()),
        "epsilon_max": float(epsilon.max()),
        "epsilon_mean": float(epsilon.mean()),
        "epsilon_std": float(epsilon.std()),
        "contrast_centroid_y_m": float((contrast * yy).sum() / contrast_sum) if contrast_sum > 1.0e-12 else 0.0,
        "negative_y_contrast_fraction": float(contrast[yy < 0.0].sum() / contrast_sum) if contrast_sum > 1.0e-12 else 0.0,
        "xx": xx,
        "yy": yy,
        "epsilon": epsilon,
    }


def main() -> None:
    if OUTPUT_DIR.exists():
        raise FileExistsError(f"Refusing to overwrite existing result directory: {OUTPUT_DIR}")
    if not DATA_FILE.exists():
        raise FileNotFoundError(DATA_FILE)

    measurement = load_measurement()
    config = TrainConfig(
        frequency_hz=FREQUENCY_HZ,
        incident_phase_sign=-1.0,
        eps_min=1.0,
        eps_max=4.0,
        eps_initial=EPS_INITIAL,
        learning_rate=LEARNING_RATE,
        field_hidden_layers=7,
        field_hidden_units=112,
        eps_hidden_layers=5,
        eps_hidden_units=96,
        fourier_bands=6,
        fourier_max_frequency=10.0,
        random_seed=SEED,
        dtype="float32",
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    target = TargetSpec(
        name="Fresnel training ROI only",
        kind="none",
        eps_background=1.0,
        eps_object=1.0,
        roi_half_width=ROI_HALF_WIDTH_M,
    )
    set_global_seed(SEED)
    device = torch.device(config.device)
    k0 = 2.0 * math.pi * FREQUENCY_HZ / 3.0e8
    _axis, _gx, _gy, grid_xy_np, domain_green_np = make_grid(k0)
    incident_grid_np, amplitude_metadata = plane_wave_incident_cache(measurement, grid_xy_np, k0)
    receiver_delta = measurement["xy"][:, None, :] - grid_xy_np[None, :, :]
    receiver_green_np = -0.25j * hankel2(0, k0 * np.linalg.norm(receiver_delta, axis=2)) * (2.0 * ROI_HALF_WIDTH_M / GRID_SIZE) ** 2

    receiver_indices = torch.empty((36, 49), dtype=torch.long, device=device)
    directions_by_view = np.empty((36, 2), dtype=np.float32)
    for view in range(1, 37):
        rows = np.flatnonzero(measurement["views"] == view)
        receiver_indices[view - 1] = torch.as_tensor(rows, dtype=torch.long, device=device)
        directions_by_view[view - 1] = measurement["directions"][rows[0]]
    tensors = {
        "grid_xy": torch.as_tensor(grid_xy_np, dtype=torch.float32, device=device),
        "domain_green": torch.as_tensor(domain_green_np, dtype=torch.complex64, device=device),
        "receiver_green": torch.as_tensor(receiver_green_np, dtype=torch.complex64, device=device),
        "receiver_xy": torch.as_tensor(measurement["xy"], dtype=torch.float32, device=device),
        "receiver_indices": receiver_indices,
        "directions_by_view": torch.as_tensor(directions_by_view, dtype=torch.float32, device=device),
        "incident_grid": torch.as_tensor(incident_grid_np, dtype=torch.complex64, device=device),
        "scattered_two": torch.as_tensor(two_from_complex(measurement["scattered"]), dtype=torch.float32, device=device),
        "scattered_complex": torch.as_tensor(measurement["scattered"], dtype=torch.complex64, device=device),
        "view_ids_by_record": torch.as_tensor(measurement["views"] - 1, dtype=torch.long, device=device),
    }
    model = DoubleBranchPINN(config, target).to(device=device, dtype=torch.float32)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    OUTPUT_DIR.mkdir(parents=False)
    image_dir = OUTPUT_DIR / "epsilon_images"
    image_dir.mkdir()
    (OUTPUT_DIR / "config.json").write_text(
        json.dumps(
            {
                "config": asdict(config),
                "data": "1 GHz, Es=Etotal-Eincident, 36 views x 49 receivers",
                "geometry": "R(-theta): source angle=-theta; receiver angle=raw receiver angle-theta",
                "incident": "per-view single complex-amplitude fitted plane wave from measured incident only",
                "model": "author FieldBranch + author EpsilonBranch",
                "loss": {
                    "data": f"{WEIGHT_DATA} * author robust receiver scattered-field MSE",
                    "consistency": "0.5*(normalized ||Es_grid-G_D w||^2 + normalized ||G_R w-Es_obs||^2), w=k0^2*(epsilon-1)*(Einc+Es_grid)",
                    "PDE": 0,
                    "TV": 0,
                    "LEP": 0,
                    "material_or_geometry_prior": 0,
                },
                "grid": "32x32",
                "views_per_step": VIEWS_PER_STEP,
                "incident_amplitudes": amplitude_metadata,
                "selection": "physical data+consistency only; no GT/RE/SSIM",
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    history: list[dict[str, float]] = []
    best_total = float("inf")
    started = time.time()
    for step in range(1, STEPS + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        view_ids = batch_view_ids(step, device)
        batch_rows = tensors["receiver_indices"][view_ids].reshape(-1)
        prediction = model.scattered_field(tensors["receiver_xy"][batch_rows], tensors["directions_by_view"][view_ids].repeat_interleave(49, dim=0))
        data = robust_data_loss(prediction, tensors["scattered_two"][batch_rows])
        consistency = consistency_loss(
            model,
            tensors["grid_xy"],
            tensors["domain_green"],
            tensors["receiver_green"],
            tensors["receiver_indices"],
            view_ids,
            tensors["directions_by_view"],
            tensors["incident_grid"],
            tensors["scattered_complex"],
        )
        data_eps_grad = epsilon_gradient_norm(data, model.epsilon_branch.parameters())
        consistency_eps_grad = epsilon_gradient_norm(consistency, model.epsilon_branch.parameters())
        total = WEIGHT_DATA * data + consistency
        if not bool(torch.isfinite(total)):
            raise FloatingPointError(f"Non-finite objective at step {step}.")
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % SAVE_EVERY != 0:
            continue
        metrics = evaluate(model, measurement, tensors)
        row = {
            "step": float(step),
            "data_loss": float(metrics["data_loss"]),
            "consistency_loss": float(metrics["consistency_loss"]),
            "total_physics_loss": float(WEIGHT_DATA * metrics["data_loss"] + metrics["consistency_loss"]),
            "epsilon_min": float(metrics["epsilon_min"]),
            "epsilon_max": float(metrics["epsilon_max"]),
            "epsilon_mean": float(metrics["epsilon_mean"]),
            "epsilon_std": float(metrics["epsilon_std"]),
            "data_epsilon_grad_norm": data_eps_grad,
            "consistency_epsilon_grad_norm": consistency_eps_grad,
            "contrast_centroid_y_m": float(metrics["contrast_centroid_y_m"]),
            "negative_y_contrast_fraction": float(metrics["negative_y_contrast_fraction"]),
            "elapsed_s": time.time() - started,
        }
        history.append(row)
        save_epsilon_image(metrics["xx"], metrics["yy"], metrics["epsilon"], image_dir / f"epsilon_step_{step:04d}.png", step)
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "step": step, "config": asdict(config)}, OUTPUT_DIR / f"checkpoint_step_{step:04d}.pt")
        if row["total_physics_loss"] < best_total:
            best_total = row["total_physics_loss"]
            torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "step": step, "config": asdict(config)}, OUTPUT_DIR / "best_checkpoint.pt")
        print(json.dumps(row, ensure_ascii=False), flush=True)

    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "step": STEPS, "config": asdict(config)}, OUTPUT_DIR / "final_checkpoint.pt")
    with (OUTPUT_DIR / "history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)
    final = history[-1]
    final["stop_reason"] = "epsilon_max_below_1.2_at_step_500" if final["epsilon_max"] < 1.2 else "completed_500_steps"
    final["best_total_physics_loss"] = best_total
    (OUTPUT_DIR / "metrics.json").write_text(json.dumps(final, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
