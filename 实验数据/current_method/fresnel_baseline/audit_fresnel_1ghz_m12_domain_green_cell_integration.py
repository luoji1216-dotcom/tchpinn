from __future__ import annotations

"""Audit Fresnel exact-LS domain Green self and near-cell quadrature.

This is forward-only.  It compares the current domain Green matrix against
diagonal-zero and source-cell integrated variants for a true eps=3 cylinder.
"""

import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.special import hankel2

import audit_fresnel_1ghz_rminus_m12_incident_ratio_calibration as base
import run_fresnel_1ghz_m12_exp_iwt_synthetic_closure as closure
from square_target.pinn_pixel_inverse_core import TargetSpec, TrainConfig, fit_fourier_bessel_incident, make_direct_ls_tensors, make_integral_tensors


HERE = Path(__file__).resolve().parent
OUTPUT_DIR = HERE / "audit_fresnel_1GHz_m12_GD_cell_integration_v2"
GRID_SIZES = (32, 48, 64)
GAUSS_ORDER = 16
NEAR_RADIUS_CELLS = 1
EPS = 1.0e-15


def outgoing_green(distance: np.ndarray, k0: float) -> np.ndarray:
    return -0.25j * hankel2(0, k0 * distance)


def infer_grid_axis(quad_xy: np.ndarray) -> np.ndarray:
    axis = np.unique(np.round(quad_xy[:, 0], decimals=14))
    if axis.size * axis.size != quad_xy.shape[0]:
        raise ValueError("Expected a square tensor-product source grid.")
    return axis


def current_exp_iwt_domain_green(quad_xy: np.ndarray, area_weight: float, k0: float) -> np.ndarray:
    cell_radius = math.sqrt(area_weight / math.pi)
    distance = np.linalg.norm(quad_xy[:, None, :] - quad_xy[None, :, :], axis=2)
    used_distance = np.where(distance < 1.0e-12, cell_radius, distance)
    return outgoing_green(used_distance, k0)


def cell_average_green_for_offset(offset_x: float, offset_y: float, dx: float, k0: float) -> complex:
    nodes, weights = np.polynomial.legendre.leggauss(GAUSS_ORDER)
    local = 0.5 * dx * nodes
    wx = 0.5 * dx * weights
    yy, xx = np.meshgrid(local, local, indexing="ij")
    wy, wx_grid = np.meshgrid(wx, wx, indexing="ij")
    distance = np.sqrt((offset_x + xx) ** 2 + (offset_y + yy) ** 2)
    values = outgoing_green(distance, k0)
    integral = np.sum(values * wx_grid * wy)
    return complex(integral / (dx * dx))


def integrated_near_domain_green(quad_xy: np.ndarray, area_weight: float, k0: float) -> tuple[np.ndarray, dict[str, Any]]:
    axis = infer_grid_axis(quad_xy)
    n_grid = axis.size
    dx = math.sqrt(area_weight)
    center_green = outgoing_green(np.linalg.norm(quad_xy[:, None, :] - quad_xy[None, :, :], axis=2), k0)
    offsets: dict[tuple[int, int], complex] = {}
    for oy in range(-NEAR_RADIUS_CELLS, NEAR_RADIUS_CELLS + 1):
        for ox in range(-NEAR_RADIUS_CELLS, NEAR_RADIUS_CELLS + 1):
            offsets[(oy, ox)] = cell_average_green_for_offset(ox * dx, oy * dx, dx, k0)
    green = center_green.copy()
    for iy in range(n_grid):
        for ix in range(n_grid):
            target_index = iy * n_grid + ix
            for oy in range(-NEAR_RADIUS_CELLS, NEAR_RADIUS_CELLS + 1):
                jy = iy + oy
                if jy < 0 or jy >= n_grid:
                    continue
                for ox in range(-NEAR_RADIUS_CELLS, NEAR_RADIUS_CELLS + 1):
                    jx = ix + ox
                    if jx < 0 or jx >= n_grid:
                        continue
                    source_index = jy * n_grid + jx
                    green[target_index, source_index] = offsets[(oy, ox)]
    metadata = {
        "definition": "G_ij is the source-cell average of -i/4 H0^(2)(k|r_i-r'|). The LS solve still multiplies by dA.",
        "gauss_order_per_axis": GAUSS_ORDER,
        "integrated_offsets": [
            {"dy_cells": int(oy), "dx_cells": int(ox), "real": float(value.real), "imag": float(value.imag), "abs": float(abs(value))}
            for (oy, ox), value in sorted(offsets.items())
        ],
    }
    return green, metadata


def forward_with_domain_green(
    domain_green: np.ndarray,
    receiver_green: np.ndarray,
    epsilon: np.ndarray,
    incident_rhs: np.ndarray,
    area_weight: float,
    k0: float,
) -> tuple[np.ndarray, dict[str, float]]:
    chi = (epsilon - 1.0).reshape(-1, 1)
    scale = k0**2 * area_weight
    n_cells = epsilon.size
    system = np.eye(n_cells, dtype=np.complex128) - scale * domain_green * chi.reshape(1, -1)
    total = np.linalg.solve(system, incident_rhs)
    residual = system @ total - incident_rhs
    source = scale * chi * total
    prediction = receiver_green @ source
    residual_metrics = {
        "linear_system_relative_residual": float(np.linalg.norm(residual) / (np.linalg.norm(incident_rhs) + EPS)),
        "linear_system_max_abs_residual": float(np.max(np.abs(residual))),
    }
    return prediction, residual_metrics


def metrics(prediction: np.ndarray, observation: np.ndarray) -> dict[str, float]:
    phase = np.angle(np.exp(1j * (np.angle(prediction) - np.angle(observation))))
    return {
        "complex_relative_l2": float(np.linalg.norm(prediction - observation) / (np.linalg.norm(observation) + EPS)),
        "amplitude_relative_error": float(np.linalg.norm(np.abs(prediction) - np.abs(observation)) / (np.linalg.norm(np.abs(observation)) + EPS)),
        "phase_mae_rad": float(np.mean(np.abs(phase))),
        "phase_mae_deg": float(np.degrees(np.mean(np.abs(phase)))),
    }


def build_problem(grid_size: int) -> dict[str, Any]:
    measurement = base.load_measurement()
    indices = base.view_indices(measurement["views"])
    receiver_xy = base.mapped_receiver_xy(measurement)
    config = TrainConfig(
        frequency_hz=1.0e9,
        eps_min=1.0,
        eps_max=4.0,
        integral_grid_size=grid_size,
        integral_sampling_mode="static",
        integral_internal_field_mode="coupled",
        weight_integral_data=1.0,
        dtype="float64",
        device="cpu",
    )
    target = TargetSpec(name="Fresnel G_D audit", kind="circle", eps_background=1.0, eps_object=3.0, roi_half_width=0.09)

    class AuditObservation:
        xy = receiver_xy

    integral = make_integral_tensors(AuditObservation(), target, config, torch.device("cpu"), torch.float64)
    direct_ls = make_direct_ls_tensors(integral, config, torch.device("cpu"), torch.float64)
    closure.base.replace_with_exp_iwt_green(direct_ls, receiver_xy, config.k0)
    quad_xy = integral.quad_xy.detach().cpu().numpy()
    receiver_green = torch.complex(integral.green_re, integral.green_im).detach().cpu().numpy()
    incident_rhs = []
    for view in range(1, 37):
        rows = indices[view]
        fit = fit_fourier_bessel_incident(receiver_xy[rows], measurement["incident"][rows], config.k0, base.M12_ORDER, base.M12_RIDGE)
        incident_rhs.append(fit.evaluate(quad_xy))
    return {
        "measurement": measurement,
        "indices": indices,
        "receiver_xy": receiver_xy,
        "config": config,
        "integral": integral,
        "direct_ls": direct_ls,
        "quad_xy": quad_xy,
        "receiver_green": receiver_green,
        "incident_rhs": np.column_stack(incident_rhs),
        "epsilon": base.true_epsilon(integral.quad_xy).detach().cpu().numpy(),
    }


def flatten_by_view(matrix: np.ndarray, indices: dict[int, np.ndarray]) -> np.ndarray:
    return np.concatenate([matrix[indices[view], view - 1] for view in range(1, 37)])


def main() -> None:
    if OUTPUT_DIR.exists():
        raise FileExistsError(f"Refusing to overwrite existing audit output: {OUTPUT_DIR}")
    OUTPUT_DIR.mkdir(parents=False)
    rows: list[dict[str, Any]] = []
    summary: dict[str, Any] = {
        "scope": "Forward audit only; no training and no checkpoint changes.",
        "time_convention": "exp(+i omega t)",
        "green": "-i/4 H0^(2)(k*r)",
        "current_GD_audit": "Fresnel exact-LS replace_with_exp_iwt_green uses self distance sqrt(dA/pi), not distance=1e-9.",
        "grid_results": {},
    }
    for grid_size in GRID_SIZES:
        problem = build_problem(grid_size)
        config = problem["config"]
        quad_xy = problem["quad_xy"]
        area_weight = float(problem["integral"].area_weight)
        current_green = problem["direct_ls"].domain_green.detach().cpu().numpy()
        reconstructed_current = current_exp_iwt_domain_green(quad_xy, area_weight, config.k0)
        current_diag = np.diag(current_green)
        manual_diag = np.diag(reconstructed_current)
        diagonal_zero = current_green.copy()
        np.fill_diagonal(diagonal_zero, 0.0)
        integrated_green, integration_metadata = integrated_near_domain_green(quad_xy, area_weight, config.k0)
        variants = {
            "current_self_radius_point_value": current_green,
            "diagonal_zero": diagonal_zero,
            "cell_integrated_self_and_neighbors": integrated_green,
        }
        observed = problem["measurement"]["scattered"]
        receiver_green = problem["receiver_green"]
        epsilon = problem["epsilon"]
        incident_rhs = problem["incident_rhs"]
        grid_summary: dict[str, Any] = {
            "dx_m": float(math.sqrt(area_weight)),
            "area_weight": area_weight,
            "current_diagonal_uses_distance_1e-9": False,
            "current_self_distance_m": float(math.sqrt(area_weight / math.pi)),
            "current_diagonal_mean": {"real": float(np.mean(current_diag.real)), "imag": float(np.mean(current_diag.imag)), "abs": float(np.mean(np.abs(current_diag)))},
            "manual_current_reconstruction_max_abs_diff": float(np.max(np.abs(current_green - reconstructed_current))),
            "manual_current_diagonal_max_abs_diff": float(np.max(np.abs(current_diag - manual_diag))),
            "cell_integration": integration_metadata,
            "variants": {},
        }
        for name, domain_green in variants.items():
            prediction_matrix, residual_metrics = forward_with_domain_green(
                domain_green=domain_green,
                receiver_green=receiver_green,
                epsilon=epsilon,
                incident_rhs=incident_rhs,
                area_weight=area_weight,
                k0=config.k0,
            )
            prediction = flatten_by_view(prediction_matrix, problem["indices"])
            metric_row = {
                "grid_size": grid_size,
                "variant": name,
                **metrics(prediction, observed),
                **residual_metrics,
            }
            rows.append(metric_row)
            grid_summary["variants"][name] = metric_row
        summary["grid_results"][str(grid_size)] = grid_summary
    with (OUTPUT_DIR / "green_domain_forward_audit.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (OUTPUT_DIR / "green_domain_forward_audit_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
