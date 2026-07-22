from __future__ import annotations

"""Geometry-mapping audit for dielTM_dec8f.exp with per-mapping M12 refits.

No training is performed.  The three tested mappings are:

A. no extra rotation:
   source_angle=(view-1)*10 deg, receiver_angle=(receiver_id-1)*5 deg
B. rotate original antenna coordinates by R(-theta):
   source_angle=0, receiver_angle=(receiver_id-1)*5 deg - theta
C. rotate original antenna coordinates by R(+theta):
   source_angle=2*theta, receiver_angle=(receiver_id-1)*5 deg + theta
"""

import csv
import json
import math
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import audit_fresnel_1ghz_rminus_m12_incident_ratio_calibration as base
import run_fresnel_1ghz_m12_exp_iwt_synthetic_closure as closure
from square_target.pinn_pixel_inverse_core import TargetSpec, TrainConfig, fit_fourier_bessel_incident, make_direct_ls_tensors, make_integral_tensors


HERE = Path(__file__).resolve().parent
OUTPUT_DIR = HERE / "audit_fresnel_1GHz_geometry_mappings_m12"
GRID_SIZE = 64
RHO_M = 0.74
EPS = 1.0e-15


def wrap_to_pi(angle: np.ndarray) -> np.ndarray:
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def mapping_angles(measurement: dict[str, np.ndarray], mapping: str) -> tuple[np.ndarray, np.ndarray]:
    theta = measurement["target_rotation"]
    receiver_raw = measurement["receiver_angles_raw"]
    if mapping == "A_no_extra_rotation":
        return theta, receiver_raw
    if mapping == "B_rotate_antennas_Rminus_theta":
        return theta - theta, receiver_raw - theta
    if mapping == "C_rotate_antennas_Rplus_theta":
        return theta + theta, receiver_raw + theta
    raise ValueError(f"Unknown mapping: {mapping}")


def xy_from_angles(angle: np.ndarray) -> np.ndarray:
    return RHO_M * np.column_stack((np.cos(angle), np.sin(angle)))


def metrics(prediction: np.ndarray, observation: np.ndarray) -> dict[str, float]:
    phase = np.angle(np.exp(1j * (np.angle(prediction) - np.angle(observation))))
    return {
        "complex_relative_l2": float(np.linalg.norm(prediction - observation) / (np.linalg.norm(observation) + EPS)),
        "amplitude_relative_error": float(np.linalg.norm(np.abs(prediction) - np.abs(observation)) / (np.linalg.norm(np.abs(observation)) + EPS)),
        "phase_mae_rad": float(np.mean(np.abs(phase))),
        "phase_mae_deg": float(np.degrees(np.mean(np.abs(phase)))),
    }


def flatten_prediction(prediction_matrix: np.ndarray, indices: dict[int, np.ndarray]) -> np.ndarray:
    return np.concatenate([prediction_matrix[indices[view], view - 1] for view in range(1, 37)])


def fit_m12_for_mapping(
    measurement: dict[str, np.ndarray],
    indices: dict[int, np.ndarray],
    receiver_xy: np.ndarray,
    quad_xy: np.ndarray,
    k0: float,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    rhs: list[np.ndarray] = []
    fit_rows: list[dict[str, Any]] = []
    for view in range(1, 37):
        rows = indices[view]
        fit = fit_fourier_bessel_incident(receiver_xy[rows], measurement["incident"][rows], k0, base.M12_ORDER, base.M12_RIDGE)
        rhs.append(fit.evaluate(quad_xy))
        fit_rows.append({"view": view, **fit.fit_metrics})
    return np.column_stack(rhs), fit_rows


def forward_prediction(receiver_xy: np.ndarray, incident_rhs: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, TrainConfig, Any]:
    config = TrainConfig(
        frequency_hz=1.0e9,
        eps_min=1.0,
        eps_max=4.0,
        integral_grid_size=GRID_SIZE,
        integral_sampling_mode="static",
        integral_internal_field_mode="coupled",
        weight_integral_data=1.0,
        dtype="float32",
        device="cpu",
    )
    target = TargetSpec(name="Fresnel geometry mapping audit", kind="circle", eps_background=1.0, eps_object=3.0, roi_half_width=0.09)

    class AuditObservation:
        xy = receiver_xy

    integral = make_integral_tensors(AuditObservation(), target, config, torch.device("cpu"), torch.float32)
    direct_ls = make_direct_ls_tensors(integral, config, torch.device("cpu"), torch.float32)
    closure.base.replace_with_exp_iwt_green(direct_ls, receiver_xy, config.k0)
    epsilon = base.true_epsilon(integral.quad_xy)
    prediction = base.forward_predictions(direct_ls, epsilon, incident_rhs, config.k0)
    receiver_green = torch.complex(integral.green_re, integral.green_im).detach().cpu().numpy()
    return prediction, receiver_green, integral.quad_xy.detach().cpu().numpy(), epsilon.detach().cpu().numpy(), config, integral


def adjoint_map(
    receiver_green: np.ndarray,
    incident_rhs: np.ndarray,
    observed: np.ndarray,
    indices: dict[int, np.ndarray],
    area_weight: float,
    k0: float,
) -> np.ndarray:
    observed_flat = np.concatenate([observed[indices[view]] for view in range(1, 37)])
    obs_norm = np.linalg.norm(observed_flat) + EPS
    scale = k0**2 * area_weight
    n_cells = incident_rhs.shape[0]
    values = np.empty(n_cells, dtype=np.float64)
    for cell in range(n_cells):
        column_parts = []
        for view in range(1, 37):
            rows = indices[view]
            column_parts.append(scale * receiver_green[rows, cell] * incident_rhs[cell, view - 1])
        column = np.concatenate(column_parts)
        values[cell] = abs(np.vdot(column, observed_flat)) / ((np.linalg.norm(column) + EPS) * obs_norm)
    return values


def save_adjoint_image(values: np.ndarray, quad_xy: np.ndarray, path: Path, title: str) -> dict[str, Any]:
    n_grid = int(round(math.sqrt(values.size)))
    image = values.reshape(n_grid, n_grid)
    peak_index = int(np.argmax(values))
    peak_xy = quad_xy[peak_index]
    weights = values / (values.sum() + EPS)
    centroid = weights @ quad_xy
    extent = [float(quad_xy[:, 0].min()), float(quad_xy[:, 0].max()), float(quad_xy[:, 1].min()), float(quad_xy[:, 1].max())]
    fig, ax = plt.subplots(figsize=(5.2, 4.8), constrained_layout=True)
    handle = ax.imshow(image, origin="lower", extent=extent, aspect="equal", cmap="magma")
    ax.scatter([peak_xy[0]], [peak_xy[1]], c="cyan", s=24, label="peak")
    ax.set_title(title)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.legend(loc="upper right")
    fig.colorbar(handle, ax=ax, label="normalized adjoint")
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return {
        "peak_x_m": float(peak_xy[0]),
        "peak_y_m": float(peak_xy[1]),
        "centroid_x_m": float(centroid[0]),
        "centroid_y_m": float(centroid[1]),
        "peak_value": float(values[peak_index]),
    }


def beta_rows_for_mapping(mapping: str, measurement: dict[str, np.ndarray], indices: dict[int, np.ndarray], source_angle: np.ndarray, receiver_angle: np.ndarray) -> list[dict[str, Any]]:
    rows_out: list[dict[str, Any]] = []
    for view in range(1, 37):
        rows = indices[view]
        beta = wrap_to_pi(receiver_angle[rows] - source_angle[rows])
        rows_out.append({
            "mapping": mapping,
            "view": view,
            "source_angle_deg": float(np.degrees(source_angle[rows[0]]) % 360.0),
            "beta_min_deg": float(np.degrees(beta.min())),
            "beta_max_deg": float(np.degrees(beta.max())),
            "beta_first_deg": float(np.degrees(beta[0])),
            "beta_last_deg": float(np.degrees(beta[-1])),
            "beta_values_deg": " ".join(f"{value:.6g}" for value in np.degrees(beta)),
        })
    return rows_out


def main() -> None:
    if OUTPUT_DIR.exists():
        raise FileExistsError(f"Refusing to overwrite existing audit output: {OUTPUT_DIR}")
    OUTPUT_DIR.mkdir(parents=False)
    measurement = base.load_measurement()
    indices = base.view_indices(measurement["views"])
    mappings = ("A_no_extra_rotation", "B_rotate_antennas_Rminus_theta", "C_rotate_antennas_Rplus_theta")
    overall_rows: list[dict[str, Any]] = []
    beta_rows_all: list[dict[str, Any]] = []
    fit_rows_all: list[dict[str, Any]] = []
    summary: dict[str, Any] = {
        "scope": "Geometry mapping audit only. No training, no checkpoint changes.",
        "data": "dielTM_dec8f.exp frequency index 1, Es=Etotal-Eincident.",
        "true_cylinder_evaluation_only": {"center_m": list(base.TRUE_CENTER), "radius_m": base.TRUE_RADIUS_M, "epsilon": base.EPS_OBJECT, "ls_grid": f"{GRID_SIZE}x{GRID_SIZE}"},
        "incident": {"model": "per-view M=12 Fourier-Bessel", "ridge": base.M12_RIDGE, "refit": "refit separately for each mapping using that mapping's receiver coordinates"},
        "mappings": {},
    }
    for mapping in mappings:
        source_angle, receiver_angle = mapping_angles(measurement, mapping)
        receiver_xy = xy_from_angles(receiver_angle)
        # Build integral tensors once to get quadrature points for the M12 fits.
        config_probe = TrainConfig(frequency_hz=1.0e9, eps_min=1.0, eps_max=4.0, integral_grid_size=GRID_SIZE, integral_sampling_mode="static", integral_internal_field_mode="coupled", weight_integral_data=1.0, dtype="float32", device="cpu")
        target_probe = TargetSpec(name="probe", kind="circle", eps_background=1.0, eps_object=3.0, roi_half_width=0.09)

        class ProbeObservation:
            xy = receiver_xy

        integral_probe = make_integral_tensors(ProbeObservation(), target_probe, config_probe, torch.device("cpu"), torch.float32)
        quad_xy = integral_probe.quad_xy.detach().cpu().numpy()
        incident_rhs, fit_rows = fit_m12_for_mapping(measurement, indices, receiver_xy, quad_xy, config_probe.k0)
        prediction_matrix, receiver_green, quad_xy, _epsilon, config, integral = forward_prediction(receiver_xy, incident_rhs)
        predicted_flat = flatten_prediction(prediction_matrix, indices)
        observed_flat = measurement["scattered"]
        forward_metrics = metrics(predicted_flat, observed_flat)
        adjoint_values = adjoint_map(receiver_green, incident_rhs, measurement["scattered"], indices, integral.area_weight, config.k0)
        adjoint_info = save_adjoint_image(adjoint_values, quad_xy, OUTPUT_DIR / f"adjoint_{mapping}.png", mapping)
        beta_rows = beta_rows_for_mapping(mapping, measurement, indices, source_angle, receiver_angle)
        beta_rows_all.extend(beta_rows)
        for row in fit_rows:
            fit_rows_all.append({"mapping": mapping, **row})
        first_beta = np.fromstring(beta_rows[0]["beta_values_deg"], sep=" ")
        beta_identical_to_view1 = all(np.allclose(np.fromstring(row["beta_values_deg"], sep=" "), first_beta, atol=1e-10) for row in beta_rows)
        mapping_summary = {
            "source_angle_view_1_9_18_27_deg": {
                str(view): float(np.degrees(source_angle[indices[view][0]]) % 360.0)
                for view in (1, 9, 18, 27)
            },
            "receiver_angle_first_last_view_1_9_18_27_deg": {
                str(view): [
                    float(np.degrees(receiver_angle[indices[view][0]]) % 360.0),
                    float(np.degrees(receiver_angle[indices[view][-1]]) % 360.0),
                ]
                for view in (1, 9, 18, 27)
            },
            "beta_identical_across_views": bool(beta_identical_to_view1),
            "beta_view1_minmax_deg": [float(beta_rows[0]["beta_min_deg"]), float(beta_rows[0]["beta_max_deg"])],
            "forward_metrics": forward_metrics,
            "adjoint": adjoint_info,
            "mean_m12_incident_fit_l2": float(np.mean([row["complex_l2"] for row in fit_rows])),
            "max_m12_incident_fit_l2": float(np.max([row["complex_l2"] for row in fit_rows])),
        }
        summary["mappings"][mapping] = mapping_summary
        overall_rows.append({
            "mapping": mapping,
            **forward_metrics,
            **adjoint_info,
            "beta_identical_across_views": bool(beta_identical_to_view1),
            "beta_view1_min_deg": float(beta_rows[0]["beta_min_deg"]),
            "beta_view1_max_deg": float(beta_rows[0]["beta_max_deg"]),
            "mean_m12_incident_fit_l2": mapping_summary["mean_m12_incident_fit_l2"],
            "max_m12_incident_fit_l2": mapping_summary["max_m12_incident_fit_l2"],
        })
        print(
            f"{mapping}: L2={forward_metrics['complex_relative_l2']:.6f}, "
            f"phase={forward_metrics['phase_mae_deg']:.3f} deg, "
            f"adjoint_peak=({adjoint_info['peak_x_m']:.4f},{adjoint_info['peak_y_m']:.4f}), "
            f"beta_identical={beta_identical_to_view1}",
            flush=True,
        )
    with (OUTPUT_DIR / "geometry_mapping_forward_adjoint_audit.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(overall_rows[0]))
        writer.writeheader()
        writer.writerows(overall_rows)
    with (OUTPUT_DIR / "geometry_mapping_beta_by_view.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(beta_rows_all[0]))
        writer.writeheader()
        writer.writerows(beta_rows_all)
    with (OUTPUT_DIR / "geometry_mapping_m12_fit_by_view.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fit_rows_all[0]))
        writer.writeheader()
        writer.writerows(fit_rows_all)
    best = min(overall_rows, key=lambda row: float(row["complex_relative_l2"]))
    summary["best_by_complex_l2"] = best
    summary["A_beta_statement"] = (
        "A has the same beta values for every view; if these values are the actual 60-300 deg receiver arc, "
        "then the raw file is already represented in a fixed-target bistatic geometry and applying another R(-theta) is a duplicate rotation."
        if summary["mappings"]["A_no_extra_rotation"]["beta_identical_across_views"]
        else "A beta values are not identical across views."
    )
    (OUTPUT_DIR / "geometry_mapping_forward_adjoint_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
