from __future__ import annotations

"""Oracle pre-subtraction incident scalar audit for Fresnel 1 GHz.

For each view this solves the diagnostic least-squares scalar:

    Es_cal = Etotal_measured - c_v * Eincident_measured

where c_v is chosen to make Es_cal close to true-cylinder exact-LS scattering.
The true-cylinder field is used only for this audit and not for training.
"""

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

import audit_fresnel_1ghz_rminus_m12_incident_ratio_calibration as base
import run_fresnel_1ghz_m12_exp_iwt_synthetic_closure as closure
from square_target.pinn_pixel_inverse_core import TargetSpec, TrainConfig, fit_fourier_bessel_incident, make_direct_ls_tensors, make_integral_tensors


HERE = Path(__file__).resolve().parent
OUTPUT_DIR = HERE / "audit_fresnel_1GHz_pre_subtraction_incident_scalar_oracle"
GRID_SIZE = 64
EPS = 1.0e-15


def metrics(prediction: np.ndarray, observation: np.ndarray) -> dict[str, float]:
    phase = np.angle(np.exp(1j * (np.angle(prediction) - np.angle(observation))))
    return {
        "complex_relative_l2": float(np.linalg.norm(prediction - observation) / (np.linalg.norm(observation) + EPS)),
        "amplitude_relative_error": float(np.linalg.norm(np.abs(prediction) - np.abs(observation)) / (np.linalg.norm(np.abs(observation)) + EPS)),
        "phase_mae_rad": float(np.mean(np.abs(phase))),
        "phase_mae_deg": float(np.degrees(np.mean(np.abs(phase)))),
    }


def true_forward_prediction() -> tuple[dict[str, np.ndarray], dict[int, np.ndarray], np.ndarray]:
    measurement = base.load_measurement()
    indices = base.view_indices(measurement["views"])
    receiver_xy = base.mapped_receiver_xy(measurement)
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
    target = TargetSpec(name="Fresnel pre-subtraction scalar audit", kind="circle", eps_background=1.0, eps_object=3.0, roi_half_width=0.09)

    class AuditObservation:
        xy = receiver_xy

    integral = make_integral_tensors(AuditObservation(), target, config, torch.device("cpu"), torch.float32)
    direct_ls = make_direct_ls_tensors(integral, config, torch.device("cpu"), torch.float32)
    closure.base.replace_with_exp_iwt_green(direct_ls, receiver_xy, config.k0)
    quad_xy = integral.quad_xy.detach().cpu().numpy()
    incident_rhs: list[np.ndarray] = []
    for view in range(1, 37):
        rows = indices[view]
        fit = fit_fourier_bessel_incident(receiver_xy[rows], measurement["incident"][rows], config.k0, base.M12_ORDER, base.M12_RIDGE)
        incident_rhs.append(fit.evaluate(quad_xy))
    prediction = base.forward_predictions(direct_ls, base.true_epsilon(integral.quad_xy), np.column_stack(incident_rhs), config.k0)
    return measurement, indices, prediction


def main() -> None:
    if OUTPUT_DIR.exists():
        raise FileExistsError(f"Refusing to overwrite existing audit output: {OUTPUT_DIR}")
    measurement, indices, prediction_matrix = true_forward_prediction()
    calibrated_scattered = measurement["scattered"].copy()
    scalar_rows: list[dict[str, Any]] = []
    per_view_metric_rows: list[dict[str, Any]] = []
    for view in range(1, 37):
        rows = indices[view]
        total = measurement["total"][rows]
        incident = measurement["incident"][rows]
        target_scattered = prediction_matrix[rows, view - 1]
        # Minimize ||Etotal - c*Eincident - Es_LS_true||^2.
        rhs = total - target_scattered
        c_value = np.vdot(incident, rhs) / (np.vdot(incident, incident) + EPS)
        calibrated = total - c_value * incident
        calibrated_scattered[rows] = calibrated
        scalar_rows.append({
            "view": view,
            "c_real": float(c_value.real),
            "c_imag": float(c_value.imag),
            "c_abs": float(abs(c_value)),
            "c_phase_rad": float(np.angle(c_value)),
            "c_phase_deg": float(np.degrees(np.angle(c_value))),
            "abs_minus_1": float(abs(c_value) - 1.0),
        })
        per_view_metric_rows.append({
            "view": view,
            "state": "before",
            **metrics(target_scattered, measurement["scattered"][rows]),
        })
        per_view_metric_rows.append({
            "view": view,
            "state": "after_oracle_c",
            **metrics(target_scattered, calibrated),
        })

    predicted_flat = np.concatenate([prediction_matrix[indices[view], view - 1] for view in range(1, 37)])
    measured_flat = measurement["scattered"]
    calibrated_flat = calibrated_scattered
    before = metrics(predicted_flat, measured_flat)
    after = metrics(predicted_flat, calibrated_flat)
    abs_minus = np.array([row["abs_minus_1"] for row in scalar_rows], dtype=np.float64)
    phase_deg = np.array([row["c_phase_deg"] for row in scalar_rows], dtype=np.float64)
    OUTPUT_DIR.mkdir(parents=False)
    with (OUTPUT_DIR / "pre_subtraction_scalar_c_by_view.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(scalar_rows[0]))
        writer.writeheader()
        writer.writerows(scalar_rows)
    with (OUTPUT_DIR / "pre_subtraction_scalar_forward_metrics_by_view.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(per_view_metric_rows[0]))
        writer.writeheader()
        writer.writerows(per_view_metric_rows)
    np.save(OUTPUT_DIR / "es_calibrated_oracle.npy", calibrated_scattered)
    summary = {
        "scope": "Oracle diagnostic only. c_v is fitted using true-cylinder exact-LS scattering and must not be used directly in formal training.",
        "definition": "Es_cal=Etotal_measured-c_v*Eincident_measured; one complex scalar c_v per view.",
        "geometry_and_incident": "R(-theta), per-view M=12 incident for true-cylinder LS forward, exp(+iwt), -i/4 H0^(2).",
        "true_cylinder_evaluation_only": {"center_m": list(base.TRUE_CENTER), "radius_m": base.TRUE_RADIUS_M, "epsilon": base.EPS_OBJECT, "ls_grid": f"{GRID_SIZE}x{GRID_SIZE}"},
        "overall_before": before,
        "overall_after_oracle_c": after,
        "c_abs_range": [float(min(row["c_abs"] for row in scalar_rows)), float(max(row["c_abs"] for row in scalar_rows))],
        "abs_c_minus_1_range": [float(np.min(abs_minus)), float(np.max(abs_minus))],
        "phase_c_deg_range": [float(np.min(phase_deg)), float(np.max(phase_deg))],
        "phase_c_abs_max_deg": float(np.max(np.abs(phase_deg))),
        "interpretation_hint": "If c_v stays within a few percent and a few degrees while L2 drops strongly, total/incident acquisition alignment is a likely dominant error.",
    }
    (OUTPUT_DIR / "pre_subtraction_scalar_oracle_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
