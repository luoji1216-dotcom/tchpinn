from __future__ import annotations

"""True-cylinder diagnostic for one shared bistatic-angle transfer function.

This is intentionally a diagnostic calibration: the coefficients are fitted
to true-cylinder forward predictions and measured scattered fields only to
quantify the residual measurement/operator mismatch.  It performs no PINN
training and does not alter any observation used by inversion.
"""

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

import audit_fresnel_1ghz_rminus_m12_incident_ratio_calibration as base
from square_target.pinn_pixel_inverse_core import TargetSpec, TrainConfig, fit_fourier_bessel_incident, make_direct_ls_tensors, make_integral_tensors


HERE = Path(__file__).resolve().parent
OUTPUT_DIR = HERE / "audit_fresnel_1GHz_rminus_m12_global_bistatic_calibration"
GRID_SIZE = 64
CALIBRATION_ORDERS = (0, 1, 2, 3)


def wrap_to_pi(angle: np.ndarray) -> np.ndarray:
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def shared_bistatic_calibration(prediction: np.ndarray, observation: np.ndarray, beta: np.ndarray, order: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit Es_measured approximately c(beta) * Es_true-cylinder for all views jointly."""
    modes = np.arange(-order, order + 1, dtype=np.int64)
    basis = np.exp(1j * beta[:, None] * modes[None, :])
    design = prediction[:, None] * basis
    coefficients, _residuals, _rank, _singular = np.linalg.lstsq(design, observation, rcond=None)
    calibration = basis @ coefficients
    return calibration * prediction, calibration, coefficients


def main() -> None:
    if OUTPUT_DIR.exists():
        raise FileExistsError(f"Refusing to overwrite existing audit output: {OUTPUT_DIR}")
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
    target = TargetSpec(name="Fresnel shared-bistatic calibration audit", kind="circle", eps_background=1.0, eps_object=3.0, roi_half_width=0.09)

    class AuditObservation:
        xy = receiver_xy

    integral = make_integral_tensors(AuditObservation(), target, config, torch.device("cpu"), torch.float32)
    direct_ls = make_direct_ls_tensors(integral, config, torch.device("cpu"), torch.float32)
    base.replace_with_exp_iwt_green(direct_ls, receiver_xy, config.k0)
    quad_xy = integral.quad_xy.detach().cpu().numpy()
    rhs: list[np.ndarray] = []
    for view in range(1, 37):
        rows = indices[view]
        fit = fit_fourier_bessel_incident(receiver_xy[rows], measurement["incident"][rows], config.k0, base.M12_ORDER, base.M12_RIDGE)
        rhs.append(fit.evaluate(quad_xy))
    prediction_matrix = base.forward_predictions(direct_ls, base.true_epsilon(integral.quad_xy), np.column_stack(rhs), config.k0)
    prediction = np.concatenate([prediction_matrix[indices[view], view - 1] for view in range(1, 37)])
    observation = measurement["scattered"]

    source_angle = -measurement["target_rotation"]
    receiver_angle = measurement["receiver_angles_raw"] - measurement["target_rotation"]
    beta = wrap_to_pi(receiver_angle - source_angle)
    expected_beta = wrap_to_pi(measurement["receiver_angles_raw"])
    if not np.allclose(beta, expected_beta, atol=1.0e-12):
        raise AssertionError("R(-theta) bistatic-angle cancellation failed.")

    overall_rows: list[dict[str, Any]] = []
    per_view_rows: list[dict[str, Any]] = []
    coefficient_rows: list[dict[str, Any]] = []
    for order in CALIBRATION_ORDERS:
        calibrated, transfer, coefficients = shared_bistatic_calibration(prediction, observation, beta, order)
        overall_rows.append({"K": order, **base.metrics(calibrated, observation)})
        for mode, coefficient in zip(range(-order, order + 1), coefficients):
            coefficient_rows.append({
                "K": order,
                "mode": mode,
                "coefficient_real": float(coefficient.real),
                "coefficient_imag": float(coefficient.imag),
                "coefficient_abs": float(abs(coefficient)),
                "coefficient_phase_rad": float(np.angle(coefficient)),
            })
        offset = 0
        for view in range(1, 37):
            n = indices[view].size
            section = slice(offset, offset + n)
            per_view_rows.append({
                "K": order,
                "view": view,
                "transfer_abs_min": float(np.abs(transfer[section]).min()),
                "transfer_abs_max": float(np.abs(transfer[section]).max()),
                "transfer_phase_min_rad": float(np.angle(transfer[section]).min()),
                "transfer_phase_max_rad": float(np.angle(transfer[section]).max()),
                **base.metrics(calibrated[section], observation[section]),
            })
            offset += n

    OUTPUT_DIR.mkdir(parents=False)
    with (OUTPUT_DIR / "global_bistatic_calibration_overall.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(overall_rows[0])); writer.writeheader(); writer.writerows(overall_rows)
    with (OUTPUT_DIR / "global_bistatic_calibration_per_view.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(per_view_rows[0])); writer.writeheader(); writer.writerows(per_view_rows)
    with (OUTPUT_DIR / "global_bistatic_calibration_coefficients.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(coefficient_rows[0])); writer.writeheader(); writer.writerows(coefficient_rows)
    report = {
        "scope": "True-cylinder forward diagnostic only. No PINN training, no observation rewrite for inversion, and no checkpoint modification.",
        "geometry": {
            "mapping": "R(-theta)",
            "source_angle": "-theta",
            "receiver_angle": "phi-theta",
            "beta": "wrap(receiver_angle-source_angle)=wrap(phi) in [-pi, pi]",
        },
        "incident": {"model": "per-view M=12 Fourier-Bessel", "ridge": base.M12_RIDGE},
        "calibration": "One global c(beta)=sum_{m=-K}^{K} b_m exp(i*m*beta) fit jointly over all 36x49 scattered samples by complex least squares.",
        "true_cylinder_evaluation_only": {"center_m": list(base.TRUE_CENTER), "radius_m": base.TRUE_RADIUS_M, "epsilon": base.EPS_OBJECT, "ls_grid": "64x64"},
        "uncalibrated_reference_complex_l2": 0.682706,
        "overall_forward_metrics": overall_rows,
        "best_K_by_complex_l2": min(overall_rows, key=lambda row: float(row["complex_relative_l2"])),
    }
    (OUTPUT_DIR / "summary.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
