from __future__ import annotations

"""Incident-only receiver calibration audit for the Fresnel 1 GHz cylinder.

The Fourier coefficients in ``c_v(phi)`` are fitted solely from the empty
incident data ratio Einc_measured / Einc_M12.  Scattered data and the true
cylinder are used only after that fit, for the forward-consistency report.
"""

import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.special import hankel2


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
SQUARE_REPRO_ROOT = REPO_ROOT / "square-target-repro"
if str(SQUARE_REPRO_ROOT) not in sys.path:
    sys.path.insert(0, str(SQUARE_REPRO_ROOT))

from square_target.pinn_pixel_inverse_core import (  # noqa: E402
    TargetSpec,
    TrainConfig,
    fit_fourier_bessel_incident,
    make_direct_ls_tensors,
    make_integral_tensors,
)


DATA_FILE = HERE / "fresnel_2001" / "dielTM_dec8f.exp"
OUTPUT_DIR = HERE / "audit_fresnel_1GHz_rminus_m12_incident_ratio_calibration"
GRID_SIZE = 64
RHO_M = 0.74
EPS_BACKGROUND = 1.0
EPS_OBJECT = 3.0
TRUE_CENTER = (0.0, -0.03)
TRUE_RADIUS_M = 0.015
M12_ORDER = 12
M12_RIDGE = 1.0e-3
CALIBRATION_ORDERS = (1, 2)


def outgoing_green(distance: np.ndarray, k0: float) -> np.ndarray:
    """2D outgoing Green function under the exp(+i omega t) convention."""
    return -0.25j * hankel2(0, k0 * distance)


def load_measurement() -> dict[str, np.ndarray]:
    raw = np.loadtxt(DATA_FILE, comments="#")
    data = raw[raw[:, 2].astype(int) == 1]
    views = data[:, 0].astype(int)
    receiver_ids = data[:, 1].astype(int)
    total = data[:, 3] + 1j * data[:, 4]
    incident = data[:, 5] + 1j * data[:, 6]
    return {
        "views": views,
        "receiver_ids": receiver_ids,
        "receiver_angles_raw": np.deg2rad((receiver_ids - 1) * 5.0),
        "target_rotation": np.deg2rad((views - 1) * 10.0),
        "total": total,
        "incident": incident,
        "scattered": total - incident,
    }


def mapped_receiver_xy(measurement: dict[str, np.ndarray]) -> np.ndarray:
    """R(-theta) mapping from the rotating-target experiment to fixed target coordinates."""
    angle = measurement["receiver_angles_raw"] - measurement["target_rotation"]
    return RHO_M * np.column_stack((np.cos(angle), np.sin(angle)))


def view_indices(views: np.ndarray) -> dict[int, np.ndarray]:
    indices = {view: np.flatnonzero(views == view) for view in range(1, 37)}
    if any(rows.size != 49 for rows in indices.values()):
        raise ValueError("Expected 36 views with exactly 49 receiver samples each.")
    return indices


def true_epsilon(quad_xy: torch.Tensor) -> torch.Tensor:
    radius_sq = (quad_xy[:, 0] - TRUE_CENTER[0]).square() + (quad_xy[:, 1] - TRUE_CENTER[1]).square()
    return torch.where(radius_sq <= TRUE_RADIUS_M**2, torch.full_like(radius_sq, EPS_OBJECT), torch.ones_like(radius_sq))


def replace_with_exp_iwt_green(direct_ls: Any, receiver_xy: np.ndarray, k0: float) -> None:
    integral = direct_ls.integral
    quad_xy = integral.quad_xy.detach().cpu().numpy()
    cell_radius = math.sqrt(integral.area_weight / math.pi)
    domain_distance = np.linalg.norm(quad_xy[:, None, :] - quad_xy[None, :, :], axis=2)
    domain_distance[domain_distance < 1.0e-12] = cell_radius
    receiver_distance = np.linalg.norm(receiver_xy[:, None, :] - quad_xy[None, :, :], axis=2)
    direct_ls.domain_green.copy_(torch.as_tensor(outgoing_green(domain_distance, k0), dtype=direct_ls.domain_green.dtype))
    receiver_green = outgoing_green(receiver_distance, k0)
    integral.green_re.copy_(torch.as_tensor(receiver_green.real, dtype=integral.green_re.dtype))
    integral.green_im.copy_(torch.as_tensor(receiver_green.imag, dtype=integral.green_im.dtype))


def forward_predictions(direct_ls: Any, epsilon: torch.Tensor, incident_rhs: np.ndarray, k0: float) -> np.ndarray:
    integral = direct_ls.integral
    n_source = epsilon.numel()
    complex_dtype = direct_ls.domain_green.dtype
    chi = (epsilon.reshape(n_source, 1) - EPS_BACKGROUND).to(complex_dtype)
    scale = float(k0**2 * integral.area_weight)
    system = torch.eye(n_source, dtype=complex_dtype) - scale * direct_ls.domain_green * chi.reshape(1, -1)
    with torch.no_grad():
        internal = torch.linalg.solve(system, torch.as_tensor(incident_rhs, dtype=complex_dtype))
        source = scale * chi * internal
        receiver_green = torch.complex(integral.green_re, integral.green_im).to(complex_dtype)
        return (receiver_green @ source).cpu().numpy()


def fit_incident_ratio_calibration(ratio: np.ndarray, phi: np.ndarray, order: int) -> tuple[np.ndarray, np.ndarray]:
    """Unweighted complex Fourier least squares for the measured incident ratio."""
    modes = np.arange(-order, order + 1, dtype=np.int64)
    basis = np.exp(1j * phi[:, None] * modes[None, :])
    coefficients, _residuals, _rank, _singular = np.linalg.lstsq(basis, ratio, rcond=None)
    return basis @ coefficients, coefficients


def metrics(prediction: np.ndarray, observation: np.ndarray) -> dict[str, float]:
    phase = np.angle(np.exp(1j * (np.angle(prediction) - np.angle(observation))))
    return {
        "complex_relative_l2": float(np.linalg.norm(prediction - observation) / np.linalg.norm(observation)),
        "amplitude_relative_error": float(np.linalg.norm(np.abs(prediction) - np.abs(observation)) / np.linalg.norm(np.abs(observation))),
        "phase_mae_rad": float(np.mean(np.abs(phase))),
        "phase_mae_deg": float(np.degrees(np.mean(np.abs(phase)))),
    }


def main() -> None:
    if OUTPUT_DIR.exists():
        raise FileExistsError(f"Refusing to overwrite existing audit output: {OUTPUT_DIR}")
    measurement = load_measurement()
    indices = view_indices(measurement["views"])
    receiver_xy = mapped_receiver_xy(measurement)
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
    target = TargetSpec(
        name="Fresnel incident-ratio calibration audit",
        kind="circle",
        eps_background=EPS_BACKGROUND,
        eps_object=EPS_OBJECT,
        roi_half_width=0.09,
    )

    class AuditObservation:
        xy = receiver_xy

    integral = make_integral_tensors(AuditObservation(), target, config, torch.device("cpu"), torch.float32)
    direct_ls = make_direct_ls_tensors(integral, config, torch.device("cpu"), torch.float32)
    replace_with_exp_iwt_green(direct_ls, receiver_xy, config.k0)
    quad_xy = integral.quad_xy.detach().cpu().numpy()

    incident_rhs: list[np.ndarray] = []
    m12_receiver: dict[int, np.ndarray] = {}
    calibration_values: dict[int, dict[int, np.ndarray]] = {}
    coefficient_rows: list[dict[str, Any]] = []
    incident_rows: list[dict[str, Any]] = []
    for view in range(1, 37):
        rows = indices[view]
        fit = fit_fourier_bessel_incident(receiver_xy[rows], measurement["incident"][rows], config.k0, M12_ORDER, M12_RIDGE)
        fitted_receiver = fit.evaluate(receiver_xy[rows])
        m12_receiver[view] = fitted_receiver
        incident_rhs.append(fit.evaluate(quad_xy))
        ratio = measurement["incident"][rows] / fitted_receiver
        phi = np.arctan2(receiver_xy[rows, 1], receiver_xy[rows, 0])
        incident_rows.append({"view": view, "record_type": "M12_incident_fit", **metrics(fitted_receiver, measurement["incident"][rows])})
        for order in CALIBRATION_ORDERS:
            fitted_ratio, coefficients = fit_incident_ratio_calibration(ratio, phi, order)
            calibration_values.setdefault(order, {})[view] = fitted_ratio
            incident_rows.append({
                "view": view,
                "record_type": f"incident_ratio_K{order}",
                "ratio_mean_abs": float(np.mean(np.abs(ratio))),
                "ratio_std_abs": float(np.std(np.abs(ratio))),
                "ratio_fit_relative_l2": float(np.linalg.norm(fitted_ratio - ratio) / np.linalg.norm(ratio)),
                "incident_after_calibration_complex_relative_l2": metrics(measurement["incident"][rows] / fitted_ratio, fitted_receiver)["complex_relative_l2"],
            })
            for mode, coefficient in zip(range(-order, order + 1), coefficients):
                coefficient_rows.append({
                    "view": view, "K": order, "mode": mode,
                    "coefficient_real": float(coefficient.real), "coefficient_imag": float(coefficient.imag),
                    "coefficient_abs": float(abs(coefficient)), "coefficient_phase_rad": float(np.angle(coefficient)),
                })

    prediction = forward_predictions(direct_ls, true_epsilon(integral.quad_xy), np.column_stack(incident_rhs), config.k0)
    variants: dict[str, np.ndarray] = {"original": measurement["scattered"]}
    for order in CALIBRATION_ORDERS:
        calibrated_total = measurement["total"].copy()
        calibrated_incident = measurement["incident"].copy()
        for view in range(1, 37):
            rows = indices[view]
            c_value = calibration_values[order][view]
            calibrated_total[rows] /= c_value
            calibrated_incident[rows] /= c_value
        variants[f"incident_ratio_K{order}"] = calibrated_total - calibrated_incident

    forward_rows: list[dict[str, Any]] = []
    overall: dict[str, dict[str, float]] = {}
    for name, observed in variants.items():
        predicted_views = [prediction[indices[view], view - 1] for view in range(1, 37)]
        observed_views = [observed[indices[view]] for view in range(1, 37)]
        overall[name] = metrics(np.concatenate(predicted_views), np.concatenate(observed_views))
        for view, pred_view, observed_view in zip(range(1, 37), predicted_views, observed_views):
            forward_rows.append({"variant": name, "view": view, **metrics(pred_view, observed_view)})
        forward_rows.append({"variant": name, "view": "ALL", **overall[name]})

    OUTPUT_DIR.mkdir(parents=False)
    with (OUTPUT_DIR / "incident_ratio_calibration_coefficients.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(coefficient_rows[0])); writer.writeheader(); writer.writerows(coefficient_rows)
    with (OUTPUT_DIR / "incident_ratio_fit_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = list(dict.fromkeys(key for row in incident_rows for key in row)); writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(incident_rows)
    with (OUTPUT_DIR / "incident_ratio_calibration_forward_audit.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(forward_rows[0])); writer.writeheader(); writer.writerows(forward_rows)
    report = {
        "scope": "Forward audit only; no PINN/CSI training, no checkpoint modification, and no ground truth used to fit calibration.",
        "geometry": "R(-theta): receiver target-fixed angle = raw receiver angle - target rotation.",
        "measurement": "Etotal and Eincident are Re+iIm; Es=Etotal-Eincident; exp(+i omega t).",
        "incident_model": {"basis": "per-view Fourier-Bessel M=12", "ridge": M12_RIDGE},
        "calibration": {
            "definition": "ratio=Eincident_measured/Eincident_M12; c_v(phi)=sum_m b_vm exp(i*m*phi), fitted independently per view from ratio only.",
            "orders": list(CALIBRATION_ORDERS),
            "application": "Etotal_cal=Etotal_measured/c; Eincident_cal=Eincident_measured/c; Es_cal=Etotal_cal-Eincident_cal.",
        },
        "true_cylinder_evaluation_only": {"center_m": list(TRUE_CENTER), "radius_m": TRUE_RADIUS_M, "epsilon": EPS_OBJECT, "ls_grid": f"{GRID_SIZE}x{GRID_SIZE}"},
        "overall_forward_metrics": overall,
        "reference_original_complex_l2": 0.682706,
    }
    (OUTPUT_DIR / "summary.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
