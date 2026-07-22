from __future__ import annotations

"""Complex low-rank diagnostic of the M=12 true-cylinder residual."""

import csv
import json
from pathlib import Path

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import audit_fresnel_1ghz_rminus_m12_incident_ratio_calibration as base
from square_target.pinn_pixel_inverse_core import TargetSpec, TrainConfig, fit_fourier_bessel_incident, make_direct_ls_tensors, make_integral_tensors


HERE = Path(__file__).resolve().parent
OUTPUT_DIR = HERE / "audit_fresnel_1GHz_rminus_m12_residual_svd"
GRID_SIZE = 64
ENERGY_RANKS = (1, 2, 3, 5)
RECONSTRUCTION_RANKS = (1, 2, 3)


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
    target = TargetSpec(name="Fresnel residual SVD audit", kind="circle", eps_background=1.0, eps_object=3.0, roi_half_width=0.09)

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
    prediction = base.forward_predictions(direct_ls, base.true_epsilon(integral.quad_xy), np.column_stack(rhs), config.k0)

    measured_matrix = np.empty((36, 49), dtype=np.complex128)
    predicted_matrix = np.empty((36, 49), dtype=np.complex128)
    for view in range(1, 37):
        rows = indices[view]
        # Preserve the raw-file order inside each 49-sample view.  The data do
        # not expose one identical receiver-angle subset after R(-theta) for
        # every view, so forcing a synthetic cross-view angular registration
        # would distort the requested 36x49 measurement matrix.
        measured_matrix[view - 1] = measurement["scattered"][rows]
        predicted_matrix[view - 1] = prediction[rows, view - 1]

    residual = measured_matrix - predicted_matrix
    u, singular_values, vh = np.linalg.svd(residual, full_matrices=False)
    total_residual_energy = float(np.sum(singular_values**2))
    measured_norm = float(np.linalg.norm(measured_matrix))
    baseline_metrics = base.metrics(predicted_matrix.reshape(-1), measured_matrix.reshape(-1))
    rows: list[dict[str, float | int]] = []
    for rank in ENERGY_RANKS:
        rows.append({
            "rank": rank,
            "singular_value": float(singular_values[rank - 1]),
            "cumulative_residual_energy": float(np.sum(singular_values[:rank] ** 2) / total_residual_energy),
        })
    removal_rows: list[dict[str, float | int]] = []
    for rank in RECONSTRUCTION_RANKS:
        low_rank = (u[:, :rank] * singular_values[:rank]) @ vh[:rank, :]
        corrected_prediction = predicted_matrix + low_rank
        corrected = base.metrics(corrected_prediction.reshape(-1), measured_matrix.reshape(-1))
        removal_rows.append({
            "rank": rank,
            "residual_energy_removed": float(np.linalg.norm(low_rank) ** 2 / total_residual_energy),
            "remaining_residual_complex_l2": float(np.linalg.norm(residual - low_rank) / measured_norm),
            **corrected,
        })

    OUTPUT_DIR.mkdir(parents=False)
    np.save(OUTPUT_DIR / "residual_complex_36x49.npy", residual)
    np.save(OUTPUT_DIR / "singular_values.npy", singular_values)
    with (OUTPUT_DIR / "residual_svd_energy.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    with (OUTPUT_DIR / "residual_low_rank_removal.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(removal_rows[0])); writer.writeheader(); writer.writerows(removal_rows)

    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.3), constrained_layout=True)
    amplitude = np.abs(residual)
    phase = np.angle(residual)
    amplitude_image = axes[0].imshow(amplitude, aspect="auto", origin="lower", cmap="magma")
    axes[0].set_title("Residual amplitude")
    axes[0].set_xlabel("Receiver ID")
    axes[0].set_ylabel("View")
    fig.colorbar(amplitude_image, ax=axes[0], fraction=0.046, pad=0.04)
    phase_image = axes[1].imshow(phase, aspect="auto", origin="lower", cmap="twilight", vmin=-np.pi, vmax=np.pi)
    axes[1].set_title("Residual phase (rad)")
    axes[1].set_xlabel("Receiver ID")
    axes[1].set_ylabel("View")
    fig.colorbar(phase_image, ax=axes[1], fraction=0.046, pad=0.04)
    axes[2].semilogy(np.arange(1, singular_values.size + 1), singular_values, marker="o", ms=3)
    axes[2].set_title("Complex residual singular values")
    axes[2].set_xlabel("Singular-value index")
    axes[2].set_ylabel("Singular value")
    axes[2].grid(True, which="both", alpha=0.3)
    fig.savefig(OUTPUT_DIR / "residual_amplitude_phase_singular_values.png", dpi=220)
    plt.close(fig)

    report = {
        "scope": "True-cylinder residual SVD diagnostic only. No PINN/CSI training, no observation modification, and no nuisance model was added.",
        "geometry": "R(-theta), per-view M=12 Fourier-Bessel incident, 64x64 exp(+i omega t) LS forward model.",
        "residual_definition": "R[view, receiver_sample] = Es_measured - Es_forward_true_cylinder; rows are views 1..36 and columns are the 49 samples in their original file order within each view. No unverified cross-view receiver-angle registration is imposed.",
        "true_cylinder_evaluation_only": {"center_m": list(base.TRUE_CENTER), "radius_m": base.TRUE_RADIUS_M, "epsilon": base.EPS_OBJECT},
        "baseline": baseline_metrics,
        "cumulative_explained_residual_energy": rows,
        "low_rank_removal_metrics": removal_rows,
    }
    (OUTPUT_DIR / "summary.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
