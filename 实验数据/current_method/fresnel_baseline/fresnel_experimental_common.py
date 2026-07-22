from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Literal, Tuple

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[3]
SQUARE_REPRO_ROOT = REPO_ROOT / "square-target-repro"
if str(SQUARE_REPRO_ROOT) not in sys.path:
    sys.path.insert(0, str(SQUARE_REPRO_ROOT))

from square_target.pinn_pixel_inverse_core import (  # noqa: E402
    ObservationSet,
    TargetSpec,
    TrainConfig,
    estimate_incident_amplitude,
    train_double_branch_pinn_from_observations,
)


FieldMode = Literal["difference", "first_pair", "second_pair"]

SOURCE_RADIUS_M = 0.720
RECEIVER_RADIUS_M = 0.760
VIEW_STEP_DEG = 10.0
RECEIVER_STEP_DEG = 5.0


def fresnel_dieltm_target() -> TargetSpec:
    return TargetSpec(
        name="Fresnel decentered dielectric cylinder",
        kind="circle",
        eps_background=1.0,
        eps_object=3.0,
        roi_half_width=0.09,
        circle_radius=0.015,
        circle_center_x=0.0,
        circle_center_y=-0.030,
    )


def fresnel_default_config(
    frequency_index: int = 1,
    epochs: int = 40000,
    device: str = "auto",
    seed: int = 20260604,
) -> TrainConfig:
    return TrainConfig(
        frequency_hz=float(frequency_index) * 1.0e9,
        incident_amplitude=1.0,
        incident_phase_sign=-1.0,
        estimate_incident_amplitude=False,
        eps_min=1.0,
        eps_max=4.0,
        eps_initial=1.02,
        domain_radius=0.12,
        max_points_per_direction=0,
        data_batch_per_direction=49,
        n_pde=3072,
        n_boundary=768,
        n_tv_grid=72,
        epochs_adam=epochs,
        learning_rate=8.0e-4,
        weight_data=200.0,
        weight_pde=1.0,
        weight_boundary=0.01,
        weight_tv=4.0e-4,
        weight_contrast_l1=1.0e-6,
        robust_data_weighting=True,
        gradient_clip_norm=1.0,
        field_hidden_layers=7,
        field_hidden_units=112,
        eps_hidden_layers=5,
        eps_hidden_units=96,
        fourier_bands=6,
        fourier_max_frequency=10.0,
        plot_grid_size=240,
        log_every=100,
        checkpoint_every=2000,
        random_seed=seed,
        dtype="float32",
        device=device,
        noise_level=0.0,
    )


def _load_exp_numeric(path: Path) -> np.ndarray:
    data = np.loadtxt(path, comments="#")
    if data.ndim != 2 or data.shape[1] != 7:
        raise ValueError(f"{path} must contain 7 numeric columns after the 10-line header.")
    return data


def _complex_pairs(data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    total = data[:, 3].astype(np.float64) + 1j * data[:, 4].astype(np.float64)
    incident = data[:, 5].astype(np.float64) + 1j * data[:, 6].astype(np.float64)
    return total, incident


def _field_by_mode(total: np.ndarray, incident: np.ndarray, mode: FieldMode) -> np.ndarray:
    if mode == "difference":
        return total - incident
    if mode == "first_pair":
        return total
    if mode == "second_pair":
        return incident
    raise ValueError(f"Unsupported field mode: {mode}")


def _angles_from_indices(views: np.ndarray, receivers: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    source_theta = np.deg2rad((views.astype(np.float64) - 1.0) * VIEW_STEP_DEG)
    receiver_theta = np.deg2rad((receivers.astype(np.float64) - 1.0) * RECEIVER_STEP_DEG)
    return source_theta, receiver_theta


def _receiver_xy(receiver_theta: np.ndarray, receiver_radius: float) -> np.ndarray:
    return np.column_stack(
        (
            receiver_radius * np.cos(receiver_theta),
            receiver_radius * np.sin(receiver_theta),
        )
    ).astype(np.float64)


def _propagation_directions(source_theta: np.ndarray) -> np.ndarray:
    return np.column_stack((-np.cos(source_theta), -np.sin(source_theta))).astype(np.float64)


def _thin_per_view(
    data: np.ndarray,
    max_receivers_per_view: int,
    seed: int,
) -> np.ndarray:
    if max_receivers_per_view <= 0:
        return data

    rng = np.random.default_rng(seed)
    kept: List[np.ndarray] = []
    for view in np.unique(data[:, 0].astype(int)):
        view_data = data[data[:, 0].astype(int) == view]
        if view_data.shape[0] <= max_receivers_per_view:
            kept.append(view_data)
            continue
        order = np.sort(rng.choice(view_data.shape[0], max_receivers_per_view, replace=False))
        kept.append(view_data[order])
    return np.vstack(kept)


def load_fresnel_dieltm_observations(
    exp_path: Path,
    config: TrainConfig,
    frequency_index: int = 1,
    field_mode: FieldMode = "difference",
    receiver_radius: float = RECEIVER_RADIUS_M,
    max_receivers_per_view: int = 0,
) -> ObservationSet:
    data = _load_exp_numeric(exp_path)
    freq_ids = data[:, 2].astype(int)
    if frequency_index not in set(freq_ids.tolist()):
        raise ValueError(f"Frequency index {frequency_index} is not present in {exp_path}.")

    data = data[freq_ids == frequency_index]
    data = _thin_per_view(data, max_receivers_per_view, config.random_seed)

    views = data[:, 0].astype(int)
    receivers = data[:, 1].astype(int)
    source_theta, receiver_theta = _angles_from_indices(views, receivers)
    xy = _receiver_xy(receiver_theta, receiver_radius)
    directions = _propagation_directions(source_theta)
    total, incident = _complex_pairs(data)
    scattered = _field_by_mode(total, incident, field_mode)

    labels = np.asarray([f"view_{view:02d}" for view in views], dtype=object)
    direction_labels = [f"view_{view:02d}" for view in sorted(np.unique(views).tolist())]

    incident_amplitudes: Dict[str, complex] = {}
    for label in direction_labels:
        view_mask = labels == label
        direction = tuple(directions[view_mask][0])
        incident_amplitudes[label] = estimate_incident_amplitude(
            xy[view_mask],
            incident[view_mask],
            direction,
            config.k0,
            config.incident_phase_sign,
        )

    return ObservationSet(
        xy=xy,
        total=total,
        scattered=scattered,
        directions=directions,
        labels=labels,
        receiver_radius=float(receiver_radius),
        direction_labels=direction_labels,
        incident_amplitudes=incident_amplitudes,
        incident_fits={},
    )


def train_fresnel_dieltm_case(
    exp_path: Path,
    output_dir: Path,
    frequency_index: int = 1,
    field_mode: FieldMode = "difference",
    epochs: int = 40000,
    device: str = "auto",
    seed: int = 20260604,
    max_receivers_per_view: int = 0,
) -> Dict[str, float]:
    target = fresnel_dieltm_target()
    config = fresnel_default_config(
        frequency_index=frequency_index,
        epochs=epochs,
        device=device,
        seed=seed,
    )
    obs = load_fresnel_dieltm_observations(
        exp_path=exp_path,
        config=config,
        frequency_index=frequency_index,
        field_mode=field_mode,
        max_receivers_per_view=max_receivers_per_view,
    )
    return train_double_branch_pinn_from_observations(
        obs=obs,
        target=target,
        config=config,
        output_dir=output_dir,
    )
