"""Core network, Green operators, and losses for formal J-epsilon inversion.

This module is a repository-local packaging of the validated implementation
from the outer ``austria_synthetic_forward`` workspace.  The equations,
parameterization, BP initialization, and gradient routing are unchanged.
"""
from __future__ import annotations

import json
import math
import sys
from dataclasses import fields
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from scipy.signal import fftconvolve
from scipy.special import hankel1


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
sys.path.insert(0, str(REPO_ROOT / "square_target"))
import pinn_pixel_inverse_core as inverse_core  # noqa: E402

TINY = 1.0e-30


def load_method_config(path: Path) -> tuple[dict, inverse_core.TrainConfig]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    allowed = {item.name for item in fields(inverse_core.TrainConfig)}
    overrides = {
        "frequency_hz": payload["frequency_hz"],
        "c0": payload["c0"],
        "incident_amplitude": payload["incident_amplitude"],
        "incident_phase_sign": payload["incident_phase_sign"],
        "observation_imag_sign": payload["observation_imag_sign"],
        "eps_min": payload["epsilon_min"],
        "eps_max": payload["epsilon_max"],
        "field_hidden_layers": payload["field_hidden_layers"],
        "field_hidden_units": payload["field_hidden_units"],
        "eps_hidden_layers": payload["epsilon_hidden_layers"],
        "eps_hidden_units": payload["epsilon_hidden_units"],
        "fourier_bands": payload["fourier_bands"],
        "fourier_max_frequency": payload["fourier_max_frequency"],
        "field_parameterization": "direct",
        "field_envelope_parameterization": "complex",
        "gradient_clip_norm": payload["gradient_clip_norm"],
        "dtype": "float32",
        "device": "auto",
        "max_points_per_direction": 836,
        "incident_mode": "analytic",
        "incident_reference_dir": None,
        "random_seed": payload["seed"],
    }
    return payload, inverse_core.TrainConfig(
        **{key: value for key, value in overrides.items() if key in allowed}
    )


class JEpsilonNetwork(torch.nn.Module):
    """Complex contrast-current branch plus angle-independent epsilon branch."""

    def __init__(
        self, config: inverse_core.TrainConfig, roi_half_width: float
    ) -> None:
        super().__init__()
        self.roi_half_width = float(roi_half_width)
        self.j_branch = inverse_core.FieldBranch(config, roi_half_width)
        j_final = self.j_branch.mlp.net[-1]
        if not isinstance(j_final, torch.nn.Linear):
            raise TypeError("J branch final layer must be Linear")
        with torch.no_grad():
            j_final.weight.zero_()
            j_final.bias.zero_()

        num_bands = (
            config.fourier_bands
            if config.epsilon_fourier_bands is None
            else config.epsilon_fourier_bands
        )
        max_frequency = (
            config.fourier_max_frequency
            if config.epsilon_fourier_max_frequency is None
            else config.epsilon_fourier_max_frequency
        )
        self.epsilon_features = inverse_core.FourierFeatureMap(
            2, num_bands, max_frequency, include_input=True
        )
        self.epsilon_mlp = inverse_core.MLP(
            input_dim=self.epsilon_features.output_dim,
            output_dim=1,
            hidden_layers=config.eps_hidden_layers,
            hidden_units=config.eps_hidden_units,
            activation="silu",
        )
        epsilon_final = self.epsilon_mlp.net[-1]
        if not isinstance(epsilon_final, torch.nn.Linear):
            raise TypeError("epsilon branch final layer must be Linear")
        with torch.no_grad():
            epsilon_final.weight.zero_()
            epsilon_final.bias.fill_(-4.0)

    def current(
        self, xy: torch.Tensor, directions: torch.Tensor
    ) -> torch.Tensor:
        values = self.j_branch(xy, directions)
        return torch.complex(values[..., 0], values[..., 1])

    def epsilon(self, xy: torch.Tensor) -> torch.Tensor:
        raw = self.epsilon_mlp(
            self.epsilon_features(xy / self.roi_half_width)
        )
        return 1.0 + 3.0 * torch.sigmoid(raw)


def branch_parameters(
    model: JEpsilonNetwork, branch: str
) -> list[torch.nn.Parameter]:
    module = model.j_branch if branch == "J" else model.epsilon_mlp
    return list(module.parameters())


def set_branch_trainable(
    model: JEpsilonNetwork, branch: str
) -> list[torch.nn.Parameter]:
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    selected = branch_parameters(model, branch)
    for parameter in selected:
        parameter.requires_grad_(True)
    return selected


def equivalent_disk_self_integral(k0: float, cell_area: float) -> complex:
    radius = math.sqrt(cell_area / math.pi)
    return (
        0.5j * math.pi * radius * hankel1(1, k0 * radius) / k0
        - 1.0 / k0**2
    )


class GreenOperators:
    """Outgoing H1 domain operator GD and dense receiver operator GS."""

    def __init__(
        self, axis: np.ndarray, receiver_xy: np.ndarray, k0: float
    ) -> None:
        self.axis = np.asarray(axis, dtype=np.float64)
        self.n = int(self.axis.size)
        self.spacing = float(self.axis[1] - self.axis[0])
        self.cell_area = self.spacing**2
        self.k0 = float(k0)
        offsets = np.arange(-(self.n - 1), self.n, dtype=np.float64)
        offsets *= self.spacing
        oy, ox = np.meshgrid(offsets, offsets, indexing="ij")
        distance = np.sqrt(ox**2 + oy**2)
        point_radius = math.sqrt(self.cell_area / math.pi)
        safe = np.where(distance < 1.0e-15, point_radius, distance)
        self.kernel = (
            self.k0**2
            * self.cell_area
            * 0.25j
            * hankel1(0, self.k0 * safe)
        ).astype(np.complex128)
        center = self.n - 1
        self.kernel[center, center] = (
            self.k0**2
            * equivalent_disk_self_integral(self.k0, self.cell_area)
        )
        yy, xx = np.meshgrid(self.axis, self.axis, indexing="ij")
        source_xy = np.column_stack((xx.reshape(-1), yy.reshape(-1)))
        delta = (
            np.asarray(receiver_xy, dtype=np.float64)[:, None, :]
            - source_xy[None, :, :]
        )
        receiver_distance = np.linalg.norm(delta, axis=2)
        self.GS = (
            self.k0**2
            * self.cell_area
            * 0.25j
            * hankel1(0, self.k0 * receiver_distance)
        ).astype(np.complex128)

    def gd(self, currents: np.ndarray) -> np.ndarray:
        return np.stack(
            [fftconvolve(current, self.kernel, mode="same") for current in currents]
        )

    def gs(self, currents: np.ndarray) -> np.ndarray:
        return (self.GS @ currents.reshape(currents.shape[0], -1).T).T

    def gs_adjoint(self, receiver_fields: np.ndarray) -> np.ndarray:
        flat = (self.GS.conj().T @ receiver_fields.T).T
        return flat.reshape(receiver_fields.shape[0], self.n, self.n)


class TorchGreenOperators:
    """Differentiable complex64 actions matching GreenOperators."""

    def __init__(self, operators: GreenOperators, device: torch.device) -> None:
        self.n = operators.n
        self.full = 3 * self.n - 2
        self.crop = self.n - 1
        kernel = torch.as_tensor(
            operators.kernel, dtype=torch.complex64, device=device
        )
        self.kernel_fft = torch.fft.fft2(
            kernel, s=(self.full, self.full)
        )
        self.gs_matrix = torch.as_tensor(
            operators.GS, dtype=torch.complex64, device=device
        )

    def gd(self, currents: torch.Tensor) -> torch.Tensor:
        transformed = torch.fft.fft2(
            currents, s=(self.full, self.full), dim=(-2, -1)
        )
        full = torch.fft.ifft2(
            transformed * self.kernel_fft[None], dim=(-2, -1)
        )
        start = self.crop
        return full[:, start : start + self.n, start : start + self.n]

    def gs(self, currents: torch.Tensor) -> torch.Tensor:
        return currents.reshape(currents.shape[0], -1) @ self.gs_matrix.T


def back_projection(
    measured: np.ndarray, operators: GreenOperators
) -> np.ndarray:
    raw = operators.gs_adjoint(measured)
    currents = np.empty_like(raw)
    for direction in range(measured.shape[0]):
        predicted_raw = operators.gs(raw[direction : direction + 1])[0]
        numerator = float(
            np.real(np.vdot(predicted_raw, measured[direction]))
        )
        denominator = float(np.vdot(predicted_raw, predicted_raw).real)
        currents[direction] = numerator / max(denominator, TINY) * raw[direction]
    return currents


def fd4_laplacian(field: torch.Tensor, spacing: float) -> torch.Tensor:
    return (
        -field[:, 2:-2, 4:]
        + 16.0 * field[:, 2:-2, 3:-1]
        - 30.0 * field[:, 2:-2, 2:-2]
        + 16.0 * field[:, 2:-2, 1:-3]
        - field[:, 2:-2, :-4]
        - field[:, 4:, 2:-2]
        + 16.0 * field[:, 3:-1, 2:-2]
        - 30.0 * field[:, 2:-2, 2:-2]
        + 16.0 * field[:, 1:-3, 2:-2]
        - field[:, :-4, 2:-2]
    ) / (12.0 * spacing**2)


def normalized_terms(
    currents: torch.Tensor,
    epsilon: torch.Tensor,
    incident: torch.Tensor,
    measured: torch.Tensor,
    operators: TorchGreenOperators,
    spacing: float,
    k0: float,
) -> Dict[str, torch.Tensor]:
    gd_current = operators.gd(currents)
    total = incident + gd_current
    predicted_receiver = operators.gs(currents)
    data_residual = predicted_receiver - measured
    chi = epsilon - 1.0
    state = currents - chi[None] * total
    laplacian = fd4_laplacian(total, spacing)
    mass = (k0**2) * epsilon[None, 2:-2, 2:-2] * total[:, 2:-2, 2:-2]
    pde = laplacian + mass
    data_scale = torch.mean(torch.abs(measured).square()).detach().clamp_min(TINY)
    state_scale = (
        torch.mean(torch.abs(currents).square())
        + torch.mean(torch.abs(chi[None] * total).square())
    ).detach().clamp_min(TINY)
    pde_scale = (
        torch.mean(torch.abs(laplacian).square())
        + torch.mean(torch.abs(mass).square())
    ).detach().clamp_min(TINY)
    return {
        "data_loss": torch.mean(torch.abs(data_residual).square()) / data_scale,
        "state_loss": torch.mean(torch.abs(state).square()) / state_scale,
        "pde_loss": torch.mean(torch.abs(pde).square()) / pde_scale,
        "predicted_receiver": predicted_receiver,
        "total_field": total,
    }


def tv_edge_loss(epsilon: torch.Tensor) -> torch.Tensor:
    dx = epsilon[:, 1:] - epsilon[:, :-1]
    dy = epsilon[1:, :] - epsilon[:-1, :]
    return 0.5 * (
        torch.mean(torch.sqrt(dx.square() + 1.0e-8))
        + torch.mean(torch.sqrt(dy.square() + 1.0e-8))
    )
