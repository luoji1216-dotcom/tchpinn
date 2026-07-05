from __future__ import annotations

import csv
import copy
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    import torch
    from torch import nn
except ModuleNotFoundError:  # Allows static parsing on machines without torch.
    torch = None

    class _MissingTorchNN:
        class Module:
            pass

    nn = _MissingTorchNN()


Direction = Tuple[float, float]


@dataclass
class TargetSpec:
    name: str
    kind: str
    eps_background: float
    eps_object: float
    roi_half_width: float = 0.5
    square_side: float = 0.4
    circle_radius: float = 0.10
    circle_center_x: float = 0.0
    circle_center_y: float = 0.30
    circle_center_offset_x: float = 0.15
    ring_center_y: float = -0.10
    ring_inner_radius: float = 0.175
    ring_outer_radius: float = 0.275


@dataclass
class TrainConfig:
    frequency_hz: float = 0.3e9
    c0: float = 3.0e8
    incident_amplitude: float = 0.1
    incident_phase_sign: float = -1.0
    observation_imag_sign: float = 1.0
    estimate_incident_amplitude: bool = False
    eps_min: float = 1.0
    eps_max: float = 5.0
    eps_initial: float = 1.05
    domain_radius: Optional[float] = None
    max_points_per_direction: int = 1200
    data_batch_per_direction: int = 256
    n_pde: int = 2048
    n_boundary: int = 512
    n_tv_grid: int = 48
    epochs_adam: int = 10000
    epochs_lbfgs: int = 0
    lbfgs_steps: int = 0
    lbfgs_lr: float = 1.0
    lbfgs_max_iter: int = 20
    lbfgs_history_size: int = 50
    learning_rate: float = 1.0e-3
    # epochs_adam: int = 30000
    # learning_rate: float = 1.0e-3
    field_lr: Optional[float] = None
    epsilon_lr: Optional[float] = None
    freeze_epsilon_steps: int = 0
    weight_data: float = 100.0
    weight_pde: float = 0.004
    weight_boundary: float = 0.04
    weight_integral_data: float = 0.0
    weight_tv: float = 0.025
    weight_contrast_l1: float = 1.0e-3
    binary_push_weight: float = 0.0
    epsilon_prior_weight: float = 0.0
    weight_edge_preserving: float = 0.0
    background_anchor_weight: float = 0.0
    background_anchor_from: Optional[str] = None
    background_anchor_threshold: float = 1.3
    robust_data_weighting: bool = True
    loss_preset: str = "current"
    lambda_f: float = 1.0
    lambda_d: float = 100.0
    lambda_ep: float = 100.0
    adaptive_gamma: float = 1.0
    adaptive_delta: float = 1.0e-8
    edge_delta: float = 1.0e-3
    gradient_clip_norm: float = 1.0
    field_hidden_layers: int = 6
    field_hidden_units: int = 96
    eps_hidden_layers: int = 5
    eps_hidden_units: int = 96
    fourier_bands: int = 5
    fourier_max_frequency: float = 8.0
    integral_grid_size: int = 32
    plot_grid_size: int = 220
    log_every: int = 100
    checkpoint_every: int = 2000
    random_seed: int = 20260430
    dtype: str = "float32"
    device: str = "auto"
    noise_level: float = 0.0
    resume_checkpoint: Optional[str] = None
    resume_epsilon_from: Optional[str] = None

    @property
    def wavelength(self) -> float:
        return self.c0 / self.frequency_hz

    @property
    def k0(self) -> float:
        return 2.0 * math.pi / self.wavelength


@dataclass
class ObservationSet:
    xy: np.ndarray
    total: np.ndarray
    scattered: np.ndarray
    directions: np.ndarray
    labels: np.ndarray
    receiver_radius: float
    direction_labels: List[str]
    incident_amplitudes: Dict[str, complex]


@dataclass
class ObservationTensors:
    xy: "torch.Tensor"
    target: "torch.Tensor"
    directions: "torch.Tensor"
    amplitudes: "torch.Tensor"
    indices_by_label: Dict[str, "torch.Tensor"]
    unique_directions: "torch.Tensor"
    unique_amplitudes: "torch.Tensor"
    receiver_radius: float


@dataclass
class IntegralTensors:
    quad_xy: "torch.Tensor"
    green_re: "torch.Tensor"
    green_im: "torch.Tensor"
    area_weight: float


def require_torch() -> None:
    if torch is None:
        raise RuntimeError(
            "PyTorch is required for the double-branch PINN. Install torch in the "
            "Python environment used to run the training script."
        )


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def resolve_device(device: str) -> "torch.device":
    require_torch()
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def resolve_dtype(dtype: str) -> "torch.dtype":
    require_torch()
    if dtype == "float64":
        return torch.float64
    if dtype == "float32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {dtype}")


def parse_direction_from_name(path: Path) -> Tuple[str, Direction]:
    name = path.stem.strip().lower().replace(" ", "")
    aliases = {
        "+x": ("+x", (1.0, 0.0)),
        "x+": ("+x", (1.0, 0.0)),
        "-x": ("-x", (-1.0, 0.0)),
        "x-": ("-x", (-1.0, 0.0)),
        "+y": ("+y", (0.0, 1.0)),
        "y+": ("+y", (0.0, 1.0)),
        "-y": ("-y", (0.0, -1.0)),
        "y-": ("-y", (0.0, -1.0)),
    }
    if name in aliases:
        return aliases[name]
    raise ValueError(
        f"Cannot infer incident direction from file name {path.name!r}. "
        "Use names such as +x.txt, -x.txt, +y.txt, -y.txt."
    )


def _direction_sort_key(item: Tuple[Path, str, Direction]) -> int:
    order = {"+x": 0, "-x": 1, "+y": 2, "-y": 3}
    return order[item[1]]


def load_fem_table(path: Path, imag_sign: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    if imag_sign not in (1.0, -1.0):
        raise ValueError("imag_sign must be +1.0 or -1.0.")
    data = np.loadtxt(path, skiprows=2)
    if data.ndim != 2 or data.shape[1] < 9:
        raise ValueError(f"{path} must contain at least 9 columns after the header.")
    xy = data[:, 0:2].astype(np.float64)
    ez = data[:, 7].astype(np.float64) + 1j * imag_sign * data[:, 8].astype(np.float64)
    return xy, ez


def incident_field_numpy(
    xy: np.ndarray,
    direction: Direction,
    k0: float,
    amplitude: complex,
    phase_sign: float,
) -> np.ndarray:
    direction_arr = np.asarray(direction, dtype=np.float64)
    phase = phase_sign * k0 * (xy @ direction_arr)
    return amplitude * np.exp(1j * phase)


def estimate_incident_amplitude(
    xy: np.ndarray,
    total: np.ndarray,
    direction: Direction,
    k0: float,
    phase_sign: float,
) -> complex:
    phase = incident_field_numpy(xy, direction, k0, 1.0 + 0.0j, phase_sign)
    denom = np.vdot(phase, phase)
    if abs(denom) < 1e-14:
        return 0.0 + 0.0j
    return np.vdot(phase, total) / denom


def load_observations(
    data_dir: Path,
    config: TrainConfig,
    direction_labels: Optional[Sequence[str]] = None,
) -> ObservationSet:
    files_with_dirs: List[Tuple[Path, str, Direction]] = []
    requested = set(direction_labels or [])
    for path in data_dir.glob("*.txt"):
        label, direction = parse_direction_from_name(path)
        if requested and label not in requested:
            continue
        files_with_dirs.append((path, label, direction))
    files_with_dirs.sort(key=_direction_sort_key)
    if not files_with_dirs:
        raise FileNotFoundError(f"No FEM txt files found in {data_dir}.")

    rng = np.random.default_rng(config.random_seed)
    xy_all: List[np.ndarray] = []
    total_all: List[np.ndarray] = []
    scattered_all: List[np.ndarray] = []
    directions_all: List[np.ndarray] = []
    labels_all: List[np.ndarray] = []
    labels_sorted: List[str] = []
    amplitudes: Dict[str, complex] = {}

    for path, label, direction in files_with_dirs:
        xy, total = load_fem_table(path, imag_sign=config.observation_imag_sign)
        if config.max_points_per_direction > 0 and xy.shape[0] > config.max_points_per_direction:
            # Uniform angular thinning keeps the observation circle balanced.
            keep = np.linspace(0, xy.shape[0] - 1, config.max_points_per_direction).round().astype(int)
            xy = xy[keep]
            total = total[keep]

        if config.estimate_incident_amplitude:
            amplitude = estimate_incident_amplitude(
                xy, total, direction, config.k0, config.incident_phase_sign
            )
        else:
            amplitude = complex(config.incident_amplitude, 0.0)

        inc = incident_field_numpy(xy, direction, config.k0, amplitude, config.incident_phase_sign)
        scattered = total - inc
        print(
            f"{label}: mean|total|={np.mean(np.abs(total)):.6e}, "
            f"mean|incident|={np.mean(np.abs(inc)):.6e}, "
            f"mean|scattered|={np.mean(np.abs(scattered)):.6e}",
            flush=True,
        )
        if config.noise_level > 0:
            rms = np.sqrt(np.mean(np.abs(scattered) ** 2))
            sigma = config.noise_level * max(rms, 1e-12)
            scattered = scattered + sigma / math.sqrt(2.0) * (
                rng.standard_normal(scattered.shape) + 1j * rng.standard_normal(scattered.shape)
            )

        xy_all.append(xy)
        total_all.append(total)
        scattered_all.append(scattered)
        directions_all.append(np.tile(np.asarray(direction, dtype=np.float64), (xy.shape[0], 1)))
        labels_all.append(np.full(xy.shape[0], label, dtype=object))
        labels_sorted.append(label)
        amplitudes[label] = amplitude

    xy_cat = np.vstack(xy_all)
    total_cat = np.concatenate(total_all)
    scattered_cat = np.concatenate(scattered_all)
    directions_cat = np.vstack(directions_all)
    labels_cat = np.concatenate(labels_all)
    receiver_radius = float(np.median(np.linalg.norm(xy_cat, axis=1)))
    return ObservationSet(
        xy=xy_cat,
        total=total_cat,
        scattered=scattered_cat,
        directions=directions_cat,
        labels=labels_cat,
        receiver_radius=receiver_radius,
        direction_labels=labels_sorted,
        incident_amplitudes=amplitudes,
    )


def to_observation_tensors(
    obs: ObservationSet,
    device: "torch.device",
    dtype: "torch.dtype",
) -> ObservationTensors:
    require_torch()
    xy = torch.as_tensor(obs.xy, dtype=dtype, device=device)
    target_np = np.column_stack((obs.scattered.real, obs.scattered.imag))
    target = torch.as_tensor(target_np, dtype=dtype, device=device)
    directions = torch.as_tensor(obs.directions, dtype=dtype, device=device)
    amplitudes_np = np.zeros((obs.xy.shape[0], 2), dtype=np.float64)
    indices_by_label = {}
    for label in obs.direction_labels:
        idx_np = np.flatnonzero(obs.labels == label)
        indices_by_label[label] = torch.as_tensor(idx_np, dtype=torch.long, device=device)
        amp = obs.incident_amplitudes[label]
        amplitudes_np[idx_np, 0] = amp.real
        amplitudes_np[idx_np, 1] = amp.imag
    amplitudes = torch.as_tensor(amplitudes_np, dtype=dtype, device=device)
    unique_dirs_np = []
    unique_amp_np = []
    for label in obs.direction_labels:
        idx = np.flatnonzero(obs.labels == label)[0]
        unique_dirs_np.append(obs.directions[idx])
        amp = obs.incident_amplitudes[label]
        unique_amp_np.append([amp.real, amp.imag])
    unique_directions = torch.as_tensor(np.vstack(unique_dirs_np), dtype=dtype, device=device)
    unique_amplitudes = torch.as_tensor(np.vstack(unique_amp_np), dtype=dtype, device=device)
    return ObservationTensors(
        xy=xy,
        target=target,
        directions=directions,
        amplitudes=amplitudes,
        indices_by_label=indices_by_label,
        unique_directions=unique_directions,
        unique_amplitudes=unique_amplitudes,
        receiver_radius=obs.receiver_radius,
    )


class FourierFeatureMap(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_bands: int,
        max_frequency: float,
        include_input: bool = True,
    ) -> None:
        super().__init__()
        self.include_input = include_input
        if num_bands <= 0:
            frequencies = torch.empty(0)
        elif num_bands == 1:
            frequencies = torch.tensor([max_frequency], dtype=torch.float32)
        else:
            frequencies = torch.logspace(
                0.0, math.log2(max_frequency), steps=num_bands, base=2.0
            )
        self.register_buffer("frequencies", frequencies)
        self.output_dim = (input_dim if include_input else 0) + 2 * input_dim * num_bands

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        parts = []
        if self.include_input:
            parts.append(x)
        if self.frequencies.numel() > 0:
            angles = math.pi * x[..., None, :] * self.frequencies[:, None]
            parts.append(torch.sin(angles).flatten(start_dim=-2))
            parts.append(torch.cos(angles).flatten(start_dim=-2))
        return torch.cat(parts, dim=-1)


class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_layers: int,
        hidden_units: int,
        activation: str = "tanh",
    ) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        dim = input_dim
        for _ in range(hidden_layers):
            layers.append(nn.Linear(dim, hidden_units))
            if activation == "silu":
                layers.append(nn.SiLU())
            elif activation == "relu":
                layers.append(nn.ReLU())
            elif activation == "tanh":
                layers.append(nn.Tanh())
            else:
                raise ValueError(f"Unsupported activation: {activation}")
            dim = hidden_units
        layers.append(nn.Linear(dim, output_dim))
        self.net = nn.Sequential(*layers)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        return self.net(x)


class FieldBranch(nn.Module):
    def __init__(self, config: TrainConfig, roi_half_width: float) -> None:
        super().__init__()
        self.k0 = config.k0
        self.roi_half_width = roi_half_width
        self.spatial_features = FourierFeatureMap(
            input_dim=2,
            num_bands=config.fourier_bands,
            max_frequency=config.fourier_max_frequency,
            include_input=True,
        )
        input_dim = self.spatial_features.output_dim + 2 + 2
        self.mlp = MLP(
            input_dim=input_dim,
            output_dim=2,
            hidden_layers=config.field_hidden_layers,
            hidden_units=config.field_hidden_units,
            activation="tanh",
        )

    def forward(self, xy: "torch.Tensor", directions: "torch.Tensor") -> "torch.Tensor":
        xy_scaled = xy / self.roi_half_width
        phase = self.k0 * torch.sum(xy * directions, dim=1, keepdim=True)
        features = torch.cat(
            [
                self.spatial_features(xy_scaled),
                directions,
                torch.sin(phase),
                torch.cos(phase),
            ],
            dim=1,
        )
        return self.mlp(features)


class EpsilonBranch(nn.Module):
    def __init__(self, config: TrainConfig, target: TargetSpec) -> None:
        super().__init__()
        self.eps_min = config.eps_min
        self.eps_max = config.eps_max
        self.eps_background = target.eps_background
        self.roi_half_width = target.roi_half_width
        self.spatial_features = FourierFeatureMap(
            input_dim=2,
            num_bands=config.fourier_bands,
            max_frequency=config.fourier_max_frequency,
            include_input=True,
        )
        self.mlp = MLP(
            input_dim=self.spatial_features.output_dim,
            output_dim=1,
            hidden_layers=config.eps_hidden_layers,
            hidden_units=config.eps_hidden_units,
            activation="silu",
        )
        p = (config.eps_initial - config.eps_min) / (config.eps_max - config.eps_min)
        p = min(max(p, 1e-5), 1.0 - 1e-5)
        final_layer = self.mlp.net[-1]
        if isinstance(final_layer, nn.Linear):
            nn.init.zeros_(final_layer.weight)
            nn.init.constant_(final_layer.bias, math.log(p / (1.0 - p)))

    def forward(self, xy: "torch.Tensor") -> "torch.Tensor":
        xy_scaled = xy / self.roi_half_width
        raw = self.mlp(self.spatial_features(xy_scaled))
        eps = self.eps_min + (self.eps_max - self.eps_min) * torch.sigmoid(raw)
        inside = torch.max(torch.abs(xy), dim=1, keepdim=True).values <= self.roi_half_width
        background = torch.full_like(eps, float(self.eps_background))
        return torch.where(inside, eps, background)

class DoubleBranchPINN(nn.Module):
    def __init__(self, config: TrainConfig, target: TargetSpec) -> None:
        super().__init__()
        self.field_branch = FieldBranch(config, target.roi_half_width)
        self.epsilon_branch = EpsilonBranch(config, target)

    def scattered_field(self, xy: "torch.Tensor", directions: "torch.Tensor") -> "torch.Tensor":
        return self.field_branch(xy, directions)

    def epsilon(self, xy: "torch.Tensor") -> "torch.Tensor":
        return self.epsilon_branch(xy)


def load_epsilon_branch_only(
    model: DoubleBranchPINN,
    checkpoint_path: str,
    device: "torch.device",
) -> None:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    checkpoint_model = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    epsilon_state = extract_epsilon_branch_state(checkpoint_model, checkpoint_path)
    model.epsilon_branch.load_state_dict(epsilon_state)
    print(
        f"epsilon branch loaded from {checkpoint_path} ({len(epsilon_state)} tensors)",
        flush=True,
    )
    print("field branch reinitialized", flush=True)


def extract_epsilon_branch_state(checkpoint_model: dict, checkpoint_path: str) -> dict:
    epsilon_state = {
        key.removeprefix("epsilon_branch."): value
        for key, value in checkpoint_model.items()
        if key.startswith("epsilon_branch.")
    }
    if not epsilon_state:
        raise ValueError(f"No epsilon_branch parameters found in checkpoint: {checkpoint_path}")
    return epsilon_state


def incident_field_torch(
    xy: "torch.Tensor",
    directions: "torch.Tensor",
    k0: float,
    amplitudes: "torch.Tensor | float",
    phase_sign: float,
) -> Tuple["torch.Tensor", "torch.Tensor"]:
    phase = phase_sign * k0 * torch.sum(xy * directions, dim=1, keepdim=True)
    cos_phase = torch.cos(phase)
    sin_phase = torch.sin(phase)
    if isinstance(amplitudes, (float, int)):
        amp_re = torch.full_like(cos_phase, float(amplitudes))
        amp_im = torch.zeros_like(cos_phase)
    else:
        amp_re = amplitudes[:, 0:1]
        amp_im = amplitudes[:, 1:2]
    inc_re = amp_re * cos_phase - amp_im * sin_phase
    inc_im = amp_re * sin_phase + amp_im * cos_phase
    return inc_re, inc_im


def laplacian_two_outputs(
    outputs: "torch.Tensor",
    xy: "torch.Tensor",
) -> Tuple["torch.Tensor", "torch.Tensor"]:
    laplacians = []
    for component in range(2):
        value = outputs[:, component : component + 1]
        grad = torch.autograd.grad(
            value.sum(), xy, create_graph=True, retain_graph=True
        )[0]
        second_terms = []
        for dim in range(2):
            grad_dim = grad[:, dim : dim + 1]
            second = torch.autograd.grad(
                grad_dim.sum(), xy, create_graph=True, retain_graph=True
            )[0][:, dim : dim + 1]
            second_terms.append(second)
        laplacians.append(second_terms[0] + second_terms[1])
    return laplacians[0], laplacians[1]


def pde_residual_loss(
    model: DoubleBranchPINN,
    xy: "torch.Tensor",
    directions: "torch.Tensor",
    amplitudes: "torch.Tensor",
    config: TrainConfig,
) -> "torch.Tensor":
    xy_req = xy.detach().clone().requires_grad_(True)
    directions = directions.detach()
    amplitudes = amplitudes.detach()
    scattered = model.scattered_field(xy_req, directions)
    lap_re, lap_im = laplacian_two_outputs(scattered, xy_req)
    eps = model.epsilon(xy_req)
    inc_re, inc_im = incident_field_torch(
        xy_req, directions, config.k0, amplitudes, config.incident_phase_sign
    )
    k2 = config.k0**2
    res_re = lap_re + k2 * eps * scattered[:, 0:1] + k2 * (eps - 1.0) * inc_re
    res_im = lap_im + k2 * eps * scattered[:, 1:2] + k2 * (eps - 1.0) * inc_im
    return torch.mean(res_re.square() + res_im.square())


def sommerfeld_boundary_loss(
    model: DoubleBranchPINN,
    xy: "torch.Tensor",
    directions: "torch.Tensor",
    config: TrainConfig,
) -> "torch.Tensor":
    xy_req = xy.detach().clone().requires_grad_(True)
    scattered = model.scattered_field(xy_req, directions.detach())
    radius = torch.linalg.norm(xy_req, dim=1, keepdim=True).clamp_min(1e-9)
    normal = xy_req / radius

    residuals = []
    for component in range(2):
        value = scattered[:, component : component + 1]
        grad = torch.autograd.grad(
            value.sum(), xy_req, create_graph=True, retain_graph=True
        )[0]
        radial_derivative = torch.sum(grad * normal, dim=1, keepdim=True)
        residuals.append(radial_derivative)

    # Outgoing scattered field: dE/dr - i*k*E ~= 0.
    res_re = residuals[0] + config.k0 * scattered[:, 1:2]
    res_im = residuals[1] - config.k0 * scattered[:, 0:1]
    return torch.mean(res_re.square() + res_im.square())


def robust_mse(diff: "torch.Tensor") -> "torch.Tensor":
    mag = torch.sqrt(torch.sum(diff.detach().square(), dim=1, keepdim=True) + 1e-12)
    scale = torch.quantile(mag.flatten(), 0.75).clamp_min(1e-6)
    weights = 1.0 / (1.0 + (mag / scale).square())
    weights = weights / weights.mean().clamp_min(1e-6)
    return torch.mean(weights * torch.sum(diff.square(), dim=1, keepdim=True))


def data_loss(
    model: DoubleBranchPINN,
    xy: "torch.Tensor",
    directions: "torch.Tensor",
    target: "torch.Tensor",
    robust: bool,
) -> "torch.Tensor":
    pred = model.scattered_field(xy, directions)
    diff = pred - target
    if robust:
        return robust_mse(diff)
    return torch.mean(torch.sum(diff.square(), dim=1, keepdim=True))


def paper_weighted_data_loss(
    model: DoubleBranchPINN,
    xy: "torch.Tensor",
    directions: "torch.Tensor",
    target: "torch.Tensor",
    gamma: float,
    delta: float,
) -> "torch.Tensor":
    pred = model.scattered_field(xy, directions)
    diff = pred - target
    residual_mag = torch.sqrt(torch.sum(diff.square(), dim=1, keepdim=True) + 1e-12)
    r_bar = torch.mean(residual_mag)
    weights = 1.0 / (1.0 + gamma * residual_mag / (r_bar + delta))
    return torch.mean(weights * residual_mag.square())


def total_variation_loss(
    model: DoubleBranchPINN,
    target: TargetSpec,
    n_grid: int,
    device: "torch.device",
    dtype: "torch.dtype",
) -> "torch.Tensor":
    if n_grid <= 1:
        return torch.zeros((), dtype=dtype, device=device)
    xs = torch.linspace(-target.roi_half_width, target.roi_half_width, n_grid, device=device, dtype=dtype)
    ys = torch.linspace(-target.roi_half_width, target.roi_half_width, n_grid, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    xy = torch.stack((xx.reshape(-1), yy.reshape(-1)), dim=1)
    eps = model.epsilon(xy).reshape(n_grid, n_grid)
    dx = eps[:, 1:] - eps[:, :-1]
    dy = eps[1:, :] - eps[:-1, :]
    beta = torch.as_tensor(1e-1, dtype=dtype, device=device)
    return torch.mean(torch.sqrt(dx.square() + beta.square())) + torch.mean(
        torch.sqrt(dy.square() + beta.square())
    )


def edge_preserving_loss(
    model: DoubleBranchPINN,
    target: TargetSpec,
    n_grid: int,
    device: "torch.device",
    dtype: "torch.dtype",
    edge_delta: float,
) -> "torch.Tensor":
    if n_grid <= 1:
        return torch.zeros((), dtype=dtype, device=device)
    xs = torch.linspace(-target.roi_half_width, target.roi_half_width, n_grid, device=device, dtype=dtype)
    ys = torch.linspace(-target.roi_half_width, target.roi_half_width, n_grid, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    xy = torch.stack((xx.reshape(-1), yy.reshape(-1)), dim=1).detach().clone().requires_grad_(True)
    eps = model.epsilon(xy)
    grad_eps = torch.autograd.grad(eps.sum(), xy, create_graph=True, retain_graph=True)[0]
    delta = torch.as_tensor(edge_delta, dtype=dtype, device=device)
    return torch.mean(torch.sqrt(grad_eps[:, 0:1].square() + grad_eps[:, 1:2].square() + delta.square()))


def contrast_l1_loss(
    model: DoubleBranchPINN,
    target: TargetSpec,
    n_points: int,
    device: "torch.device",
    dtype: "torch.dtype",
) -> "torch.Tensor":
    xy = (2.0 * torch.rand(n_points, 2, device=device, dtype=dtype) - 1.0) * target.roi_half_width
    eps = model.epsilon(xy)
    return torch.mean(torch.abs(eps - target.eps_background))


def binary_push_loss(
    model: DoubleBranchPINN,
    target: TargetSpec,
    n_points: int,
    device: "torch.device",
    dtype: "torch.dtype",
) -> "torch.Tensor":
    xy = (2.0 * torch.rand(n_points, 2, device=device, dtype=dtype) - 1.0) * target.roi_half_width
    eps = model.epsilon(xy)
    penalty = (eps - target.eps_background) * (target.eps_object - eps)
    return torch.mean(torch.clamp(penalty, min=0.0))


def epsilon_prior_loss(
    model: DoubleBranchPINN,
    epsilon_prior_branch: Optional[nn.Module],
    target: TargetSpec,
    n_points: int,
    device: "torch.device",
    dtype: "torch.dtype",
) -> "torch.Tensor":
    if epsilon_prior_branch is None:
        return torch.zeros((), dtype=dtype, device=device)
    xy = (2.0 * torch.rand(n_points, 2, device=device, dtype=dtype) - 1.0) * target.roi_half_width
    eps = model.epsilon(xy)
    with torch.no_grad():
        eps_prior = epsilon_prior_branch(xy)
    return torch.mean((eps - eps_prior).square())


def background_anchor_loss(
    model: DoubleBranchPINN,
    background_anchor_branch: Optional[nn.Module],
    target: TargetSpec,
    n_points: int,
    threshold: float,
    device: "torch.device",
    dtype: "torch.dtype",
) -> "torch.Tensor":
    if background_anchor_branch is None:
        return torch.zeros((), dtype=dtype, device=device)
    xy = (2.0 * torch.rand(n_points, 2, device=device, dtype=dtype) - 1.0) * target.roi_half_width
    with torch.no_grad():
        eps_anchor = background_anchor_branch(xy)
        mask = eps_anchor[:, 0] < threshold
    if not torch.any(mask):
        return torch.zeros((), dtype=dtype, device=device)
    eps = model.epsilon(xy[mask])
    return torch.mean((eps - target.eps_background).square())


def sample_points_in_circle(
    n_points: int,
    radius: float,
    device: "torch.device",
    dtype: "torch.dtype",
) -> "torch.Tensor":
    theta = 2.0 * math.pi * torch.rand(n_points, 1, device=device, dtype=dtype)
    r = radius * torch.sqrt(torch.rand(n_points, 1, device=device, dtype=dtype))
    return torch.cat((r * torch.cos(theta), r * torch.sin(theta)), dim=1)


def sample_collocation_points(
    n_points: int,
    target: TargetSpec,
    radius: float,
    device: "torch.device",
    dtype: "torch.dtype",
) -> "torch.Tensor":
    n_roi = n_points // 2
    n_outer = n_points - n_roi
    roi = (2.0 * torch.rand(n_roi, 2, device=device, dtype=dtype) - 1.0) * target.roi_half_width
    outer = sample_points_in_circle(n_outer, radius, device, dtype)
    return torch.cat((roi, outer), dim=0)


def sample_boundary_points(
    n_points: int,
    radius: float,
    device: "torch.device",
    dtype: "torch.dtype",
) -> "torch.Tensor":
    theta = 2.0 * math.pi * torch.rand(n_points, 1, device=device, dtype=dtype)
    return torch.cat((radius * torch.cos(theta), radius * torch.sin(theta)), dim=1)


def sample_directions(
    unique_directions: "torch.Tensor",
    n_points: int,
) -> "torch.Tensor":
    idx = torch.randint(0, unique_directions.shape[0], (n_points,), device=unique_directions.device)
    return unique_directions[idx]


def sample_direction_batch(
    unique_directions: "torch.Tensor",
    unique_amplitudes: "torch.Tensor",
    n_points: int,
) -> Tuple["torch.Tensor", "torch.Tensor"]:
    idx = torch.randint(0, unique_directions.shape[0], (n_points,), device=unique_directions.device)
    return unique_directions[idx], unique_amplitudes[idx]


def sample_observation_batch(
    obs: ObservationTensors,
    batch_per_direction: int,
) -> Tuple["torch.Tensor", "torch.Tensor", "torch.Tensor", "torch.Tensor", "torch.Tensor"]:
    indices = []
    for label, label_indices in obs.indices_by_label.items():
        n_take = min(batch_per_direction, label_indices.numel())
        perm = torch.randperm(label_indices.numel(), device=label_indices.device)[:n_take]
        indices.append(label_indices[perm])
    idx = torch.cat(indices, dim=0)
    return idx, obs.xy[idx], obs.directions[idx], obs.amplitudes[idx], obs.target[idx]


def make_integral_tensors(
    obs: ObservationSet,
    target: TargetSpec,
    config: TrainConfig,
    device: "torch.device",
    dtype: "torch.dtype",
) -> Optional[IntegralTensors]:
    if config.weight_integral_data <= 0.0:
        return None
    if config.integral_grid_size <= 1:
        raise ValueError("integral_grid_size must be greater than 1 when integral loss is enabled.")

    from scipy.special import hankel1

    n_grid = int(config.integral_grid_size)
    half = float(target.roi_half_width)
    dx = 2.0 * half / n_grid
    coords = np.linspace(-half + 0.5 * dx, half - 0.5 * dx, n_grid)
    xx, yy = np.meshgrid(coords, coords)
    quad_xy_np = np.column_stack((xx.reshape(-1), yy.reshape(-1))).astype(np.float64)

    delta = obs.xy[:, None, :] - quad_xy_np[None, :, :]
    distance = np.linalg.norm(delta, axis=2).clip(min=1e-9)
    green = 0.25j * hankel1(0, config.k0 * distance)
    quad_xy = torch.as_tensor(quad_xy_np, dtype=dtype, device=device)
    green_re = torch.as_tensor(green.real, dtype=dtype, device=device)
    green_im = torch.as_tensor(green.imag, dtype=dtype, device=device)
    return IntegralTensors(
        quad_xy=quad_xy,
        green_re=green_re,
        green_im=green_im,
        area_weight=float(dx * dx),
    )


def volume_integral_data_loss(
    model: DoubleBranchPINN,
    data_indices: "torch.Tensor",
    data_dirs: "torch.Tensor",
    data_target: "torch.Tensor",
    integral: IntegralTensors,
    obs_tensors: ObservationTensors,
    target: TargetSpec,
    config: TrainConfig,
) -> "torch.Tensor":
    eps = model.epsilon(integral.quad_xy)
    contrast = eps - float(target.eps_background)
    losses = []
    k2_area = (config.k0**2) * integral.area_weight

    for dir_idx in range(obs_tensors.unique_directions.shape[0]):
        direction = obs_tensors.unique_directions[dir_idx : dir_idx + 1]
        mask = torch.all(data_dirs == direction, dim=1)
        if not torch.any(mask):
            continue

        quad_dirs = direction.expand(integral.quad_xy.shape[0], 2)
        quad_amps = obs_tensors.unique_amplitudes[dir_idx : dir_idx + 1].expand(
            integral.quad_xy.shape[0], 2
        )
        scattered_quad = model.scattered_field(integral.quad_xy, quad_dirs)
        inc_re, inc_im = incident_field_torch(
            integral.quad_xy,
            quad_dirs,
            config.k0,
            quad_amps,
            config.incident_phase_sign,
        )
        total_re = inc_re + scattered_quad[:, 0:1]
        total_im = inc_im + scattered_quad[:, 1:2]
        source_re = (k2_area * contrast * total_re).squeeze(1)
        source_im = (k2_area * contrast * total_im).squeeze(1)

        green_re = integral.green_re[data_indices[mask]]
        green_im = integral.green_im[data_indices[mask]]
        pred_re = torch.matmul(green_re, source_re) - torch.matmul(green_im, source_im)
        pred_im = torch.matmul(green_re, source_im) + torch.matmul(green_im, source_re)
        pred = torch.stack((pred_re, pred_im), dim=1)
        diff = pred - data_target[mask]
        losses.append(torch.mean(torch.sum(diff.square(), dim=1, keepdim=True)))

    if not losses:
        return torch.zeros((), dtype=integral.quad_xy.dtype, device=integral.quad_xy.device)
    return torch.stack(losses).mean()


def target_mask(target: TargetSpec, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    if target.kind == "square":
        half = target.square_side / 2.0
        return (np.abs(x) <= half) & (np.abs(y) <= half)

    if target.kind == "circle":
        return (
            (x - target.circle_center_x) ** 2
            + (y - target.circle_center_y) ** 2
            <= target.circle_radius**2
        )
    if target.kind == "austria":
        left_circle = (
            (x + target.circle_center_offset_x) ** 2
            + (y - target.circle_center_y) ** 2
            <= target.circle_radius**2
        )
        right_circle = (
            (x - target.circle_center_offset_x) ** 2
            + (y - target.circle_center_y) ** 2
            <= target.circle_radius**2
        )
        ring_r = np.sqrt(x**2 + (y - target.ring_center_y) ** 2)
        ring = (ring_r >= target.ring_inner_radius) & (ring_r <= target.ring_outer_radius)
        return left_circle | right_circle | ring
    raise ValueError(f"Unsupported target kind: {target.kind}")


def true_epsilon_grid(target: TargetSpec, grid_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    xs = np.linspace(-target.roi_half_width, target.roi_half_width, grid_size)
    ys = np.linspace(-target.roi_half_width, target.roi_half_width, grid_size)
    xx, yy = np.meshgrid(xs, ys)
    eps = np.full_like(xx, target.eps_background, dtype=np.float64)
    eps[target_mask(target, xx, yy)] = target.eps_object
    return xx, yy, eps


def reconstruct_epsilon(
    model: DoubleBranchPINN,
    target: TargetSpec,
    grid_size: int,
    device: "torch.device",
    dtype: "torch.dtype",
    batch_size: int = 8192,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    xs = np.linspace(-target.roi_half_width, target.roi_half_width, grid_size)
    ys = np.linspace(-target.roi_half_width, target.roi_half_width, grid_size)
    xx, yy = np.meshgrid(xs, ys)
    xy_np = np.column_stack((xx.reshape(-1), yy.reshape(-1)))
    values = []
    model.eval()
    with torch.no_grad():
        for start in range(0, xy_np.shape[0], batch_size):
            batch = torch.as_tensor(xy_np[start : start + batch_size], dtype=dtype, device=device)
            values.append(model.epsilon(batch).detach().cpu().numpy())
    eps = np.vstack(values).reshape(grid_size, grid_size)
    return xx, yy, eps


def relative_error(pred: np.ndarray, truth: np.ndarray) -> float:
    denom = np.linalg.norm(truth.ravel())
    if denom < 1e-12:
        return float("nan")
    return float(np.linalg.norm((pred - truth).ravel()) / denom)


def global_ssim(pred: np.ndarray, truth: np.ndarray, data_range: float) -> float:
    pred = pred.astype(np.float64)
    truth = truth.astype(np.float64)
    mu_x = pred.mean()
    mu_y = truth.mean()
    var_x = pred.var()
    var_y = truth.var()
    cov_xy = ((pred - mu_x) * (truth - mu_y)).mean()
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    numerator = (2 * mu_x * mu_y + c1) * (2 * cov_xy + c2)
    denominator = (mu_x**2 + mu_y**2 + c1) * (var_x + var_y + c2)
    return float(numerator / denominator)


def threshold_epsilon_map(
    eps: np.ndarray,
    target: TargetSpec,
    threshold: Optional[float] = None,
) -> np.ndarray:
    """Binarize epsilon with the midpoint between background and object permittivity.

    The square-target ground truth is binary: eps_background outside the target and
    eps_object inside it.  When no explicit threshold is supplied, the midpoint
    is the neutral cutoff between those two values.
    """
    cutoff = 0.5 * (target.eps_background + target.eps_object) if threshold is None else threshold
    return np.where(eps >= cutoff, target.eps_object, target.eps_background).astype(np.float64)


def epsilon_metrics(pred: np.ndarray, truth: np.ndarray, target: TargetSpec) -> Dict[str, float]:
    data_range = target.eps_object - target.eps_background
    thresholded = threshold_epsilon_map(pred, target)
    return {
        "rel_error_continuous": relative_error(pred, truth),
        "ssim_continuous": global_ssim(pred, truth, data_range),
        "rel_error_thresholded": relative_error(thresholded, truth),
        "ssim_thresholded": global_ssim(thresholded, truth, data_range),
    }


def evaluate_epsilon_reconstruction(
    model: DoubleBranchPINN,
    target: TargetSpec,
    config: TrainConfig,
    device: "torch.device",
    dtype: "torch.dtype",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, float]]:
    was_training = model.training
    x, y, pred = reconstruct_epsilon(model, target, config.plot_grid_size, device, dtype)
    if was_training:
        model.train()
    _, _, truth = true_epsilon_grid(target, config.plot_grid_size)
    thresholded = threshold_epsilon_map(pred, target)
    return x, y, pred, truth, thresholded, epsilon_metrics(pred, truth, target)


EVALUATION_FIELDNAMES = [
    "epoch",
    "total_loss",
    "data_loss",
    "pde_loss",
    "boundary_loss",
    "tv_loss",
    "lf_loss",
    "ldw_loss",
    "lep_loss",
    "rel_error_continuous",
    "ssim_continuous",
    "rel_error_thresholded",
    "ssim_thresholded",
    "checkpoint_path",
]


def make_evaluation_row(
    *,
    epoch: int,
    loss_items: Dict[str, float],
    metrics: Dict[str, float],
    checkpoint_path: str = "",
) -> Dict[str, object]:
    return {
        "epoch": int(epoch),
        "total_loss": float(loss_items.get("total", float("nan"))),
        "data_loss": float(loss_items.get("data", float("nan"))),
        "pde_loss": float(loss_items.get("pde", float("nan"))),
        "boundary_loss": float(loss_items.get("boundary", float("nan"))),
        "tv_loss": float(loss_items.get("tv", float("nan"))),
        "lf_loss": float(loss_items.get("lf", float("nan"))),
        "ldw_loss": float(loss_items.get("ldw", float("nan"))),
        "lep_loss": float(loss_items.get("lep", float("nan"))),
        "rel_error_continuous": float(metrics["rel_error_continuous"]),
        "ssim_continuous": float(metrics["ssim_continuous"]),
        "rel_error_thresholded": float(metrics["rel_error_thresholded"]),
        "ssim_thresholded": float(metrics["ssim_thresholded"]),
        "checkpoint_path": checkpoint_path,
    }


def configure_matplotlib() -> None:
    import matplotlib as mpl

    mpl.use("Agg", force=True)
    mpl.rcParams.update(
        {
            "font.family": "Times New Roman",
            "font.size": 18,
            "axes.labelsize": 20,
            "xtick.labelsize": 16,
            "ytick.labelsize": 16,
            "axes.linewidth": 1.0,
            "figure.dpi": 160,
            "savefig.dpi": 220,
        }
    )


def plot_epsilon_image(
    x: np.ndarray,
    y: np.ndarray,
    eps: np.ndarray,
    truth: np.ndarray,
    target: TargetSpec,
    out_path: Path,
    title: str,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
) -> None:
    import matplotlib.pyplot as plt

    configure_matplotlib()
    vmin = target.eps_background if vmin is None else vmin
    vmax = target.eps_object if vmax is None else vmax
    fig, ax = plt.subplots(figsize=(5.92, 5.0))
    im = ax.imshow(
        eps,
        extent=[x.min(), x.max(), y.min(), y.max()],
        origin="lower",
        cmap="jet",
        vmin=vmin,
        vmax=vmax,
        interpolation="bilinear",
    )
    level = 0.5 * (target.eps_background + target.eps_object)
    ax.contour(x, y, truth, levels=[level], colors="k", linewidths=2.0)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-target.roi_half_width, target.roi_half_width)
    ax.set_ylim(-target.roi_half_width, target.roi_half_width)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(title)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("Relative Permittivity")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_comparison(
    x: np.ndarray,
    y: np.ndarray,
    truth: np.ndarray,
    pred: np.ndarray,
    target: TargetSpec,
    out_path: Path,
    title: str,
) -> None:
    import matplotlib.pyplot as plt

    configure_matplotlib()
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.8), constrained_layout=True)
    level = 0.5 * (target.eps_background + target.eps_object)
    for ax, data, label in zip(axes, [truth, pred], ["True", "Double-branch PINN"]):
        im = ax.imshow(
            data,
            extent=[x.min(), x.max(), y.min(), y.max()],
            origin="lower",
            cmap="jet",
            vmin=target.eps_background,
            vmax=target.eps_object,
            interpolation="bilinear",
        )
        ax.contour(x, y, truth, levels=[level], colors="k", linewidths=1.8)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(-target.roi_half_width, target.roi_half_width)
        ax.set_ylim(-target.roi_half_width, target.roi_half_width)
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_title(label, fontsize=17, pad=6)
    fig.suptitle(title, fontsize=18, y=1.01)
    cb = fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.046, pad=0.03)
    cb.set_label("Relative Permittivity")
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_loss(history: List[Dict[str, float]], out_path: Path) -> None:
    if not history:
        return
    import matplotlib.pyplot as plt

    configure_matplotlib()
    steps = np.asarray([row["step"] for row in history], dtype=np.float64)
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    for key in ["total", "data", "integral_data", "pde", "boundary", "tv"]:
        if key not in history[0]:
            continue
        vals = np.asarray([row[key] for row in history], dtype=np.float64)
        ax.semilogy(steps, vals, linewidth=2, label=key)
    ax.set_xlabel("Iterations")
    ax.set_ylabel("Loss")
    ax.grid(False)
    ax.legend(loc="best", frameon=True)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def save_history_csv(history: List[Dict[str, object]], out_path: Path) -> None:
    if not history:
        return
    keys = list(history[0].keys())
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(history)


def save_evaluation_csv(rows: List[Dict[str, object]], out_path: Path) -> None:
    if not rows:
        return
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=EVALUATION_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def json_ready(value):
    if isinstance(value, complex):
        return {"real": value.real, "imag": value.imag}
    if isinstance(value, dict):
        return {k: json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    return value


def save_torch_file(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        torch.save(payload, f)


def compute_loss_terms(
    model: DoubleBranchPINN,
    *,
    data_xy: "torch.Tensor",
    data_dirs: "torch.Tensor",
    data_target: "torch.Tensor",
    data_indices: "torch.Tensor",
    pde_xy: "torch.Tensor",
    pde_dirs: "torch.Tensor",
    pde_amps: "torch.Tensor",
    bc_xy: "torch.Tensor",
    bc_dirs: "torch.Tensor",
    integral_tensors: Optional[IntegralTensors],
    obs_tensors: ObservationTensors,
    target: TargetSpec,
    config: TrainConfig,
    device: "torch.device",
    dtype: "torch.dtype",
    epsilon_prior_branch: Optional[nn.Module] = None,
    background_anchor_branch: Optional[nn.Module] = None,
) -> Dict[str, "torch.Tensor"]:
    loss_data = data_loss(model, data_xy, data_dirs, data_target, robust=config.robust_data_weighting)
    loss_pde = pde_residual_loss(model, pde_xy, pde_dirs, pde_amps, config)
    loss_boundary = sommerfeld_boundary_loss(model, bc_xy, bc_dirs, config)
    if integral_tensors is None:
        loss_integral = torch.zeros((), dtype=dtype, device=device)
    else:
        loss_integral = volume_integral_data_loss(
            model, data_indices, data_dirs, data_target, integral_tensors, obs_tensors, target, config
        )
    loss_tv = total_variation_loss(model, target, config.n_tv_grid, device, dtype)
    loss_l1 = contrast_l1_loss(model, target, 1024, device, dtype)
    loss_binary_push = binary_push_loss(model, target, 1024, device, dtype)
    loss_epsilon_prior = epsilon_prior_loss(
        model, epsilon_prior_branch, target, 1024, device, dtype
    )
    loss_background_anchor = background_anchor_loss(
        model,
        background_anchor_branch,
        target,
        1024,
        config.background_anchor_threshold,
        device,
        dtype,
    )
    loss_lep = edge_preserving_loss(model, target, config.n_tv_grid, device, dtype, config.edge_delta)
    loss_lf = loss_pde + loss_boundary
    if config.loss_preset == "paper":
        loss_ldw = paper_weighted_data_loss(
            model,
            data_xy,
            data_dirs,
            data_target,
            gamma=config.adaptive_gamma,
            delta=config.adaptive_delta,
        )
        total = config.lambda_f * loss_lf + config.lambda_d * loss_ldw + config.lambda_ep * loss_lep
    else:
        loss_ldw = torch.zeros((), dtype=dtype, device=device)
        total = (
            config.weight_data * loss_data
            + config.weight_pde * loss_pde
            + config.weight_boundary * loss_boundary
            + config.weight_integral_data * loss_integral
            + config.weight_tv * loss_tv
            + config.weight_contrast_l1 * loss_l1
            + config.binary_push_weight * loss_binary_push
            + config.epsilon_prior_weight * loss_epsilon_prior
            + config.weight_edge_preserving * loss_lep
            + config.background_anchor_weight * loss_background_anchor
        )
    return {
        "total": total,
        "data": loss_data,
        "integral_data": loss_integral,
        "pde": loss_pde,
        "boundary": loss_boundary,
        "tv": loss_tv,
        "contrast_l1": loss_l1,
        "binary_push": loss_binary_push,
        "epsilon_prior": loss_epsilon_prior,
        "background_anchor": loss_background_anchor,
        "lf": loss_lf,
        "ldw": loss_ldw,
        "lep": loss_lep,
    }


def train_double_branch_pinn(
    data_dir: Path,
    target: TargetSpec,
    config: TrainConfig,
    output_dir: Path,
    direction_labels: Optional[Sequence[str]] = None,
) -> Dict[str, float]:
    obs = load_observations(data_dir, config, direction_labels=direction_labels)
    return train_double_branch_pinn_from_observations(
        obs=obs,
        target=target,
        config=config,
        output_dir=output_dir,
    )


def train_double_branch_pinn_from_observations(
    obs: ObservationSet,
    target: TargetSpec,
    config: TrainConfig,
    output_dir: Path,
) -> Dict[str, float]:
    require_torch()
    set_global_seed(config.random_seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(config.device)
    dtype = resolve_dtype(config.dtype)

    obs_tensors = to_observation_tensors(obs, device=device, dtype=dtype)
    integral_tensors = make_integral_tensors(obs, target, config, device=device, dtype=dtype)
    radius = config.domain_radius or obs.receiver_radius * 0.98

    model = DoubleBranchPINN(config, target).to(device=device, dtype=dtype)
    resume_step = 0
    if config.resume_checkpoint and config.resume_epsilon_from:
        raise ValueError("--resume-checkpoint and --resume-epsilon-from cannot be used together.")
    if config.resume_checkpoint:
        checkpoint = torch.load(config.resume_checkpoint, map_location=device, weights_only=False)
        checkpoint_model = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
        model.load_state_dict(checkpoint_model)
        if isinstance(checkpoint, dict):
            resume_step = int(checkpoint.get("step", 0) or 0)
        print(f"Loaded checkpoint: {config.resume_checkpoint}", flush=True)
    elif config.resume_epsilon_from:
        load_epsilon_branch_only(model, config.resume_epsilon_from, device)
    epsilon_prior_branch: Optional[nn.Module] = None
    if config.epsilon_prior_weight > 0.0:
        epsilon_prior_branch = copy.deepcopy(model.epsilon_branch).to(device=device, dtype=dtype)
        epsilon_prior_branch.eval()
        for param in epsilon_prior_branch.parameters():
            param.requires_grad_(False)
        print("epsilon prior initialized from current epsilon branch", flush=True)
    background_anchor_branch: Optional[nn.Module] = None
    if config.background_anchor_weight > 0.0:
        if not config.background_anchor_from:
            raise ValueError("--background-anchor-from is required when --background-anchor-weight > 0.")
        anchor_checkpoint = torch.load(config.background_anchor_from, map_location=device, weights_only=False)
        anchor_model = (
            anchor_checkpoint["model"]
            if isinstance(anchor_checkpoint, dict) and "model" in anchor_checkpoint
            else anchor_checkpoint
        )
        background_anchor_branch = copy.deepcopy(model.epsilon_branch).to(device=device, dtype=dtype)
        background_anchor_branch.load_state_dict(
            extract_epsilon_branch_state(anchor_model, config.background_anchor_from)
        )
        background_anchor_branch.eval()
        for param in background_anchor_branch.parameters():
            param.requires_grad_(False)
        print(
            f"background anchor loaded from {config.background_anchor_from} "
            f"(threshold={config.background_anchor_threshold:g})",
            flush=True,
        )
    metadata = {
        "target": asdict(target),
        "config": asdict(config),
        "k0": config.k0,
        "wavelength": config.wavelength,
        "receiver_radius": obs.receiver_radius,
        "direction_labels": obs.direction_labels,
        "incident_amplitudes": json_ready(obs.incident_amplitudes),
    }
    with (output_dir / "run_config.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    history: List[Dict[str, object]] = []
    evaluation_history: List[Dict[str, object]] = []
    last_loss_items: Dict[str, float] = {}
    start_time = time.time()
    model.train()

    # ==================================================
    # Stage 1: Adam optimization.
    # ==================================================
    if config.epochs_adam <= 0:
        print("=== Stage 1: Adam skipped ===", flush=True)
    else:
        print("=== Stage 1: Adam optimization starts ===", flush=True)
    use_split_lrs = (
        config.field_lr is not None
        or config.epsilon_lr is not None
        or config.freeze_epsilon_steps > 0
    )
    if use_split_lrs:
        field_base_lr = config.field_lr if config.field_lr is not None else config.learning_rate
        epsilon_base_lr = config.epsilon_lr if config.epsilon_lr is not None else config.learning_rate
        optimizer_adam = torch.optim.Adam(
            [
                {"params": model.field_branch.parameters(), "lr": field_base_lr, "name": "field"},
                {"params": model.epsilon_branch.parameters(), "lr": epsilon_base_lr, "name": "epsilon"},
            ]
        )
        scheduler = None
        print(
            f"Using split Adam learning rates: field_lr={field_base_lr:g}, "
            f"epsilon_lr={epsilon_base_lr:g}, freeze_epsilon_steps={config.freeze_epsilon_steps}",
            flush=True,
        )
    else:
        optimizer_adam = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer_adam, T_max=max(config.epochs_adam, 1), eta_min=config.learning_rate * 0.05
        )

    def cosine_lr(base_lr: float, index: int, total_steps: int) -> float:
        progress = min(max(index, 0), max(total_steps - 1, 1)) / max(total_steps - 1, 1)
        factor = 0.05 + 0.5 * (1.0 - 0.05) * (1.0 + math.cos(math.pi * progress))
        return base_lr * factor

    for step in range(1, config.epochs_adam + 1):
        if use_split_lrs:
            field_base_lr = config.field_lr if config.field_lr is not None else config.learning_rate
            epsilon_base_lr = config.epsilon_lr if config.epsilon_lr is not None else config.learning_rate
            remaining_eps_steps = max(config.epochs_adam - config.freeze_epsilon_steps, 1)
            for group in optimizer_adam.param_groups:
                if group.get("name") == "field":
                    group["lr"] = cosine_lr(field_base_lr, step - 1, config.epochs_adam)
                elif step <= config.freeze_epsilon_steps:
                    group["lr"] = 0.0
                else:
                    group["lr"] = cosine_lr(
                        epsilon_base_lr,
                        step - config.freeze_epsilon_steps - 1,
                        remaining_eps_steps,
                    )
        optimizer_adam.zero_grad(set_to_none=True)

        # Sample points and compute the same weighted loss terms used by Adam.
        data_indices, data_xy, data_dirs, _data_amps, data_target = sample_observation_batch(
            obs_tensors, config.data_batch_per_direction
        )
        pde_xy = sample_collocation_points(config.n_pde, target, radius, device, dtype)
        pde_dirs, pde_amps = sample_direction_batch(
            obs_tensors.unique_directions, obs_tensors.unique_amplitudes, config.n_pde
        )
        bc_xy = sample_boundary_points(config.n_boundary, obs.receiver_radius, device, dtype)
        bc_dirs = sample_directions(obs_tensors.unique_directions, config.n_boundary)

        losses = compute_loss_terms(
            model,
            data_xy=data_xy,
            data_dirs=data_dirs,
            data_target=data_target,
            data_indices=data_indices,
            pde_xy=pde_xy,
            pde_dirs=pde_dirs,
            pde_amps=pde_amps,
            bc_xy=bc_xy,
            bc_dirs=bc_dirs,
            integral_tensors=integral_tensors,
            obs_tensors=obs_tensors,
            target=target,
            config=config,
            device=device,
            dtype=dtype,
            epsilon_prior_branch=epsilon_prior_branch,
            background_anchor_branch=background_anchor_branch,
        )
        total = losses["total"]

        total.backward()
        if config.gradient_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_norm)
        optimizer_adam.step()
        if scheduler is not None:
            scheduler.step()

        loss_items = {
            "total": float(total.detach().cpu()),
            "data": float(losses["data"].detach().cpu()),
            "integral_data": float(losses["integral_data"].detach().cpu()),
            "pde": float(losses["pde"].detach().cpu()),
            "boundary": float(losses["boundary"].detach().cpu()),
            "tv": float(losses["tv"].detach().cpu()),
            "contrast_l1": float(losses["contrast_l1"].detach().cpu()),
            "binary_push": float(losses["binary_push"].detach().cpu()),
            "epsilon_prior": float(losses["epsilon_prior"].detach().cpu()),
            "background_anchor": float(losses["background_anchor"].detach().cpu()),
            "lf": float(losses["lf"].detach().cpu()),
            "ldw": float(losses["ldw"].detach().cpu()),
            "lep": float(losses["lep"].detach().cpu()),
        }
        last_loss_items = loss_items

        checkpoint_path = ""
        global_step = resume_step + step
        checkpoint_due = config.checkpoint_every > 0 and global_step % config.checkpoint_every == 0
        if checkpoint_due:
            checkpoint_file = output_dir / f"checkpoint_adam_{global_step:06d}.pt"
            save_torch_file(
                {"model": model.state_dict(), "config": asdict(config), "target": asdict(target), "step": global_step},
                checkpoint_file,
            )
            checkpoint_path = str(checkpoint_file)

        # Log metrics and save checkpoints on the existing schedule.
        should_log = step == 1 or step % config.log_every == 0
        if should_log or checkpoint_due:
            _, _, _, _, _, metric_items = evaluate_epsilon_reconstruction(model, target, config, device, dtype)
            evaluation_history.append(
                make_evaluation_row(
                    epoch=global_step,
                    loss_items=loss_items,
                    metrics=metric_items,
                    checkpoint_path=checkpoint_path,
                )
            )
            row = {
                "step": float(global_step),
                **loss_items,
                "rel_error_continuous": metric_items["rel_error_continuous"],
                "ssim_continuous": metric_items["ssim_continuous"],
                "rel_error_thresholded": metric_items["rel_error_thresholded"],
                "ssim_thresholded": metric_items["ssim_thresholded"],
                "elapsed_s": float(time.time() - start_time),
            }
            history.append(row)
            if should_log:
                if config.loss_preset == "paper":
                    print(
                        "Adam step={step:6.0f} total={total:.4e} lf={lf:.4e} "
                        "ldw={ldw:.4e} lep={lep:.4e} "
                        "rel={rel_error_continuous:.4f} rel_thr={rel_error_thresholded:.4f}".format(**row),
                        flush=True,
                    )
                else:
                    print(
                        "Adam step={step:6.0f} total={total:.4e} data={data:.4e} "
                        "int={integral_data:.4e} pde={pde:.4e} "
                        "bc={boundary:.4e} tv={tv:.4e} "
                        "rel={rel_error_continuous:.4f} rel_thr={rel_error_thresholded:.4f}".format(**row),
                        flush=True,
                    )
    lbfgs_steps = int(config.lbfgs_steps if config.lbfgs_steps > 0 else config.epochs_lbfgs)
    if lbfgs_steps <= 0:
        print("\n=== Stage 2: L-BFGS disabled; keeping Adam result ===", flush=True)
    else:
        # Stage 2: optional L-BFGS refinement after Adam.
        print("\n=== Stage 2: L-BFGS refinement starts ===", flush=True)
        optimizer_lbfgs_refine = torch.optim.LBFGS(
            model.parameters(),
            lr=config.lbfgs_lr,
            max_iter=config.lbfgs_max_iter,
            history_size=config.lbfgs_history_size,
            line_search_fn="strong_wolfe",
            tolerance_grad=1e-12,
            tolerance_change=1e-12,
        )
        current_loss_items: Dict[str, float] = {}

        for lbfgs_step in range(1, lbfgs_steps + 1):
            data_indices, data_xy, data_dirs, _data_amps, data_target = sample_observation_batch(
                obs_tensors, config.data_batch_per_direction
            )
            pde_xy = sample_collocation_points(config.n_pde, target, radius, device, dtype)
            pde_dirs, pde_amps = sample_direction_batch(
                obs_tensors.unique_directions, obs_tensors.unique_amplitudes, config.n_pde
            )
            bc_xy = sample_boundary_points(config.n_boundary, obs.receiver_radius, device, dtype)
            bc_dirs = sample_directions(obs_tensors.unique_directions, config.n_boundary)

            def lbfgs_refine_closure():
                optimizer_lbfgs_refine.zero_grad(set_to_none=True)
                losses = compute_loss_terms(
                    model,
                    data_xy=data_xy,
                    data_dirs=data_dirs,
                    data_target=data_target,
                    data_indices=data_indices,
                    pde_xy=pde_xy,
                    pde_dirs=pde_dirs,
                    pde_amps=pde_amps,
                    bc_xy=bc_xy,
                    bc_dirs=bc_dirs,
                    integral_tensors=integral_tensors,
                    obs_tensors=obs_tensors,
                    target=target,
                    config=config,
                    device=device,
                    dtype=dtype,
                    epsilon_prior_branch=epsilon_prior_branch,
                    background_anchor_branch=background_anchor_branch,
                )
                total = losses["total"]
                total.backward()
                current_loss_items.update(
                    {
                        "data": float(losses["data"].detach().cpu()),
                        "integral_data": float(losses["integral_data"].detach().cpu()),
                        "pde": float(losses["pde"].detach().cpu()),
                        "boundary": float(losses["boundary"].detach().cpu()),
                        "tv": float(losses["tv"].detach().cpu()),
                        "contrast_l1": float(losses["contrast_l1"].detach().cpu()),
                        "binary_push": float(losses["binary_push"].detach().cpu()),
                        "epsilon_prior": float(losses["epsilon_prior"].detach().cpu()),
                        "background_anchor": float(losses["background_anchor"].detach().cpu()),
                        "lf": float(losses["lf"].detach().cpu()),
                        "ldw": float(losses["ldw"].detach().cpu()),
                        "lep": float(losses["lep"].detach().cpu()),
                    }
                )
                return total

            global_step = resume_step + config.epochs_adam + lbfgs_step
            loss_val = optimizer_lbfgs_refine.step(lbfgs_refine_closure)
            loss_items = {
                "total": float(loss_val.detach().cpu()),
                **current_loss_items,
            }
            last_loss_items = loss_items
            checkpoint_path = ""
            checkpoint_due = config.checkpoint_every > 0 and global_step % config.checkpoint_every == 0
            if checkpoint_due:
                checkpoint_file = output_dir / f"checkpoint_lbfgs_{global_step:06d}.pt"
                save_torch_file(
                    {"model": model.state_dict(), "config": asdict(config), "target": asdict(target), "step": global_step},
                    checkpoint_file,
                )
                checkpoint_path = str(checkpoint_file)

            should_log = lbfgs_step == 1 or lbfgs_step % max(config.log_every, 1) == 0
            if should_log or checkpoint_due:
                _, _, _, _, _, metric_items = evaluate_epsilon_reconstruction(model, target, config, device, dtype)
                evaluation_history.append(
                    make_evaluation_row(
                        epoch=global_step,
                        loss_items=loss_items,
                        metrics=metric_items,
                        checkpoint_path=checkpoint_path,
                    )
                )
                row = {
                    "step": float(global_step),
                    **loss_items,
                    "rel_error_continuous": metric_items["rel_error_continuous"],
                    "ssim_continuous": metric_items["ssim_continuous"],
                    "rel_error_thresholded": metric_items["rel_error_thresholded"],
                    "ssim_thresholded": metric_items["ssim_thresholded"],
                    "elapsed_s": float(time.time() - start_time),
                }
                history.append(row)
                if config.loss_preset == "paper":
                    print(
                        "LBFGS step={step:6.0f} total={total:.4e} lf={lf:.4e} "
                        "ldw={ldw:.4e} lep={lep:.4e} "
                        "rel={rel_error_continuous:.4f} rel_thr={rel_error_thresholded:.4f}".format(**row),
                        flush=True,
                    )
                else:
                    print(
                        "LBFGS step={step:6.0f} total={total:.4e} data={data:.4e} "
                        "int={integral_data:.4e} pde={pde:.4e} "
                        "bc={boundary:.4e} tv={tv:.4e} "
                        "rel={rel_error_continuous:.4f} rel_thr={rel_error_thresholded:.4f}".format(**row),
                        flush=True,
                    )

        lbfgs_checkpoint = output_dir / "model_lbfgs_final.pt"
        save_torch_file(
            {
                "model": model.state_dict(),
                "config": asdict(config),
                "target": asdict(target),
                "step": resume_step + config.epochs_adam + lbfgs_steps,
            },
            lbfgs_checkpoint,
        )
        _, _, pred_lbfgs, _, _, lbfgs_metric_items = evaluate_epsilon_reconstruction(
            model, target, config, device, dtype
        )
        np.save(output_dir / "epsilon_reconstruction_lbfgs.npy", pred_lbfgs)
        lbfgs_metrics = {
            "relative_error": lbfgs_metric_items["rel_error_continuous"],
            "ssim": lbfgs_metric_items["ssim_continuous"],
            **lbfgs_metric_items,
            "elapsed_s": float(time.time() - start_time),
            "noise_level": config.noise_level,
        }
        with (output_dir / "metrics_lbfgs.json").open("w", encoding="utf-8") as f:
            json.dump(lbfgs_metrics, f, indent=2, ensure_ascii=False)

    final_checkpoint = output_dir / "model_final.pt"
    final_step = resume_step + config.epochs_adam + lbfgs_steps
    save_torch_file(
        {
            "model": model.state_dict(),
            "config": asdict(config),
            "target": asdict(target),
            "step": final_step,
        },
        final_checkpoint,
    )

    x, y, pred, truth, thresholded, metric_items = evaluate_epsilon_reconstruction(
        model, target, config, device, dtype
    )
    if not evaluation_history or int(evaluation_history[-1]["epoch"]) != final_step:
        evaluation_history.append(
            make_evaluation_row(
                epoch=final_step,
                loss_items=last_loss_items,
                metrics=metric_items,
                checkpoint_path=str(final_checkpoint),
            )
        )
    elif evaluation_history[-1].get("checkpoint_path"):
        evaluation_history.append(
            make_evaluation_row(
                epoch=final_step,
                loss_items=last_loss_items,
                metrics=metric_items,
                checkpoint_path=str(final_checkpoint),
            )
        )
    else:
        evaluation_history[-1]["checkpoint_path"] = str(final_checkpoint)

    save_history_csv(history, output_dir / "loss_history.csv")
    save_evaluation_csv(evaluation_history, output_dir / "evaluation_metrics.csv")

    np.save(output_dir / "epsilon_reconstruction.npy", pred)
    np.save(output_dir / "epsilon_thresholded.npy", thresholded)
    np.save(output_dir / "epsilon_truth.npy", truth)
    np.save(output_dir / "x_grid.npy", x)
    np.save(output_dir / "y_grid.npy", y)

    metrics = {
        "relative_error": metric_items["rel_error_continuous"],
        "ssim": metric_items["ssim_continuous"],
        **metric_items,
        "elapsed_s": float(time.time() - start_time),
        "noise_level": config.noise_level,
    }
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    plot_epsilon_image(
        x,
        y,
        truth,
        truth,
        target,
        output_dir / "true_epsilon.png",
        title="True",
    )
    plot_epsilon_image(
        x,
        y,
        pred,
        truth,
        target,
        output_dir / "pinn_reconstruction.png",
        title="Double-branch PINN",
    )
    plot_comparison(
        x,
        y,
        truth,
        pred,
        target,
        output_dir / "comparison.png",
        title=f"{config.frequency_hz / 1e9:.1f} GHz",
    )
    plot_loss(history, output_dir / "loss_curve.png")
    return metrics


def train_noise_sweep(
    data_dir: Path,
    target: TargetSpec,
    base_config: TrainConfig,
    output_root: Path,
    noise_levels: Sequence[float],
    direction_labels: Optional[Sequence[str]] = None,
) -> Dict[str, Dict[str, float]]:
    all_metrics: Dict[str, Dict[str, float]] = {}
    for noise in noise_levels:
        cfg = TrainConfig(**asdict(base_config))
        cfg.noise_level = float(noise)
        suffix = f"noise_{int(round(100.0 * noise)):02d}pct"
        out_dir = output_root / suffix
        metrics = train_double_branch_pinn(
            data_dir=data_dir,
            target=target,
            config=cfg,
            output_dir=out_dir,
            direction_labels=direction_labels,
        )
        all_metrics[suffix] = metrics
    output_root.mkdir(parents=True, exist_ok=True)
    with (output_root / "noise_sweep_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2, ensure_ascii=False)
    return all_metrics
