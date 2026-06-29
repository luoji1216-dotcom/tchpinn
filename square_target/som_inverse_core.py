from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
from scipy.linalg import solve
from scipy.ndimage import gaussian_filter
from scipy.special import hankel1
from scipy.sparse.linalg import LinearOperator, lsmr

from pinn_pixel_inverse_core import (
    ObservationSet,
    TargetSpec,
    TrainConfig,
    configure_matplotlib,
    global_ssim,
    incident_field_numpy,
    plot_epsilon_image,
    relative_error,
    true_epsilon_grid,
)


@dataclass
class SomConfig:
    frequency_hz: float = 0.3e9
    c0: float = 3.0e8
    observation_imag_sign: float = -1.0
    incident_phase_sign: float = 1.0
    grid_size: int = 56
    ridge: float = 2.0e-3
    som_iters: int = 0
    truncated_modes: int = 0
    eps_min: float = 1.0
    eps_max: float = 4.0
    contrast_gain: float = 1.0
    smooth_sigma: float = 0.0
    max_lsmr_iters: int = 800
    lsmr_tol: float = 1.0e-5
    internal_solver_grid_limit: int = 44
    output_prefix: str = "som"
    green_convention: str = "h1_i4"

    @property
    def wavelength(self) -> float:
        return self.c0 / self.frequency_hz

    @property
    def k0(self) -> float:
        return 2.0 * math.pi / self.wavelength


def build_som_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--grid-size", type=int, default=56)
    parser.add_argument("--ridge", type=float, default=2.0e-3)
    parser.add_argument("--som-iters", type=int, default=0)
    parser.add_argument("--truncated-modes", type=int, default=0)
    parser.add_argument("--max-lsmr-iters", type=int, default=800)
    parser.add_argument("--lsmr-tol", type=float, default=1.0e-5)
    parser.add_argument("--observation-imag-sign", type=float, default=-1.0, choices=[1.0, -1.0])
    parser.add_argument("--phase-sign", type=float, default=1.0)
    parser.add_argument("--eps-max", type=float, default=4.0)
    parser.add_argument(
        "--contrast-gain",
        type=float,
        default=1.0,
        help=(
            "Optional multiplicative gain applied to the solved contrast before clipping. "
            "Use this only as a convention/calibration knob for FEM data."
        ),
    )
    parser.add_argument(
        "--smooth-sigma",
        type=float,
        default=0.0,
        help="Optional Gaussian smoothing sigma, in pixels, applied to the final permittivity map.",
    )
    parser.add_argument(
        "--green-convention",
        choices=["h1_i4", "h1_minus_i4", "h2_i4", "h2_minus_i4"],
        default="h1_i4",
        help=(
            "2-D Green-function convention. Synthetic FEM exports with Ez=Re-iIm "
            "often image better with h2_minus_i4."
        ),
    )
    parser.add_argument("--output-dir", default=None)
    return parser


def make_roi_grid(target: TargetSpec, grid_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    half = float(target.roi_half_width)
    dx = 2.0 * half / grid_size
    coords = np.linspace(-half + 0.5 * dx, half - 0.5 * dx, grid_size)
    xx, yy = np.meshgrid(coords, coords)
    xy = np.column_stack((xx.reshape(-1), yy.reshape(-1))).astype(np.float64)
    return xx, yy, xy, float(dx * dx)


def outgoing_green(
    receiver_xy: np.ndarray,
    source_xy: np.ndarray,
    k0: float,
    convention: str = "h1_i4",
) -> np.ndarray:
    distance = np.linalg.norm(receiver_xy[:, None, :] - source_xy[None, :, :], axis=2)
    distance = np.maximum(distance, 1.0e-9)
    kr = k0 * distance
    if convention == "h1_i4":
        return 0.25j * hankel1(0, kr)
    if convention == "h1_minus_i4":
        return -0.25j * hankel1(0, kr)
    if convention == "h2_i4":
        return 0.25j * np.conjugate(hankel1(0, kr))
    if convention == "h2_minus_i4":
        return -0.25j * np.conjugate(hankel1(0, kr))
    raise ValueError(f"Unsupported Green-function convention: {convention}")


def incident_matrix(
    cell_xy: np.ndarray,
    directions: Sequence[Tuple[float, float]],
    amplitudes: Sequence[complex],
    config: SomConfig,
) -> np.ndarray:
    fields = []
    for direction, amplitude in zip(directions, amplitudes):
        fields.append(
            incident_field_numpy(
                cell_xy,
                direction,
                config.k0,
                amplitude,
                config.incident_phase_sign,
            )
        )
    return np.column_stack(fields)


def unique_observation_directions(obs: ObservationSet) -> Tuple[np.ndarray, np.ndarray, Dict[str, int]]:
    dirs = []
    amps = []
    label_to_col: Dict[str, int] = {}
    for col, label in enumerate(obs.direction_labels):
        idx = np.flatnonzero(obs.labels == label)[0]
        dirs.append(obs.directions[idx])
        amps.append(obs.incident_amplitudes[label])
        label_to_col[label] = col
    return np.asarray(dirs, dtype=np.float64), np.asarray(amps, dtype=np.complex128), label_to_col


def build_observation_operator(
    obs: ObservationSet,
    cell_xy: np.ndarray,
    config: SomConfig,
    incident_cells: np.ndarray,
    total_cells: Optional[np.ndarray] = None,
) -> np.ndarray:
    green = outgoing_green(obs.xy, cell_xy, config.k0, config.green_convention)
    _, _, label_to_col = unique_observation_directions(obs)
    label_cols = np.asarray([label_to_col[str(label)] for label in obs.labels], dtype=np.int64)
    if total_cells is None:
        source_field = incident_cells[:, label_cols].T
    else:
        source_field = total_cells[:, label_cols].T
    return (config.k0**2) * green * source_field


def solve_real_contrast(
    operator: np.ndarray,
    data: np.ndarray,
    area_weight: float,
    target: TargetSpec,
    config: SomConfig,
) -> np.ndarray:
    a = area_weight * operator
    y = data.astype(np.complex128)
    n_unknowns = a.shape[1]
    ridge_sqrt = math.sqrt(max(config.ridge, 0.0))

    if config.truncated_modes > 0:
        real_a = np.vstack((a.real, a.imag))
        real_y = np.concatenate((y.real, y.imag))
        u, s, vt = np.linalg.svd(real_a, full_matrices=False)
        modes = min(config.truncated_modes, s.size)
        coeff = (u[:, :modes].T @ real_y) / np.maximum(s[:modes], 1.0e-12)
        contrast = vt[:modes, :].T @ coeff
    else:
        def matvec(x: np.ndarray) -> np.ndarray:
            z = a @ x
            if ridge_sqrt > 0:
                return np.concatenate((z.real, z.imag, ridge_sqrt * x))
            return np.concatenate((z.real, z.imag))

        def rmatvec(v: np.ndarray) -> np.ndarray:
            data_len = a.shape[0]
            complex_part = v[:data_len] + 1j * v[data_len : 2 * data_len]
            out = np.real(np.conjugate(a).T @ complex_part)
            if ridge_sqrt > 0:
                out = out + ridge_sqrt * v[2 * data_len :]
            return out

        m_rows = 2 * a.shape[0] + (n_unknowns if ridge_sqrt > 0 else 0)
        rhs = np.concatenate((y.real, y.imag, np.zeros(n_unknowns) if ridge_sqrt > 0 else []))
        linear_operator = LinearOperator(
            (m_rows, n_unknowns),
            matvec=matvec,
            rmatvec=rmatvec,
            dtype=np.float64,
        )
        result = lsmr(
            linear_operator,
            rhs,
            atol=config.lsmr_tol,
            btol=config.lsmr_tol,
            maxiter=config.max_lsmr_iters,
        )
        contrast = result[0]

    contrast = np.nan_to_num(contrast, nan=0.0, posinf=0.0, neginf=0.0)
    return np.clip(contrast, config.eps_min - target.eps_background, config.eps_max - target.eps_background)


def internal_total_fields(
    cell_xy: np.ndarray,
    contrast: np.ndarray,
    incident_cells: np.ndarray,
    area_weight: float,
    config: SomConfig,
) -> np.ndarray:
    grid_limit = config.internal_solver_grid_limit**2
    if cell_xy.shape[0] > grid_limit:
        return incident_cells
    green = outgoing_green(cell_xy, cell_xy, config.k0, config.green_convention)
    np.fill_diagonal(green, 0.0 + 0.0j)
    system = np.eye(cell_xy.shape[0], dtype=np.complex128) - (
        config.k0**2 * area_weight * green * contrast[None, :]
    )
    fields = []
    for col in range(incident_cells.shape[1]):
        fields.append(solve(system, incident_cells[:, col], assume_a="gen"))
    return np.column_stack(fields)


def run_som_inversion(
    obs: ObservationSet,
    target: TargetSpec,
    config: SomConfig,
    output_dir: Path,
) -> Dict[str, float]:
    output_dir.mkdir(parents=True, exist_ok=True)
    start = time.time()
    xx, yy, cell_xy, area_weight = make_roi_grid(target, config.grid_size)
    directions, amplitudes, _ = unique_observation_directions(obs)
    incident_cells = incident_matrix(
        cell_xy,
        [tuple(d) for d in directions],
        list(amplitudes),
        config,
    )
    total_cells: Optional[np.ndarray] = None
    contrast = np.zeros(cell_xy.shape[0], dtype=np.float64)

    # 迭代增益修正：只有纯Born近似（0次迭代）使用contrast_gain，迭代模式下增益为1
    gain = config.contrast_gain if config.som_iters == 0 else 1.0

    for iteration in range(config.som_iters + 1):
        # 构建观测算子：首次用入射场（Born近似），后续用更新后的内部总场
        operator = build_observation_operator(obs, cell_xy, config, incident_cells, total_cells)

        # 求解对比度
        contrast = solve_real_contrast(operator, obs.scattered, area_weight, target, config)

        # 应用增益 + 物理范围裁剪
        contrast = np.clip(
            contrast * gain,
            config.eps_min - target.eps_background,
            config.eps_max - target.eps_background,
        )

        # 调试打印：确认迭代是否生效（运行没问题后可以注释掉）
        print(f"迭代 {iteration}: 对比度最大值 = {np.max(contrast):.4f}, 范数 = {np.linalg.norm(contrast):.4f}")

        # 非最后一次迭代：更新内部总场，供下一次构建算子使用
        if iteration < config.som_iters:
            total_cells = internal_total_fields(cell_xy, contrast, incident_cells, area_weight, config)

    eps = target.eps_background + contrast.reshape(config.grid_size, config.grid_size)
    # total_cells: Optional[np.ndarray] = None
    # contrast = np.zeros(cell_xy.shape[0], dtype=np.float64)
    # for iteration in range(max(config.som_iters, 0) + 1):
    #     operator = build_observation_operator(obs, cell_xy, config, incident_cells, total_cells)
    #     contrast = solve_real_contrast(operator, obs.scattered, area_weight, target, config)
    #     contrast = np.clip(
    #         contrast * config.contrast_gain,
    #         config.eps_min - target.eps_background,
    #         config.eps_max - target.eps_background,
    #     )
    #     if iteration < config.som_iters:
    #         total_cells = internal_total_fields(cell_xy, contrast, incident_cells, area_weight, config)
    #
    # eps = target.eps_background + contrast.reshape(config.grid_size, config.grid_size)
    if config.smooth_sigma > 0.0:
        eps = gaussian_filter(eps, sigma=config.smooth_sigma)
        eps = np.clip(eps, config.eps_min, config.eps_max)
    _, _, truth = true_epsilon_grid(target, config.grid_size)
    np.save(output_dir / f"{config.output_prefix}_epsilon_reconstruction.npy", eps)
    np.save(output_dir / f"{config.output_prefix}_epsilon_truth.npy", truth)
    np.save(output_dir / f"{config.output_prefix}_x_grid.npy", xx)
    np.save(output_dir / f"{config.output_prefix}_y_grid.npy", yy)

    metrics = {
        "relative_error": relative_error(eps, truth),
        "ssim": global_ssim(eps, truth, target.eps_object - target.eps_background),
        "elapsed_s": float(time.time() - start),
        "grid_size": config.grid_size,
        "som_iters": config.som_iters,
        "contrast_gain": config.contrast_gain,
        "smooth_sigma": config.smooth_sigma,
    }
    (output_dir / f"{config.output_prefix}_metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (output_dir / f"{config.output_prefix}_run_config.json").write_text(
        json.dumps({"target": asdict(target), "config": asdict(config)}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    plot_epsilon_image(
        xx,
        yy,
        eps,
        truth,
        target,
        output_dir / f"{config.output_prefix}_reconstruction.png",
        title="SOM",
        vmin=target.eps_background,
        vmax=target.eps_object,
    )
    plot_som_comparison(
        xx,
        yy,
        truth,
        eps,
        target,
        output_dir / f"{config.output_prefix}_comparison.png",
        title=f"SOM, {config.frequency_hz / 1e9:.1f} GHz",
    )
    return metrics


def plot_som_comparison(
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
    for ax, data, label in zip(axes, [truth, pred], ["True", "SOM"]):
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


def som_config_from_args(args: argparse.Namespace, frequency_hz: float, eps_max: float) -> SomConfig:
    return SomConfig(
        frequency_hz=frequency_hz,
        observation_imag_sign=args.observation_imag_sign,
        incident_phase_sign=args.phase_sign,
        grid_size=args.grid_size,
        ridge=args.ridge,
        som_iters=args.som_iters,
        truncated_modes=args.truncated_modes,
        eps_max=args.eps_max if hasattr(args, "eps_max") else eps_max,
        contrast_gain=args.contrast_gain,
        smooth_sigma=args.smooth_sigma,
        max_lsmr_iters=args.max_lsmr_iters,
        lsmr_tol=args.lsmr_tol,
        green_convention=args.green_convention,
    )
