from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
from scipy.special import hankel1

ROOT = Path(__file__).resolve().parents[1]
CORE_ROOT = ROOT / "square_target"
if str(CORE_ROOT) not in sys.path:
    sys.path.insert(0, str(CORE_ROOT))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from austria_optimized_common import build_target  # noqa: E402
from pinn_pixel_inverse_core import (  # noqa: E402
    TrainConfig,
    incident_field_numpy,
    load_fem_table,
    parse_direction_from_name,
    target_mask,
    train_double_branch_pinn,
)


HERE = Path(__file__).resolve().parent
DIRECTIONS = ("+x", "-x", "+y", "-y")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate code-operator-consistent 1GHz Austria synthetic data and run "
            "the current direct PINN inversion flow on it."
        )
    )
    parser.add_argument("--source-data-dir", default=str(HERE / "data_1GHz"))
    parser.add_argument("--synthetic-data-dir", default=str(HERE / "data_1GHz_synthetic_self_consistency"))
    parser.add_argument("--output-dir", default=str(HERE / "results_5_3_3_austria_1GHz_synthetic_self_consistency"))
    parser.add_argument("--forward-grid-size", type=int, default=36)
    parser.add_argument("--epochs", type=int, default=3000)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="float32", choices=["float32", "float64"])
    parser.add_argument("--lr", type=float, default=4.0e-4)
    parser.add_argument("--weight-data", type=float, default=0.0)
    parser.add_argument("--weight-pde", type=float, default=0.006)
    parser.add_argument("--weight-boundary", type=float, default=0.02)
    parser.add_argument("--weight-integral-data", type=float, default=50.0)
    parser.add_argument("--weight-tv", type=float, default=0.001)
    parser.add_argument("--weight-contrast-l1", type=float, default=8.0e-2)
    parser.add_argument("--integral-internal-field-mode", choices=["coupled", "detach_field", "incident_only"], default="detach_field")
    parser.add_argument("--max-points-per-direction", type=int, default=836)
    parser.add_argument("--data-batch-per-direction", type=int, default=256)
    parser.add_argument("--n-pde", type=int, default=1280)
    parser.add_argument("--n-boundary", type=int, default=320)
    parser.add_argument("--integral-grid-size", type=int, default=36)
    parser.add_argument("--plot-grid-size", type=int, default=220)
    parser.add_argument("--log-every", type=int, default=200)
    parser.add_argument("--checkpoint-every", type=int, default=0)
    parser.add_argument("--eps-initial", type=float, default=1.5)
    parser.add_argument("--eps-max", type=float, default=4.0)
    parser.add_argument("--incident-amplitude", type=float, default=0.1)
    parser.add_argument("--skip-generate", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    return parser


def make_forward_grid(target, n_grid: int) -> tuple[np.ndarray, float, np.ndarray]:
    half = float(target.roi_half_width)
    dx = 2.0 * half / n_grid
    coords = np.linspace(-half + 0.5 * dx, half - 0.5 * dx, n_grid)
    xx, yy = np.meshgrid(coords, coords)
    xy = np.column_stack((xx.reshape(-1), yy.reshape(-1))).astype(np.float64)
    eps = np.full(xy.shape[0], target.eps_background, dtype=np.float64)
    eps[target_mask(target, xy[:, 0], xy[:, 1])] = target.eps_object
    return xy, float(dx * dx), eps


def green_matrix(dst_xy: np.ndarray, src_xy: np.ndarray, k0: float) -> np.ndarray:
    delta = dst_xy[:, None, :] - src_xy[None, :, :]
    distance = np.linalg.norm(delta, axis=2).clip(min=1.0e-9)
    return 0.25j * hankel1(0, k0 * distance)


def solve_total_field(
    *,
    cell_xy: np.ndarray,
    area_weight: float,
    eps: np.ndarray,
    direction: tuple[float, float],
    k0: float,
    amplitude: complex,
) -> np.ndarray:
    incident = incident_field_numpy(cell_xy, direction, k0, amplitude, phase_sign=1.0)
    contrast = eps - 1.0
    operator = green_matrix(cell_xy, cell_xy, k0) * ((k0**2) * area_weight * contrast[None, :])
    system = np.eye(cell_xy.shape[0], dtype=np.complex128) - operator
    return np.linalg.solve(system, incident)


def synthetic_scattered(
    *,
    obs_xy: np.ndarray,
    cell_xy: np.ndarray,
    area_weight: float,
    eps: np.ndarray,
    total_cells: np.ndarray,
    k0: float,
) -> np.ndarray:
    contrast = eps - 1.0
    source = (k0**2) * area_weight * contrast * total_cells
    return green_matrix(obs_xy, cell_xy, k0) @ source


def write_total_field_table(path: Path, xy: np.ndarray, total: np.ndarray) -> None:
    table = np.zeros((xy.shape[0], 9), dtype=np.float64)
    table[:, 0:2] = xy
    table[:, 7] = total.real
    table[:, 8] = -total.imag
    with path.open("w", encoding="utf-8") as f:
        f.write("% synthetic total field generated by diagnose_1ghz_synthetic_self_consistency.py\n")
        f.write("% columns: x y z unused unused unused unused real(Ez) imag_storage\n")
        np.savetxt(f, table, fmt="%.12e")


def generate_synthetic_data(args: argparse.Namespace, target, config: TrainConfig) -> dict[str, object]:
    source_dir = Path(args.source_data_dir)
    out_dir = Path(args.synthetic_data_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cell_xy, area_weight, eps = make_forward_grid(target, args.forward_grid_size)
    rows = []

    for label in DIRECTIONS:
        path = source_dir / f"{label}.txt"
        parsed, direction = parse_direction_from_name(path)
        obs_xy, _ = load_fem_table(path, imag_sign=config.observation_imag_sign)
        if config.max_points_per_direction > 0 and obs_xy.shape[0] > config.max_points_per_direction:
            keep = np.linspace(0, obs_xy.shape[0] - 1, config.max_points_per_direction).round().astype(int)
            obs_xy = obs_xy[keep]
        total_cells = solve_total_field(
            cell_xy=cell_xy,
            area_weight=area_weight,
            eps=eps,
            direction=direction,
            k0=config.k0,
            amplitude=complex(args.incident_amplitude, 0.0),
        )
        scattered = synthetic_scattered(
            obs_xy=obs_xy,
            cell_xy=cell_xy,
            area_weight=area_weight,
            eps=eps,
            total_cells=total_cells,
            k0=config.k0,
        )
        incident = incident_field_numpy(
            obs_xy, direction, config.k0, complex(args.incident_amplitude, 0.0), phase_sign=1.0
        )
        total_obs = incident + scattered
        write_total_field_table(out_dir / f"{label}.txt", obs_xy, total_obs)
        rows.append(
            {
                "direction": label,
                "parsed": parsed,
                "n_obs": int(obs_xy.shape[0]),
                "mean_abs_incident": float(np.mean(np.abs(incident))),
                "mean_abs_scattered": float(np.mean(np.abs(scattered))),
                "mean_abs_total": float(np.mean(np.abs(total_obs))),
                "cell_total_rel_residual": float(
                    np.linalg.norm(
                        total_cells
                        - incident_field_numpy(
                            cell_xy,
                            direction,
                            config.k0,
                            complex(args.incident_amplitude, 0.0),
                            phase_sign=1.0,
                        )
                        - synthetic_scattered(
                            obs_xy=cell_xy,
                            cell_xy=cell_xy,
                            area_weight=area_weight,
                            eps=eps,
                            total_cells=total_cells,
                            k0=config.k0,
                        )
                    )
                    / np.linalg.norm(total_cells)
                ),
            }
        )

    manifest = {
        "frequency_hz": config.frequency_hz,
        "k0": config.k0,
        "incident_amplitude": args.incident_amplitude,
        "incident_phase_sign": 1.0,
        "observation_imag_sign_for_loader": -1.0,
        "forward_grid_size": args.forward_grid_size,
        "area_weight": area_weight,
        "target": asdict(target),
        "directions": rows,
    }
    with (out_dir / "synthetic_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    return manifest


def train_on_synthetic(args: argparse.Namespace, target, config: TrainConfig) -> dict[str, float]:
    return train_double_branch_pinn(
        data_dir=Path(args.synthetic_data_dir),
        target=target,
        config=config,
        output_dir=Path(args.output_dir),
        direction_labels=DIRECTIONS,
    )


def main() -> None:
    args = build_parser().parse_args()
    target = build_target(0.0)
    config = TrainConfig(
        frequency_hz=1.0e9,
        incident_amplitude=args.incident_amplitude,
        incident_phase_sign=1.0,
        observation_imag_sign=-1.0,
        estimate_incident_amplitude=False,
        eps_min=1.0,
        eps_max=args.eps_max,
        eps_initial=args.eps_initial,
        epochs_adam=args.epochs,
        learning_rate=args.lr,
        device=args.device,
        dtype=args.dtype,
        max_points_per_direction=args.max_points_per_direction,
        data_batch_per_direction=args.data_batch_per_direction,
        n_pde=args.n_pde,
        n_boundary=args.n_boundary,
        integral_grid_size=args.integral_grid_size,
        plot_grid_size=args.plot_grid_size,
        log_every=args.log_every,
        checkpoint_every=args.checkpoint_every,
        robust_data_weighting=True,
        weight_data=args.weight_data,
        weight_pde=args.weight_pde,
        weight_boundary=args.weight_boundary,
        weight_integral_data=args.weight_integral_data,
        weight_tv=args.weight_tv,
        weight_contrast_l1=args.weight_contrast_l1,
        integral_internal_field_mode=args.integral_internal_field_mode,
        random_seed=20260430,
        noise_level=0.0,
    )

    print(f"frequency_hz={config.frequency_hz}")
    print(f"k0={config.k0}")
    print(f"source_data_dir={Path(args.source_data_dir).resolve()}")
    print(f"synthetic_data_dir={Path(args.synthetic_data_dir).resolve()}")
    print(f"output_dir={Path(args.output_dir).resolve()}")
    print("synthetic_forward=discrete Lippmann-Schwinger solve with G=+i/4 H0 and source=+k0^2(eps-1)E")
    print("training_flow=train_double_branch_pinn, direct integral settings")

    manifest = None
    if not args.skip_generate:
        manifest = generate_synthetic_data(args, target, config)
        print(json.dumps({"synthetic_manifest": manifest}, indent=2, ensure_ascii=False))
    if args.skip_train:
        return
    metrics = train_on_synthetic(args, target, config)
    print(json.dumps({"target": asdict(target), "config": asdict(config), "metrics": metrics}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
