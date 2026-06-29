from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CORE_ROOT = ROOT / "square_target"
if str(CORE_ROOT) not in sys.path:
    sys.path.insert(0, str(CORE_ROOT))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pinn_pixel_inverse_core import (  # noqa: E402
    TargetSpec,
    TrainConfig,
    load_fem_table,
    parse_direction_from_name,
    train_double_branch_pinn,
)


def build_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--epochs", type=int, default=30000, help="Adam iterations.")
    parser.add_argument("--device", default="auto", help="'auto', 'cpu', 'cuda', or 'cuda:0'.")
    parser.add_argument("--dtype", default="float32", choices=["float32", "float64"])
    parser.add_argument("--lr", type=float, default=8.0e-4, help="Adam learning rate.")
    parser.add_argument("--incident-amplitude", type=float, default=0.1)
    parser.add_argument("--phase-sign", type=float, default=1.0)
    parser.add_argument(
        "--observation-imag-sign",
        type=float,
        default=-1.0,
        choices=[1.0, -1.0],
        help="Use +1 for Ez=Re+iIm, or -1 for Ez=Re-iIm.",
    )
    parser.add_argument("--noise-level", type=float, default=0.0)
    parser.add_argument("--eps-initial", type=float, default=1.35)
    parser.add_argument("--max-points-per-direction", type=int, default=836)
    parser.add_argument("--data-batch-per-direction", type=int, default=256)
    parser.add_argument("--n-pde", type=int, default=1280)
    parser.add_argument("--n-boundary", type=int, default=320)
    parser.add_argument("--n-tv-grid", type=int, default=40)
    parser.add_argument("--integral-grid-size", type=int, default=36)
    parser.add_argument("--plot-grid-size", type=int, default=220)
    parser.add_argument("--log-every", type=int, default=200)
    parser.add_argument("--checkpoint-every", type=int, default=0)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--no-robust-data", action="store_true")
    parser.add_argument("--diagnose-imag-sign", action="store_true")
    parser.add_argument("--weight-data", type=float, default=120.0)
    parser.add_argument("--weight-pde", type=float, default=0.02)
    parser.add_argument("--weight-boundary", type=float, default=0.02)
    parser.add_argument("--weight-integral-data", type=float, default=500.0)
    parser.add_argument("--weight-tv", type=float, default=0.006)
    parser.add_argument("--weight-contrast-l1", type=float, default=0.0)
    parser.add_argument("--field-hidden-layers", type=int, default=5)
    parser.add_argument("--field-hidden-units", type=int, default=88)
    parser.add_argument("--eps-hidden-layers", type=int, default=5)
    parser.add_argument("--eps-hidden-units", type=int, default=96)
    parser.add_argument("--fourier-bands", type=int, default=5)
    parser.add_argument("--fourier-max-frequency", type=float, default=8.0)
    return parser


def build_target(noise_level: float) -> TargetSpec:
    return TargetSpec(
        name=f"Austria target, {int(round(100.0 * noise_level))}% noise",
        kind="austria",
        eps_background=1.0,
        eps_object=3.0,
        roi_half_width=0.5,
        circle_radius=0.10,
        circle_center_y=0.30,
        circle_center_offset_x=0.15,
        ring_center_y=-0.10,
        ring_inner_radius=0.175,
        ring_outer_radius=0.275,
    )


def diagnose_imaginary_sign(data_dir: Path, frequency_hz: float, phase_sign: float) -> None:
    k0 = 2.0 * math.pi * frequency_hz / 3.0e8
    for sign in (1.0, -1.0):
        print(f"\nObservation convention: Ez = Re {'+' if sign > 0 else '-'} i Im")
        for file_name in ("+x.txt", "-x.txt", "+y.txt", "-y.txt"):
            path = data_dir / file_name
            label, direction = parse_direction_from_name(path)
            xy, field = load_fem_table(path, imag_sign=sign)
            direction_arr = np.asarray(direction, dtype=np.float64)
            flipped_arr = -direction_arr

            def fit(dvec: np.ndarray) -> tuple[float, complex]:
                phase = np.exp(1j * phase_sign * k0 * (xy @ dvec))
                amp = np.vdot(phase, field) / np.vdot(phase, phase)
                res = np.linalg.norm(field - amp * phase) / np.linalg.norm(field)
                return float(res), complex(amp)

            expected_res, expected_amp = fit(direction_arr)
            flipped_res, flipped_amp = fit(flipped_arr)
            print(
                f"  {label:>2s}: n={xy.shape[0]:5d}, "
                f"expected_res={expected_res:.4f}, "
                f"amp={expected_amp.real:+.5f}{expected_amp.imag:+.5f}j; "
                f"flipped_res={flipped_res:.4f}, "
                f"amp={flipped_amp.real:+.5f}{flipped_amp.imag:+.5f}j"
            )


def run_optimized_austria(
    *,
    default_data_dir: Path,
    default_output_dir: str,
    description: str,
) -> None:
    args = build_parser(description).parse_args()
    data_dir = Path(args.data_dir).resolve() if args.data_dir else default_data_dir.resolve()
    output_dir = default_data_dir / (args.output_dir or default_output_dir)

    if not data_dir.exists():
        raise FileNotFoundError(
            f"FEM data directory does not exist: {data_dir}. "
            "Expected +x.txt, -x.txt, +y.txt and -y.txt in that directory."
        )

    frequency_hz = 0.3e9
    if args.diagnose_imag_sign:
        diagnose_imaginary_sign(data_dir, frequency_hz, args.phase_sign)
        return

    target = build_target(args.noise_level)
    config = TrainConfig(
        frequency_hz=frequency_hz,
        incident_amplitude=args.incident_amplitude,
        incident_phase_sign=args.phase_sign,
        observation_imag_sign=args.observation_imag_sign,
        estimate_incident_amplitude=False,
        eps_min=1.0,
        eps_max=4.0,
        eps_initial=args.eps_initial,
        epochs_adam=args.epochs,
        learning_rate=args.lr,
        device=args.device,
        dtype=args.dtype,
        max_points_per_direction=args.max_points_per_direction,
        data_batch_per_direction=args.data_batch_per_direction,
        n_pde=args.n_pde,
        n_boundary=args.n_boundary,
        n_tv_grid=args.n_tv_grid,
        integral_grid_size=args.integral_grid_size,
        plot_grid_size=args.plot_grid_size,
        log_every=args.log_every,
        checkpoint_every=args.checkpoint_every,
        robust_data_weighting=not args.no_robust_data,
        weight_data=args.weight_data,
        weight_pde=args.weight_pde,
        weight_boundary=args.weight_boundary,
        weight_integral_data=args.weight_integral_data,
        weight_tv=args.weight_tv,
        weight_contrast_l1=args.weight_contrast_l1,
        field_hidden_layers=args.field_hidden_layers,
        field_hidden_units=args.field_hidden_units,
        eps_hidden_layers=args.eps_hidden_layers,
        eps_hidden_units=args.eps_hidden_units,
        fourier_bands=args.fourier_bands,
        fourier_max_frequency=args.fourier_max_frequency,
        random_seed=20260430,
        noise_level=args.noise_level,
    )

    metrics = train_double_branch_pinn(
        data_dir=data_dir,
        target=target,
        config=config,
        output_dir=output_dir,
        direction_labels=("+x", "-x", "+y", "-y"),
    )
    print(json.dumps({"target": asdict(target), "metrics": metrics}, indent=2, ensure_ascii=False))
