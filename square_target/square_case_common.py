from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
ROOT = Path(__file__).resolve().parents[1]
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pinn_pixel_inverse_core import (
    TargetSpec,
    TrainConfig,
    load_fem_table,
    parse_direction_from_name,
    train_double_branch_pinn,
)


def build_square_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Compatibility alias for --epochs-adam.",
    )
    parser.add_argument("--epochs-adam", type=int, default=30000, help="Adam iterations.")
    parser.add_argument(
        "--epochs-lbfgs",
        type=int,
        default=0,
        help="Compatibility alias for --lbfgs-steps.",
    )
    parser.add_argument(
        "--lbfgs-steps",
        type=int,
        default=None,
        help="Optional L-BFGS refinement steps after Adam. Use 0 to keep the Adam-only result.",
    )
    parser.add_argument("--lbfgs-lr", type=float, default=1.0, help="L-BFGS learning rate.")
    parser.add_argument("--lbfgs-max-iter", type=int, default=20, help="Internal iterations per L-BFGS step.")
    parser.add_argument("--lbfgs-history-size", type=int, default=50, help="L-BFGS history size.")
    parser.add_argument("--device", default="auto", help="'auto', 'cpu', or 'cuda'.")
    parser.add_argument("--dtype", default="float32", choices=["float32", "float64"])
    parser.add_argument("--lr", type=float, default=1.0e-3, help="Adam learning rate.")
    parser.add_argument("--incident-amplitude", type=float, default=0.1)
    parser.add_argument("--phase-sign", type=float, default=1.0)
    parser.add_argument("--eps-initial", type=float, default=1.5)
    parser.add_argument("--eps-max", type=float, default=5.0)
    parser.add_argument("--domain-radius", type=float, default=None)
    parser.add_argument(
        "--observation-imag-sign",
        type=float,
        default=-1.0,
        choices=[1.0, -1.0],
        help="Use +1 for Ez=Re+iIm, or -1 for Ez=Re-iIm.",
    )
    parser.add_argument(
        "--estimate-incident-amplitude",
        action="store_true",
        help="Estimate one complex incident amplitude per direction.",
    )
    parser.add_argument(
        "--diagnose-imag-sign",
        action="store_true",
        help="Fit plane waves with both imaginary-part signs and exit.",
    )
    parser.add_argument("--max-points-per-direction", type=int, default=1200)
    parser.add_argument("--data-batch-per-direction", type=int, default=256)
    parser.add_argument("--n-pde", type=int, default=1024)
    parser.add_argument("--n-boundary", type=int, default=256)
    parser.add_argument("--n-tv-grid", type=int, default=32)
    parser.add_argument("--integral-grid-size", type=int, default=32)
    parser.add_argument("--plot-grid-size", type=int, default=220)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--checkpoint-every", type=int, default=2000)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--data-dir", default=None)
    parser.add_argument(
        "--resume-checkpoint",
        default=None,
        help="Load a saved checkpoint/model before training or L-BFGS refinement.",
    )
    parser.add_argument("--no-robust-data", action="store_true")
    parser.add_argument("--loss-preset", choices=["current", "paper"], default="current")
    parser.add_argument("--lambda-f", type=float, default=1.0)
    parser.add_argument("--lambda-d", type=float, default=100.0)
    parser.add_argument("--lambda-ep", type=float, default=100.0)
    parser.add_argument("--adaptive-gamma", type=float, default=1.0)
    parser.add_argument("--adaptive-delta", type=float, default=1.0e-8)
    parser.add_argument("--edge-delta", type=float, default=1.0e-3)
    parser.add_argument("--weight-data", type=float, default=100.0)
    parser.add_argument("--weight-pde", type=float, default=0.02)
    parser.add_argument("--weight-boundary", type=float, default=0.02)
    parser.add_argument("--weight-integral-data", type=float, default=500.0)
    parser.add_argument("--weight-tv", type=float, default=0.01)
    parser.add_argument("--weight-contrast-l1", type=float, default=0.0)
    parser.add_argument("--field-hidden-layers", type=int, default=5)
    parser.add_argument("--field-hidden-units", type=int, default=80)
    parser.add_argument("--eps-hidden-layers", type=int, default=5)
    parser.add_argument("--eps-hidden-units", type=int, default=96)
    parser.add_argument("--fourier-bands", type=int, default=5)
    parser.add_argument("--fourier-max-frequency", type=float, default=8.0)
    return parser


def _fit_plane_wave(
    xy: np.ndarray,
    field: np.ndarray,
    direction: tuple[float, float],
    k0: float,
    phase_sign: float,
) -> tuple[float, complex]:
    direction_arr = np.asarray(direction, dtype=np.float64)
    phase = np.exp(1j * phase_sign * k0 * (xy @ direction_arr))
    amplitude = np.vdot(phase, field) / np.vdot(phase, phase)
    residual = np.linalg.norm(field - amplitude * phase) / np.linalg.norm(field)
    return float(residual), complex(amplitude)


def diagnose_imaginary_sign(data_dir: Path, frequency_hz: float, phase_sign: float) -> None:
    k0 = 2.0 * math.pi * frequency_hz / 3.0e8
    for sign in (1.0, -1.0):
        print(f"\nObservation convention: Ez = Re {'+' if sign > 0 else '-'} i Im")
        for file_name in ("+x.txt", "-x.txt", "+y.txt", "-y.txt"):
            path = data_dir / file_name
            if not path.exists():
                continue
            label, direction = parse_direction_from_name(path)
            xy, field = load_fem_table(path, imag_sign=sign)
            expected_res, expected_amp = _fit_plane_wave(
                xy, field, direction, k0, phase_sign
            )
            flipped = (-direction[0], -direction[1])
            flipped_res, flipped_amp = _fit_plane_wave(xy, field, flipped, k0, phase_sign)
            print(
                f"  {label:>2s}: expected_dir_res={expected_res:.4f}, "
                f"amp={expected_amp.real:+.5f}{expected_amp.imag:+.5f}j; "
                f"flipped_dir_res={flipped_res:.4f}, "
                f"amp={flipped_amp.real:+.5f}{flipped_amp.imag:+.5f}j"
            )


def run_square_case(
    *,
    frequency_hz: float,
    default_data_dir: Path,
    default_output_dir: str,
    description: str,
    direction_labels: tuple[str, ...] = ("+x", "-x"),
) -> None:
    args = build_square_parser(description).parse_args()
    data_dir = Path(args.data_dir).resolve() if args.data_dir else default_data_dir.resolve()
    output_dir = default_data_dir / (args.output_dir or default_output_dir)

    if not data_dir.exists():
        raise FileNotFoundError(
            f"FEM data directory does not exist: {data_dir}. "
            "Expected +x.txt and -x.txt in that directory."
        )

    if args.diagnose_imag_sign:
        diagnose_imaginary_sign(data_dir, frequency_hz, args.phase_sign)
        return

    target = TargetSpec(
        name="Regular square target",
        kind="square",
        eps_background=1.0,
        eps_object=4.0,
        roi_half_width=0.5,
        square_side=0.4,
    )
    lbfgs_steps = args.lbfgs_steps if args.lbfgs_steps is not None else args.epochs_lbfgs
    config = TrainConfig(
        frequency_hz=frequency_hz,
        incident_amplitude=args.incident_amplitude,
        incident_phase_sign=args.phase_sign,
        observation_imag_sign=args.observation_imag_sign,
        estimate_incident_amplitude=args.estimate_incident_amplitude,
        eps_min=1.0,
        eps_max=args.eps_max,
        eps_initial=args.eps_initial,
        domain_radius=args.domain_radius,
        epochs_adam=args.epochs if args.epochs is not None else args.epochs_adam,
        epochs_lbfgs=0,
        lbfgs_steps=lbfgs_steps,
        lbfgs_lr=args.lbfgs_lr,
        lbfgs_max_iter=args.lbfgs_max_iter,
        lbfgs_history_size=args.lbfgs_history_size,
        learning_rate=args.lr,
        device=args.device,
        dtype=args.dtype,
        resume_checkpoint=args.resume_checkpoint,
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
        loss_preset=args.loss_preset,
        lambda_f=args.lambda_f,
        lambda_d=args.lambda_d,
        lambda_ep=args.lambda_ep,
        adaptive_gamma=args.adaptive_gamma,
        adaptive_delta=args.adaptive_delta,
        edge_delta=args.edge_delta,
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
    )

    metrics = train_double_branch_pinn(
        data_dir=data_dir,
        target=target,
        config=config,
        output_dir=output_dir,
        direction_labels=direction_labels,
    )
    print(json.dumps({"target": asdict(target), "metrics": metrics}, indent=2, ensure_ascii=False))
