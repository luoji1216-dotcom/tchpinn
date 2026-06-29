from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CORE_ROOT = ROOT / "square_target"
if str(CORE_ROOT) not in sys.path:
    sys.path.insert(0, str(CORE_ROOT))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pinn_pixel_inverse_core import TargetSpec, TrainConfig, train_noise_sweep


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Section 5.3.3: double-branch PINN pixel-level inverse scattering "
            "for the Austria target. FEM files are total Ez fields; columns 8 "
            "and 9 are read as real/imaginary parts."
        )
    )
    parser.add_argument("--epochs", type=int, default=35000, help="Adam iterations per noise level.")
    parser.add_argument("--device", default="auto", help="'auto', 'cpu', or 'cuda'.")
    parser.add_argument("--dtype", default="float32", choices=["float32", "float64"])
    parser.add_argument("--lr", type=float, default=8.0e-4, help="Adam learning rate.")
    parser.add_argument("--incident-amplitude", type=float, default=0.1)
    parser.add_argument("--phase-sign", type=float, default=-1.0)
    parser.add_argument(
        "--estimate-incident-amplitude",
        action="store_true",
        help="Estimate one complex incident amplitude per FEM direction before subtracting the incident field.",
    )
    parser.add_argument(
        "--noise-levels",
        type=float,
        nargs="+",
        default=[0.0, 0.25, 0.50],
        help="Relative complex Gaussian noise levels added to observed scattered fields.",
    )
    parser.add_argument("--max-points-per-direction", type=int, default=1200)
    parser.add_argument("--data-batch-per-direction", type=int, default=256)
    parser.add_argument("--n-pde", type=int, default=2560)
    parser.add_argument("--n-boundary", type=int, default=640)
    parser.add_argument("--n-tv-grid", type=int, default=56)
    parser.add_argument("--plot-grid-size", type=int, default=240)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--checkpoint-every", type=int, default=2500)
    parser.add_argument("--output-dir", default="results_5_3_3_austria")
    parser.add_argument("--no-robust-data", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    data_dir = Path(__file__).resolve().parent
    output_dir = data_dir / args.output_dir

    target = TargetSpec(
        name="Austria target",
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
    config = TrainConfig(
        frequency_hz=0.3e9,
        incident_amplitude=args.incident_amplitude,
        incident_phase_sign=args.phase_sign,
        estimate_incident_amplitude=args.estimate_incident_amplitude,
        eps_min=1.0,
        eps_max=4.0,
        eps_initial=1.04,
        epochs_adam=args.epochs,
        learning_rate=args.lr,
        device=args.device,
        dtype=args.dtype,
        max_points_per_direction=args.max_points_per_direction,
        data_batch_per_direction=args.data_batch_per_direction,
        n_pde=args.n_pde,
        n_boundary=args.n_boundary,
        n_tv_grid=args.n_tv_grid,
        plot_grid_size=args.plot_grid_size,
        log_every=args.log_every,
        checkpoint_every=args.checkpoint_every,
        robust_data_weighting=not args.no_robust_data,
        weight_data=120.0,
        weight_pde=1.0,
        weight_boundary=0.05,
        weight_tv=3.0e-4,
        weight_contrast_l1=2.0e-5,
        random_seed=20260430,
    )

    metrics = train_noise_sweep(
        data_dir=data_dir,
        target=target,
        base_config=config,
        output_root=output_dir,
        noise_levels=args.noise_levels,
        direction_labels=("+x", "-x", "+y", "-y"),
    )
    print(json.dumps({"target": asdict(target), "metrics": metrics}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
