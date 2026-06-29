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

from pinn_pixel_inverse_core import TargetSpec, TrainConfig, train_double_branch_pinn  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Paper-faithful research entry for thesis section 5.3.3, Austria target, 0.3 GHz. "
            "It uses the thesis-style double-branch PINN with total-field fitting, "
            "Helmholtz residuals, robust observation weighting, TV regularization, "
            "and optional Lippmann-Schwinger data consistency."
        )
    )
    parser.add_argument("--confirm-long-run", action="store_true")
    parser.add_argument("--epochs", type=int, default=30000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="float32", choices=["float32", "float64"])
    parser.add_argument("--noise-level", type=float, default=0.0)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--plot-grid-size", type=int, default=240)
    parser.add_argument("--log-every", type=int, default=200)
    parser.add_argument("--checkpoint-every", type=int, default=5000)
    parser.add_argument(
        "--observation-imag-sign",
        type=float,
        default=-1.0,
        choices=[1.0, -1.0],
        help="Use -1 for Ez=Re-iIm, which matched the 0.3 GHz FEM exports best.",
    )
    parser.add_argument("--phase-sign", type=float, default=1.0)
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


def main() -> None:
    args = build_parser().parse_args()
    here = Path(__file__).resolve().parent
    target = build_target(args.noise_level)
    output_dir = args.output_dir or (
        f"results_5_3_3_austria_0_3GHz_noise_{int(round(100.0 * args.noise_level)):02d}pct_paper_faithful"
    )

    config = TrainConfig(
        frequency_hz=0.3e9,
        incident_amplitude=0.1,
        incident_phase_sign=args.phase_sign,
        observation_imag_sign=args.observation_imag_sign,
        estimate_incident_amplitude=False,
        eps_min=1.0,
        eps_max=4.0,
        eps_initial=1.35,
        max_points_per_direction=836,
        data_batch_per_direction=256,
        n_pde=2048,
        n_boundary=512,
        n_tv_grid=56,
        integral_grid_size=40,
        epochs_adam=args.epochs,
        learning_rate=8.0e-4,
        weight_data=120.0,
        weight_pde=0.02,
        weight_boundary=0.02,
        weight_integral_data=500.0,
        weight_tv=0.006,
        weight_contrast_l1=0.0,
        robust_data_weighting=True,
        gradient_clip_norm=1.0,
        field_hidden_layers=6,
        field_hidden_units=96,
        eps_hidden_layers=5,
        eps_hidden_units=96,
        fourier_bands=6,
        fourier_max_frequency=10.0,
        plot_grid_size=args.plot_grid_size,
        log_every=args.log_every,
        checkpoint_every=args.checkpoint_every,
        random_seed=20260430,
        dtype=args.dtype,
        device=args.device,
        noise_level=args.noise_level,
    )

    metadata = {
        "data_dir": str(here),
        "direction_labels": ["+x", "-x", "+y", "-y"],
        "target": asdict(target),
        "config": asdict(config),
        "output_dir": str(here / output_dir),
    }
    if not args.confirm_long_run:
        print(json.dumps(metadata, indent=2, ensure_ascii=False))
        raise SystemExit("Pass --confirm-long-run to start this long research-grade run.")

    metrics = train_double_branch_pinn(
        data_dir=here,
        target=target,
        config=config,
        output_dir=here / output_dir,
        direction_labels=("+x", "-x", "+y", "-y"),
    )
    print(json.dumps({"target": asdict(target), "metrics": metrics}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
