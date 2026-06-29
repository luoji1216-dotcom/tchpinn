from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CORE_ROOT = ROOT / "square_target"
if str(CORE_ROOT) not in sys.path:
    sys.path.insert(0, str(CORE_ROOT))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pinn_pixel_inverse_core import TargetSpec, TrainConfig, load_observations  # noqa: E402
from som_inverse_core import build_som_parser, run_som_inversion, som_config_from_args  # noqa: E402


def main() -> None:
    here = Path(__file__).resolve().parent
    parser = build_som_parser("SOM inversion for the 0.3 GHz Austria target.")
    parser.set_defaults(
        grid_size=72,
        ridge=4.0e-3,
        som_iters=300,
        eps_max=3.0,
        green_convention="h2_minus_i4",
    )
    parser.add_argument("--noise-level", type=float, default=0.0)
    args = parser.parse_args()
    output_dir = here / (
        args.output_dir
        or f"results_5_3_3_austria_0_3GHz_som_noise_{int(round(args.noise_level * 100)):02d}pct"
    )

    target = TargetSpec(
        name=f"Austria target, {int(round(100.0 * args.noise_level))}% noise",
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
    train_config = TrainConfig(
        frequency_hz=0.3e9,
        incident_amplitude=0.1,
        incident_phase_sign=args.phase_sign,
        observation_imag_sign=args.observation_imag_sign,
        estimate_incident_amplitude=False,
        max_points_per_direction=836,
        noise_level=args.noise_level,
    )
    obs = load_observations(
        here,
        train_config,
        direction_labels=("+x", "-x", "+y", "-y"),
    )
    som_config = som_config_from_args(args, frequency_hz=0.3e9, eps_max=3.0)
    metrics = run_som_inversion(obs, target, som_config, output_dir)
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
