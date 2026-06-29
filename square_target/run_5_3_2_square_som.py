from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pinn_pixel_inverse_core import TargetSpec, TrainConfig, load_observations  # noqa: E402
from som_inverse_core import build_som_parser, run_som_inversion, som_config_from_args  # noqa: E402


def main() -> None:
    here = Path(__file__).resolve().parent
    parser = build_som_parser("SOM inversion for the 0.3 GHz square target.")
    parser.set_defaults(
        grid_size=56,
        ridge=2.0e-3,
        som_iters=5,
        eps_max=4.0,
        green_convention="h2_minus_i4",
        contrast_gain=1.0,
        smooth_sigma=2.0,
    )
    args = parser.parse_args()
    output_dir = here / (args.output_dir or "results_5_3_2_square_0_3GHz_som")

    target = TargetSpec(
        name="Regular square target",
        kind="square",
        eps_background=1.0,
        eps_object=4.0,
        roi_half_width=0.5,
        square_side=0.4,
    )
    train_config = TrainConfig(
        frequency_hz=0.3e9,
        incident_amplitude=0.1,
        incident_phase_sign=args.phase_sign,
        observation_imag_sign=args.observation_imag_sign,
        estimate_incident_amplitude=False,
        max_points_per_direction=836,
    )
    obs = load_observations(here, train_config, direction_labels=("+x", "-x"))
    som_config = som_config_from_args(args, frequency_hz=0.3e9, eps_max=4.0)
    metrics = run_som_inversion(obs, target, som_config, output_dir)
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
