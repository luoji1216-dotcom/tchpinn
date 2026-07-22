from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
SQUARE_REPRO_ROOT = REPO_ROOT / "square-target-repro"
if str(SQUARE_REPRO_ROOT) not in sys.path:
    sys.path.insert(0, str(SQUARE_REPRO_ROOT))

from download_fresnel_data import ensure_fresnel_data  # noqa: E402
from fresnel_experimental_common import (  # noqa: E402
    fresnel_default_config,
    fresnel_dieltm_target,
    load_fresnel_dieltm_observations,
)
from square_target.som_inverse_core import build_som_parser, run_som_inversion, som_config_from_args  # noqa: E402


def main() -> None:
    here = Path(__file__).resolve().parent
    parser = build_som_parser("SOM inversion for Fresnel 5.3.4 experimental data.")
    parser.set_defaults(
        grid_size=72,
        ridge=8.0e-3,
        som_iters=0,
        eps_max=4.0,
        phase_sign=-1.0,
        observation_imag_sign=1.0,
    )
    parser.add_argument("--frequency-index", type=int, default=1, choices=range(1, 9))
    parser.add_argument("--field-mode", choices=["difference", "first_pair", "second_pair"], default="difference")
    parser.add_argument(
        "--data-file",
        type=Path,
        default=here / "fresnel_2001" / "dielTM_dec8f.exp",
    )
    parser.add_argument("--download-missing", action="store_true")
    args = parser.parse_args()

    data_file = args.data_file
    if not data_file.exists():
        if not args.download_missing:
            raise FileNotFoundError(
                f"{data_file} does not exist. Run download_fresnel_data.py first or pass "
                "--download-missing."
            )
        data_file = ensure_fresnel_data(data_file.parent, data_file.name, force=False)

    output_dir = here / (
        args.output_dir
        or f"results_5_3_4_fresnel_dieltm_{args.frequency_index}GHz_som"
    )
    target = fresnel_dieltm_target()
    train_config = fresnel_default_config(
        frequency_index=args.frequency_index,
        epochs=1,
        device="cpu",
    )
    train_config.incident_phase_sign = args.phase_sign
    obs = load_fresnel_dieltm_observations(
        exp_path=data_file,
        config=train_config,
        frequency_index=args.frequency_index,
        field_mode=args.field_mode,
    )
    som_config = som_config_from_args(
        args,
        frequency_hz=float(args.frequency_index) * 1.0e9,
        eps_max=4.0,
    )
    metrics = run_som_inversion(obs, target, som_config, output_dir)
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
