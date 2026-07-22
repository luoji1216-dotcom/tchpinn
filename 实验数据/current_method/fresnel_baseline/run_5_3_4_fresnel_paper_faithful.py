from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
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
from square_target.pinn_pixel_inverse_core import train_double_branch_pinn_from_observations  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Paper-faithful research entry for thesis section 5.3.4, Fresnel experimental data. "
            "It keeps the double-branch PINN workflow and adds the same robust data, "
            "TV and integral-consistency options used in the synthetic examples."
        )
    )
    parser.add_argument(
        "--data-file",
        type=Path,
        default=Path(__file__).resolve().parent / "fresnel_2001" / "dielTM_dec8f.exp",
    )
    parser.add_argument("--download-missing", action="store_true")
    parser.add_argument("--frequency-index", type=int, default=1, choices=range(1, 9), metavar="{1..8}")
    parser.add_argument(
        "--field-mode",
        choices=["difference", "first_pair", "second_pair"],
        default="difference",
    )
    parser.add_argument("--max-receivers-per-view", type=int, default=0)
    parser.add_argument("--confirm-long-run", action="store_true")
    parser.add_argument("--epochs", type=int, default=40000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=20260604)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--plot-grid-size", type=int, default=240)
    parser.add_argument("--log-every", type=int, default=200)
    parser.add_argument("--checkpoint-every", type=int, default=5000)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    data_file = args.data_file
    if not data_file.exists():
        if not args.download_missing:
            raise FileNotFoundError(
                f"{data_file} does not exist. Run download_fresnel_data.py first or pass --download-missing."
            )
        data_file = ensure_fresnel_data(data_file.parent, data_file.name, force=False)

    target = fresnel_dieltm_target()
    config = fresnel_default_config(
        frequency_index=args.frequency_index,
        epochs=args.epochs,
        device=args.device,
        seed=args.seed,
    )
    config.weight_pde = 0.05
    config.weight_boundary = 0.02
    config.weight_tv = 8.0e-4
    config.weight_integral_data = 200.0
    config.integral_grid_size = 48
    config.n_pde = 3072
    config.n_boundary = 768
    config.n_tv_grid = 72
    config.field_hidden_layers = 7
    config.field_hidden_units = 112
    config.eps_hidden_layers = 5
    config.eps_hidden_units = 96
    config.fourier_bands = 6
    config.fourier_max_frequency = 10.0
    config.plot_grid_size = args.plot_grid_size
    config.log_every = args.log_every
    config.checkpoint_every = args.checkpoint_every

    obs = load_fresnel_dieltm_observations(
        exp_path=data_file,
        config=config,
        frequency_index=args.frequency_index,
        field_mode=args.field_mode,
        max_receivers_per_view=args.max_receivers_per_view,
    )
    output_dir = args.output_dir or (
        Path(__file__).resolve().parent
        / f"results_5_3_4_fresnel_dielTM_{args.frequency_index}GHz_paper_faithful"
    )

    metadata = {
        "data_file": str(data_file),
        "field_mode": args.field_mode,
        "frequency_index": args.frequency_index,
        "num_observations": int(obs.xy.shape[0]),
        "num_views": len(obs.direction_labels),
        "target": asdict(target),
        "config": asdict(config),
        "output_dir": str(output_dir),
    }
    if not args.confirm_long_run:
        print(json.dumps(metadata, indent=2, ensure_ascii=False))
        raise SystemExit("Pass --confirm-long-run to start this long research-grade run.")

    metrics = train_double_branch_pinn_from_observations(
        obs=obs,
        target=target,
        config=config,
        output_dir=output_dir,
    )
    print(json.dumps({"target": asdict(target), "metrics": metrics}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
