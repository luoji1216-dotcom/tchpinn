from __future__ import annotations

import argparse
import json
from pathlib import Path

from download_fresnel_data import ensure_fresnel_data
from fresnel_experimental_common import train_fresnel_dieltm_case


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Chapter 5.3.4 experimental-data imaging validation with the "
            "double-branch PINN."
        )
    )
    parser.add_argument(
        "--data-file",
        type=Path,
        default=Path(__file__).resolve().parent / "fresnel_2001" / "dielTM_dec8f.exp",
        help="Path to the Fresnel dielTM_dec8f.exp file.",
    )
    parser.add_argument(
        "--download-missing",
        action="store_true",
        help="Download dielTM_dec8f.exp from the official IOP supplementary data page if missing.",
    )
    parser.add_argument(
        "--frequency-index",
        type=int,
        default=1,
        choices=range(1, 9),
        metavar="{1..8}",
        help="Frequency index in the Fresnel file. 1 means 1 GHz; default follows thesis 5.3.4.",
    )
    parser.add_argument(
        "--field-mode",
        choices=["difference", "first_pair", "second_pair"],
        default="difference",
        help=(
            "Use total-incident, total only, or incident only as the training field. "
            "The official file columns make 'difference' the scattered field."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "results_5_3_4_fresnel_dieltm_1GHz",
        help="Directory for checkpoints, metrics, and reconstruction figures.",
    )
    parser.add_argument("--epochs", type=int, default=40000, help="Adam training iterations.")
    parser.add_argument(
        "--device",
        default="auto",
        help="auto, cuda, cuda:0, or cpu. Default chooses CUDA when available.",
    )
    parser.add_argument("--seed", type=int, default=20260604, help="Random seed.")
    parser.add_argument(
        "--max-receivers-per-view",
        type=int,
        default=0,
        help="Optional receiver thinning per view. 0 keeps all 49 receivers.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_file = args.data_file
    if not data_file.exists():
        if not args.download_missing:
            raise FileNotFoundError(
                f"{data_file} does not exist. Run download_fresnel_data.py first or pass "
                "--download-missing."
            )
        data_file = ensure_fresnel_data(data_file.parent, data_file.name, force=False)

    if args.frequency_index != 1 and "1GHz" in str(args.output_dir):
        args.output_dir = args.output_dir.with_name(
            args.output_dir.name.replace("1GHz", f"{args.frequency_index}GHz")
        )

    metrics = train_fresnel_dieltm_case(
        exp_path=data_file,
        output_dir=args.output_dir,
        frequency_index=args.frequency_index,
        field_mode=args.field_mode,  # type: ignore[arg-type]
        epochs=args.epochs,
        device=args.device,
        seed=args.seed,
        max_receivers_per_view=args.max_receivers_per_view,
    )
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
