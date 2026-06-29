from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Iterable, List, Optional


DEFAULT_CSV = Path(__file__).resolve().parent / "results_5_3_2_square_0_3GHz" / "evaluation_metrics.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare continuous and thresholded PINN epsilon metrics."
    )
    parser.add_argument(
        "csv_path",
        nargs="?",
        default=str(DEFAULT_CSV),
        help="Path to evaluation_metrics.csv.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for generated plots. Defaults to the CSV directory.",
    )
    parser.add_argument("--no-plots", action="store_true", help="Only print best epochs.")
    return parser.parse_args()


def to_float(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def load_rows(csv_path: Path) -> List[dict]:
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"No rows found in {csv_path}")
    return rows


def valid_rows(rows: Iterable[dict], metric: str) -> List[dict]:
    return [row for row in rows if not math.isnan(to_float(row.get(metric)))]


def best_row(rows: List[dict], metric: str, *, higher_is_better: bool) -> Optional[dict]:
    candidates = valid_rows(rows, metric)
    if not candidates:
        return None
    return max(candidates, key=lambda row: to_float(row[metric])) if higher_is_better else min(
        candidates, key=lambda row: to_float(row[metric])
    )


def print_best(label: str, rows: List[dict], metric: str, *, higher_is_better: bool) -> None:
    row = best_row(rows, metric, higher_is_better=higher_is_better)
    if row is None:
        print(f"{label}: no valid values")
        return
    checkpoint = row.get("checkpoint_path", "")
    suffix = f", checkpoint={checkpoint}" if checkpoint else ""
    print(f"{label}: epoch={row.get('epoch')}, {metric}={to_float(row[metric]):.6g}{suffix}")


def metric_series(rows: List[dict], metric: str) -> tuple[list[float], list[float]]:
    epochs: list[float] = []
    values: list[float] = []
    for row in rows:
        epoch = to_float(row.get("epoch"))
        value = to_float(row.get(metric))
        if math.isnan(epoch) or math.isnan(value):
            continue
        epochs.append(epoch)
        values.append(value)
    return epochs, values


def plot_curves(rows: List[dict], output_dir: Path) -> None:
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    for metric, label in (
        ("rel_error_continuous", "continuous"),
        ("rel_error_thresholded", "thresholded"),
    ):
        epochs, values = metric_series(rows, metric)
        ax.plot(epochs, values, marker="o", linewidth=1.8, markersize=3.5, label=label)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Relative error")
    ax.legend()
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    rel_path = output_dir / "pinn_relative_error_curve.png"
    fig.savefig(rel_path, dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    for metric, label in (
        ("ssim_continuous", "continuous"),
        ("ssim_thresholded", "thresholded"),
    ):
        epochs, values = metric_series(rows, metric)
        ax.plot(epochs, values, marker="o", linewidth=1.8, markersize=3.5, label=label)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("SSIM")
    ax.legend()
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    ssim_path = output_dir / "pinn_ssim_curve.png"
    fig.savefig(ssim_path, dpi=180)
    plt.close(fig)

    print(f"Wrote plots: {rel_path}")
    print(f"Wrote plots: {ssim_path}")


def main() -> None:
    args = parse_args()
    csv_path = Path(args.csv_path).resolve()
    rows = load_rows(csv_path)

    print(f"Loaded {len(rows)} rows from {csv_path}")
    print_best("Best continuous relative error", rows, "rel_error_continuous", higher_is_better=False)
    print_best("Best thresholded relative error", rows, "rel_error_thresholded", higher_is_better=False)
    print_best("Best continuous SSIM", rows, "ssim_continuous", higher_is_better=True)
    print_best("Best thresholded SSIM", rows, "ssim_thresholded", higher_is_better=True)

    if not args.no_plots:
        output_dir = Path(args.output_dir).resolve() if args.output_dir else csv_path.parent
        plot_curves(rows, output_dir)


if __name__ == "__main__":
    main()
