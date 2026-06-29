from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np

from pinn_pixel_inverse_core import (
    DoubleBranchPINN,
    TargetSpec,
    TrainConfig,
    global_ssim,
    reconstruct_epsilon,
    relative_error,
    threshold_epsilon_map,
    true_epsilon_grid,
)


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
    parser.add_argument(
        "--checkpoint-threshold-sweep",
        action="store_true",
        help="Reconstruct each saved checkpoint and scan thresholded metrics from 1.2 to 3.0.",
    )
    parser.add_argument("--device", default="cpu", help="'cpu', 'cuda', or 'auto' for checkpoint sweep.")
    parser.add_argument("--dtype", default="float32", choices=["float32", "float64"])
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


def checkpoint_rows(rows: List[dict]) -> List[dict]:
    filtered: List[dict] = []
    for row in rows:
        checkpoint_path = (row.get("checkpoint_path") or "").strip()
        if not checkpoint_path:
            continue
        if Path(checkpoint_path).exists():
            filtered.append(row)
    return filtered


def threshold_sweep_rows(
    epsilon_pred: np.ndarray,
    epsilon_truth: np.ndarray,
    target: TargetSpec,
    truth_object_pixels: int,
) -> List[dict]:
    rows: List[dict] = []
    data_range = target.eps_object - target.eps_background
    thresholds = np.arange(1.2, 3.0 + 0.5 * 0.05, 0.05, dtype=np.float64)
    for threshold in thresholds:
        threshold_value = float(threshold)
        thresholded = threshold_epsilon_map(epsilon_pred, target, threshold=threshold_value)
        predicted_object_pixels = int(np.count_nonzero(thresholded >= threshold_value))
        area_ratio = (
            float(predicted_object_pixels / truth_object_pixels) if truth_object_pixels else float("nan")
        )
        rows.append(
            {
                "threshold": threshold_value,
                "rel_error_thresholded": relative_error(thresholded, epsilon_truth),
                "ssim_thresholded": global_ssim(thresholded, epsilon_truth, data_range),
                "predicted_object_pixels": predicted_object_pixels,
                "predicted_to_truth_area_ratio": area_ratio,
            }
        )
    return rows


def load_checkpoint_model(checkpoint_path: Path, *, device: str, dtype_name: str) -> tuple[DoubleBranchPINN, TargetSpec, TrainConfig, "torch.device", "torch.dtype"]:
    import torch

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or "model" not in checkpoint or "config" not in checkpoint or "target" not in checkpoint:
        raise ValueError(f"Checkpoint must contain model/config/target: {checkpoint_path}")

    target = TargetSpec(**checkpoint["target"])
    config_payload = dict(checkpoint["config"])
    config_payload["device"] = device
    config_payload["dtype"] = dtype_name
    config = TrainConfig(**config_payload)

    torch_device = torch.device("cpu" if device == "auto" else device)
    torch_dtype = torch.float64 if dtype_name == "float64" else torch.float32
    model = DoubleBranchPINN(config, target).to(device=torch_device, dtype=torch_dtype)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, target, config, torch_device, torch_dtype


def run_checkpoint_threshold_sweep(rows: List[dict], *, csv_path: Path, output_dir: Path, device: str, dtype_name: str) -> None:
    candidates = checkpoint_rows(rows)
    if not candidates:
        print("No existing checkpoint_path entries found for threshold sweep.")
        return

    summary_rows: List[dict] = []
    for row in candidates:
        checkpoint_path = Path(str(row["checkpoint_path"])).resolve()
        model, target, config, torch_device, torch_dtype = load_checkpoint_model(
            checkpoint_path, device=device, dtype_name=dtype_name
        )
        x, y, epsilon_pred = reconstruct_epsilon(model, target, config.plot_grid_size, torch_device, torch_dtype)
        _, _, epsilon_truth = true_epsilon_grid(target, config.plot_grid_size)
        truth_threshold = 0.5 * (target.eps_background + target.eps_object)
        truth_object_pixels = int(np.count_nonzero(epsilon_truth >= truth_threshold))
        sweep = threshold_sweep_rows(epsilon_pred, epsilon_truth, target, truth_object_pixels)
        best = min(sweep, key=lambda item: item["rel_error_thresholded"])
        summary_rows.append(
            {
                "epoch": int(to_float(row.get("epoch"))),
                "checkpoint_path": str(checkpoint_path),
                "best_threshold": float(best["threshold"]),
                "best_rel_error_thresholded": float(best["rel_error_thresholded"]),
                "best_ssim_thresholded": float(best["ssim_thresholded"]),
                "best_predicted_object_pixels": int(best["predicted_object_pixels"]),
                "best_predicted_to_truth_area_ratio": float(best["predicted_to_truth_area_ratio"]),
            }
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "all_checkpoint_threshold_sweep_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "epoch",
                "checkpoint_path",
                "best_threshold",
                "best_rel_error_thresholded",
                "best_ssim_thresholded",
                "best_predicted_object_pixels",
                "best_predicted_to_truth_area_ratio",
            ],
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    global_best = min(summary_rows, key=lambda item: item["best_rel_error_thresholded"])
    print(f"Wrote checkpoint threshold sweep summary: {summary_path}")
    print(
        "Global best checkpoint: epoch={epoch}, threshold={threshold:.2f}, rel_error={rel:.6g}, ssim={ssim:.6g}, checkpoint={checkpoint}".format(
            epoch=global_best["epoch"],
            threshold=global_best["best_threshold"],
            rel=global_best["best_rel_error_thresholded"],
            ssim=global_best["best_ssim_thresholded"],
            checkpoint=global_best["checkpoint_path"],
        )
    )


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
    else:
        output_dir = Path(args.output_dir).resolve() if args.output_dir else csv_path.parent

    if args.checkpoint_threshold_sweep:
        run_checkpoint_threshold_sweep(
            rows,
            csv_path=csv_path,
            output_dir=output_dir,
            device=args.device,
            dtype_name=args.dtype,
        )


if __name__ == "__main__":
    main()
