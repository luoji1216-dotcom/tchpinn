from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from pinn_pixel_inverse_core import (
    DoubleBranchPINN,
    TargetSpec,
    TrainConfig,
    configure_matplotlib,
    global_ssim,
    plot_epsilon_image,
    relative_error,
    reconstruct_epsilon,
    threshold_epsilon_map,
    true_epsilon_grid,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Diagnose a square_target checkpoint.")
    parser.add_argument(
        "checkpoint",
        nargs="?",
        default="square_target/results_5_3_2_square_0_3GHz_metric/checkpoint_adam_017000.pt",
        help="Path to a checkpoint .pt file.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory to store diagnostic figures. Defaults to <checkpoint_dir>/checkpoint_diagnosis.",
    )
    parser.add_argument("--grid-size", type=int, default=None, help="Override epsilon reconstruction grid size.")
    parser.add_argument("--device", default="cpu", help="'cpu', 'cuda', or 'auto'.")
    parser.add_argument("--dtype", default="float32", choices=["float32", "float64"])
    parser.add_argument(
        "--fixed-threshold",
        type=float,
        default=None,
        help="Override threshold used for thresholded export and metrics.",
    )
    parser.add_argument(
        "--final-reproduction",
        action="store_true",
        help="Write the requested final reproduction artifacts into a final_reproduction directory.",
    )
    return parser


def _to_target(payload: dict) -> TargetSpec:
    return TargetSpec(**payload)


def _to_config(payload: dict, args: argparse.Namespace) -> TrainConfig:
    payload = dict(payload)
    payload["device"] = args.device
    payload["dtype"] = args.dtype
    if args.grid_size is not None:
        payload["plot_grid_size"] = args.grid_size
        payload["integral_grid_size"] = args.grid_size
    return TrainConfig(**payload)


def _print_stats(name: str, arr: np.ndarray) -> None:
    print(
        f"{name}: min={arr.min():.6g}, max={arr.max():.6g}, mean={arr.mean():.6g}, std={arr.std():.6g}",
        flush=True,
    )


def _threshold_sweep_rows(
    epsilon_pred: np.ndarray,
    epsilon_truth: np.ndarray,
    target: TargetSpec,
    truth_object_pixels: int,
) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    data_range = target.eps_object - target.eps_background
    thresholds = np.arange(1.2, 3.0 + 0.5 * 0.05, 0.05, dtype=np.float64)
    for threshold in thresholds:
        thresholded = threshold_epsilon_map(epsilon_pred, target, threshold=float(threshold))
        pred_object_pixels = int(np.count_nonzero(thresholded >= threshold))
        area_ratio = float(pred_object_pixels / truth_object_pixels) if truth_object_pixels else float("nan")
        rows.append(
            {
                "threshold": float(threshold),
                "rel_error_thresholded": relative_error(thresholded, epsilon_truth),
                "ssim_thresholded": global_ssim(thresholded, epsilon_truth, data_range),
                "predicted_object_pixels": float(pred_object_pixels),
                "predicted_to_truth_area_ratio": area_ratio,
            }
        )
    return rows


def _save_abs_error_image(
    x: np.ndarray,
    y: np.ndarray,
    error_map: np.ndarray,
    epsilon_truth: np.ndarray,
    target: TargetSpec,
    threshold: float,
    out_path: Path,
    title: str,
    colorbar_label: str,
) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5.92, 5.0))
    im = ax.imshow(
        error_map,
        extent=[x.min(), x.max(), y.min(), y.max()],
        origin="lower",
        cmap="magma",
        interpolation="bilinear",
    )
    ax.contour(x, y, epsilon_truth, levels=[threshold], colors="c", linewidths=1.8)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-target.roi_half_width, target.roi_half_width)
    ax.set_ylim(-target.roi_half_width, target.roi_half_width)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(title)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label(colorbar_label)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = build_parser().parse_args()
    checkpoint_path = Path(args.checkpoint).resolve()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    import torch

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or "model" not in checkpoint or "config" not in checkpoint or "target" not in checkpoint:
        raise ValueError(
            "Checkpoint must contain 'model', 'config', and 'target' entries."
        )

    target = _to_target(checkpoint["target"])
    config = _to_config(checkpoint["config"], args)

    device = torch.device("cpu" if args.device == "auto" else args.device)
    dtype = torch.float64 if args.dtype == "float64" else torch.float32

    model = DoubleBranchPINN(config, target).to(device=device, dtype=dtype)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    grid_size = config.plot_grid_size if args.grid_size is None else args.grid_size
    x, y, epsilon_pred = reconstruct_epsilon(model, target, grid_size, device, dtype)
    _, _, epsilon_truth = true_epsilon_grid(target, grid_size)
    threshold = args.fixed_threshold
    if threshold is None:
        threshold = 0.5 * (target.eps_background + target.eps_object)
    epsilon_pred_thresholded = threshold_epsilon_map(epsilon_pred, target, threshold=threshold)
    epsilon_abs_error = np.abs(epsilon_pred - epsilon_truth)
    epsilon_abs_error_thresholded = np.abs(epsilon_pred_thresholded - epsilon_truth)

    truth_threshold = 0.5 * (target.eps_background + target.eps_object)
    object_mask = epsilon_truth >= truth_threshold
    background_mask = ~object_mask
    thresholded_object_pixels = int(np.count_nonzero(epsilon_pred_thresholded >= truth_threshold))
    truth_object_pixels = int(np.count_nonzero(object_mask))
    ratio = float(thresholded_object_pixels / truth_object_pixels) if truth_object_pixels else float("nan")
    data_range = target.eps_object - target.eps_background
    rel_error_continuous = relative_error(epsilon_pred, epsilon_truth)
    ssim_continuous = global_ssim(epsilon_pred, epsilon_truth, data_range)
    rel_error_thresholded = relative_error(epsilon_pred_thresholded, epsilon_truth)
    ssim_thresholded = global_ssim(epsilon_pred_thresholded, epsilon_truth, data_range)

    print(f"Checkpoint: {checkpoint_path}")
    print(f"Grid size: {grid_size}")
    print(f"Threshold: {threshold:.6g}")
    _print_stats("epsilon_pred", epsilon_pred)
    _print_stats("epsilon_truth", epsilon_truth)
    _print_stats("epsilon_pred[target]", epsilon_pred[object_mask])
    _print_stats("epsilon_pred[background]", epsilon_pred[background_mask])
    print(
        "thresholded pixels: pred={pred}, truth={truth}, ratio={ratio:.6g}".format(
            pred=thresholded_object_pixels,
            truth=truth_object_pixels,
            ratio=ratio,
        ),
        flush=True,
    )

    if args.final_reproduction:
        output_dir = Path(args.output_dir) if args.output_dir else checkpoint_path.parent / "final_reproduction"
    else:
        output_dir = Path(args.output_dir) if args.output_dir else checkpoint_path.parent / "checkpoint_diagnosis"
    output_dir.mkdir(parents=True, exist_ok=True)
    configure_matplotlib()

    plot_epsilon_image(
        x,
        y,
        epsilon_truth,
        epsilon_truth,
        target,
        output_dir / "epsilon_truth.png",
        title="Truth",
    )
    plot_epsilon_image(
        x,
        y,
        epsilon_pred,
        epsilon_truth,
        target,
        output_dir / "epsilon_pred_continuous.png",
        title="Predicted continuous epsilon",
    )
    thresholded_name = (
        f"epsilon_pred_thresholded_thr{str(threshold).replace('.', 'p')}.png"
        if args.final_reproduction
        else "epsilon_pred_thresholded.png"
    )
    plot_epsilon_image(
        x,
        y,
        epsilon_pred_thresholded,
        epsilon_truth,
        target,
        output_dir / thresholded_name,
        title="Predicted thresholded epsilon",
    )

    import matplotlib.pyplot as plt

    _save_abs_error_image(
        x,
        y,
        epsilon_abs_error,
        epsilon_truth,
        target,
        truth_threshold,
        output_dir / ("epsilon_abs_error_continuous.png" if args.final_reproduction else "epsilon_abs_error.png"),
        title="Absolute error",
        colorbar_label="|epsilon_pred - epsilon_truth|",
    )
    if args.final_reproduction:
        _save_abs_error_image(
            x,
            y,
            epsilon_abs_error_thresholded,
            epsilon_truth,
            target,
            truth_threshold,
            output_dir / f"epsilon_abs_error_thresholded_thr{str(threshold).replace('.', 'p')}.png",
            title="Thresholded absolute error",
            colorbar_label="|epsilon_pred_thresholded - epsilon_truth|",
        )

    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    bins = min(60, max(20, epsilon_pred.size // 40))
    ax.hist(epsilon_pred.ravel(), bins=bins, color="#4477aa", alpha=0.85, label="pred")
    ax.axvline(threshold, color="#cc3311", linestyle="--", linewidth=2.0, label="threshold")
    ax.axvline(target.eps_background, color="#777777", linestyle=":", linewidth=1.5, label="background")
    ax.axvline(target.eps_object, color="#222222", linestyle=":", linewidth=1.5, label="object")
    ax.set_xlabel("epsilon")
    ax.set_ylabel("count")
    ax.set_title("Predicted epsilon histogram")
    ax.legend(loc="best", frameon=True)
    fig.tight_layout()
    fig.savefig(output_dir / "epsilon_pred_histogram.png", bbox_inches="tight")
    plt.close(fig)

    summary = {
        "checkpoint": str(checkpoint_path),
        "threshold": threshold,
        "grid_size": grid_size,
        "thresholded_object_pixels": thresholded_object_pixels,
        "truth_object_pixels": truth_object_pixels,
        "thresholded_to_truth_ratio": ratio,
        "rel_error_continuous": rel_error_continuous,
        "ssim_continuous": ssim_continuous,
        "rel_error_thresholded": rel_error_thresholded,
        "ssim_thresholded": ssim_thresholded,
        "epsilon_pred": {
            "min": float(epsilon_pred.min()),
            "max": float(epsilon_pred.max()),
            "mean": float(epsilon_pred.mean()),
            "std": float(epsilon_pred.std()),
        },
        "epsilon_truth": {
            "min": float(epsilon_truth.min()),
            "max": float(epsilon_truth.max()),
            "mean": float(epsilon_truth.mean()),
            "std": float(epsilon_truth.std()),
        },
        "target_region_pred": {
            "mean": float(epsilon_pred[object_mask].mean()),
            "std": float(epsilon_pred[object_mask].std()),
        },
        "background_region_pred": {
            "mean": float(epsilon_pred[background_mask].mean()),
            "std": float(epsilon_pred[background_mask].std()),
        },
    }

    sweep_rows = _threshold_sweep_rows(epsilon_pred, epsilon_truth, target, truth_object_pixels)
    best_row = min(sweep_rows, key=lambda row: row["rel_error_thresholded"])
    with (output_dir / "threshold_sweep.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "threshold",
                "rel_error_thresholded",
                "ssim_thresholded",
                "predicted_object_pixels",
                "predicted_to_truth_area_ratio",
            ],
        )
        writer.writeheader()
        writer.writerows(sweep_rows)
    summary["threshold_sweep_best_rel_error"] = {
        "threshold": float(best_row["threshold"]),
        "rel_error_thresholded": float(best_row["rel_error_thresholded"]),
        "ssim_thresholded": float(best_row["ssim_thresholded"]),
        "predicted_object_pixels": int(best_row["predicted_object_pixels"]),
        "predicted_to_truth_area_ratio": float(best_row["predicted_to_truth_area_ratio"]),
    }
    with (output_dir / "diagnosis_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    if args.final_reproduction:
        epoch = int(checkpoint.get("step", 0) or 0)
        final_metrics = {
            "checkpoint_path": str(checkpoint_path),
            "epoch": epoch,
            "threshold": float(threshold),
            "rel_error_continuous": float(rel_error_continuous),
            "ssim_continuous": float(ssim_continuous),
            "rel_error_thresholded": float(rel_error_thresholded),
            "ssim_thresholded": float(ssim_thresholded),
            "predicted_object_pixels": int(thresholded_object_pixels),
            "truth_object_pixels": int(truth_object_pixels),
            "predicted_to_truth_area_ratio": float(ratio),
        }
        with (output_dir / "final_metrics.json").open("w", encoding="utf-8") as f:
            json.dump(final_metrics, f, indent=2, ensure_ascii=False)
        with (output_dir / "final_metrics.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(final_metrics.keys()))
            writer.writeheader()
            writer.writerow(final_metrics)
        print(
            "Final metrics: epoch={epoch}, threshold={threshold:.2f}, rel_cont={rel_cont:.6g}, "
            "ssim_cont={ssim_cont:.6g}, rel_thr={rel_thr:.6g}, ssim_thr={ssim_thr:.6g}, area_ratio={area_ratio:.6g}".format(
                epoch=final_metrics["epoch"],
                threshold=final_metrics["threshold"],
                rel_cont=final_metrics["rel_error_continuous"],
                ssim_cont=final_metrics["ssim_continuous"],
                rel_thr=final_metrics["rel_error_thresholded"],
                ssim_thr=final_metrics["ssim_thresholded"],
                area_ratio=final_metrics["predicted_to_truth_area_ratio"],
            ),
            flush=True,
        )

    print(
        "Best thresholded relative error: threshold={threshold:.2f}, rel_error={rel_error:.6g}".format(
            threshold=best_row["threshold"],
            rel_error=best_row["rel_error_thresholded"],
        ),
        flush=True,
    )
    print(f"Saved diagnostics to: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
