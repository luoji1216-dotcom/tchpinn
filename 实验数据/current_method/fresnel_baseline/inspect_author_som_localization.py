from __future__ import annotations

"""GT-free localization summary for an already completed author SOM run."""

import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


OUTPUT_DIR = Path(__file__).resolve().parent / "results_author_som_1GHz_difference_quick"


def main() -> None:
    epsilon = np.load(OUTPUT_DIR / "som_epsilon_reconstruction.npy")
    x = np.load(OUTPUT_DIR / "som_x_grid.npy")
    y = np.load(OUTPUT_DIR / "som_y_grid.npy")
    contrast = np.maximum(epsilon - 1.0, 0.0)
    total_contrast = float(contrast.sum())
    if total_contrast <= 0.0:
        raise ValueError("SOM reconstruction has no positive contrast.")
    peak_rc = np.unravel_index(int(np.argmax(epsilon)), epsilon.shape)
    peak_xy = [float(x[peak_rc]), float(y[peak_rc])]
    centroid = [
        float((contrast * x).sum() / total_contrast),
        float((contrast * y).sum() / total_contrast),
    ]
    strong = contrast >= 0.5 * float(contrast.max())
    strong_weights = contrast * strong
    strong_total = float(strong_weights.sum())
    strong_centroid = [
        float((strong_weights * x).sum() / strong_total),
        float((strong_weights * y).sum() / strong_total),
    ]
    metrics = json.loads((OUTPUT_DIR / "som_metrics.json").read_text(encoding="utf-8"))
    summary = {
        "scope": "Post-processing of author SOM output only. No ground truth is read or used.",
        "elapsed_s_from_author_run": metrics["elapsed_s"],
        "epsilon_min": float(epsilon.min()),
        "epsilon_max": float(epsilon.max()),
        "epsilon_mean": float(epsilon.mean()),
        "epsilon_std": float(epsilon.std()),
        "contrast_centroid_m": centroid,
        "high_contrast_centroid_m": strong_centroid,
        "peak_m": peak_xy,
        "positive_contrast_fraction_at_negative_y": float(contrast[y < 0.0].sum() / total_contrast),
        "peak_is_negative_y": bool(peak_xy[1] < 0.0),
        "contrast_centroid_is_negative_y": bool(centroid[1] < 0.0),
    }
    (OUTPUT_DIR / "som_gt_free_localization.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.6), constrained_layout=True)
    extent = [float(x.min()), float(x.max()), float(y.min()), float(y.max())]
    panels = ((epsilon, "Continuous epsilon", 1.0, 4.0), (contrast, "Continuous contrast", 0.0, 3.0))
    for ax, (field, title, vmin, vmax) in zip(axes, panels):
        image = ax.imshow(field, extent=extent, origin="lower", cmap="jet", vmin=vmin, vmax=vmax, interpolation="bilinear")
        ax.scatter([peak_xy[0]], [peak_xy[1]], marker="x", c="white", s=45, linewidths=1.2, label="peak")
        ax.scatter([centroid[0]], [centroid[1]], marker="+", c="white", s=65, linewidths=1.5, label="contrast centroid")
        ax.set_aspect("equal")
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_title(title)
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    axes[0].legend(loc="upper right", fontsize=8)
    fig.savefig(OUTPUT_DIR / "som_continuous_epsilon_contrast.png", dpi=220)
    plt.close(fig)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
