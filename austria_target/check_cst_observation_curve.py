from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CORE_ROOT = ROOT / "square_target"
if str(CORE_ROOT) not in sys.path:
    sys.path.insert(0, str(CORE_ROOT))

from pinn_pixel_inverse_core import load_fem_table, parse_direction_from_name  # noqa: E402


HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data_1GHz"
DIRECTIONS = ("+x", "-x", "+y", "-y")
FREQUENCY_HZ = 1.0e9
C0 = 3.0e8
K0 = 2.0 * math.pi * FREQUENCY_HZ / C0


def angle_span_deg(theta: np.ndarray) -> tuple[float, float, float, float]:
    theta_mod = np.mod(theta, 2.0 * math.pi)
    theta_sorted = np.sort(theta_mod)
    gaps = np.diff(np.concatenate([theta_sorted, theta_sorted[:1] + 2.0 * math.pi]))
    largest_gap = float(np.max(gaps))
    coverage = 2.0 * math.pi - largest_gap
    return (
        float(np.rad2deg(theta_sorted[0])),
        float(np.rad2deg(theta_sorted[-1])),
        float(np.rad2deg(coverage)),
        float(np.rad2deg(largest_gap)),
    )


def fit_incident(xy: np.ndarray, field: np.ndarray, direction: tuple[float, float]) -> complex:
    dvec = np.asarray(direction, dtype=np.float64)
    phase = np.exp(1j * K0 * (xy @ dvec))
    return complex(np.vdot(phase, field) / np.vdot(phase, phase))


def print_direction(label: str, xy: np.ndarray, field: np.ndarray, amplitude: complex) -> dict[str, float | str]:
    radius = np.linalg.norm(xy, axis=1)
    theta = np.arctan2(xy[:, 1], xy[:, 0])
    angle_min, angle_max, coverage, largest_gap = angle_span_deg(theta)
    full_circle = coverage >= 350.0
    abs_field = np.abs(field)
    row: dict[str, float | str] = {
        "direction": label,
        "n": int(xy.shape[0]),
        "x_min": float(xy[:, 0].min()),
        "x_max": float(xy[:, 0].max()),
        "y_min": float(xy[:, 1].min()),
        "y_max": float(xy[:, 1].max()),
        "radius_min": float(radius.min()),
        "radius_max": float(radius.max()),
        "radius_mean": float(radius.mean()),
        "radius_std": float(radius.std()),
        "angle_min_deg": angle_min,
        "angle_max_deg": angle_max,
        "angle_coverage_deg": coverage,
        "largest_angle_gap_deg": largest_gap,
        "full_360": str(full_circle),
        "field_abs_min": float(abs_field.min()),
        "field_abs_max": float(abs_field.max()),
        "field_abs_mean": float(abs_field.mean()),
        "fit_amp_real": float(amplitude.real),
        "fit_amp_imag": float(amplitude.imag),
        "fit_amp_abs": float(abs(amplitude)),
    }
    print(
        f"{label}: n={row['n']} "
        f"x=[{row['x_min']:.8g},{row['x_max']:.8g}] "
        f"y=[{row['y_min']:.8g},{row['y_max']:.8g}] "
        f"radius min/max/mean/std="
        f"{row['radius_min']:.8g}/{row['radius_max']:.8g}/"
        f"{row['radius_mean']:.8g}/{row['radius_std']:.8g} "
        f"angle deg min/max/coverage/gap="
        f"{row['angle_min_deg']:.3f}/{row['angle_max_deg']:.3f}/"
        f"{row['angle_coverage_deg']:.3f}/{row['largest_angle_gap_deg']:.3f} "
        f"full_360={row['full_360']} "
        f"|Fieldz| min/max/mean="
        f"{row['field_abs_min']:.8g}/{row['field_abs_max']:.8g}/{row['field_abs_mean']:.8g} "
        f"A={amplitude.real:+.8e}{amplitude.imag:+.8e}j |A|={abs(amplitude):.8e}",
        flush=True,
    )
    return row


def save_plot(rows_xy: dict[str, np.ndarray], out_path: Path) -> None:
    import matplotlib as mpl

    mpl.use("Agg", force=True)
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    ax_xy = axes[0, 0]
    ax_r_index = axes[0, 1]
    ax_r_angle = axes[1, 0]
    ax_hist = axes[1, 1]

    for label, xy in rows_xy.items():
        radius = np.linalg.norm(xy, axis=1)
        theta_deg = np.rad2deg(np.mod(np.arctan2(xy[:, 1], xy[:, 0]), 2.0 * math.pi))
        ax_xy.scatter(xy[:, 0], xy[:, 1], s=6, label=label, alpha=0.75)
        ax_r_index.plot(radius, label=label, linewidth=1.0)
        ax_r_angle.scatter(theta_deg, radius, s=5, label=label, alpha=0.7)
        ax_hist.hist(radius, bins=40, alpha=0.45, label=label)

    ax_xy.set_title("receiver points")
    ax_xy.set_xlabel("x (m)")
    ax_xy.set_ylabel("y (m)")
    ax_xy.set_aspect("equal", adjustable="box")
    ax_xy.legend()

    ax_r_index.set_title("radius by row index")
    ax_r_index.set_xlabel("row index")
    ax_r_index.set_ylabel("radius (m)")
    ax_r_index.legend()

    ax_r_angle.set_title("radius by angle")
    ax_r_angle.set_xlabel("angle (deg)")
    ax_r_angle.set_ylabel("radius (m)")
    ax_r_angle.set_xlim(0.0, 360.0)
    ax_r_angle.legend()

    ax_hist.set_title("radius histogram")
    ax_hist.set_xlabel("radius (m)")
    ax_hist.set_ylabel("count")
    ax_hist.legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> None:
    print(f"data_dir={DATA_DIR.resolve()}")
    print("Fieldz convention: load_fem_table(..., imag_sign=-1.0), Ez = col8 - i col9")
    print(f"incident fit: A * exp(+i k d dot r), frequency_hz={FREQUENCY_HZ:.8e}, k0={K0:.8e}")
    rows_xy: dict[str, np.ndarray] = {}
    for label in DIRECTIONS:
        path = DATA_DIR / f"{label}.txt"
        parsed, direction = parse_direction_from_name(path)
        xy, field = load_fem_table(path, imag_sign=-1.0)
        amp = fit_incident(xy, field, direction)
        rows_xy[parsed] = xy
        print_direction(parsed, xy, field, amp)
    out_path = HERE / "cst_observation_curve_check.png"
    save_plot(rows_xy, out_path)
    print(f"wrote={out_path}")


if __name__ == "__main__":
    main()
