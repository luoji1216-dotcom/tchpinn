from __future__ import annotations

from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RAW_ROOT = ROOT / "austria_rezult"
OUT_DIR = Path(__file__).resolve().parent / "data_1GHz"

DIRECTIONS = {
    "+x": "1_+x",
    "-x": "1_-x",
    "+y": "1_+y",
    "-y": "1_-y",
}

HEADER = (
    "            x [m]            y [m]            z [m]   FieldxRe [V/m]   "
    "FieldxIm [V/m]   FieldyRe [V/m]   FieldyIm [V/m]   FieldzRe [V/m]   "
    "FieldzIm [V/m]\n"
    "---------------------------------------------------------------------------------------------------------------------------------------------------------"
)


def load_curve(path: Path) -> np.ndarray:
    rows: list[tuple[float, float]] = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        parts = line.split()
        if len(parts) != 2:
            continue
        try:
            rows.append((float(parts[0]), float(parts[1])))
        except ValueError:
            continue
    if not rows:
        raise ValueError(f"No numeric curve rows found in {path}")
    return np.asarray(rows, dtype=np.float64)


def build_direction(raw_dir: Path) -> np.ndarray:
    curves = [load_curve(raw_dir / f"{idx}.txt") for idx in range(1, 10)]
    reference_s = curves[0][:, 0]
    for idx, curve in enumerate(curves[1:], start=2):
        if curve.shape != curves[0].shape:
            raise ValueError(f"Curve {idx} shape {curve.shape} differs from curve 1 {curves[0].shape}")
        if not np.allclose(curve[:, 0], reference_s, rtol=0.0, atol=1e-8):
            raise ValueError(f"Curve {idx} arc-length coordinates differ from curve 1")
    return np.column_stack([curve[:, 1] for curve in curves])


def summarize(label: str, data: np.ndarray) -> None:
    xy = data[:, :2]
    z = data[:, 2]
    ez_re = data[:, 7]
    ez_im = data[:, 8]
    print(
        f"{label}: n={data.shape[0]}, "
        f"x=[{xy[:, 0].min():.6g}, {xy[:, 0].max():.6g}], "
        f"y=[{xy[:, 1].min():.6g}, {xy[:, 1].max():.6g}], "
        f"max|z|={np.abs(z).max():.3g}, "
        f"FieldzRe mean/std=[{ez_re.mean():.6g}, {ez_re.std():.6g}], "
        f"FieldzIm mean/std=[{ez_im.mean():.6g}, {ez_im.std():.6g}], "
        f"Fieldz abs max={np.hypot(ez_re, ez_im).max():.6g}"
    )


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for label, dirname in DIRECTIONS.items():
        raw_dir = RAW_ROOT / dirname
        if not raw_dir.exists():
            raise FileNotFoundError(raw_dir)
        data = build_direction(raw_dir)
        out_path = OUT_DIR / f"{label}.txt"
        np.savetxt(out_path, data, header=HEADER, comments="", fmt="% .10e")
        summarize(label, data)
        print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
