from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


EXPECTED = [
    ("X", "real:cartesian"),
    ("Y", "real:cartesian"),
    ("Z", "real:cartesian"),
    ("e-field (f=3) [pw], x", "real:cartesian"),
    ("e-field (f=3) [pw], x", "imag:cartesian"),
    ("e-field (f=3) [pw], y", "real:cartesian"),
    ("e-field (f=3) [pw], y", "imag:cartesian"),
    ("e-field (f=3) [pw], z", "real:cartesian"),
    ("e-field (f=3) [pw], z", "imag:cartesian"),
]


def parse_cst_xy(path: Path) -> tuple[dict[str, str], np.ndarray]:
    metadata: dict[str, str] = {}
    rows: list[tuple[float, float]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            metadata[key.strip()] = value.strip()
            continue
        stripped = line.strip()
        if not stripped or stripped[0] not in "+-.0123456789":
            continue
        parts = stripped.split()
        if len(parts) < 2:
            continue
        try:
            rows.append((float(parts[0]), float(parts[1])))
        except ValueError:
            continue

    data = np.asarray(rows, dtype=np.float64)
    if data.ndim != 2 or data.shape[1] != 2 or data.shape[0] == 0:
        raise ValueError(f"{path} does not contain CST XY numeric data.")
    declared = int(metadata.get("Npoints", "-1"))
    if declared != data.shape[0]:
        raise ValueError(f"{path} declares Npoints={declared}, parsed {data.shape[0]} rows.")
    return metadata, data


def merge_direction(input_dir: Path, output_path: Path) -> int:
    curves: list[np.ndarray] = []
    for idx, (expected_label, expected_result) in enumerate(EXPECTED, start=1):
        path = input_dir / f"{idx}.txt"
        if not path.exists():
            raise FileNotFoundError(path)
        metadata, data = parse_cst_xy(path)
        label = metadata.get("Curvelabel")
        result_type = metadata.get("Result type")
        if label != expected_label or result_type != expected_result:
            raise ValueError(
                f"{path} has Curvelabel={label!r}, Result type={result_type!r}; "
                f"expected {expected_label!r}, {expected_result!r}."
            )
        curves.append(data)

    reference = curves[0][:, 0]
    for idx, curve in enumerate(curves[1:], start=2):
        if curve.shape != curves[0].shape:
            raise ValueError(f"{input_dir / f'{idx}.txt'} shape differs from 1.txt.")
        if not np.allclose(curve[:, 0], reference, rtol=0.0, atol=1.0e-12):
            raise ValueError(f"{input_dir / f'{idx}.txt'} length parameter column differs from 1.txt.")

    merged = np.column_stack([curve[:, 1] for curve in curves])
    if not np.isfinite(merged[:, 7]).all() or not np.isfinite(merged[:, 8]).all():
        raise ValueError(f"{input_dir} contains non-finite FieldzRe/FieldzIm values.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    header = "Prepared square 3 GHz CST data\nx y z FieldxRe FieldxIm FieldyRe FieldyIm FieldzRe FieldzIm"
    np.savetxt(output_path, merged, fmt="%.15g", header=header, comments="# ")
    return int(merged.shape[0])


def default_source_root(repo_root: Path) -> Path:
    preferred = repo_root / "square_result"
    fallback = repo_root / "square_rezult"
    if preferred.exists():
        return preferred
    return fallback


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Prepare square 3 GHz CST exports for PINN training.")
    parser.add_argument("--source-root", type=Path, default=default_source_root(repo_root))
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "data_3GHz")
    args = parser.parse_args()

    source_root = args.source_root
    counts = {
        "+x": merge_direction(source_root / "3_+x", args.output_dir / "+x.txt"),
        "-x": merge_direction(source_root / "3_-x", args.output_dir / "-x.txt"),
        "+y": merge_direction(source_root / "3_+y", args.output_dir / "+y.txt"),
        "-y": merge_direction(source_root / "3_-y", args.output_dir / "-y.txt"),
    }
    for label, count in counts.items():
        print(f"{label}: {count} points")


if __name__ == "__main__":
    main()
