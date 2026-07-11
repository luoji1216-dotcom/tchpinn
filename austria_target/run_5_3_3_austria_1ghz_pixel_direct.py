from __future__ import annotations

import sys
from pathlib import Path

from austria_optimized_common import run_optimized_austria


def ensure_default_flag(flag: str, value: str | None = None) -> None:
    if any(arg == flag or arg.startswith(f"{flag}=") for arg in sys.argv[1:]):
        return
    sys.argv.append(flag)
    if value is not None:
        sys.argv.append(value)


if __name__ == "__main__":
    ensure_default_flag("--epsilon-parameterization", "pixel")
    ensure_default_flag("--eps-max", "3.0")
    ensure_default_flag("--pixel-grid-size", "128")
    ensure_default_flag("--output-dir", "results_5_3_3_austria_1GHz_pixel_direct")
    here = Path(__file__).resolve().parent
    run_optimized_austria(
        default_data_dir=here / "data_1GHz",
        default_output_dir="results_5_3_3_austria_1GHz_pixel_direct",
        description=(
            "Pixel-epsilon double-branch PINN for direct Austria target inversion at 1 GHz. "
            "The field branch remains the standard neural branch; epsilon is a trainable grid."
        ),
        frequency_hz=1.0e9,
        output_base_dir=here,
    )
