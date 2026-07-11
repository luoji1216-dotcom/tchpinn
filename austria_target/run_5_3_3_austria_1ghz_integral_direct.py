from __future__ import annotations

import math
import sys
from pathlib import Path

from austria_optimized_common import run_optimized_austria


def ensure_arg(name: str, value: str) -> None:
    if name not in sys.argv:
        sys.argv.extend([name, value])


if __name__ == "__main__":
    here = Path(__file__).resolve().parent
    data_dir = here / "data_1GHz"
    frequency_hz = 1.0e9
    k0 = 2.0 * math.pi * frequency_hz / 3.0e8

    ensure_arg("--lr", "4e-4")
    ensure_arg("--weight-data", "0.0")
    ensure_arg("--eps-initial", "1.5")
    ensure_arg("--checkpoint-every", "500")

    print(f"frequency_hz={frequency_hz}")
    print(f"k0={k0}")
    print(f"data_dir={data_dir.resolve()}")
    print("directions=+x,-x,+y,-y")
    print("resume_checkpoint=None")
    print("resume_epsilon_from=None")
    print("ordinary_field_data_weight default=0.0")

    run_optimized_austria(
        default_data_dir=data_dir,
        default_output_dir="results_5_3_3_austria_1GHz_integral_direct",
        description=(
            "Integral-direct double-branch PINN for the Austria target at 1 GHz. "
            "The ordinary field data loss is disabled by default so epsilon is "
            "mainly constrained by the volume integral data loss."
        ),
        frequency_hz=frequency_hz,
        output_base_dir=here,
    )
