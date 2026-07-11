from __future__ import annotations

from pathlib import Path

from austria_optimized_common import run_optimized_austria


if __name__ == "__main__":
    here = Path(__file__).resolve().parent
    run_optimized_austria(
        default_data_dir=here / "data_1GHz",
        default_output_dir="results_5_3_3_austria_1GHz_optimized",
        description=(
            "Optimized double-branch PINN for the Austria target at 1 GHz. "
            "The default convention is Ez=Re-iIm with phase_sign=+1."
        ),
        frequency_hz=1.0e9,
        output_base_dir=here,
    )
