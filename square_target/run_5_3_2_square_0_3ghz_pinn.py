from __future__ import annotations

from pathlib import Path

from square_case_common import run_square_case


if __name__ == "__main__":
    here = Path(__file__).resolve().parent
    run_square_case(
        frequency_hz=0.3e9,
        default_data_dir=here,
        default_output_dir="results_5_3_2_square_0_3GHz",
        description=(
            "Section 5.3.2 square target at 0.3 GHz. Uses current FEM files "
            "+x.txt and -x.txt. The optimized default convention is Ez=Re-iIm "
            "with phase_sign=+1."
        ),
    )
