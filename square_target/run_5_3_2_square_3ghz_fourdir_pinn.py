from __future__ import annotations

from pathlib import Path

from square_case_common import run_square_case


if __name__ == "__main__":
    here = Path(__file__).resolve().parent
    run_square_case(
        frequency_hz=3.0e9,
        default_data_dir=here / "data_3GHz",
        default_output_dir="results_5_3_2_square_3GHz_fourdir_metric",
        description=(
            "Section 5.3.2 square target at 3 GHz with +x, -x, +y, and -y "
            "from ./data_3GHz or from --data-dir."
        ),
        direction_labels=("+x", "-x", "+y", "-y"),
    )
