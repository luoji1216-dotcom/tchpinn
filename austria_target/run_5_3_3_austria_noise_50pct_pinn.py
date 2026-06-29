from __future__ import annotations
import sys
from pathlib import Path

from austria_optimized_common import run_optimized_austria

if __name__ == "__main__":
    here = Path(__file__).resolve().parent
    # 强制注入0噪声命令行参数，写死噪声水平
    sys.argv.extend(["--noise-level", "0.5"])

    run_optimized_austria(
        default_data_dir=here,
        default_output_dir="results_5_3_3_austria_noise_50pct_optimized",
        description="Section 5.3.3 Austria target at 50% Gaussian noise (optimized version).",
    )