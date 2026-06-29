from __future__ import annotations
import sys
from pathlib import Path

from austria_optimized_common import run_optimized_austria

if __name__ == "__main__":
    here = Path(__file__).resolve().parent
    # 批量注入命令行参数
    sys.argv.extend([
        "--noise-level", "0.0",          # 0噪声
        "--weight-tv", "0.0001",         # TV正则权重
        "--epochs", "3000"              # 迭代步数，修改这里的数字即可调整
    ])

    run_optimized_austria(
        default_data_dir=here,
        default_output_dir="results_5_3_3_austria_noise_0pct_optimized",
        description="Section 5.3.3 Austria target at 0% Gaussian noise (optimized version).",
    )