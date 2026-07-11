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
    frequency_hz = 1.0e9
    k0 = 2.0 * math.pi * frequency_hz / 3.0e8
    data_dir = here / "data_1GHz"
    default_epsilon = here / "results_5_3_3_austria_0_3GHz_opt_v4_30000" / "checkpoint_adam_012000.pt"

    has_resume_checkpoint = "--resume-checkpoint" in sys.argv
    has_resume_epsilon = "--resume-epsilon-from" in sys.argv

    ensure_arg("--data-dir", str(data_dir))
    if not has_resume_checkpoint and not has_resume_epsilon:
        ensure_arg("--resume-epsilon-from", str(default_epsilon))
    ensure_arg("--weight-integral-data", "500")
    ensure_arg("--checkpoint-every", "500")

    print(f"frequency_hz={frequency_hz}")
    print(f"k0={k0}")
    print(f"data_dir={data_dir}")
    print("directions=+x,-x,+y,-y")
    resume_checkpoint = "None"
    resume_epsilon = "None"
    if "--resume-checkpoint" in sys.argv:
        resume_checkpoint = sys.argv[sys.argv.index("--resume-checkpoint") + 1]
    if "--resume-epsilon-from" in sys.argv:
        resume_epsilon = sys.argv[sys.argv.index("--resume-epsilon-from") + 1]
    print(f"resume_epsilon_from={resume_epsilon}")
    print(f"resume_checkpoint={resume_checkpoint}")
    if resume_checkpoint == "None":
        print("field branch reinitialized by epsilon-only resume")
    else:
        print("field branch loaded from continuation checkpoint")

    run_optimized_austria(
        default_data_dir=data_dir,
        default_output_dir="results_5_3_3_austria_1GHz_continuation",
        description=(
            "Austria 0.3GHz to 1GHz continuation. Only epsilon_branch is "
            "initialized from the 0.3GHz checkpoint; field_branch is fresh."
        ),
        frequency_hz=frequency_hz,
        output_base_dir=here,
    )
