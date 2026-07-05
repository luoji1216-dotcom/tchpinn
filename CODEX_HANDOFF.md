# CODEX Handoff

## Repository

- Path: `T:\work\tchpinn\square-target-repro`
- Current HEAD: `d31b5a2 Add Austria optimized resume checkpoint option`
- Current branch status: `main...origin/main`
- Current working tree summary:
  - Modified: `square_target/run_5_3_2_square_3ghz_pinn.py`
  - Untracked raw/data/result items include `square_rezult/`, `square_target/data_3GHz*/`, `square_target/diag_square_3GHz*/`, and `square_target/prepare_square_3ghz_data.py`
  - Do not commit generated training results unless explicitly requested.

## Square 0.3GHz

- Status: reproduced successfully.
- Best checkpoint:
  - `square_target/results_5_3_2_square_0_3GHz_metric/checkpoint_adam_014000.pt`
- Continuous reconstruction:
  - RE ~= `0.23046`
  - SSIM ~= `0.90973`
- Thresholded reconstruction, threshold `1.85`, for visualization/structure analysis only:
  - RE ~= `0.13596`
  - SSIM ~= `0.97394`
- Evaluation rule: table/main metrics should use continuous epsilon reconstruction. Thresholded metrics are not paper-table metrics.

## Square 3GHz

- Current data:
  - `square_target/data_3GHz/+x.txt`
  - `square_target/data_3GHz/-x.txt`
- Current setup uses only two incident directions: `+x` and `-x`.
- Data preparation source:
  - raw CST export directory: `square_rezult/3_+x/` and `square_rezult/3_-x/`
  - each direction had `1.txt` to `9.txt`
- Data diagnosis conclusion:
  - 0.3GHz and 3GHz both map to the same 9-column field format: `x y z FieldxRe FieldxIm FieldyRe FieldyIm FieldzRe FieldzIm`
  - observation circle radius and coordinate range match, about `0.75 m`
  - 3GHz has denser sampling, `3341` points per direction versus `836` for 0.3GHz
  - `+x/-x` direction mapping is consistent
  - `observation_imag_sign=-1` and `phase_sign=+1` are consistent with plane-wave direction diagnostics
  - `+x` and mirrored `-x` fields satisfy square symmetry very well
  - raw CST header indicates `e-field (f=3) [pw], z`, so there is no obvious frequency/field-type export error
- Failure conclusion:
  - The 3GHz two-direction failure is more likely due to insufficient inverse constraints at high frequency, not corrupted data.
- Best-but-unusable result:
  - sampling: C group, left/right main arcs, about `836` points per direction
  - `weight_pde = baseline / 100 = 0.0002`
  - `weight_integral_data = 3000`
  - `weight_boundary = 0.02`
  - `lr = 1e-4`
  - step `3000`
  - continuous RE ~= `0.5223`
  - continuous SSIM ~= `0.3360`
  - thresholded RE ~= `0.6152`
  - thresholded SSIM ~= `0.2113`
- Visual conclusion:
  - Image is still poor and does not recover a usable square structure.
  - Continuing the same setup to 6000 steps made metrics and thresholded structure worse.
- Recommendation:
  - Do not continue two-direction small hyperparameter sweeps or long training.
  - Either add `+y/-y` and run a four-direction 3GHz check, or stop the square 3GHz branch.

## Austria 0.3GHz

- Current recommended result: optimized baseline at 12k.
- Recommended checkpoint:
  - `austria_target/results_5_3_3_austria_0_3GHz_opt_v4_30000/checkpoint_adam_012000.pt`
- Metrics:
  - continuous RE ~= `0.3311`
  - continuous SSIM ~= `0.7596`
- Visual conclusion:
  - Austria structure is recovered, but the two upper circles and lower ring are connected/sticky.
- Tried but did not beat optimized baseline:
  - lowering `weight_tv`
  - lowering `weight_integral_data`
  - resume fine-tuning with `weight_contrast_l1`
  - binary epsilon prior
  - epsilon-network Fourier/high-frequency capacity changes
  - `paper_faithful` loss experiments
- Keep the Austria 0.3GHz result as a non-paper experiment unless the data provenance changes.

## Austria Next Step

- Priority: regenerate Austria 1GHz CST data with four directions `+x`, `-x`, `+y`, `-y` if the goal is paper Table 5.2 reproduction.
- Continuing 0.3GHz Austria small tuning is unlikely to pay off.

## Do Not Commit

- Raw CST export directories, especially `square_rezult/`
- Failed experiment result directories
- `square_target/diag_square_3GHz*/`
- `square_target/data_3GHz_*` temporary sampled data directories
- Checkpoints, `.pt`, `.pth`, `.npy`, `__pycache__`, and result folders

## Possible Future Commits

- Already committed and worth keeping:
  - Austria optimized `--resume-checkpoint`
- Consider only after successful four-direction validation:
  - `square_target/prepare_square_3ghz_data.py`
  - `square_target/run_5_3_2_square_3ghz_pinn.py`
  - cleaned 3GHz data-preparation documentation

