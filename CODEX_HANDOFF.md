# CODEX Handoff

## Square 3GHz Four-Direction Carrier-Envelope Archive

- Four-direction carrier-envelope support is implemented in
  `square_target/pinn_pixel_inverse_core.py` and exposed by
  `square_target/square_case_common.py` through
  `--field-parameterization direct|carrier_envelope`.
- Direct mode retains the existing scattered-field MLP path. Carrier mode uses
  `Es = exp(i * incident_phase_sign * k0 * d.r) * U` and the normalized
  envelope PDE. Direct-to-carrier continuation loads only the compatible
  epsilon branch when the field parameterizations differ.
- Four-direction entry point:
  `square_target/run_5_3_2_square_3ghz_fourdir_pinn.py` with
  `+x/-x/+y/-y`. The two-direction entry remains `run_5_3_2_square_3ghz_pinn.py`.
- Carrier PDE calibration uses `weight_pde=2000`; the direct PDE residual has
  a different scale and retains its original `weight_pde=0.0002`.
- Best carrier result is the initial four-direction 3000-step continuation:
  - checkpoint:
    `square_target/data_3GHz/results_5_3_2_square_3GHz_fourdir_carrier_from_directeps_freeze1k_joint2k/model_final.pt`
  - initialize epsilon from the four-direction direct checkpoint, reinitialize
    the carrier field, freeze epsilon for 1000 steps, then jointly train 2000.
  - continuous RE/SSIM: `0.364978 / 0.737389`
  - thresholded RE/SSIM: `0.453354 / 0.653491`
  - square mean/max: `2.660631 / 4.620285`; background mean: `1.119809`.
- A continuation through global step 14000 did not improve the carrier result.
  It lowered raw field/PDE/boundary losses while reducing square mean and
  degrading structure. The common checkpoint table is:
  `square_target/data_3GHz/results_5_3_2_square_3GHz_fourdir_carrier_from_directeps_freeze1k_joint2k_continue_to14k/checkpoint_evaluation_summary.csv`.
- Fixed-state diagnostics at the 3000-step carrier checkpoint show persistent
  field-epsilon compensation: with true epsilon fixed, field refit reached
  integral loss about `2.0e-3`, versus about `9.6e-4` for the low-amplitude
  joint solution. With carrier field frozen, epsilon did not fill the square;
  its integral-data epsilon-gradient square/background ratio was about `0.224`.
- Do not treat longer carrier training as an amplitude-recovery method. The
  archived four-direction carrier result is informative but remains below the
  existing 0.3GHz-to-3GHz continuation best listed below.

## Square 3GHz Continuation

- Current best result: `B prior=3e-4 at 12000`
- Checkpoint:
  - `square_target/diag_square_3GHz_fourdir_continuation_eps0p3_12000_B_prior3e4/checkpoint_adam_012000.pt`

## Metrics

- continuous RE/SSIM: `0.205448 / 0.930618`
- thresholded RE/SSIM: `0.224312 / 0.926563`
- background mean/std: `1.137205 / 0.147469`
- center mean/max: `4.004725 / 4.137980`
- edge std: `0.628064`

## Effective Method

- Initialize epsilon branch from successful 0.3GHz square checkpoint.
- Reinitialize field branch for 3GHz.
- Freeze epsilon early.
- Continue with small epsilon learning rate.
- Keep epsilon prior enabled.

## Not Recommended

- Lep edge-preserving regularization.
- Contrast L1.
- Binary push.
- Background anchor.
- Full observation points instead of the 1200-point per-direction sampling.

## Next Step

- Austria 1GHz direct current best: `B53000`.
  - Archive: `rezult/austria_1GHz_direct_B53000_best/`
  - Checkpoint: `austria_target/results_5_3_3_austria_1GHz_direct_lineB_detach52000_detach_to60000/checkpoint_adam_053000.pt`
  - `mode=detach_field`
  - `weight_pde=0.006`
  - `weight_integral_data=50`
  - `weight_contrast_l1=8e-2`
  - `weight_tv=0.001`
  - `lr=1e-6`
  - continuous RE/SSIM: `0.393698 / 0.546493`
  - thresholded RE/SSIM: `0.479391 / 0.488895`
  - This is the current best direct single-frequency checkpoint by structure.
  - Later direct refinements did not exceed B53000:
    - `B53000 -> 57000` amplitude/integral/L1 adjustments.
    - Structure-first low-integral plus strong-L1 runs.
    - Epsilon-only sharpen from B53000.
    - Phase-binary prior.
    - Pseudo separation using B53000 high-confidence cores.
    - Pixel-epsilon direct.
    - Learnable per-direction integral alpha.
  - Pixel-epsilon direct failed to recover Austria structure, so the failure is not mainly epsilon-MLP expressivity.
  - Learnable integral alpha barely moved from `1+0i` and did not improve structure.
  - Current direct judgment:
    - Four-direction 1GHz direct can recover the Austria large outline.
    - It does not reliably separate the two upper circles, lower ring, and left/right/lower artifacts.
    - More small tuning of `lr`, L1, TV, or integral weight is not worth continuing.
- Stop Austria 1GHz single-frequency direct inversion.
- Diagnostics show this is not a simple parameter issue:
  - At 1GHz, true Austria epsilon is better than background, but worse than the trained pseudo solution under the current coupled integral/operator loss.
  - The single-frequency direct setup lets the field branch and epsilon adapt to a non-physical solution.
- Do not continue 1GHz single-frequency direct.
- Stop ordinary Austria 1GHz continuation long training.
- Current recommended Austria 1GHz continuation result: `C12000`.
  - Result archive: `rezult/austria_1GHz_continuation_C12000_best/`
  - Checkpoint: `austria_target/results_5_3_3_austria_1GHz_continuation_c9000_to12000_C_int25_eps2e7/checkpoint_adam_012000.pt`
  - `weight_integral_data=25`
  - `field_lr=2e-5`
  - `epsilon_lr=2e-7`
  - `epsilon_prior_weight=1e-3`
  - continuous RE/SSIM: `0.319877 / 0.757427`
  - thresholded RE/SSIM: `0.353353 / 0.776654`
- Backup Austria 1GHz continuation result: `C9000`.
  - Checkpoint: `austria_target/results_5_3_3_austria_1GHz_continuation_c6000_to9000_C_halflr_int50/checkpoint_adam_009000.pt`
  - `weight_integral_data=50`
  - `field_lr=5e-5`
  - `epsilon_lr=5e-7`
  - continuous RE/SSIM: `0.322231 / 0.759463`
  - thresholded RE/SSIM: `0.355710 / 0.775904`
- Austria 1GHz conclusion:
  - Austria 1GHz random/direct single-frequency training fails or produces pseudo-solutions.
  - 0.3GHz -> 1GHz continuation is clearly more stable.
  - 1GHz fine-tuning can only weakly act on epsilon.
  - The most stable strategy is long epsilon freeze, tiny epsilon learning rate, strong epsilon prior, and low integral weight.
  - C12000 slightly improves metrics over C9000, but the structure is slightly smoother.
  - Do not continue ordinary long training to 15000 without a new strategy.
- Austria 1GHz operator/data alignment diagnostics completed today:
  - `data_1GHz` `Fieldz` is total-field-like, not pure scattered field.
  - Fixed incident subtraction with amplitude `0.1` is inaccurate.
  - Least-squares fitted incident amplitudes are approximately:
    - `+x`: `0.0883 - 0.0013j`
    - `-x`: `0.0885 - 0.0013j`
    - `+y`: `0.0870 - 0.0010j`
    - `-y`: `0.0894 + 0.0055j`
  - Added `--incident-amplitude-mode fixed|fit_per_direction`.
    - Default is `fixed`, preserving old behavior.
    - `fit_per_direction` estimates one complex incident amplitude per direction and is worth keeping as a correctness option.
  - 3000-step random-direct smoke tests:
    - `fixed=0.1`: final continuous RE/SSIM `0.556054 / 0.005236`; epsilon stayed at background.
    - `fit_per_direction`: final continuous RE/SSIM `0.556054 / 0.005236`; epsilon also stayed at background.
    - `fit_per_direction` slightly reduced integral loss but did not change the reconstruction meaningfully.
  - Operator convention sweep:
    - Recommended convention remains:
      - `Einc = exp(+i k d·r)`
      - `G = +i/4 H0`
      - `source = +k0^2`
      - `contrast = epsilon - 1`
    - No simple sign/convention combination made true epsilon outperform B53000.
  - Fixed-epsilon field refit:
    - Even after freezing true epsilon and retraining only the field branch, true epsilon still underperformed the B53000 pseudo solution.
    - This indicates the current 1GHz loss/operator favors a field-epsilon pseudo self-consistent solution.
- Current diagnosis:
  - Incident amplitude mismatch is a real issue, but not the main cause of 1GHz random-direct failure.
  - More likely causes:
    - CST Austria geometry and code truth geometry are not exactly the same.
    - The 2D volume-integral operator does not match the CST 1GHz simulation model.
    - The learned field branch gives a field-epsilon pseudo-solution bypass.
- Tomorrow priority:
  - Do not train first.
  - Start new session by reading `CODEX_HANDOFF.md` and `git status`.
  - Build `austria_target/diagnose_1ghz_geometry_sweep.py`.
  - Sweep small changes to the code true Austria geometry and check whether true/operator loss can approach or beat B53000.

## Do Not Commit

- `square_rezult/`
- `square_target/data_3GHz/`
- `square_target/diag_square_3GHz_*/`
- `austria_rezult/`
- `austria_target/data_1GHz/`
- `austria_target/results_*/`
