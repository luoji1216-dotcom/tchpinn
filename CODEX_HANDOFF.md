# CODEX Handoff

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

- From B12000, test `weight_integral_data = 2000 / 1000 / 500`.

## Do Not Commit

- `square_rezult/`
- `square_target/data_3GHz/`
- `square_target/diag_square_3GHz_*/`
