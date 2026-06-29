# Square Target Reproduction

This repository is a focused extraction for reproducing the square-target part of thesis Section 5.3.2.

Only the square-target material is included:

- `paper_chapter5_square.pdf`: thesis chapter excerpt used as the reference.
- `square_target/+x.txt` and `square_target/-x.txt`: FEM total-field observations.
- `square_target/run_5_3_2_square_0_3ghz_pinn.py`: double-branch PINN entry point for the 0.3 GHz square case.
- `square_target/pinn_pixel_inverse_core.py`: double-branch PINN implementation.
- `square_target/run_5_3_2_square_som.py` and `square_target/som_inverse_core.py`: SOM baseline implementation.

The original single-branch PINN baseline is not included here.

## Current Finding

The thesis reports for the 0.3 GHz square target:

| Method | Relative error | SSIM |
| --- | ---: | ---: |
| SOM | 0.1377 | 0.928 |
| Double-branch PINN | 0.1292 | 0.944 |

With the current double-branch PINN code, pure Adam training does not yet reproduce the reported `0.1292` continuous-map error. The best continuous-output checkpoints observed so far are around:

| Run | Relative error | SSIM |
| --- | ---: | ---: |
| Adam, old small-network config, 10000 steps | about 0.229 | about 0.912 |
| Adam, current 30000-step run, best checkpoint around 14000 steps | about 0.229 | about 0.913 |

After thresholding the reconstructed permittivity map into background/object values, the geometric error can drop to about `0.156`, which is much closer to the thesis table. This suggests the thesis result may involve stronger binary/edge priors, post-processing, or undocumented settings.

## Environment

The code was tested with:

- Python environment: `T:\anaconda3\envs\py\python.exe`
- PyTorch CUDA available

Install dependencies:

```powershell
pip install -r requirements.txt
```

## Run Double-Branch PINN

From the repository root:

```powershell
T:\anaconda3\envs\py\python.exe .\square_target\run_5_3_2_square_0_3ghz_pinn.py --device cuda --output-dir results_square_0_3GHz_adam
```

Useful quick test:

```powershell
T:\anaconda3\envs\py\python.exe .\square_target\run_5_3_2_square_0_3ghz_pinn.py --device cuda --epochs 300 --checkpoint-every 0 --output-dir results_quick_check
```

Imaginary-part convention diagnostic:

```powershell
T:\anaconda3\envs\py\python.exe .\square_target\run_5_3_2_square_0_3ghz_pinn.py --diagnose-imag-sign
```

Current convention used by default:

```text
Ez = FieldzRe - i * FieldzIm
phase_sign = +1
```

## Run SOM

```powershell
T:\anaconda3\envs\py\python.exe .\square_target\run_5_3_2_square_som.py --output-dir results_square_0_3GHz_som
```

## Notes For Further Work

The raw continuous PINN output tends to recover the target location and high-permittivity interior, but the square edges remain too diffuse. The main open question is whether the thesis metric was computed on the continuous output, a thresholded output, or an output using additional binary/edge constraints.

Likely next experiments:

1. Add an explicit binary contrast prior on the epsilon branch.
2. Track best validation/error checkpoint instead of assuming 30000 Adam steps is optimal.
3. Compare continuous-map and thresholded-map metrics side by side for every checkpoint.
