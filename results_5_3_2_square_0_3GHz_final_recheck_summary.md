# Square 0.3GHz Final Recheck Summary

## Selected Results

Formal main result:

```text
E: square_target/results_5_3_2_square_0_3GHz_tvlep_binary_high_E_best
```

Edge-fill contrast result:

```text
G: square_target/results_5_3_2_square_0_3GHz_E_edgefill_G_tv001_high015_thr050_1k
```

## Comparison Table

| Result | Role | Continuous RE/SSIM | Best threshold RE/SSIM | Best threshold | True square min/mean/max | Center mean/max | Edge mean | Background mean | Note |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| TV+Lep 14k | thresholded structure best | 0.213694 / 0.919121 | 0.062314 / 0.994544 | 2.32 | 1.861825 / 3.155735 / 3.396869 | 3.343505 / 3.396869 | 3.093146 | 1.047581 | Best thresholded structure, but epsilon amplitude is low. |
| early metric | center-amplitude reference | 0.238185 / 0.908110 | 0.151920 / 0.967426 | 1.64 | 1.287686 / 3.333007 / 4.134728 | 4.075866 / 4.134728 | 3.085387 | 1.024656 | Center epsilon is close to 4, but it overshoots and has weak edge / whole-object structure. |
| E | current comprehensive best | 0.171311 / 0.952462 | 0.106657 / 0.983925 | 2.20 | 1.585770 / 3.465424 / 3.934785 | 3.804970 / 3.909725 | 3.352242 | 1.035583 | Current formal main result: higher square mean, center close to 4 without overshoot, stable background, best continuous metric. |
| G | edge-fill contrast | 0.175957 / 0.950525 | 0.117164 / 0.980596 | 2.10 | 1.551311 / 3.489301 / 4.094863 | 3.853086 / 4.025705 | 3.368039 | 1.029852 | Edge and square mean improve over E, but eps max exceeds 4 and continuous/thresholded metrics are worse. |

## Interpretation

1. TV+Lep 14k has the best thresholded structure, but the reconstructed epsilon amplitude is low. Its square mean is only `3.155735`, and its center max is `3.396869`.

2. The early metric result has center epsilon close to 4, but this is not a balanced reconstruction. It overshoots to `4.134728`, while the square edge remains weaker (`edge mean = 3.085387`) and the whole-object structure metrics are worse.

3. E is the current comprehensive best. It has the best continuous metric among the selected candidates (`0.171311 / 0.952462`), a high square mean (`3.465424`), a center close to 4 without overshoot (`3.804970 / 3.909725`), stable background (`1.035583`), and no obvious boundary expansion.

4. G is useful as an edge-fill contrast. It increases square mean from E's `3.465424` to `3.489301` and edge mean from `3.352242` to `3.368039`, but it also pushes epsilon above 4 (`eps max = 4.094863`) and has worse continuous and thresholded metrics. This indicates that lowering smoothing regularization can fill edges, but it introduces overshoot risk.

## Final Conclusion

Dielectric inversion quality cannot be judged only by thresholded structure. Reports must include:

- center mean / max
- true square mean
- true square edge mean
- background mean
- continuous RE / SSIM
- threshold sweep best RE / SSIM

E is the current formal main result for square 0.3GHz. G should be kept as an amplitude / edge-enhancement contrast, not as the official best.

