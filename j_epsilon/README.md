# Physics-informed J-epsilon dual-branch inversion

Formal Austria 0.3 GHz implementation with:

- a direction-dependent complex contrast-current branch `J(x,y,direction)`;
- an angle-independent bounded epsilon branch;
- `Es = GS @ J`;
- `Etot = Einc + GD @ J`;
- normalized receiver-data, state, and fourth-order finite-difference PDE
  losses;
- BP current initialization;
- strict alternating J and epsilon optimization.

The committed formal configuration is the selected no-sparsity baseline
(`contrast_sparsity_weight = 0`).

Minimal startup check:

```powershell
python -B j_epsilon/run_formal_austria.py --dry-run
```

Training expects the six-direction CST tables outside the repository by
default at `../data_austra/0.3GHz_six_direction`. Override with `--data-dir`.
