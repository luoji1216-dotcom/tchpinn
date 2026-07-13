# Hard-Boundary Note

## Implementation

The optional outer-taper hard-boundary field parameterization has been implemented as:

```text
Es(x, y, theta) = B(r) * Z(x, y, theta)
```

where `Z` is the raw field-branch output and `B(r)` is an outer radial taper. The option is controlled by:

```text
--field-hard-boundary outer_taper
--field-envelope-start-radius <R_start>
--field-envelope-outer-radius <R_outer>
```

The default mode remains disabled:

```text
--field-hard-boundary none
```

## Clean PDE Region

In the clean hard-boundary version, the ordinary Helmholtz PDE residual is evaluated only inside the physical region:

```text
r <= pde_physical_radius
```

When `field_hard_boundary=outer_taper` and `--pde-physical-radius` is not specified, the code defaults to:

```text
pde_physical_radius = field_envelope_start_radius
```

This avoids sampling ordinary PDE collocation points in the taper region:

```text
field_envelope_start_radius < r < field_envelope_outer_radius
```

That distinction matters because after `Es = B(r) * Z`, the Helmholtz residual in the taper region includes derivative terms from `B(r)`. Treating that region as an ordinary physical PDE region artificially penalizes the envelope rather than the field model.

## Experiment Summary

For square 0.3GHz hard-boundary experiments:

- When ordinary PDE collocation sampled into the taper region, `square max` could approach `4`, but PDE and boundary losses became much worse.
- After restricting ordinary PDE to the clean physical region, PDE dropped from about `0.872` to about `0.194`.
- After that correction, `square max` was about `2.962`, so the previous amplitude increase disappeared.
- Therefore, the earlier amplitude recovery was mainly caused by an unclean PDE/taper coupling effect. It should not be treated as a final physical solution.

## Current Recommendation

Keep hard-boundary support as a paper-structure component and optional experiment switch.

Do not use the taper-region PDE coupling as an amplitude-recovery mechanism. Epsilon amplitude recovery should instead be pursued through a separate route, such as:

- amplitude prior
- mask-amplitude parameterization
- another explicit contrast calibration mechanism

