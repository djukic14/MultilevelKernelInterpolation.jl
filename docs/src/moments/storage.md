# Moment traits and storage

A **moment collection** stores the integrated matrices. A **moment** stores
the mutable coefficients used during multiplication. They are different
objects with different lifetimes.

| Object | Purpose |
| :--- | :--- |
| `MultilevelKernelInterpolation.TestMomentCollection`, `MultilevelKernelInterpolation.TrialMomentCollection` | Read-only leaf matrices in the core |
| `MultilevelKernelInterpolation.PolynomialMoment` | Mutable coefficient arrays in a workspace |
| `MultilevelKernelInterpolation.ToeplitzPolynomialMoment` | Ordinary coefficients plus padded FFT arrays |
| `MultilevelKernelInterpolation.BsplineDensePolynomialMoment` | Raw coefficients plus reduced-basis scratch, without FFTs |

The component count is not the spatial dimension. For example, a scalar
moment has one component even on a 3-D mesh; a divergence-augmented vector
moment has four.

## Which query selects what?

- `MultilevelKernelInterpolation.testmoment` and `MultilevelKernelInterpolation.trialmoment` select the integrand for each side.
- `MultilevelKernelInterpolation.i2otranslation(op)` selects how the kernel contracts those components.
- `MultilevelKernelInterpolation.defaultmomenttype` selects the ordinary or Toeplitz-compatible moment type.
- `H2Factory.i2otrait` selects how the stored coupling is applied.
- `H2Factory.modulation` selects any basis conversion and FFT around I2O.

These decisions work together, but are not interchangeable. An FFT buffer
does not make a nonuniform grid Toeplitz; a scalar kernel type does not imply
a single moment component for every operator formulation.

## Raw and converted moments

I2I and O2O use the raw polynomial basis. Buffered B-spline I2O temporarily
uses a reduced basis; Toeplitz I2O also uses padded frequency-domain arrays.
The [modulation policies](../coupling/bspline.md) make those conversions.

H2Factory handles pool allocation and workspace management; see its
[core and workspace](https://djukic14.github.io/H2Factory.jl/dev/maps/core-workspace/).

The current adjoint leaf selection rejects genuinely complex collections.
The map's kernel may be complex while its collections are real: inspecting
`eltype(A)` is not enough to determine the leaf operation.
