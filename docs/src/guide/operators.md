# Supported operators

Choose the operator first: it determines which moments and kernel MultilevelKernelInterpolation needs.
The constructor also needs a BEAST function space appropriate to the problem.
The space examples below are not an exhaustive list of supported spaces.

## Helmholtz

| BEAST operator | Dimension | Map route |
| :--- | :--- | :--- |
| `Helmholtz3D.singlelayer` | 3-D | Galerkin or Petrov |
| `Helmholtz3D.doublelayer` / `doublelayer_transposed` | 3-D | Petrov: the two moment operators differ |
| `Helmholtz3D.hypersingular` | 3-D | Galerkin or Petrov |
| `Helmholtz2D.singlelayer` | 2-D | Galerkin or Petrov |
| `Helmholtz2D.doublelayer` / `doublelayer_transposed` | 2-D | Petrov: the two moment operators differ |

Scalar spaces include `BEAST.lagrangecxd0(mesh)` and
`BEAST.lagrangec0d1(mesh)`. Their suitability depends on the integral equation.
The 2-D hypersingular operator has no implemented moment route.

The default `InterpolationStrategy` uses a 3-D polynomial. For a 2-D problem, provide a
2-D basis explicitly, for example
`MultilevelKernelInterpolation.BarycentricLagrangePolynomialEquidistant(dim=2, order=6)`.

## Maxwell in 3-D

BEAST's `Maxwell3D.singlelayer`, `Maxwell3D.doublelayer`, and rotated
double-layer operator have moment factorizations in MultilevelKernelInterpolation. For example, RWG
functions are constructed with `BEAST.raviartthomas(mesh)`.

The wrappers below provide alternative factorizations. They change which
derivatives live in the moments and which live in the interpolated kernel.

| MultilevelKernelInterpolation constructor | Interpolated kernel |
| :--- | :--- |
| `MultilevelKernelInterpolation.efie(...)` / `MultilevelKernelInterpolation.MWEFIE3D(...)` | Dyadic Green's function |
| `MultilevelKernelInterpolation.MWHyperSingular3D(...)` | Hessian of the Green's function |
| `MultilevelKernelInterpolation.MWWeaklySingular3D(...)` | Scalar Green's function |
| `MultilevelKernelInterpolation.MWDoubleLayer3D(...)` | Cross-product form of the kernel gradient |

`MultilevelKernelInterpolation.HH3DDoubleLayerFDBIOGradientI2O` and
`MultilevelKernelInterpolation.HH3DDoubleLayerTransposedFDBIOGradientI2O` similarly move the Helmholtz
double-layer derivative into I2O. These need the Petrov route.

The wrappers already supply the coefficient signs for their factorization.
Transpose and adjoint are handled by the map; no additional sign correction
is needed when applying those views.

## Galerkin or Petrov?

`GalerkinKernelInterpolation` requires equal test and trial moment operators. The same BEAST
space on both sides is not sufficient if the factorization uses different
moments. `PetrovGalerkinKernelInterpolation(op, space, space)` handles that case too.

See [Constructing a map](../getting-started.md) for the constructors, and the
[operator reference](../api/operators.md) for individual types.
