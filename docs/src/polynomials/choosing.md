# Choosing a basis

The interpolation basis lives on each tree box.

## The main choices

For an FFT-based method with protruding supports, start with
`MultilevelKernelInterpolation.BufferedBsplinePolynomialEquidistant`: it combines buffers with the
stabilized B-spline route. For a simpler dense reference, start with
Chebyshev Lagrange. Equidistant Lagrange is a useful low-to-moderate-degree
Toeplitz choice and is the current map default; avoid treating higher degree
as an automatic improvement.

| Basis | Parameters | I2O choices |
| :--- | :--- | :--- |
| Equidistant Lagrange | Degree | Dense or Toeplitz |
| Chebyshev Lagrange | Degree and Chebyshev node family | Dense |
| Equidistant B-splines | Degree and basis count | Dense or Toeplitz, with basis conversion |
| Buffered B-splines | Degree, basis count, buffer policy | Dense or Toeplitz, with stabilization and conversion |

`MultilevelKernelInterpolation.BsplinePolynomialEquidistant` is the plain, unbuffered basis; it does
not select the buffered stabilization route. `MultilevelKernelInterpolation.ScaledShiftedPolynomial`
wraps another basis with a coordinate map, rather than introducing a new family.

Lagrange basis functions extend across the whole box. Their degree fixes the
number of nodes: degree `p` gives `p + 1` nodes on each axis.

B-splines have local support. Their degree `p` and number of basis functions
`b` are separate choices, with `b > p`. Increasing `b` refines the knots without
raising the degree. Buffering and stabilization can make the stored basis
count differ from the number of kernel sampling points.

## Explore the functions

Change the degree, basis count, or buffer and press **Update**. The box is
`[0,1]`; the shaded regions are the B-spline buffer outside it. Lagrange uses
`p+1` functions. B-splines keep degree and basis count separate.

```@raw html
<div class="basis-explorer" data-kind="basis"></div>
```

The browser evaluates the basis functions with the same node and knot
conventions as MultilevelKernelInterpolation. These plots show the basis before the sampling-matrix
solves used in B-spline I2O; they do not show a stabilized kernel approximation.

The explorer accepts degrees beyond the package's current stabilization
policy. That policy covers B-spline degrees 1–10 in
`BufferedBsplinePolynomialEquidistant`; a curve in the explorer does not
mean the corresponding stabilized configuration can be constructed.

The corresponding Julia calls are:

```julia
lagrange = MultilevelKernelInterpolation.BarycentricLagrangePolynomialEquidistant(dim=1, order=4)
bspline = MultilevelKernelInterpolation.BufferedBsplinePolynomialEquidistant(
    dim=1, degree=3, nbasis=9, buffer=MultilevelKernelInterpolation.ExplicitBuffer(2))
values = zeros(first(MultilevelKernelInterpolation.numpolynomials(bspline)))
MultilevelKernelInterpolation.evaluateall1dpolynomial!(values, bspline, 0.5, 1)
```

## From the reference interval to a box

`MultilevelKernelInterpolation.ScaledShiftedPolynomial` places a basis on a physical interval:

```math
x=a\xi+s,\qquad \ell^{\mathrm{box}}_j(x)=\ell_j((x-s)/a).
```

For example, this maps `[0,1]` to `[-1,1]` without changing the basis family:

```@example scaled_basis
using MultilevelKernelInterpolation
base = MultilevelKernelInterpolation.BarycentricLagrangePolynomialEquidistant(dim=1, order=4)
physical = MultilevelKernelInterpolation.ScaledShiftedPolynomial(base, 2.0, -1.0)
referencevalues = zeros(5)
physicalvalues = zeros(5)
MultilevelKernelInterpolation.evaluateall1dpolynomial!(referencevalues, base, 0.25, 1)
MultilevelKernelInterpolation.evaluateall1dpolynomial!(physicalvalues, physical, -0.5, 1)
@assert referencevalues ≈ physicalvalues
nothing # hide
```

The shift can be a scalar or a vector with one entry per axis. Derivatives
include the factor `1/a`. Map construction normally creates these physical
leaf bases through `MultilevelKernelInterpolation.assembletreepolynomials`.

## Pass the basis to a strategy

Constructors with explicit dimensions are available for both families:

```julia
lagrange = MultilevelKernelInterpolation.BarycentricLagrangePolynomialEquidistant(dim=3, order=6)
chebyshev = MultilevelKernelInterpolation.BarycentricLagrangePolynomialChebyshev2(dim=3, order=6)
bspline = MultilevelKernelInterpolation.BufferedBsplinePolynomialEquidistant(
    dim=3, degree=3, nbasis=9,
    buffer=MultilevelKernelInterpolation.AutoBufferFromProtrusion(),
)
```

Use `MultilevelKernelInterpolation.PolynomialStrategy(op; polynomial=bspline)` to select one. The
[approximation guide](../guide/approximation.md) shows the complete
`InterpolationStrategy` call, including the matching I2O strategy.

## What to read next

- [Lagrange interpolation](lagrange.md): node choices, barycentric evaluation,
  and support outside a box.
- [B-splines and stabilization](bsplines.md): buffers, knot removal, and the
  conversion used by I2O.
- [Level translations](../transfers/i2i.md): how the parent basis is chosen
  and represented by its children.

These are starting choices, not accuracy guarantees. The useful degree and
basis count depend on the kernel, geometry, and quadrature.
