```@meta
CurrentModule = MultilevelKernelInterpolation
```

# Multilevel Kernel Interpolation

MultilevelKernelInterpolation uses polynomial interpolation to approximate well-separated interactions
in kernel-based matrices, such as those arising from the boundary element
method (BEM). Nearby interactions are assembled directly.
The result is a linear map: apply it with `A * x` or `mul!(y, A, x)` without
forming the full dense matrix.

The built-in assemblers use BEAST's integral operators and function spaces.
The interpolation and translation machinery is not limited to boundary elements;
other discretizations can supply their own moment assembly.

The package implements the stabilized B-spline method described in the preprint
listed below, alongside Lagrange-based interpolation methods.
The following example introduces its factorization; the later pages connect
the polynomial bases, moments, and translations to the code.

## The factorization

As an example, consider the vector-potential matrix entry for a test function
`f_m` and a trial function `f_n`

```math
A_{mn}=\int_{\operatorname{supp}\mathbf f_m}
       \int_{\operatorname{supp}\mathbf f_n}
       \mathbf f_m(\mathbf x)\cdot
       g(\mathbf x,\mathbf y)\,\mathbf f_n(\mathbf y)
       \,\mathrm dS_{\mathbf y}\,\mathrm dS_{\mathbf x}.
```

For well-separated boxes, approximate the scalar kernel with their box bases

```math
g(\mathbf x,\mathbf y)\approx
\sum_{i,j}\ell_{i,c_x}(\mathbf x)\,W_{ij}\,\ell_{j,c_y}(\mathbf y).
```

Substituting this expansion separates the two integrals. Define one moment
vector per Cartesian component

```math
[\mathbf m_{d,n}]_j =
    \int_{\operatorname{supp}\mathbf f_n}
    f_{n,d}(\mathbf y)\,\ell_{j,c_y}(\mathbf y)\,\mathrm dS_{\mathbf y}.
```

The matrix entry then becomes

```math
A_{mn}\approx\sum_{d=1}^{3}
    \mathbf m_{d,m}^{\mathsf T}\,W\,\mathbf m_{d,n}.
```

The **trial moments** on the right collect the source into polynomial
coefficients. **`W`** couples two boxes. The **test moments** on the left
project the result back to the test functions. Other operators use different
moment integrands, but keep this separation.

For Lagrange interpolation, `W` contains kernel samples. B-splines also
require sampling-matrix solves. I2I and O2O reuse these moments across tree
levels, so distant interactions need not all be evaluated at the leaves.

## Start here

- [Getting started](getting-started.md) shows the construction and multiplication calls.
- [Choosing a basis](polynomials/choosing.md) compares Lagrange and B-splines.
- [Choosing an approximation](guide/approximation.md) puts those choices into a strategy.
- [Level transfers](transfers/i2i.md) explains which interpolation identities are exact.
- [Moment integrals](moments/integrals.md) gives the integrands used by the assemblers.

The guide includes configuration snippets and interactive [basis plots](polynomials/choosing.md).
It does not run matrix assembly, scattering solves, or numerical studies during a docs build.

## Where the packages fit

BEAST supplies the integral operators, function spaces, and quadrature. H2Trees
organizes their geometry. H2Factory runs the tree passes and manages scratch
storage. MultilevelKernelInterpolation supplies the interpolation bases, moments, and translations.

To add another operator, start with [Adding an operator](extending/operators.md).
The [API reference](api/maps.md) contains the individual methods and their docstrings.

## References

- W. Hackbusch and S. Börm, “H²-matrix approximation of integral operators by
  interpolation,” *Applied Numerical Mathematics*, 43(1–2), 129–143, 2002.
- D. T. Schobert and T. F. Eibert, “Low-frequency surface integral equation
  solution by multilevel Green's function interpolation with fast Fourier
  transform acceleration,” *IEEE Transactions on Antennas and Propagation*,
  60(3), 1440–1449, 2012.
- D. Jukić, B. Hofmann, T. F. Eibert, and S. B. Adrian,
  [“A Stabilized Multilevel B-Spline-Based Fast Integral Method for the Solution
  of the Electric Field Integral Equation”](https://arxiv.org/abs/2606.13482), 2026.
