# Dense I2O

I2O applies the kernel between two well-separated boxes at the same interaction
step. With Lagrange bases its matrix is simply

```math
G_{ij}=g(x_i,y_j),\qquad o_i=\sum_jG_{ij}m_j.
```

Rows belong to the receiving box; columns belong to the source box.
For vector kernels, the operator also couples moment components.

```julia
translators = MultilevelKernelInterpolation.TranslatorStrategy(i2ostrategy=MultilevelKernelInterpolation.DenseI2OStrategy())
```

`MultilevelKernelInterpolation.I2OTranslationCollection` constructs the stored couplings. This is a
matrix between interpolation grids, **not** the full matrix being approximated.
It is useful as a reference route and supports nonuniform Chebyshev grids.

In `D` dimensions with `b` points per axis, a dense scalar coupling has
`b^(2D)` entries. [Toeplitz I2O](toeplitz.md) avoids this storage when the
kernel and grids admit a difference representation.

## Reusing symmetries

`MultilevelKernelInterpolation.DenseI2OStrategy(symmetryreduced=true)` shares couplings between
symmetry-related lattice directions. It adds no approximation. The current
dense reduction supports Galerkin, non-B-spline bases and supported scalar or
3-by-3 kernels; unsupported combinations error. It is off by default.

For B-splines, the dense route also performs [basis conversions](bspline.md).
