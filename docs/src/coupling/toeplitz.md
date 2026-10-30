# Toeplitz I2O

When the kernel depends only on the displacement between two points, uniform
grids with matching spacing give a Toeplitz coupling: grid pairs with the same index difference
share the same kernel value. MultilevelKernelInterpolation stores these values and applies the coupling
with FFTs instead of storing a dense matrix.

```julia
translators = MultilevelKernelInterpolation.TranslatorStrategy(i2ostrategy=MultilevelKernelInterpolation.ToeplitzI2OStrategy())
```

Use equidistant Lagrange or a compatible B-spline basis. Chebyshev grids do
not satisfy the regular-spacing requirement. Petrov maps also require
compatible grids on both sides; construction checks this explicitly.

For `b` points per axis in 3-D, the difference tensor has `(2b-1)^3` entries,
rather than `b^6` dense entries. FFT storage and scratch add their own cost.

## Where the conversions happen

The map allocates the FFT plans and buffers during construction. Its modulation
hooks perform the forward and inverse transforms; no separate FFT call is needed
when applying the map.

Toeplitz and dense routes apply the same chosen interpolation approximation,
up to floating-point error. Symmetry reduction is a separate option:
`MultilevelKernelInterpolation.ToeplitzI2OStrategy(symmetryreduced=true)` reuses lattice directions.
For B-splines, the [modulation also includes sampling solves](bspline.md).
