# Getting started

An MultilevelKernelInterpolation map approximates a kernel-based matrix. Construct it once, then reuse
it for multiplications, for example inside an iterative solver.

## Construct a map

The following BEM example uses a surface mesh you have already loaded as `mesh`:

```julia
using MultilevelKernelInterpolation, LinearAlgebra
import BEAST

space = BEAST.raviartthomas(mesh)
op = MultilevelKernelInterpolation.efie(; wavenumber=1.0)
A = GalerkinKernelInterpolation(op, space)
```

This call assembles leaf moments, near interactions, and translation data.

`GalerkinKernelInterpolation` returns a map with its own reusable workspace:

```julia
x = ones(eltype(A), size(A, 2))
y = zeros(eltype(A), size(A, 1))
mul!(y, A, x)
```

`A * x`, `transpose(A) * y`, and `adjoint(A) * y` are also available. Transpose
and adjoint reuse the constructed approximation; they do not assemble another map.

For different test and trial spaces, use:

```julia
strategy = MultilevelKernelInterpolation.InterpolationStrategy(op, testspace, trialspace)
A = PetrovGalerkinKernelInterpolation(op, testspace, trialspace; strategy)
```

Toeplitz I2O requires matching interpolation grids on the two sides.
Construction checks `MultilevelKernelInterpolation.toeplitzcompatible`; use `MultilevelKernelInterpolation.DenseI2OStrategy()`
if those grids differ. H2Factory explains the
[core and workspace](https://djukic14.github.io/H2Factory.jl/dev/maps/core-workspace/)
and how to give simultaneous products separate scratch storage.

## Supply a tree

If you already built a compatible H2Trees tree, pass it after the space:

```julia
A = GalerkinKernelInterpolation(op, space, tree; strategy)
```

Otherwise the constructor builds the tree. Keep the basis-function index
ordering consistent with the space; tree values refer to those indices.

## Change the configuration

Pass a strategy through `strategy=`

```julia
strategy = MultilevelKernelInterpolation.InterpolationStrategy(op, space)
A = GalerkinKernelInterpolation(op, space; strategy)
```

The [approximation guide](guide/approximation.md) shows which strategy to change.
