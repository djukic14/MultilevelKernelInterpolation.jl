# Adding an operator

An operator tells MultilevelKernelInterpolation which moments to integrate and which kernel to sample
between boxes. If a new operator can use an existing factorization, most of
the translation machinery can stay unchanged.

## Define the factorization

These are the dispatch points. `op` is your operator and `poly` is the
interpolation basis; the table describes required results, not method bodies.

| Method | Result |
| :--- | :--- |
| `MultilevelKernelInterpolation.testmoment(op, poly)` | Moment operator for the receiving side |
| `MultilevelKernelInterpolation.trialmoment(op, poly)` | Moment operator for the source side |
| `MultilevelKernelInterpolation.i2otranslation(op)` | Kernel trait, such as `MultilevelKernelInterpolation.ScalarGreen3DTranslation()` |
| `H2Factory.α(op, poly)` | Scalar or per-component factor applied by the factorization |
| `MultilevelKernelInterpolation.αtranspose(op, poly)` | Transpose factor, if different from the default |

Equal moments may instead be supplied through `MultilevelKernelInterpolation.moment(op, poly)`;
the default test and trial methods both call it. `αtranspose` defaults to
`H2Factory.α(op, poly)`. The adjoint coefficient is its complex conjugate.

The built-in moments are described in the [operator reference](../api/operators.md).
Their component count is not the spatial dimension. For example,
`LaplaceMomentDiv{4,T}` contains three vector components and one divergence
component for a 3-D Maxwell operator.

## Supply the kernel

Existing traits select the layout: a scalar matrix, a vector of component
matrices, or a 3-by-3 matrix of component matrices. Reuse a trait only when
both its kernel and component action match the intended factorization.

For a new scalar kernel trait derived from `MultilevelKernelInterpolation.ScalarI2OTranslation`, define:

```julia
MultilevelKernelInterpolation.kernel(op, trait, testpoint, trialpoint)
```

The existing scalar sampling loop calls it for each pair. A new block kernel
also needs the corresponding `MultilevelKernelInterpolation.kernelbuffer`, `MultilevelKernelInterpolation.computekernel!`, and
storage selection through `MultilevelKernelInterpolation.eltypei2otranslation`. `MultilevelKernelInterpolation.kernel!` is the
in-place evaluator used by the existing block traits.

Symmetry reduction additionally needs `MultilevelKernelInterpolation.symmetrycomponentaction(trait)`.
A matrix of matrices alone does not say whether a kernel transforms as a
true tensor or a pseudo-tensor.

## Connect assembly

The standard builders use `BEAST.scalartype(op)` and the propagation parameter
from `MultilevelKernelInterpolation.operatorgamma(op)`. The latter normally calls `BEAST.gamma(op)`;
the existing 3-D Green's-function evaluators also call `BEAST.gamma` directly.

Near interactions still need an assembly route. A kernel sampling method
alone does not make an arbitrary object a complete BEAST operator. Either
provide the BEAST assembly interface or select a custom near-interaction
assembler through `InterpolationStrategy`.

## Check the two sides

Use the Petrov constructor if the moment operators differ. Check forward,
transpose, and adjoint against an independent reference before relying on
the new factorization against a small dense reference.

For a different leaf integration routine, continue with
[Custom moment assembly](moments.md). For different translation storage or
application, see [Custom translations](translations.md).
