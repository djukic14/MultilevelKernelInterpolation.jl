# Assembling moment collections

`MultilevelKernelInterpolation.TestMomentCollections` and `MultilevelKernelInterpolation.TrialMomentCollections` build one
collection per leaf. They select a moment operator, integrate it, and wrap
the resulting matrices. Test and trial collections may differ in a Petrov map.

```julia
testcollections = MultilevelKernelInterpolation.TestMomentCollections(
    op, testtree, testleafpolynomials, testspace;
    quadstrat=BEAST.SingleNumQStrat(7),
)
trialcollections = MultilevelKernelInterpolation.TrialMomentCollections(
    op, trialtree, trialleafpolynomials, trialspace;
    quadstrat=BEAST.SingleNumQStrat(7),
)
```

These are construction calls, not per-multiplication operations. The normal
map constructor makes them for you.

## From an integral to a matrix entry

`MultilevelKernelInterpolation.BEASTMomentAssembler` visits the elements supporting each leaf's basis
functions. At quadrature points it evaluates the box polynomials and calls
`MultilevelKernelInterpolation.integrand!` with the basis-function data. Quadrature weights and BEAST's
local-to-global assembly coefficients accumulate into the leaf matrices.

```text
leaf polynomial + basis-function data
                  ↓ integrand!
          quadrature contributions
                  ↓ local-to-global assembly
       one matrix per moment component
```

Rows index the flattened tensor-product polynomial basis. Columns follow
the basis-function indices in `H2Trees.values(tree, leaf)`. Collections
follow `H2Trees.leaves(tree)` order. Axis 1 varies fastest when flattening.

## Choose quadrature or replace the assembler

```julia
assembly = MultilevelKernelInterpolation.MomentAssemblyStrategy(
    momentquadstrategy=BEAST.SingleNumQStrat(7),
    MomentAssembler=MultilevelKernelInterpolation.BEASTMomentAssembler,
)
strategy = MultilevelKernelInterpolation.InterpolationStrategy(op, space; momentassemblystrategy=assembly)
```

This quadrature setting is independent of near-interaction quadrature.
The [custom assembler page](../extending/moments.md) gives the two calls
another assembler must implement. If only the integral changes, extending
`MultilevelKernelInterpolation.integrand!` may be sufficient.

B-spline collections are converted to sparse matrices by
`MultilevelKernelInterpolation.postprocessmoment!`, reflecting the local support of the basis.
