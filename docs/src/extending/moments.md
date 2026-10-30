# Custom moment assembly

`MomentAssemblyStrategy` selects the constructor that assembles leaf
collections. Its default is `MultilevelKernelInterpolation.BEASTMomentAssembler`.

```julia
assembly = MultilevelKernelInterpolation.MomentAssemblyStrategy(
    MomentAssembler=MyAssembler,
    momentquadstrategy=quadrature,
)
strategy = MultilevelKernelInterpolation.InterpolationStrategy(op, testspace, trialspace;
    momentassemblystrategy=assembly,
)
```

Here `MyAssembler` is your constructor and `quadrature` is the quadrature
configuration it expects. This setting is independent of near-interaction
quadrature.

## The two calls to implement

The collection builder first constructs your assembler:

```julia
assembler = MyAssembler(momentoperator, space, quadstrategy, ncomponents)
```

It then calls it once for a side's leaves:

```julia
collections = assembler(tree, leafpolynomials; verbose=false)
```

Return one entry per leaf, in `H2Trees.leaves(tree)` order. Each entry contains
one matrix per moment component. Matrix rows correspond to the flattened
polynomial basis; columns correspond to the basis-function indices in
`H2Trees.values(tree, leaf)` order. The stock assembler returns an `SVector`
of matrices for each leaf.

The surrounding builder wraps these entries as `MultilevelKernelInterpolation.TestMomentCollection`
or `MultilevelKernelInterpolation.TrialMomentCollection`. B-spline collections are converted to sparse
storage by `MultilevelKernelInterpolation.postprocessmoment!`.

## Reuse BEAST integration

If only the integrand changes, an entirely new assembler may be unnecessary.
The default assembler dispatches on `MultilevelKernelInterpolation.AbstractMomentOperator{D,T}` and
calls `MultilevelKernelInterpolation.integrand!`. Here `D` is the component count and `T` its scalar
type. Gradient moments also select different polynomial-evaluation scratch.

The leaf assembly path uses `H2Factory.buildstorer` and
`H2Factory.storetrait` for storage. These matter when reusing that path with
a different storage policy, not when an assembler independently returns
the finished collections.

## Complex collections

The standard moments normally produce real leaf collections even when the
kernel is complex. Do not infer collection type from the map's scalar type.

The current MultilevelKernelInterpolation adjoint leaf selection rejects complex leaf collections.
A custom assembler that produces them also needs a suitable conjugating leaf
operation and an explicit extension of that selection. A forward product
passing is not evidence that the adjoint works.

The [moment reference](../api/moments.md) documents the collection types;
the [H2Factory guide](https://djukic14.github.io/H2Factory.jl/dev/) describes
the leaf multiplication protocol.
