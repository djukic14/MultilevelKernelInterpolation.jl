# Choosing an approximation

Start with a polynomial family and a compatible I2O implementation. Then pass
both through `MultilevelKernelInterpolation.InterpolationStrategy`. Here `op` and `space` are the operator and
space from [Getting started](../getting-started.md).

## Defaults are not interchangeable

`InterpolationStrategy(op, space)` chooses equidistant Lagrange and Toeplitz I2O.
Standalone `PolynomialStrategy(op)` chooses Chebyshev, while standalone
`TranslatorStrategy()` chooses dense I2O. Replacing just one of these pieces
can therefore change compatibility. The examples below set both choices
explicitly.

## Where the settings go

The outer configuration object is **`MultilevelKernelInterpolation.InterpolationStrategy`**. It groups several
smaller strategies. For example:

```julia
polynomials = MultilevelKernelInterpolation.PolynomialStrategy(op;
    polynomial=MultilevelKernelInterpolation.BarycentricLagrangePolynomialEquidistant3D(6),
)
translators = MultilevelKernelInterpolation.TranslatorStrategy(
    i2istrategy=MultilevelKernelInterpolation.OneDimI2IStrategy(),
    i2ostrategy=MultilevelKernelInterpolation.ToeplitzI2OStrategy(),
)
strategy = MultilevelKernelInterpolation.InterpolationStrategy(op, space;
    polynomialstrategy=polynomials,
    translatorstrategy=translators,
    momentassemblystrategy=MultilevelKernelInterpolation.MomentAssemblyStrategy(),
)
# Pass the finished strategy to the map constructor:
# A = GalerkinKernelInterpolation(op, space; strategy=strategy)
```

Each row below names both the constructor accepting the setting and the
keyword connecting it to the outer `InterpolationStrategy`.

| Choice | Pass to | Connect to `InterpolationStrategy` with | Effect |
| :--- | :--- | :--- | :--- |
| Basis and resolution | `PolynomialStrategy(op; polynomial=...)` | `polynomialstrategy=...` | Changes the far approximation |
| Growth across levels | `PolynomialStrategy(op; increase=..., hfhalfsize=...)` | `polynomialstrategy=...` | Changes when and how the basis grows |
| Leaf quadrature | `MomentAssemblyStrategy(momentquadstrategy=...)` | `momentassemblystrategy=...` | Changes moment integration |
| Near quadrature | `H2Factory.BlockBEASTNearInteractionsStrategy(quadrature)` | `nearinteractionsstrategy=...` | Changes direct near integration |
| Partition and tree passes | `H2Factory.InteractionStrategy(...)` | `nearfieldstrategy=...` | Selects pairs and traversal mode |
| Dense or Toeplitz I2O | `TranslatorStrategy(i2ostrategy=...)` | `translatorstrategy=...` | Changes coupling storage and application |
| Combined or separable I2I | `TranslatorStrategy(i2istrategy=...)` | `translatorstrategy=...` | Changes level-transfer storage |
| Execution scheduler | `InterpolationStrategy(op, space; scheduler=...)` | Direct keyword | Changes execution, not interpolation accuracy |

The shorter constructor names in this table all belong to `MultilevelKernelInterpolation`, except
the two explicitly qualified `H2Factory` constructors. Supply multiple
settings for one smaller strategy in the **same** constructor call; a second
`TranslatorStrategy` does not extend the first one.

The **map constructor**, not `InterpolationStrategy`, also accepts an explicit tree after
the space. See [Constructing a map](../getting-started.md). Changing the tree
changes which boxes interact, so it can change both cost and approximation error.

The examples below are alternative configurations, not steps to run in order.

## Equidistant Lagrange and FFTs

This spells out the choices used by the default 3-D strategy:

```julia
polynomial = MultilevelKernelInterpolation.BarycentricLagrangePolynomialEquidistant3D(6)
strategy = MultilevelKernelInterpolation.InterpolationStrategy(op, space;
    polynomialstrategy=MultilevelKernelInterpolation.PolynomialStrategy(op; polynomial),
    translatorstrategy=MultilevelKernelInterpolation.TranslatorStrategy(
        i2ostrategy=MultilevelKernelInterpolation.ToeplitzI2OStrategy()),
)
```

Toeplitz coupling uses the regular spacing of the sampling grid to apply the
kernel with FFTs. It does not replace the underlying interpolation by a
different approximation.

## Chebyshev Lagrange and dense coupling

Chebyshev nodes are not equidistant. Pair them with dense I2O:

```julia
polynomial = MultilevelKernelInterpolation.BarycentricLagrangePolynomialChebyshev3D2(6)
strategy = MultilevelKernelInterpolation.InterpolationStrategy(op, space;
    polynomialstrategy=MultilevelKernelInterpolation.PolynomialStrategy(op; polynomial),
    translatorstrategy=MultilevelKernelInterpolation.TranslatorStrategy(
        i2ostrategy=MultilevelKernelInterpolation.DenseI2OStrategy()),
)
```

Dense I2O stores the interpolation couplings, not the original dense BEM
matrix. Those couplings can nevertheless be large at high orders.

## Buffered B-splines

Choose degree and basis count separately. The automatic buffer policy covers
basis functions whose support extends outside their tree box:

```julia
polynomial = MultilevelKernelInterpolation.BufferedBsplinePolynomialEquidistant(
    dim=3, degree=3, nbasis=9,
    buffer=MultilevelKernelInterpolation.AutoBufferFromProtrusion(),
)
strategy = MultilevelKernelInterpolation.InterpolationStrategy(op, space;
    polynomialstrategy=MultilevelKernelInterpolation.PolynomialStrategy(op; polynomial),
    translatorstrategy=MultilevelKernelInterpolation.TranslatorStrategy(
        i2ostrategy=MultilevelKernelInterpolation.ToeplitzI2OStrategy()),
)
```

The builder selects B-spline basis conversion on both the dense and Toeplitz
routes. [Stabilization](../polynomials/bsplines.md) explains why this conversion
is not just an FFT.

## Other choices

| Setting | Select it with |
| :--- | :--- |
| Separable level transfers | `MultilevelKernelInterpolation.TranslatorStrategy(i2istrategy=MultilevelKernelInterpolation.OneDimI2IStrategy())` |
| Symmetry-reduced dense coupling | `MultilevelKernelInterpolation.DenseI2OStrategy(symmetryreduced=true)` |
| Symmetry-reduced Toeplitz coupling | `MultilevelKernelInterpolation.ToeplitzI2OStrategy(symmetryreduced=true)` |
| Moment quadrature | `MultilevelKernelInterpolation.MomentAssemblyStrategy(momentquadstrategy=...)` |
| Near-interaction quadrature | `H2Factory.BlockBEASTNearInteractionsStrategy(quadstrategy)` |
| Near/far partition and traversal mode | `H2Factory.InteractionStrategy(...)`, passed as `nearfieldstrategy` |

Separable transfers store one factor per axis rather than one combined
tensor-product matrix. Set `i2ostrategy` in that same `TranslatorStrategy`
call to retain the desired dense or Toeplitz choice.

Symmetry reduction reuses lattice directions without adding an approximation.
Dense reduction currently supports Galerkin maps with non-B-spline bases and
scalar or supported 3-by-3 kernels; it rejects unsupported combinations.
Toeplitz reduction has a wider set of routes. Non-lattice displacements use
unreduced couplings instead. Reduction is off by default.
