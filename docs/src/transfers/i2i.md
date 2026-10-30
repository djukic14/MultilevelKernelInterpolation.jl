# Level translations: I2I and O2O

I2I combines child moments into a parent moment. Its matrix expresses each
parent basis function in terms of the child functions:

```math
\ell_i^p(x) \approx \sum_j C_{ij}\ell_j^c(x),\qquad
m_i^p \approx \sum_{\text{children }c}\sum_j C_{ij}m_j^c.
```

Both relations are exact when the child space contains the restricted parent
functions; otherwise the transfer approximates the parent moment integral.

Configure the child basis below, then choose `noincrease(child)` or
`increase(child)` to build the parent.

The parent box is `[0,1]`; its children cover `[0,1/2]` and `[1/2,1]`.
The first two panels show their bases. The next two show one parent function
reconstructed from the children, and the difference from the original.

## Choosing the parent basis

MultilevelKernelInterpolation builds the bases from fine to coarse. `noincrease` keeps the Lagrange
degree or the B-spline count before buffering. `increase` uses:

```math
\begin{aligned}
\text{Lagrange:}\quad &p_{\mathrm{parent}}=2p_{\mathrm{child}},\\
\text{B-splines:}\quad &p_{\mathrm{parent}}=p,\qquad
 b_{\mathrm{parent}}=2b_{\mathrm{child}}-p.
\end{aligned}
```

Here `b` excludes buffer functions. The finest level keeps the starting
resolution. On coarser levels, the default requests growth when the box
side length is at least half a wavelength. Set that threshold through
`PolynomialStrategy`; `hfhalfsize` is a **half-side length**:

```julia
polynomials = MultilevelKernelInterpolation.PolynomialStrategy(op;
    polynomial=MultilevelKernelInterpolation.BarycentricLagrangePolynomialEquidistant(dim=3, order=6),
    hfhalfsize=0.25 * MultilevelKernelInterpolation.λ(op),
)
strategy = MultilevelKernelInterpolation.InterpolationStrategy(op, space; polynomialstrategy=polynomials)
```

`op` and `space` are from [Getting started](../getting-started.md). This
threshold selects where the basis grows; it is not an error tolerance.

## Lagrange

With `noincrease`, parent and child have the same degree. The child basis can
represent every parent function, so the curves agree up to rounding error.

Choose `increase` to double the parent degree. The child degree stays fixed,
so the reconstruction is now an interpolation rather than an exact identity.

```@raw html
<div class="basis-explorer" data-kind="transfer" data-family="lagrange"></div>
```

The transfer coefficients are the parent functions evaluated at the child
nodes. The browser uses the same construction as MultilevelKernelInterpolation.

```jldoctest
julia> using MultilevelKernelInterpolation

julia> child = MultilevelKernelInterpolation.BarycentricLagrangePolynomialEquidistant(dim=1, order=4);

julia> parent = MultilevelKernelInterpolation.noincrease(child);

julia> left, right = MultilevelKernelInterpolation.onedimi2itranslations(child, parent);

julia> MultilevelKernelInterpolation.testonedimensionali2i(child, parent, left, right)
true

julia> MultilevelKernelInterpolation.polynomialorders(MultilevelKernelInterpolation.increase(child))
(8,)
```

## B-splines

Here `increase` keeps the degree and changes the basis count from `b` to
`2b-p`. Six cubic splines become nine on the parent. Unlike increasing the
Lagrange degree, this can preserve an exact representation: the knots still
align with the child knots.

```@raw html
<div class="basis-explorer" data-kind="transfer" data-family="bspline"></div>
```

The buffer input represents `MultilevelKernelInterpolation.ExplicitBuffer(points)`. Both `increase`
and `noincrease` keep that count on the parent.

For a compatible unbuffered transfer:

```jldoctest
julia> using MultilevelKernelInterpolation

julia> child = MultilevelKernelInterpolation.BsplinePolynomialEquidistant(dim=1, degree=3, nbasis=6);

julia> parent = MultilevelKernelInterpolation.increase(child);

julia> left, right = MultilevelKernelInterpolation.onedimi2itranslations(child, parent);

julia> size(left), size(right)
((9, 6), (9, 6))
```

MultilevelKernelInterpolation constructs B-spline transfers by **knot insertion**. The browser uses
collocation at the child Greville points to illustrate the representation.
For nested spaces, both give the same coefficients up to rounding error.
With buffers, the restricted parent functions must still lie in the child
space. A small sampled residual is not a proof of this, and high degrees
can make the browser solve ill-conditioned.

### Buffers across levels

`AutoBufferFromProtrusion` instead recomputes the count at each level. For
protrusion `eta`, measured relative to the box side length, it chooses

```math
n_{\mathrm{buffer}}=\max\!\left(n_{\min},\left\lceil\eta(b-p)\right\rceil\right).
```

The knot spacing in box coordinates is `1/(b-p)`, so this rounds the required
extension up to whole intervals. Each side gets that many extra knots.
`n_min` is the policy's optional minimum, zero by default.

```jldoctest
julia> using MultilevelKernelInterpolation

julia> policy = MultilevelKernelInterpolation.AutoBufferFromProtrusion();

julia> MultilevelKernelInterpolation.resolvebuffer(policy, 3, 9, 0.2)
2
```

Thus `noincrease` need not keep an automatic buffer unchanged: the basis
count can stay fixed while the geometric requirement changes. The tree
supplies those requirements; the interactive example above uses an explicit
buffer so it needs no geometry. `NoBuffer()` always requests zero.

`MultilevelKernelInterpolation.OneDimI2IStrategy()` stores one transfer factor per axis; the full
translator stores tensor-product matrices.

## O2O is the transpose relation

Write the basis relation as `ell_parent ≈ C * ell_child`. A field with parent
coefficients `a` then has child coefficients `transpose(C) * a`:

```math
u(x)=a^{\mathsf T}\boldsymbol\ell^p(x)
    \approx(C^{\mathsf T}a)^{\mathsf T}\boldsymbol\ell^c(x).
```

This becomes equality for nested spaces, just as in I2I.

Forward I2I uses `C`, and forward O2O uses its **transpose**, not its adjoint.
`MultilevelKernelInterpolation.O2OTranslator` reuses the I2I collection. H2Factory handles traversal
and map orientation.
