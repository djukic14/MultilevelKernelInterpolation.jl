# Moment integrals

Moments connect the boundary-element functions to the interpolation basis.
For a scalar basis function `f_n` and a box polynomial `ℓ_i`, the basic
collection entry is

```math
F_{in}=\int_\Gamma f_n(x)\,\ell_i(x)\,\mathrm dS_x.
```

Only the support of `f_n` contributes. Multiplying by the input coefficients
gives the box moment `m_i = ∑_n F_in x_n`. The same `F` is reused for every
matrix-vector product.

For vector-valued functions, each component has its own matrix:

```math
F^{(d)}_{in}=\int_\Gamma f_{n,d}(x)\,\ell_i(x)\,\mathrm dS_x.
```

This is the moment in the opening factorization. The transpose in that
factorization is deliberate; it is not a Hermitian inner product.

## The implemented integrands

`MultilevelKernelInterpolation.testmoment(op, polynomial)` and `MultilevelKernelInterpolation.trialmoment(op, polynomial)`
select the moment operator on each side. The operators below describe what
is integrated before quadrature and assembly into the collection.

| Moment operator | Integrand for polynomial `ℓ_i` |
| :--- | :--- |
| `MultilevelKernelInterpolation.LaplaceMoment` | `f_n ℓ_i`, scalar or componentwise vector |
| `MultilevelKernelInterpolation.LaplaceMomentDiv` | `(f_{n,1}, f_{n,2}, f_{n,3}, div_Γ f_n) ℓ_i` |
| `MultilevelKernelInterpolation.NormalLaplaceMoment` | `f_n n ℓ_i`, for scalar `f_n` |
| `MultilevelKernelInterpolation.GradLaplaceMoment` | `f_n × ∇ℓ_i`, for vector `f_n` |
| `MultilevelKernelInterpolation.NormalGradLaplaceMoment` | `f_n (n ⋅ ∇ℓ_i)` for scalar `f_n`; `(f_n × n) × ∇ℓ_i` for vector `f_n` |
| `MultilevelKernelInterpolation.HyperSingularMoment` | The six components of `(f_n n, curl_Γ f_n) ℓ_i` |

Here `n` is the surface normal. The hypersingular path reads the surface-curl
data supplied by BEAST; it does not replace it with a Cartesian curl.

For example, the extra divergence component in `LaplaceMomentDiv` is

```math
F^{(4)}_{in}=\int_\Gamma
    (\operatorname{div}_\Gamma\mathbf f_n)(x)\,\ell_i(x)\,\mathrm dS_x.
```

The gradient path instead differentiates the interpolation basis:

```math
\mathbf F_{in}=\int_\Gamma
    \mathbf f_n(x)\times\nabla\ell_i(x)\,\mathrm dS_x.
```

Operator prefactors and kernel component contractions are separate from these
integrals. A moment name alone does not specify the full integral operator.

Next: [assemble these entries](assembly.md), then see how the
[traits and storage](storage.md) connect them to multiplication.
