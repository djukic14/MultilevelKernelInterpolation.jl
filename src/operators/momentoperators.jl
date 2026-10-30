abstract type AbstractMomentOperator{D,T} end
abstract type AbstractGradLaplaceMomentOperator{D,T} <: AbstractMomentOperator{D,T} end

"""
    struct LaplaceMoment

Integrate `j(x) * L_i(x)` over the boundary `Γ`, component by component.
`L_i` is an interpolation basis function and `j` may be scalar or vector-valued.
"""
struct LaplaceMoment{D,T} <: AbstractMomentOperator{D,T} end

"""
    struct LaplaceMomentDiv

Integrate `[j_x, j_y, j_z, ∇⋅j] * L_i` over the boundary.
Like `LaplaceMoment`, with an extra component for the divergence.
"""
struct LaplaceMomentDiv{D,T} <: AbstractMomentOperator{D,T} end

"""
    struct NormalLaplaceMoment

Integrate `j(x) * n(x) * L_i(x)` over the boundary, for scalar `j`
and boundary normal `n`.
"""
struct NormalLaplaceMoment{D,T} <: AbstractMomentOperator{D,T} end

"""
    struct GradLaplaceMoment

Integrate `j(x) × ∇L_i(x)` over the boundary for vector-valued `j`.
"""
struct GradLaplaceMoment{D,T} <: AbstractGradLaplaceMomentOperator{D,T} end

"""
    struct NormalGradLaplaceMoment

Integrate `j(x) * (n(x) ⋅ ∇L_i(x))` for scalar `j`.
For three-component `j`, integrate `(j(x) × n(x)) × ∇L_i(x)`.
Here `n` is the boundary normal.
"""
struct NormalGradLaplaceMoment{D,T} <: AbstractGradLaplaceMomentOperator{D,T} end

"""
    struct HyperSingularMoment

Concatenate the three normal moments and three surface-curl moments of scalar `j`:

```math
∫_Γ j(x) n(x) L_i(x) dx
```

and `∫_Γ (∇_x j(x) × n(x)) L_i(x) dx`.
The integrand reads the surface curl from BEAST's `f.curl`.
"""
struct HyperSingularMoment{D,T} <: AbstractMomentOperator{D,T} end

"""
    integrand!(buffer, polynomialmomentoperator, polynomial, polynomialvalue, f, point)

Write the moment integrand into `buffer`.
`polynomialvalue` is a basis value, or its gradient for a gradient moment.
`f` holds BEAST's basis-function data at `point`, including values, divergence, or curl.
"""
function integrand!(buffer, ::LaplaceMoment{1}, ::Any, polynomialvalue, f, point)
    return buffer .= polynomialvalue * f.value
end

function integrand!(buffer, ::LaplaceMoment, ::Any, polynomialvalue, f, point)
    buffer .= polynomialvalue
    buffer .*= f.value
    return buffer
end

function integrand!(buffer, ::LaplaceMomentDiv, ::Any, polynomialvalue, f, point)
    buffer .= polynomialvalue
    view(buffer, 1:length(f.value)) .*= f.value
    buffer[end] *= f.divergence
    return buffer
end

function integrand!(buffer, ::NormalLaplaceMoment, ::Any, polynomialvalue, f, point)
    buffer .= normal(point)
    buffer .*= polynomialvalue * f.value
    return buffer
end

function integrand!(buffer, ::GradLaplaceMoment, ::Any, gradientpolynomial, f, point)
    buffer .= cross(f.value, gradientpolynomial)
    return buffer
end

function integrand!(
    buffer, ::NormalGradLaplaceMoment{1}, ::Any, gradientpolynomial, f, point
)
    buffer .= f.value
    buffer .*= dot(normal(point), gradientpolynomial)
    return buffer
end

function integrand!(
    buffer, ::NormalGradLaplaceMoment{3}, ::Any, gradientpolynomial, f, point
)
    buffer .= cross(cross(f.value, normal(point)), gradientpolynomial)
    return buffer
end

function integrand!(buffer, ::HyperSingularMoment, ::Any, polynomialvalue, f, point)
    view(buffer, 1:3) .= normal(point)
    view(buffer, 1:3) .*= f.value
    view(buffer, 4:6) .= f.curl
    buffer .*= polynomialvalue
    return buffer
end

"""
    dims(::AbstractMomentOperator{D,T}) where {D,T}

Return the moment component count `D`, not the spatial dimension.
"""
function dims(::AbstractMomentOperator{D,T}) where {D,T}
    return D
end

function valuetype(op)
    return SVector{dims(op),scalartype(op)}
end

function BEAST.scalartype(::AbstractMomentOperator{D,T}) where {D,T}
    return T
end

function BEAST.quadrule(
    op::AbstractMomentOperator, rs, q, element, qdata, qs::BEAST.SingleNumQStrat
)
    return qdata[1, q]
end

struct BufferFunctor end

function (::BufferFunctor)(
    op::AbstractGradLaplaceMomentOperator, space, refspace, leafpolynomials, ::Val{T}
) where {T}
    zdimension = (
        numfunctions(refspace, domain(chart(geometry(space), 1))),
        numpolynomials(leafpolynomials)...,
    )
    z = Array{T}(undef, zdimension)

    return z,
    MultilevelKernelInterpolation.kernelbufferderivative(leafpolynomials, eltype(T)),
    zeros(real(eltype(T)), dims(op))
end

function (::BufferFunctor)(op, space, refspace, leafpolynomials, ::Val{T}) where {T}
    polynomial = leafpolynomials
    zdimension = (
        numfunctions(refspace, domain(chart(geometry(space), 1))),
        numpolynomials(polynomial)...,
    )
    z = Array{T}(undef, zdimension)

    return z,
    MultilevelKernelInterpolation.kernelbuffer(polynomial, eltype(T)),
    zeros(real(eltype(T)), dims(op))
end
