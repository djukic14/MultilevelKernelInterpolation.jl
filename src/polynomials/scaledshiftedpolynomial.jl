
"""
    struct ScaledShiftedPolynomial{D,P,A,I} <: AbstractTensorProductPolynomial{D, eltype(P)}

Wrap a polynomial with sampling coordinates `x[d] * scaling + shift[d]`.
Stores the inverse scale for evaluation. A scalar shift is repeated along every axis.
"""
struct ScaledShiftedPolynomial{D,P,A,I} <: AbstractTensorProductPolynomial{D,eltype(P)}
    polynomial::P
    inversescaling::A
    shift::I
end

function ScaledShiftedPolynomial(polynomial::P, scaling::A, shift::I) where {P,A,I}
    shiftvector = _shiftvector(shift, Val(dims(polynomial)))
    return ScaledShiftedPolynomial{dims(polynomial),P,A,typeof(shiftvector)}(
        polynomial, inv(scaling), shiftvector
    )
end

function _shiftvector(shift::Number, ::Val{D}) where {D}
    return SVector{D}(ntuple(_ -> shift, Val(D)))
end

function _shiftvector(shift, ::Val{D}) where {D}
    return SVector{D}(shift)
end

function polynomial(polynomial::ScaledShiftedPolynomial)
    return polynomial.polynomial
end

function inversescaling(polynomial::ScaledShiftedPolynomial)
    return polynomial.inversescaling
end

function scaling(polynomial::ScaledShiftedPolynomial)
    return inv(polynomial.inversescaling)
end

function shift(polynomial::ScaledShiftedPolynomial)
    return polynomial.shift
end

function H2Trees.center(p::ScaledShiftedPolynomial, dim::Int)
    return shift(p)[dim] + inv(inversescaling(p) * 2)
end

function H2Trees.center(p::ScaledShiftedPolynomial)
    return shift(p) .+ 1 / (inversescaling(p) * 2)
end

function Base.eltype(p::ScaledShiftedPolynomial)
    return Base.eltype(polynomial(p))
end

function Base.range(p::ScaledShiftedPolynomial, dim::Int)
    return (range(polynomial(p), dim) * scaling(p)) .+ shift(p)[dim]
end

function Base.range(p::ScaledShiftedPolynomial, dim::Int, i::Int)
    return (range(polynomial(p), dim, i) * scaling(p)) + shift(p)[dim]
end

function dims(p::ScaledShiftedPolynomial)
    return dims(polynomial(p))
end

function numpolynomials(p::ScaledShiftedPolynomial)
    return numpolynomials(polynomial(p))
end

function kernelbuffer(p::ScaledShiftedPolynomial, ::Type{T}) where {T}
    return kernelbuffer(polynomial(p), T)
end

function kernelbufferderivative(p::ScaledShiftedPolynomial, ::Type{T}) where {T}
    return kernelbufferderivative(polynomial(p), T)
end

function evaluateall1dpolynomial!(
    buffer, p::ScaledShiftedPolynomial, newsample::T, dim::Int; extrapolate=true
) where {T}
    return evaluateall1dpolynomial!(
        buffer,
        polynomial(p),
        (newsample - shift(p)[dim]) * inversescaling(p),
        dim;
        extrapolate=extrapolate,
    )
end

function evaluateall1dpolynomialderivative!(
    buffer, p::ScaledShiftedPolynomial, newsample::T, dim::Int; extrapolate::Bool=true
) where {T}
    evaluateall1dpolynomialderivative!(
        buffer,
        polynomial(p),
        (newsample - shift(p)[dim]) * inversescaling(p),
        dim;
        extrapolate=extrapolate,
    )
    buffer[2] .*= inversescaling(p)

    return buffer
end

function range!(rangebuffer, p::ScaledShiftedPolynomial, dim::Int)
    rangebuffer .= range(polynomial(p), dim)
    rangebuffer .*= scaling(p)
    rangebuffer .+= shift(p)[dim]

    return rangebuffer
end

function increase(p::ScaledShiftedPolynomial; protrusion=0.0, kwargs...)
    pro = max(protrusion, maximum(abs, shift(p)) / 2)

    return ScaledShiftedPolynomial(
        increase(polynomial(p); kwargs...), 1 + 2 * pro, fill(-pro, dims(p))
    )
end

function noincrease(p::ScaledShiftedPolynomial; protrusion=0.0, kwargs...)
    pro = max(protrusion, maximum(abs, shift(p)) / 2)

    return ScaledShiftedPolynomial(
        noincrease(polynomial(p); kwargs...), 1 + 2 * pro, fill(-pro, dims(p))
    )
end

"""
    realizedprotrusion(p::ScaledShiftedPolynomial)

Return the sampling-domain extension relative to the nominal box side length.
The stored shift already includes the extension chosen by `increase`/`noincrease`;
no buffer-count rounding is needed.
"""
function realizedprotrusion(p::ScaledShiftedPolynomial)
    return maximum(abs, shift(p))
end

function isequidistantpolynomial(p::ScaledShiftedPolynomial)
    return isequidistantpolynomial(polynomial(p))
end

function Base.CartesianIndices(p::ScaledShiftedPolynomial)
    return CartesianIndices(polynomial(p))
end

function Base.LinearIndices(p::ScaledShiftedPolynomial)
    return LinearIndices(polynomial(p))
end
