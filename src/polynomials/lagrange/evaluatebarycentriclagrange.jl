
"""
    evaluateall1dpolynomial!(result, polynomial, newsample, dim::Int; extrapolate=true)

Write all basis values along axis `dim` at `newsample` into `result`.
"""
function evaluateall1dpolynomial!(
    result, polynomial::AbstractBarycentricLagrange, newsample, dim::Int; extrapolate=true
)
    return evaluateall1dbarycentriclagrangepolynomial!(
        result,
        weights(polynomial, dim),
        range(polynomial, dim),
        newsample;
        extrapolate=extrapolate,
    )
end

"""
    evaluateall1dpolynomialderivative!(result, polynomial, newsample, dim::Int; extrapolate=true)

Write basis values and their derivatives along axis `dim` at `newsample` into `result`.
"""
function evaluateall1dpolynomialderivative!(
    result,
    polynomial::AbstractBarycentricLagrange,
    newsample,
    dim::Int;
    extrapolate::Bool=true,
)
    return evaluateall1dbarycentriclagrangepolynomialderivative!(
        result,
        weights(polynomial, dim),
        range(polynomial, dim),
        newsample;
        extrapolate=extrapolate,
    )
end

function evaluateall1dbarycentriclagrangepolynomial!(
    w, barycentricweights, samplingpoints, newsample::T; extrapolate=true
) where {T}
    for i in eachindex(w)
        if samplingpoints[i] ≈ newsample
            fill!(w, zero(T))
            w[i] = one(T)
            return w
        end

        w[i] = barycentricweights[i] / (newsample - samplingpoints[i])
    end

    w ./= sum(w)

    if !extrapolate && (newsample < 0 || newsample > 1)
        fill!(w, zero(T))
    end

    return w
end

function weights(polynomial::AbstractBarycentricLagrange)
    return polynomial.weights
end

function weights(polynomial::AbstractBarycentricLagrange, dim::Int)
    return weights(polynomial)[dim]
end

function evaluateall1dbarycentriclagrangepolynomialderivative!(
    buffer, barycentricweights, samplingpoints, newsample::T; extrapolate::Bool=true
) where {T}
    polynomial, derivative = buffer
    for i in eachindex(polynomial)
        if samplingpoints[i] ≈ newsample
            fill!(polynomial, zero(T))
            polynomial[i] = one(T)

            derivative[i] = zero(T)
            invbarycentrici = inv(barycentricweights[i])
            for j in eachindex(derivative)
                j == i && continue
                derivative[j] =
                    barycentricweights[j] * invbarycentrici /
                    (samplingpoints[i] - samplingpoints[j])
            end

            derivative[i] = -sum(derivative)
            return (polynomial, derivative)
        end

        invxminusxi = inv(newsample - samplingpoints[i])
        polynomial[i] = barycentricweights[i] * invxminusxi
        # Reuse the derivative buffer as scratch.
        derivative[i] = polynomial[i] * invxminusxi
    end

    sumpolynomial = sum(polynomial)
    polynomial ./= sumpolynomial

    sumtempstorage = sum(derivative) / sumpolynomial

    derivative .= polynomial
    derivative .*= sumtempstorage

    for i in eachindex(derivative)
        derivative[i] -=
            (barycentricweights[i] / (newsample - samplingpoints[i])^2) / sumpolynomial
    end

    if !extrapolate && (newsample < 0 || newsample > 1)
        fill!(polynomial, zero(T))
        fill!(derivative, zero(T))
    end

    return polynomial, derivative
end

function _rangebaryweightsequidistant(polynomialorder, ::Val{D}, ::Val{T}) where {D,T}
    range = T.(Base.range(0, 1; length=polynomialorder + 1))
    weight = _equidistantbarycentricweights(T, polynomialorder + 1)

    ranges = SVector{D}([range for i in 1:D])

    barycentricweights = SVector{D}([weight for i in 1:D])

    return ranges, barycentricweights
end

function _rangebaryweightschebyshev(polynomialorder, ::Val{D}, ::Val{T}) where {D,T}
    range = _gausschebyshev(T, polynomialorder + 1)
    range .= (range .+ one(T)) ./ T(2)
    weight = _chebyshevbarycentricweights(T, polynomialorder + 1)

    ranges = SVector{D}([range for i in 1:D])
    barycentricweights = SVector{D}([weight for i in 1:D])

    return ranges, barycentricweights
end

function _rangebaryweightschebyshev2(polynomialorder, ::Val{D}, ::Val{T}) where {D,T}
    range = _gausschebyshev2(T, polynomialorder + 1)
    range .= (range .+ one(T)) ./ T(2)
    weights = _chebyshevbarycentricweights2(T, polynomialorder + 1)

    ranges = SVector{D}([range for i in 1:D])
    barycentricweights = SVector{D}([weights for i in 1:D])

    return ranges, barycentricweights
end

function _equidistantbarycentricweights(::Type{T}, n::Int) where {T}
    w = Vector{T}(undef, n)

    for i in eachindex(w)
        w[i] = (-one(T))^(i - 1) * binomial(BigInt(n - 1), BigInt(i - 1))
    end

    return w
end

function _chebyshevbarycentricweights(::Type{T}, n::Int) where {T}
    w = Vector{T}(undef, n)

    for i in eachindex(w)
        w[i] = (-one(T))^(i - one(T)) * sinpi(T((T(2) * i - one(T)) / (T(2) * n)))
    end
    return w
end

function _chebyshevbarycentricweights2(::Type{T}, n::Int) where {T}
    w = ones(T, n)

    w[begin] = T(0.5)
    w[end] = T(0.5)
    for i in eachindex(w)
        w[i] *= (-one(T))^(i - 1)
    end

    return w
end

function _gausschebyshev(T, n::Int)
    return [cos(T((2 * k - 1) * π / (2 * n))) for k in n:-1:1]
end

function _gausschebyshev2(T, n::Int)
    return [cos(T(k * π / (n - 1))) for k in (n - 1):-1:0]
end
