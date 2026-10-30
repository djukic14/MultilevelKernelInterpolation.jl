module LagrangeReference
using MultilevelKernelInterpolation
using StaticArrays

macro LagrangePolynomial(name, ranges)
    quote
        struct $name{D,T} <: MultilevelKernelInterpolation.AbstractLagrange{D,T}
            ranges::SVector{D,T}

            function $name{D,T}(p::Int) where {D,T<:Number}
                ranges = $ranges(p, Val{D}(), Val{T}())[1]
                return new{D,Base.eltype(ranges)}(ranges)
            end

            function $name{D,T}(polynomialorder::Int) where {D,T<:AbstractVector}
                return $name{D,eltype(T)}(polynomialorder)
            end
        end
    end
end

@LagrangePolynomial LagrangePolynomialEquidistant MultilevelKernelInterpolation._rangebaryweightsequidistant
@LagrangePolynomial LagrangePolynomialChebyshev MultilevelKernelInterpolation._rangebaryweightschebyshev
@LagrangePolynomial LagrangePolynomialChebyshev2 MultilevelKernelInterpolation._rangebaryweightschebyshev2

"""
```math
∏_{j=1, index ≠ j}^(p) (x - x_j) / (x_index - x_j)
```

With `p` the polynomial order `x` the new sample, and the sampling points `x_j`.
"""
function evaluate1dlagrangepolynomial(
    samplingpoints, newsample::T, index::Int; extrapolate=true
) where {T}
    result = T(1)
    samplingpointatindex = samplingpoints[index]

    newsample ≈ samplingpointatindex && return result

    if extrapolate == false
        if newsample < 0 || newsample > 1
            return zero(T)
        end
    end

    for (i, sample) in enumerate(samplingpoints)
        i == index && continue

        denominator = samplingpointatindex - sample
        denominator ≈ T(0) && return T(0)
        result *= (newsample - sample) / denominator
    end
    return result
end

function evaluate1dlagrangepolynomialderivative(
    samplingpoints, newsample::T, index::Int
) where {T}
    polynomial = evaluate1dlagrangepolynomial(samplingpoints, newsample, index)
    samplingpointatindex = samplingpoints[index]

    denominator = one(T)
    nominator = zero(T)
    for (k, samplek) in enumerate(samplingpoints)
        k == index && continue

        denominator *= samplingpointatindex - samplek

        productl = one(T)
        for (l, samplel) in enumerate(samplingpoints)
            l == index && continue
            l == k && continue
            productl *= newsample - samplel
        end
        nominator += productl
    end

    return polynomial, nominator / denominator
end

function MultilevelKernelInterpolation.evaluateall1dpolynomial!(
    buffer,
    polynomial::MultilevelKernelInterpolation.AbstractLagrange,
    newsample,
    dim::Int;
    extrapolate=true,
)
    for i in eachindex(buffer)
        buffer[i] = evaluate1dlagrangepolynomial(
            MultilevelKernelInterpolation.range(polynomial, dim),
            newsample,
            i;
            extrapolate=extrapolate,
        )
    end
    return buffer
end

function MultilevelKernelInterpolation.evaluateall1dpolynomialderivative!(
    buffer, polynomial::MultilevelKernelInterpolation.AbstractLagrange, newsample, dim::Int
)
    polynomialbuffer, derivativebuffer = buffer

    for i in eachindex(polynomialbuffer)
        p, b = evaluate1dlagrangepolynomialderivative(
            MultilevelKernelInterpolation.range(polynomial, dim), newsample, i
        )
        polynomialbuffer[i] = p
        derivativebuffer[i] = b
    end

    return (polynomialbuffer, derivativebuffer)
end

end # module LangrangeReference
