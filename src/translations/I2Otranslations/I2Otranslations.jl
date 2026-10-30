"""
    translation!(tf, operator, i2otranslationtrait, testpolynomial, trialpolynomial)

Fill and return `tf` with kernel samples between the test and trial polynomial nodes.
`i2otranslationtrait` selects the kernel and component layout. Sampling-point and kernel
buffers are allocated once and reused across node pairs.
"""
function translation!(tf, operator, i2otranslationtrait, testpolynomial, trialpolynomial)
    # Each sampling buffer has its side's spatial dimension.
    buffertest = Vector{eltype(testpolynomial)}(undef, dims(testpolynomial))
    buffertrial = Vector{eltype(trialpolynomial)}(undef, dims(trialpolynomial))
    buffer = kernelbuffer(tf, i2otranslationtrait)

    triallinearindices = collect(LinearIndices(trialpolynomial))
    testlinearindices = collect(LinearIndices(testpolynomial))
    for j in triallinearindices
        getsamplingpoint!(buffertrial, trialpolynomial, j)
        for i in testlinearindices
            getsamplingpoint!(buffertest, testpolynomial, i)
            computekernel!(
                tf, i, j, operator, i2otranslationtrait, buffertest, buffertrial, buffer
            )
        end
    end

    return tf
end

"""
    kernelbuffer(tf, i2otranslationtrait)

Allocate scratch for kernel samples with the scalar type and component layout of `tf`.
"""
function kernelbuffer(::Any, ::Any) end

function kernelbuffer(tf, ::Union{∇Green3DTranslation})
    return Vector{eltype(eltype(tf))}(undef, 3)
end

function kernelbuffer(
    tf, ::Union{∇∇Green3DTranslation,DyadicGreen3DTranslation,∇CrossGreen3DTranslation}
)
    return Matrix{eltype(eltype(tf))}(undef, 3, 3)
end

"""
    computekernel!(
        tf, i, j, operator, i2otrait, buffertest, buffertrial, kernelbuffer
    )

Evaluate the kernel at `buffertest` and `buffertrial`, using `kernelbuffer` as scratch.
Write the result to entry `(i, j)` of `tf`, or the corresponding entries of its components.
"""
function computekernel!(
    tf, i, j, operator, i2o::ScalarI2OTranslation, buffertest, buffertrial, ::Any
)
    tf[i, j] = kernel(operator, i2o, buffertest, buffertrial)
    return tf
end

function computekernel!(
    tf,
    i,
    j,
    operator,
    i2o::Union{∇Green3DTranslation,DyadicGreen3DTranslation,∇∇Green3DTranslation},
    buffertest,
    buffertrial,
    kernelbuffer,
)
    kernel!(kernelbuffer, operator, i2o, buffertest, buffertrial)
    for ibuffer in eachindex(kernelbuffer)
        tf[ibuffer][i, j] = kernelbuffer[ibuffer]
    end
    return tf
end

function computekernel!(
    tf,
    i,
    j,
    operator,
    i2o::Union{∇CrossGreen3DTranslation},
    buffertest,
    buffertrial,
    kernelbuffer,
)
    kernel!(kernelbuffer, operator, i2o, buffertest, buffertrial)
    tf[1, 1][i, j] = 0
    tf[1, 2][i, j] = -kernelbuffer[3]
    tf[1, 3][i, j] = kernelbuffer[2]

    tf[2, 1][i, j] = kernelbuffer[3]
    tf[2, 2][i, j] = 0
    tf[2, 3][i, j] = -kernelbuffer[1]

    tf[3, 1][i, j] = -kernelbuffer[2]
    tf[3, 2][i, j] = kernelbuffer[1]
    tf[3, 3][i, j] = 0
    return tf
end

function kernel(operator, ::ScalarGreen3DTranslation, buffertest, buffertrial)
    return scalargreen3Dkernel(operator, buffertest, buffertrial)
end

function kernel(operator, ::ScalarGreen2DTranslation, buffertest, buffertrial)
    return scalargreen2Dkernel(operator, buffertest, buffertrial)
end

"""
    kernel!(kernelbuffer, operator, buffertest, buffertrial, i2otrait)

Evaluate `operator` at the test and trial sampling points in `buffertest` and `buffertrial`.
Store the result in `kernelbuffer`; `i2otrait` selects the kernel.
"""
function kernel!(
    kernelbuffer,
    operator,
    ::Union{∇Green3DTranslation,∇CrossGreen3DTranslation},
    buffertest,
    buffertrial,
)
    return ∇green3Dkernel!(kernelbuffer, operator, buffertest, buffertrial)
end

function kernel!(
    kernelbuffer, operator, ::Union{DyadicGreen3DTranslation}, buffertest, buffertrial
)
    return dyadicgreen3Dkernel!(kernelbuffer, operator, buffertest, buffertrial)
end

function kernel!(
    kernelbuffer, operator, ::Union{∇∇Green3DTranslation}, buffertest, buffertrial
)
    return ∇∇green3Dkernel!(kernelbuffer, operator, buffertest, buffertrial)
end

"""
Calculates g(x,y) = exp(-ik|x-y|)/(4π|x-y|).
"""
function scalargreen3Dkernel(op, x, y)
    distance = sqrt((x[1] - y[1])^2 + (x[2] - y[2])^2 + (x[3] - y[3])^2)
    return cis(im * BEAST.gamma(op) * distance) / (4π * distance)
end

"""
Evaluate `g(x,y) = -i/4 H₀⁽²⁾(k|x-y|)`, the 2-D Helmholtz Green's function.
Uses BEAST's outgoing-wave convention (`hankelh2`, `-i/4`).
For imaginary `γ`, recover a real `k` to use the faster real-argument Hankel evaluation.
"""
function scalargreen2Dkernel(op, x, y)
    γ = operatorgamma(op)
    k = iszero(real(γ)) ? imag(γ) : -im * γ

    distance = sqrt((x[1] - y[1])^2 + (x[2] - y[2])^2)

    return -im / 4 * hankelh2(0, k * distance)
end

"""
Calculates (∇/k^2)g(x,y) for the efie with g(x,y) = exp(-ik|x-y|)/(4π|x-y|).
"""
function ∇green3Dkernel!(vector, op, x::T, y::T) where {T}
    R = x - y
    normR = norm(R)
    invnormR = 1 / normR

    γ = BEAST.gamma(op)
    imk = BEAST.gamma(op) # im * k
    minusk = im * imk
    greenf = cis(minusk * normR) * invnormR / (4π)

    A = -(γ + invnormR) * greenf

    vector .= R
    vector .*= A * invnormR

    return vector
end

"""
Calculates -∇∇g(x,y) for the hyper singular kernel with g(x,y) = exp(-ik|x-y|)/(4π|x-y|).
"""
function ∇∇green3Dkernel!(dyad, op, x::T, y::T) where {T}
    R = x - y
    normR = norm(R)
    invnormR = 1 / normR
    invnormR2 = invnormR^2

    imk = BEAST.gamma(op) # im * k
    minusk = im * imk
    k = -minusk

    inv4pi = 1 / (4π)

    scalargreen = cis(minusk * normR) * invnormR * inv4pi
    A = -3 * invnormR2 - 3 * imk * invnormR + k^2
    scalaragreenA = scalargreen * A
    scalaragreenAinvnormR2 = scalaragreenA * invnormR2

    B = imk * invnormR + invnormR2
    scalargreenB = scalargreen * B

    for i in axes(dyad, 1)
        for j in axes(dyad, 2)
            dyad[i, j] = R[i] * R[j] * scalaragreenAinvnormR2
        end
    end

    for i in diagind(dyad)
        dyad[i] += scalargreenB
    end

    dyad *= -1
    return dyad
end

"""
Calculates (I+∇∇/k^2)g(x,y) for the efie with g(x,y) = exp(-ik|x-y|)/(4π|x-y|).
"""
function dyadicgreen3Dkernel!(dyad, op, x::T, y::T) where {T}
    R = x - y
    normR = norm(R)
    invnormR = 1 / normR
    invnormR2 = invnormR^2

    imk = BEAST.gamma(op) # im * k
    minusk = im * imk
    k = -minusk
    k2 = k^2

    invnormRk = invnormR / k
    invnormR2k2 = invnormR2 / k2

    inv4pi = 1 / (4π)

    scalargreen = cis(minusk * normR) * invnormR * inv4pi
    A = 3 * invnormR2k2 + 3 * im * invnormRk - 1
    scalaragreenA = scalargreen * A
    scalaragreenAinvnormR2 = scalaragreenA * invnormR2

    B = -im * invnormRk - invnormR2k2 + 1
    scalargreenB = scalargreen * B

    for i in axes(dyad, 1)
        for j in axes(dyad, 2)
            dyad[i, j] = R[i] * R[j] * scalaragreenAinvnormR2
        end
    end

    for i in diagind(dyad)
        dyad[i] += scalargreenB
    end

    return dyad
end
