# Compose a plain Toeplitz coupling with B-spline basis conversion.
# Both sides share a grid; key conversion matrices by level, not node count.
# Modulation combines basis conversion with the forward/inverse FFT.
struct BsplineToeplitzModulation{LU,TLU} <: H2Factory.AbstractI2OModulation
    interpolationmatriceslu::LU        # level => interpolation inverse
    transposeinterpolationmatriceslu::TLU
end

function ToeplitzI2OTranslationCollection(
    operator,
    tree,
    treepolynomials::A,
    translatingplan::AbstractPlan,
    testspace,
    trialspace;
    verbose=true,
    symmetryreduced=false,
) where {A<:AbstractArray{<:AbstractBspline}}
    return _bsplinetoeplitzi2otranslation(
        operator,
        tree,
        treepolynomials,
        translatingplan;
        verbose=verbose,
        couplingbuilder=_couplingbuilder(symmetryreduced),
    )
end

function ToeplitzI2OTranslationCollection(
    operator,
    tree,
    treepolynomials::A,
    translatingplan::AbstractPlan,
    space;
    verbose=true,
    symmetryreduced=false,
) where {A<:AbstractArray{<:AbstractBspline}}
    return _bsplinetoeplitzi2otranslation(
        operator,
        tree,
        treepolynomials,
        translatingplan;
        verbose=verbose,
        couplingbuilder=_couplingbuilder(symmetryreduced),
    )
end

# Read-only routes use the same B-spline selection as the pair-returning constructors.
function toeplitzi2ocoupling(
    operator,
    tree,
    treepolynomials::A,
    translatingplan::AbstractPlan,
    testspace,
    trialspace;
    verbose=true,
    symmetryreduced=false,
) where {A<:AbstractArray{<:AbstractBspline}}
    return _bsplinetoeplitzcoupling(
        operator,
        tree,
        treepolynomials,
        translatingplan;
        verbose=verbose,
        couplingbuilder=_couplingbuilder(symmetryreduced),
    )
end

function toeplitzi2ocoupling(
    operator,
    tree,
    treepolynomials::A,
    translatingplan::AbstractPlan,
    space;
    verbose=true,
    symmetryreduced=false,
) where {A<:AbstractArray{<:AbstractBspline}}
    return _bsplinetoeplitzcoupling(
        operator,
        tree,
        treepolynomials,
        translatingplan;
        verbose=verbose,
        couplingbuilder=_couplingbuilder(symmetryreduced),
    )
end

# Keep the four-argument builder plain: the B-spline builder calls it internally.
# Select B-splines on the space-taking methods to avoid recursion.
function _bsplinetoeplitzi2otranslation(
    operator,
    tree,
    treepolynomials,
    translatingplan::AbstractPlan;
    verbose=true,
    couplingbuilder=toeplitzi2ocoupling,
)
    coupling = _bsplinetoeplitzcoupling(
        operator,
        tree,
        treepolynomials,
        translatingplan;
        verbose=verbose,
        couplingbuilder=couplingbuilder,
    )
    return coupling,
    i2ochannelfor(
        coupling, treepolynomials, translatingplan, eltype(operatorgamma(operator))
    )
end

# Build read-only coupling and modulation; workspace scratch is allocated separately.
# couplingbuilder selects reduced or unreduced storage without changing modulation.
function _bsplinetoeplitzcoupling(
    operator,
    tree,
    treepolynomials,
    translatingplan::AbstractPlan;
    verbose=true,
    couplingbuilder=toeplitzi2ocoupling,
)
    i2o = couplingbuilder(operator, tree, treepolynomials, translatingplan; verbose=verbose)

    T = eltype(operatorgamma(operator))

    translatinglevels = levels(translatingplan)
    interpolationmatriceslu = []
    transposeinterpolationmatriceslu = []

    for polynomial in treepolynomials[translatinglevels]
        push!(
            interpolationmatriceslu,
            transposeinterpolationinverse(
                transposeinterpolationmatrix(polynomial), polynomial
            ),
        )
        push!(
            transposeinterpolationmatriceslu,
            interpolationinverse(interpolationmatrix(polynomial), polynomial),
        )
    end

    interpolationmatriceslu = convert(
        Vector{_getminmomenttype(interpolationmatriceslu)}, interpolationmatriceslu
    )
    transposeinterpolationmatriceslu = convert(
        Vector{_getminmomenttype(transposeinterpolationmatriceslu)},
        transposeinterpolationmatriceslu,
    )

    return H2Factory.I2OTranslation(
        i2o,
        BsplineToeplitzModulation(
            Dict(zip(translatinglevels, interpolationmatriceslu)),
            Dict(zip(translatinglevels, transposeinterpolationmatriceslu)),
        ),
    )
end

"""
    bsplinetoeplitzscratch(T, treepolynomials, translatinglevels)

Allocate Toeplitz B-spline conversion scratch as a level-ordered vector of chunk buffers.
Each buffer holds `(toreduced, res, toraw)`; each conversion set contains `D-1`
intermediates. Scratch is workspace-local.

Run inside `withschedulechunks` because allocation reads `H2Factory.nchunks(level)`.
"""
function bsplinetoeplitzscratch(T, treepolynomials, translatinglevels)
    buffers = []
    for (level, polynomial) in zip(translatinglevels, treepolynomials[translatinglevels])
        s = size(LinearIndices(polynomial))

        b = numpolynomials(polynomial)[1]
        b2 = s[1]
        D = dims(polynomial)

        # Both conversion directions use res in the reduced basis.
        buffer = (
            toreduced=separablescratch(T, b, b2, D),
            res=zeros(T, s),
            toraw=separablescratch(T, b2, b, D),
        )

        push!(buffers, [deepcopy(buffer) for _ in 1:(H2Factory.nchunks(level))])
    end
    return convert(Vector{typeof(buffers[1])}, buffers)
end

"""
    interpolationinverse(interpolationmatrix, bspline)

Build the inverse conversion for the reduced B-spline basis.
Degree 1 returns `I`. Active knot removal composes the inverse with cheap averaging.
Below the empirical order-7 crossover, store a combined Kronecker matrix; otherwise
keep the 1-D factor for separable application.
"""
function interpolationinverse(interpolationmatrix, bspline)
    degree(MultilevelKernelInterpolation.bspline(bspline, 1))[1] == 1 && return I
    invimatrix = inv(collect(interpolationmatrix))

    if length(range(bspline, 1)) != numpolynomials(bspline)[1]
        invimatrix =
            i2icheapaveraging(MultilevelKernelInterpolation.bspline(bspline, 1).bspline)' *
            invimatrix
        return SMatrix{size(invimatrix, 1),size(invimatrix, 2)}(invimatrix)
    end

    # Keep a combined matrix only below the empirical crossover, measured in 3-D.
    if numpolynomials(bspline)[1] >= 7
        return SMatrix{size(invimatrix, 1),size(invimatrix, 2)}(invimatrix)
    else
        # Form one tensor-product factor per spatial axis.
        return computend(ntuple(_ -> invimatrix, dims(bspline))...)
    end
end

function transposeinterpolationinverse(interpolationmatrix, bspline)
    if length(range(bspline, 1)) != numpolynomials(bspline)[1]
        invimatrix = inv(collect(interpolationmatrix))
        invimatrix =
            invimatrix *
            i2icheapaveraging(MultilevelKernelInterpolation.bspline(bspline, 1).bspline)
        return SMatrix{size(invimatrix, 1),size(invimatrix, 2)}(invimatrix)
    end
    return interpolationinverse(interpolationmatrix, bspline)
end

# Fuse conversion with the Toeplitz FFT. mlu == I needs only the FFT round trip.
# The shared grid needs no side-specific modulation data.
function H2Factory.modulate_in!(
    in::M,
    level::Int,
    maptrait,
    modulation::BsplineToeplitzModulation,
    coupling,
    channel,
    ::Scheduler,
    buffer,
) where {M<:AbstractToeplitzMoment}
    fftplan = buffer.fftplan
    mlu = modulation.interpolationmatriceslu[level]

    if mlu == I
        for i in eachindex(in)
            toeplitzmoment(in, i) .= 0
            fftplan * _copytobuffer!(toeplitzmoment(in, i), in[i])
        end
        return nothing
    end

    scratch = buffer.buffer

    for i in eachindex(in)
        res = separable1dinterpolations!(scratch.res, mlu, in[i], scratch.toreduced)
        toeplitzmoment(in, i) .= 0
        fftplan * _copytobuffer!(toeplitzmoment(in, i), res)
    end
    return nothing
end

function H2Factory.modulate_out!(
    out::M,
    level::Int,
    maptrait,
    modulation::BsplineToeplitzModulation,
    coupling,
    channel,
    ::Scheduler,
    buffer,
) where {M<:AbstractToeplitzMoment}
    tmlu = modulation.transposeinterpolationmatriceslu[level]
    ifftplan = buffer.ifftplan

    if tmlu == I
        for i in eachindex(out)
            _copytomoment!(out[i], ifftplan * toeplitzmoment(out, i))
        end
        return nothing
    end

    scratch = buffer.buffer
    for i in eachindex(out)
        _copytomoment!(scratch.res, ifftplan * toeplitzmoment(out, i))
        separable1dinterpolations!(out[i], tmlu, scratch.res, scratch.toraw)
    end
    return nothing
end

"""
    separable1dinterpolations!(res, inverse::SMatrix, a, buffers)
    separable1dinterpolations!(res, inverse::Matrix, a, buffers)

Convert coefficient array `a` into the basis selected by `inverse`, writing to `res`.
An `SMatrix` is a 1-D factor applied along each axis; a `Matrix` is the combined
Kronecker product applied to the flattened array.

For the separable path, reuse the `D-1` arrays from [`separablescratch`](@ref).
Axes run last-to-first. Intermediate `k` has shape `from^(D-k) × to^k`.
The combined path ignores `buffers`. Omitting scratch allocates it for one-off use.
"""
function separable1dinterpolations!(res, inverse::SMatrix, a, buffers)
    # A compile-time axis keeps the inner fibre multiplication type-stable.
    return _interpolateaxis!(res, inverse, a, buffers, Val(ndims(a)))
end

# Axis 1 writes res last. In-place use is safe for D > 1 because it reads scratch.
# D == 1 has no scratch and must not alias source and destination.
function _interpolateaxis!(res, inverse, source, buffers, ::Val{1})
    return _applyalongaxis!(res, inverse, source, Val(1))
end

function _interpolateaxis!(res, inverse, source, buffers, ::Val{d}) where {d}
    staged = _applyalongaxis!(buffers[ndims(source) - d + 1], inverse, source, Val(d))
    return _interpolateaxis!(res, inverse, staged, buffers, Val(d - 1))
end

# Multiply each fibre along axis d; other axes keep their source indices.
function _applyalongaxis!(
    dest, matrix, source::AbstractArray{<:Any,N}, ::Val{d}
) where {N,d}
    pre = CartesianIndices(ntuple(k -> axes(source, k), Val(d - 1)))
    post = CartesianIndices(ntuple(k -> axes(source, d + k), Val(N - d)))

    for J in post, I in pre
        LinearAlgebra.mul!(view(dest, I, :, J), matrix, view(source, I, :, J))
    end
    return dest
end

function separable1dinterpolations!(res, inverse::Matrix, a, buffers)
    LinearAlgebra.mul!(view(res, :), inverse, view(a, :))
    return res
end

# Allocate scratch only for the separable form; a combined matrix does not need it.
function separable1dinterpolations!(res, inverse::SMatrix, a)
    return separable1dinterpolations!(
        res, inverse, a, separablescratch(eltype(a), inverse, ndims(a))
    )
end

separable1dinterpolations!(res, inverse::Matrix, a) =
    separable1dinterpolations!(res, inverse, a, nothing)

"""
    separablescratch(T, from::Int, to::Int, D::Int)
    separablescratch(T, inverse, D::Int)

Allocate the `D-1` intermediate arrays for a separable conversion from `from` to `to`
entries per axis. Buffer `k` has shape `from^(D-k) × to^k`.
Return a vector whose entries share the concrete type `Array{T,D}`.
"""
function separablescratch(T, from::Int, to::Int, D::Int)
    return [zeros(T, ntuple(axis -> axis <= D - k ? from : to, D)) for k in 1:(D - 1)]
end

separablescratch(T, inverse, D::Int) =
    separablescratch(T, size(inverse, 2), size(inverse, 1), D)

"""
    BsplineToeplitzI2OChannel{T,B}

Pair a Toeplitz FFT channel with workspace-local B-spline conversion scratch.
"""
struct BsplineToeplitzI2OChannel{T,B}
    toeplitzi2ochannel::T
    buffers::B
end

function BsplineToeplitzI2OChannel(toeplitzi2ochannel, buffers, nodelevels)
    @assert length(buffers) == length(nodelevels)
    isempty(buffers) && return BsplineToeplitzI2OChannel([], buffers)

    bufferchannels = []
    for i in eachindex(buffers)
        push!(bufferchannels, createchannel(buffers[i]))
    end

    bufferchannels = convert(Vector{typeof(bufferchannels[1])}, bufferchannels)

    return BsplineToeplitzI2OChannel(
        toeplitzi2ochannel, Dict(zip(nodelevels, bufferchannels))
    )
end

function H2Factory.borrowbuffer(
    channel::BsplineToeplitzI2OChannel, level::Int, chunkid::Int, maptrait
)
    !haskey(channel.buffers, level) &&
        return (fftplan=EmptyChannels(), ifftplan=EmptyChannels(), buffer=EmptyChannels())
    toeplitzelememt = H2Factory.borrowbuffer(
        toeplitzchannel(channel), level, chunkid, maptrait
    )

    return (
        fftplan=toeplitzelememt.fftplan,
        ifftplan=toeplitzelememt.ifftplan,
        buffer=takebuffer!(channel, level, chunkid),
    )
end

function toeplitzchannel(b::BsplineToeplitzI2OChannel)
    return b.toeplitzi2ochannel
end

function takebuffer!(channel::BsplineToeplitzI2OChannel, level::Int, chunkid::Int)
    return H2Factory.borrowbuffer(channel.buffers[level], chunkid)
end

function takeifftplan!(
    channel::BsplineToeplitzI2OChannel, level::Int, chunkid::Int, maptrait
)
    return takeifftplan!(toeplitzchannel(channel), level, chunkid, maptrait)
end

function takefftplan!(
    channel::BsplineToeplitzI2OChannel, level::Int, chunkid::Int, maptrait
)
    return takefftplan!(toeplitzchannel(channel), level, chunkid, maptrait)
end
