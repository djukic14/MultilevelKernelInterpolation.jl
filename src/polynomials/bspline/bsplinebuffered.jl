struct BufferedBspline{B}
    bspline::B
    numbufferpoints::Int
end

function Base.eltype(bspline::BufferedBspline{B}) where {B}
    return Base.eltype(bspline.bspline.knotVec)
end

"""
    BufferedBspline(b::Int, numbufferpoints::Int, p::Int)

Create a degree-`p` B-spline with `b` original basis functions and
`numbufferpoints` extra knots beyond each end of `[0,1]`.
Return the basis together with its buffer and original basis counts.
"""
function BufferedBspline(b::Int, numbufferpoints::Int, p::Int, ::Type{T}) where {T}
    @assert b > p "$b must be greater than p $p."
    bspline = Bspline(p, T.(generateKnotVec(b + 2 * numbufferpoints, p)))
    return BufferedBspline(bspline, numbufferpoints)
end

function NURBS.degree(bspline::BufferedBspline)
    return NURBS.degree(bspline.bspline)
end

"""
    originalnumpolynomials(bspline::BufferedBspline)

Return the number of original B-spline basis functions before adding buffer knots.
"""
function originalnumpolynomials(bspline::BufferedBspline)
    return numBasisFunctions(bspline) - 2 * numbufferpoints(bspline)
end

function bspline(bspline::BufferedBspline)
    return bspline.bspline
end

function _checkbsplinedomain(bspl::BufferedBspline, x)
    extension = _leftboundary(bspl)
    lo, hi = -extension, 1 + extension
    lo <= x <= hi || throw(
        DomainError(
            x,
            "Buffered B-spline evaluation requires $lo <= x <= $hi; enlarge the buffer to cover this point.",
        ),
    )
    return nothing
end

function _bufferedcoordinate(bspl::BufferedBspline, x)
    _checkbsplinedomain(bspl, x)
    leftboundary = _leftboundary(bspl)
    # Roundoff in the coordinate change must not move a valid endpoint outside [0,1].
    return clamp((x + leftboundary) / (1 + 2 * leftboundary), 0, 1)
end

function NURBS.evalNaive(bspl::BufferedBspline, i::Int, x::Number)
    return NURBS.evalNaive(bspline(bspl), i, _bufferedcoordinate(bspl, x))
end

function NURBS.evalNaiveDerivative(bspl::BufferedBspline, i::Int, k::Int, x::Number)
    coordinate = _bufferedcoordinate(bspl, x)
    scale = 1 + 2 * _leftboundary(bspl)
    return NURBS.evalNaiveDerivative(bspline(bspl), i, k, coordinate) / scale^k
end

function _leftboundary(bspl::BufferedBspline)
    return numbufferpoints(bspl) / (originalnumpolynomials(bspl) - degree(bspl))
end

function _leftboundary(poly)
    return 0
end

function NURBS.numBasisFunctions(bspl::BufferedBspline)
    return numBasisFunctions(bspline(bspl))
end

struct BufferedBsplineStabilityPolicy{K,C}
    knots_to_remove::K
    removal_cutoffs::C
end

const DEFAULT_BUFFERED_BSPLINE_STABILITY_POLICY = BufferedBsplineStabilityPolicy(
    [0, 0, 1, 1, 2, 2, 3, 3, 4, 5], [0, 0, 20, 15, 11, 10, 15, 16, 18, 20]
)

const knotstoremoveforstability = DEFAULT_BUFFERED_BSPLINE_STABILITY_POLICY.knots_to_remove
const cutoffforremoval = DEFAULT_BUFFERED_BSPLINE_STABILITY_POLICY.removal_cutoffs

function knotstoremove(p::Int, b::Int)
    return knotstoremove(DEFAULT_BUFFERED_BSPLINE_STABILITY_POLICY, p, b)
end

function knotstoremove(policy::BufferedBsplineStabilityPolicy, p::Int, b::Int)
    p == 1 && return 0
    p == 2 && return 0

    knotstoremove = policy.knots_to_remove[p]

    if b <= p + knotstoremove * 2
        return 0
    else
        return knotstoremove
    end
end

"""
    AbstractBsplineBufferPolicy

Policy for choosing the buffer of a `BufferedBsplinePolynomialEquidistant`.
Use `NoBuffer`, `ExplicitBuffer`, or `AutoBufferFromProtrusion`;
`resolvebuffer` turns the policy into an integer count.
"""
abstract type AbstractBsplineBufferPolicy end

"""
    NoBuffer <: AbstractBsplineBufferPolicy

Use zero buffer points, even with nonzero protrusion.
Suitable only when the basis does not need to extend beyond its nominal box.
"""
struct NoBuffer <: AbstractBsplineBufferPolicy end

"""
    ExplicitBuffer(points) <: AbstractBsplineBufferPolicy

Use exactly `points` buffer points at every level, regardless of protrusion.
`points` must be nonnegative.
"""
struct ExplicitBuffer <: AbstractBsplineBufferPolicy
    points::Int

    function ExplicitBuffer(points::Int)
        points >= 0 ||
            throw(ArgumentError("ExplicitBuffer requires points >= 0, got $points"))
        return new(points)
    end
end

"""
    AutoBufferFromProtrusion(minimum=0) <: AbstractBsplineBufferPolicy

Choose the buffer from geometric protrusion, with at least `minimum` points.
`minimum` must be nonnegative. Tree construction obtains protrusion through
`H2Trees.maxprotrusion` and `H2Trees.BEASTProtrusionFunctor`.
"""
struct AutoBufferFromProtrusion <: AbstractBsplineBufferPolicy
    minimum::Int

    function AutoBufferFromProtrusion(minimum::Int)
        minimum >= 0 || throw(
            ArgumentError("AutoBufferFromProtrusion requires minimum >= 0, got $minimum"),
        )
        return new(minimum)
    end
end
AutoBufferFromProtrusion() = AutoBufferFromProtrusion(0)

"""
    resolvebuffer(policy::AbstractBsplineBufferPolicy, p, b, protrusion)

Return an integer buffer count for degree `p`, `b` basis functions, and geometric
`protrusion` measured relative to the tree box side length.
"""
function resolvebuffer(::NoBuffer, p, b, protrusion)
    return 0
end

function resolvebuffer(policy::ExplicitBuffer, p, b, protrusion)
    return policy.points
end

function resolvebuffer(policy::AutoBufferFromProtrusion, p, b, protrusion)
    return max(policy.minimum, ceil(Int, protrusion * (b - p)))
end

"""
    BufferedBsplinePolynomialEquidistant{D,T,B}

Tensor-product B-spline basis with equidistant knots and extra knots beyond `[0,1]`.
`D` is the spatial dimension, `T` the scalar type, and `B` the underlying 1-D basis type.

The requested buffer policy is stored as a field, not a type parameter: grids built
with different policies can still be Toeplitz-compatible. Policy dispatch occurs only
at construction, not during multiplication.
"""
struct BufferedBsplinePolynomialEquidistant{D,T,B,C,L} <: AbstractBspline{D,T}
    bsplines::SVector{D,B}
    ranges::SVector{D,Vector{T}}
    cartesianindices::C
    linearindices::L
    policy::AbstractBsplineBufferPolicy

    function BufferedBsplinePolynomialEquidistant{D,T}(
        polynomialorder::Int, numfunctions::Int, bufferpoints::Int
    ) where {D,T<:Number}
        return BufferedBsplinePolynomialEquidistant{D,T}(
            polynomialorder, numfunctions, bufferpoints, ExplicitBuffer(bufferpoints)
        )
    end

    function BufferedBsplinePolynomialEquidistant{D,T}(
        polynomialorder::Int,
        numfunctions::Int,
        policy::AbstractBsplineBufferPolicy;
        protrusion=0.0,
    ) where {D,T<:Number}
        bufferpoints = resolvebuffer(policy, polynomialorder, numfunctions, protrusion)
        return BufferedBsplinePolynomialEquidistant{D,T}(
            polynomialorder, numfunctions, bufferpoints, policy
        )
    end

    # Preserve the requested policy when rebuilding with a newly resolved buffer count.
    function BufferedBsplinePolynomialEquidistant{D,T}(
        polynomialorder::Int,
        numfunctions::Int,
        bufferpoints::Int,
        policy::AbstractBsplineBufferPolicy,
    ) where {D,T<:Number}
        leftboundary = T(-bufferpoints / (numfunctions - polynomialorder))
        rightboundary = one(T) - leftboundary
        bspline = BufferedBspline(numfunctions, bufferpoints, polynomialorder, T)

        b = numBasisFunctions(bspline)
        _knotstoremove = MultilevelKernelInterpolation.knotstoremove(
            polynomialorder, numfunctions
        )

        b -= 2 * _knotstoremove
        ū = T.(collect(LinRange(leftboundary, rightboundary, b)))

        return BufferedBsplinePolynomialEquidistant{D,T}(bspline, ū, policy)
    end

    function BufferedBsplinePolynomialEquidistant{D,T}(
        bspline,
        range::Vector,
        policy::AbstractBsplineBufferPolicy=ExplicitBuffer(numbufferpoints(bspline)),
    ) where {D,T}
        bsplines = [bspline for _ in 1:D]
        ranges = [range for _ in 1:D]
        c = CartesianIndices(Tuple([length(range) for range in ranges]))
        l = LinearIndices(Tuple([length(range) for range in ranges]))
        return new{D,T,eltype(bsplines),typeof(c),typeof(l)}(
            SVector{D}(bsplines), SVector{D}(ranges), c, l, policy
        )
    end

    function BufferedBsplinePolynomialEquidistant{D,T,B,C,L}(
        polynomialorder::Int, numfunctions::Int, bufferpoints::Int
    ) where {D,T,B,C,L}
        return BufferedBsplinePolynomialEquidistant{D,eltype(T)}(
            polynomialorder, numfunctions, bufferpoints
        )
    end

    function BufferedBsplinePolynomialEquidistant{D,T,B,C,L}(
        polynomialorder::Int,
        numfunctions::Int,
        bufferpoints::Int,
        policy::AbstractBsplineBufferPolicy,
    ) where {D,T,B,C,L}
        return BufferedBsplinePolynomialEquidistant{D,eltype(T)}(
            polynomialorder, numfunctions, bufferpoints, policy
        )
    end
end

# function numpolynomials(bspline::A) where {A<:BufferedBsplinePolynomialEquidistant}
#     return Tuple([length(range) for range in bspline.ranges])
# end

BufferedBsplinePolynomialEquidistant3D{T} =
    BufferedBsplinePolynomialEquidistant{3,T,B} where {B}

function BufferedBsplinePolynomialEquidistant(;
    dim::Int,
    degree::Int,
    nbasis::Int,
    buffer::Union{Int,AbstractBsplineBufferPolicy},
    T=Float64,
)
    return BufferedBsplinePolynomialEquidistant{dim,T}(degree, nbasis, buffer)
end

"""
    bufferpolicy(poly::BufferedBsplinePolynomialEquidistant)

Return the requested buffer policy, not the realized count from `numbufferpoints`.
`increase` and `noincrease` resolve this policy again for each level's protrusion.
"""
function bufferpolicy(poly::BufferedBsplinePolynomialEquidistant)
    return poly.policy
end

"""
    numbufferpoints(bspline)

Return the number of buffer points added beyond [0,1] on each side of the B-spline.
"""
function numbufferpoints(bspline::BufferedBspline)
    return bspline.numbufferpoints
end

function numbufferpoints(bspl::BufferedBsplinePolynomialEquidistant)
    return numbufferpoints(bspline(bspl, 1))
end

function numinterpolationnodes(p::BufferedBsplinePolynomialEquidistant)
    return Tuple(length.(ranges(p)))
end

function increasednumpolynomials(p::BufferedBsplinePolynomialEquidistant)
    return (numpolynomials(p)[1] - 2 * numbufferpoints(p)) * 2 - polynomialorders(p)[1]
end

"""
    realizedprotrusion(p::BufferedBsplinePolynomialEquidistant)

Return the realized domain extension relative to the nominal box side length.
Uses the constructed buffer count, including rounding, rather than the input protrusion.
`_leftboundary` computes the ratio from the underlying 1-D basis.
"""
function realizedprotrusion(p::BufferedBsplinePolynomialEquidistant)
    return _leftboundary(bspline(p, 1))
end

"""
    increase(poly::BufferedBsplinePolynomialEquidistant; protrusion=0.0)

Grow the basis for the next coarser level and resolve `bufferpolicy(poly)` against
`protrusion`. Explicit buffers stay fixed; automatic buffers can grow.
"""
function increase(poly::A; protrusion=0.0) where {A<:BufferedBsplinePolynomialEquidistant}
    p = polynomialorders(poly)[1]
    b = increasednumpolynomials(poly)
    policy = bufferpolicy(poly)
    buffer = resolvebuffer(policy, p, b, protrusion)
    return A(p, b, buffer, policy)
end

"""
    noincrease(poly::BufferedBsplinePolynomialEquidistant; protrusion=0.0)

Reset to the original basis count and resolve `bufferpolicy(poly)` against `protrusion`.
The requested policy is preserved, just as in `increase`.
"""
function noincrease(poly::A; protrusion=0.0) where {A<:BufferedBsplinePolynomialEquidistant}
    p = polynomialorders(poly)[1]
    b = originalnumpolynomials(bspline(poly, 1))
    policy = bufferpolicy(poly)
    buffer = resolvebuffer(policy, p, b, protrusion)
    return A(p, b, buffer, policy)
end

"""
    bsplinebuffersummary(treepolynomials, levels, protrusion=zeros(length(levels)))

Return one diagnostic record per level: polynomial `degree`, basis `count`, requested
buffer `policy`, `realizedbuffer`, `realizedprotrusion`, and input `protrusion`.
The three inputs must have equal lengths. Use this to compare the requested geometry
with the buffer actually built; see `bspline_buffer.md`.
"""
function bsplinebuffersummary(
    treepolynomials,
    levels,
    protrusion=zeros(eltype(first(treepolynomials)), length(levels)),
)
    if !(length(treepolynomials) == length(levels) == length(protrusion))
        throw(
            ArgumentError(
                "bsplinebuffersummary requires treepolynomials, levels, and protrusion to have " *
                "the same length, got $(length(treepolynomials)), $(length(levels)), and " *
                "$(length(protrusion)) -- a mismatch here would otherwise silently truncate to " *
                "the shortest one via `zip`, hiding exactly the kind of misalignment this " *
                "diagnostic exists to surface.",
            ),
        )
    end

    return [
        (
            level=level,
            degree=polynomialorders(poly)[1],
            count=numpolynomials(poly)[1],
            policy=bufferpolicy(poly),
            realizedbuffer=numbufferpoints(poly),
            realizedprotrusion=realizedprotrusion(poly),
            protrusion=pro,
        ) for (level, poly, pro) in zip(levels, treepolynomials, protrusion)
    ]
end
