"""
    BsplinePolynomialEquidistant{D,T,B}

Use the same equidistant 1-D B-spline basis along each of `D` axes.
`T` is the domain scalar type and `B` is the underlying basis type.
"""
struct BsplinePolynomialEquidistant{D,T,B,C,L} <: AbstractBspline{D,T}
    bsplines::SVector{D,B}
    ranges::SVector{D,Vector{T}}
    cartesianindices::C
    linearindices::L

    function BsplinePolynomialEquidistant{D,T}(
        polynomialorder::Int, numfunctions::Int
    ) where {D,T<:Number}
        ū = collect(LinRange(zero(T), one(T), numfunctions))
        kvec = generateKnotVec(numfunctions, polynomialorder)
        return BsplinePolynomialEquidistant{D,T}(Bspline(polynomialorder, kvec), ū)
    end

    function BsplinePolynomialEquidistant{D,T}(bspline, range::Vector) where {D,T}
        bsplines = [bspline for _ in 1:D]
        ranges = [range for _ in 1:D]
        c = CartesianIndices(Tuple([length(range) for range in ranges]))
        l = LinearIndices(Tuple([length(range) for range in ranges]))
        return new{D,T,eltype(bsplines),typeof(c),typeof(l)}(
            SVector{D}(bsplines), SVector{D}(ranges), c, l
        )
    end

    function BsplinePolynomialEquidistant{D,T,B,C,L}(
        polynomialorder::Int, numfunctions::Int
    ) where {D,T,B,C,L}
        return BsplinePolynomialEquidistant{D,eltype(T)}(polynomialorder, numfunctions)
    end
end

BsplinePolynomialEquidistant3D{T} = BsplinePolynomialEquidistant{3,T,B} where {B}

function BsplinePolynomialEquidistant(; dim::Int, degree::Int, nbasis::Int, T=Float64)
    return BsplinePolynomialEquidistant{dim,T}(degree, nbasis)
end

function increasednumpolynomials(p::BsplinePolynomialEquidistant)
    return numpolynomials(p)[1] * 2 - polynomialorders(p)[1]
end
