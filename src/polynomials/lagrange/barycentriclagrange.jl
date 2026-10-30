struct BarycentricLagrangePolynomialEquidistant{D,T,R,W,C,L} <:
       AbstractBarycentricLagrange{D,T}
    ranges::SVector{D,R}
    weights::SVector{D,W}
    cartesianindices::C
    linearindices::L
end

struct BarycentricLagrangePolynomialChebyshev{D,T,R,W,C,L} <:
       AbstractBarycentricLagrange{D,T}
    ranges::SVector{D,R}
    weights::SVector{D,W}
    cartesianindices::C
    linearindices::L
end

struct BarycentricLagrangePolynomialChebyshev2{D,T,R,W,C,L} <:
       AbstractBarycentricLagrange{D,T}
    ranges::SVector{D,R}
    weights::SVector{D,W}
    cartesianindices::C
    linearindices::L
end

function _barycentricindices(ranges, ::Val{D}) where {D}
    sizes = Tuple(length(ranges[dim]) for dim in 1:D)
    return CartesianIndices(sizes), LinearIndices(sizes)
end

function _barycentricpolynomial(
    ::Type{P}, ranges::SVector{D,R}, weights::SVector{D,W}, ::Type{T}
) where {P,D,R,W,T}
    cartesianindices, linearindices = _barycentricindices(ranges, Val(D))
    return P{D,T,R,W,typeof(cartesianindices),typeof(linearindices)}(
        ranges, weights, cartesianindices, linearindices
    )
end

function BarycentricLagrangePolynomialEquidistant{D,T}(
    polynomialorder::Int
) where {D,T<:Number}
    ranges, weights = _rangebaryweightsequidistant(polynomialorder, Val(D), Val(T))
    return _barycentricpolynomial(
        BarycentricLagrangePolynomialEquidistant, ranges, weights, T
    )
end

function BarycentricLagrangePolynomialChebyshev{D,T}(
    polynomialorder::Int
) where {D,T<:Number}
    ranges, weights = _rangebaryweightschebyshev(polynomialorder, Val(D), Val(T))
    return _barycentricpolynomial(
        BarycentricLagrangePolynomialChebyshev, ranges, weights, T
    )
end

function BarycentricLagrangePolynomialChebyshev2{D,T}(
    polynomialorder::Int
) where {D,T<:Number}
    ranges, weights = _rangebaryweightschebyshev2(polynomialorder, Val(D), Val(T))
    return _barycentricpolynomial(
        BarycentricLagrangePolynomialChebyshev2, ranges, weights, T
    )
end

function BarycentricLagrangePolynomialEquidistant{D,T,R,W,C,L}(
    polynomialorder::Int
) where {D,T<:Number,R,W,C,L}
    return BarycentricLagrangePolynomialEquidistant{D,T}(polynomialorder)
end

function BarycentricLagrangePolynomialChebyshev{D,T,R,W,C,L}(
    polynomialorder::Int
) where {D,T<:Number,R,W,C,L}
    return BarycentricLagrangePolynomialChebyshev{D,T}(polynomialorder)
end

function BarycentricLagrangePolynomialChebyshev2{D,T,R,W,C,L}(
    polynomialorder::Int
) where {D,T<:Number,R,W,C,L}
    return BarycentricLagrangePolynomialChebyshev2{D,T}(polynomialorder)
end

function BarycentricLagrangePolynomialEquidistant{D,T,R,W,C,L}(
    polynomialorder::Int
) where {D,T<:AbstractVector,R,W,C,L}
    return BarycentricLagrangePolynomialEquidistant{D,eltype(T)}(polynomialorder)
end

function BarycentricLagrangePolynomialChebyshev{D,T,R,W,C,L}(
    polynomialorder::Int
) where {D,T<:AbstractVector,R,W,C,L}
    return BarycentricLagrangePolynomialChebyshev{D,eltype(T)}(polynomialorder)
end

function BarycentricLagrangePolynomialChebyshev2{D,T,R,W,C,L}(
    polynomialorder::Int
) where {D,T<:AbstractVector,R,W,C,L}
    return BarycentricLagrangePolynomialChebyshev2{D,eltype(T)}(polynomialorder)
end

function BarycentricLagrangePolynomialEquidistant{D,T}(
    polynomialorder::Int
) where {D,T<:AbstractVector}
    return BarycentricLagrangePolynomialEquidistant{D,eltype(T)}(polynomialorder)
end

function BarycentricLagrangePolynomialChebyshev{D,T}(
    polynomialorder::Int
) where {D,T<:AbstractVector}
    return BarycentricLagrangePolynomialChebyshev{D,eltype(T)}(polynomialorder)
end

function BarycentricLagrangePolynomialChebyshev2{D,T}(
    polynomialorder::Int
) where {D,T<:AbstractVector}
    return BarycentricLagrangePolynomialChebyshev2{D,eltype(T)}(polynomialorder)
end

function BarycentricLagrangePolynomialEquidistant(; dim::Int, order::Int, T=Float64)
    return BarycentricLagrangePolynomialEquidistant{dim,T}(order)
end

function BarycentricLagrangePolynomialChebyshev(; dim::Int, order::Int, T=Float64)
    return BarycentricLagrangePolynomialChebyshev{dim,T}(order)
end

function BarycentricLagrangePolynomialChebyshev2(; dim::Int, order::Int, T=Float64)
    return BarycentricLagrangePolynomialChebyshev2{dim,T}(order)
end

# Barycentric Lagrange Polynomial Constructors

"""
    BarycentricLagrangePolynomialEquidistant3D(polynomialorder::Int)

Create an order-`polynomialorder` barycentric Lagrange basis on an equidistant 3-D grid.
"""
function BarycentricLagrangePolynomialEquidistant3D(polynomialorder::Int)
    return BarycentricLagrangePolynomialEquidistant{3,Float64}(polynomialorder)
end

"""
    BarycentricLagrangePolynomialChebyshev3D(polynomialorder::Int)

Create an order-`polynomialorder` barycentric Lagrange basis on a 3-D Chebyshev grid.
"""
function BarycentricLagrangePolynomialChebyshev3D(polynomialorder::Int)
    return BarycentricLagrangePolynomialChebyshev{3,Float64}(polynomialorder)
end

"""
    BarycentricLagrangePolynomialChebyshev3D2(polynomialorder::Int)

Create an order-`polynomialorder` barycentric Lagrange basis on a 3-D Chebyshev-II grid.
"""
function BarycentricLagrangePolynomialChebyshev3D2(polynomialorder::Int)
    return BarycentricLagrangePolynomialChebyshev2{3,Float64}(polynomialorder)
end
