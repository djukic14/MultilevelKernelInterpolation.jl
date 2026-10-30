"""
    AbstractTensorProductPolynomial{D,T}

Tensor-product polynomial basis in `D` dimensions with scalar type `T`.
"""
abstract type AbstractTensorProductPolynomial{D,T} end

"""
    AbstractLagrange{D,T}

Tensor-product Lagrange basis in `D` dimensions with scalar type `T`.
"""
abstract type AbstractLagrange{D,T} <: AbstractTensorProductPolynomial{D,T} end

"""
    AbstractBarycentricLagrange{D,T}

Tensor-product Lagrange basis evaluated in barycentric form.
See J. Berrut and L. Trefethen, "Barycentric Lagrange Interpolation" (2004).
"""
abstract type AbstractBarycentricLagrange{D,T} <: AbstractLagrange{D,T} end

"""
    AbstractBspline{D,T}

Tensor-product B-spline basis in `D` dimensions with scalar type `T`.
"""
abstract type AbstractBspline{D,T} <: AbstractTensorProductPolynomial{D,T} end

"""
    polynomialorders(poly)

Return the polynomial orders as a tuple, one per spatial dimension.
"""
function polynomialorders(p::A) where {D,T,A<:AbstractLagrange{D,T}}
    return ntuple(dim -> length(range(p, dim)) - 1, Val(D))
end

"""
    numpolynomials(poly)

Return the number of basis polynomials in each spatial dimension for a polynomial `poly`.
"""
function numpolynomials(p::A) where {A<:AbstractLagrange}
    return polynomialorders(p) .+ 1
end

function numinterpolationnodes(p::A) where {D,T,A<:AbstractTensorProductPolynomial{D,T}}
    return ntuple(dim -> length(range(p, dim)), Val(D))
end

function numbasisfunctions(p::AbstractTensorProductPolynomial)
    return numpolynomials(p)
end

"""
    Base.eltype(poly::A) where {D, T, A<:AbstractTensorProductPolynomial}

Return the polynomial's scalar type.
"""
function Base.eltype(polynomial::A) where {D,T,A<:AbstractTensorProductPolynomial{D,T}}
    return Base.eltype(T)
end

"""
    ranges(poly::A) where {A<:AbstractTensorProductPolynomial}

Return the interpolation nodes in each spatial dimension for a tensor product polynomial.
"""
function ranges(p::A) where {A<:AbstractTensorProductPolynomial}
    return p.ranges
end

"""
    Base.range(poly::A, dim::Int) where {A<:AbstractTensorProductPolynomial}

Return the interpolation nodes in a specific spatial dimension for a tensor product polynomial.
"""
function Base.range(p::A, dim::Int) where {A<:AbstractTensorProductPolynomial}
    return ranges(p)[dim]
end

"""
    Base.range(poly::A, dim::Int, i::Int) where {A<:AbstractTensorProductPolynomial}

Return the interpolation node at index `i` in dimension `dim` for a tensor product polynomial.
"""
function Base.range(p::A, dim::Int, i::Int) where {A<:AbstractTensorProductPolynomial}
    return ranges(p)[dim][i]
end

"""
    dims(poly::A) where {D, T, A<:AbstractTensorProductPolynomial{D, T}}

Return the spatial dimension of a tensor product polynomial.
"""
function dims(p::A) where {D,T,A<:AbstractTensorProductPolynomial{D,T}}
    return D
end

"""
    Base.LinearIndices(poly::A) where {A<:AbstractTensorProductPolynomial}

Return the linear indices of the tensor-product basis.
"""
function Base.LinearIndices(polynomial::A) where {A<:AbstractTensorProductPolynomial}
    return polynomial.linearindices
end

"""
    Base.CartesianIndices(poly::A) where {A<:AbstractTensorProductPolynomial}

Return the Cartesian indices of the tensor-product basis.
"""
function Base.CartesianIndices(polynomial::A) where {A<:AbstractTensorProductPolynomial}
    return polynomial.cartesianindices
end

"""
    kernelbuffer(poly:AbstractTensorProductPolynomial, ::Type{T}) where {T}

Allocate one `Vector{T}` per axis, sized to that axis's basis count.
"""
function kernelbuffer(polynomial::AbstractTensorProductPolynomial, ::Type{T}) where {T}
    return Vector{T}[Vector{T}(undef, i) for i in numpolynomials(polynomial)]
end

"""
    kernelbufferderivative(poly::AbstractLagrange, ::Type{T}) where {T}

Allocate a pair of `Vector{T}` buffers per axis for basis values and derivatives.
"""
function kernelbufferderivative(
    polynomial::AbstractTensorProductPolynomial, ::Type{T}
) where {T}
    return NTuple{2,Vector{T}}[
        (
            Vector{T}(undef, length(range(polynomial, i))),
            Vector{T}(undef, length(range(polynomial, i))),
        ) for i in eachindex(ranges(polynomial))
    ]
end

"""
    getsamplingpoint!(point, poly::A, i::Int) where {A<:AbstractTensorProductPolynomial}

Write the coordinates of sampling point `i` into `point` and return it.
"""
function getsamplingpoint!(
    point, polynomial::A, i::Int
) where {A<:AbstractTensorProductPolynomial}
    c = CartesianIndices(polynomial)[i]

    for dim in eachindex(point)
        point[dim] = range(polynomial, dim, c[dim])
    end
    return point
end

# Abstract Polynomial 3D
AbstractTensorProductPolynomial3D{T} = AbstractTensorProductPolynomial{3,T}
AbstractLagrange3D{T} = AbstractLagrange{3,T}
AbstractBspline3D{T} = AbstractBspline{3,T}

function evaluateallpolynomials!(
    result, polynomial::AbstractTensorProductPolynomial{D}, newsample
) where {D}
    buffers = kernelbuffer(polynomial, eltype(newsample))

    for dim in 1:D
        evaluateall1dpolynomial!(buffers[dim], polynomial, newsample[dim], dim)
    end

    for index in CartesianIndices(result)
        value = one(eltype(result))
        for dim in 1:D
            value *= buffers[dim][index[dim]]
        end
        @inbounds result[index] = value
    end

    return result
end

"""
    increase(polynomial::A) where {A<:AbstractLagrange}

Return the same Lagrange family with twice the polynomial degree.
Accept and ignore `protrusion`: plain Lagrange bases have no extended domain.
"""
function increase(polynomial::A; kwargs...) where {A<:AbstractLagrange}
    return A(2 * polynomialorders(polynomial)[1])
end

"""
    realizedprotrusion(polynomial)

Return how far the constructed sampling domain extends beyond `[0,1]^D`, relative to
the box side length. This includes any rounding or inherited extension from
`increase`/`noincrease`, unlike the geometric protrusion supplied before construction.
The fallback is zero; buffered B-splines and scaled/shifted bases override it.
"""
realizedprotrusion(::Any) = 0

"""
    evaluateallpolynomials(poly::AbstractTensorProductPolynomial{D}, samples::AbstractVector) where {D}

Evaluate the basis on the Cartesian product of `samples` along all `D` axes.
Return a matrix with `length(samples)^D` rows and `prod(numpolynomials(poly))` columns:
one row per point and one column per basis function.
"""
function evaluateallpolynomials(
    polynomial::AbstractTensorProductPolynomial{D}, samples::AbstractVector
) where {D}
    result = Array{eltype(samples)}(
        undef, ntuple(_ -> length(samples), Val(D))..., numpolynomials(polynomial)...
    )

    samplepoint = fill(samples[begin], D)
    for sampleindex in CartesianIndices(ntuple(_ -> eachindex(samples), Val(D)))
        for dim in 1:D
            samplepoint[dim] = samples[sampleindex[dim]]
        end
        evaluateallpolynomials!(
            view(result, Tuple(sampleindex)..., ntuple(_ -> Colon(), Val(D))...),
            polynomial,
            samplepoint,
        )
    end

    return reshape(result, length(samples)^D, prod(numpolynomials(polynomial)))
end
