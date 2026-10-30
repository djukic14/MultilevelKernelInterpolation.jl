"""
    bspline(p::AbstractBspline, i::Int)

Return the one-dimensional B-spline basis along axis `i`.
"""
function bspline(p::AbstractBspline, i::Int)
    return p.bsplines[i]
end

function _checkbsplinedomain(spline::NURBS.Basis, x)
    lo, hi = first(spline.knotVec), last(spline.knotVec)
    lo <= x <= hi || throw(
        DomainError(
            x,
            "B-spline evaluation requires $lo <= x <= $hi; enlarge the buffer to cover this point.",
        ),
    )
    return nothing
end

function polynomialorders(p::A) where {D,T,A<:AbstractBspline{D,T}}
    return ntuple(dim -> degree(bspline(p, dim)), Val(D))
end

function numpolynomials(polynomial::A) where {D,T,A<:AbstractBspline{D,T}}
    return ntuple(dim -> NURBS.numBasisFunctions(polynomial.bsplines[dim]), Val(D))
end

function numbasisfunctions(polynomial::A) where {A<:AbstractBspline}
    return numpolynomials(polynomial)
end

"""
    evaluateall1dpolynomial!(result, polynomial::AbstractBspline, x, dim; extrapolate=true)

Evaluate the B-spline basis at `x`. Points outside its domain throw `DomainError`;
buffered bases include their buffer. `extrapolate` is accepted for the common
polynomial interface, but does not extend a B-spline's domain.
"""
function evaluateall1dpolynomial!(
    result, polynomial::AbstractBspline, newsample::T, dim::Int; extrapolate=true
) where {T}
    spline = bspline(polynomial, dim)
    _checkbsplinedomain(spline, newsample)
    for i in eachindex(result)
        result[i] = evalNaive(spline, i, newsample)
    end
    return result
end

function evaluateall1dpolynomialderivative!(
    result, polynomial::AbstractBspline, newsample, dim::Int; extrapolate::Bool=true
)
    evaluateall1dpolynomial!(result[1], polynomial, newsample, dim; extrapolate=extrapolate)
    spline = bspline(polynomial, dim)
    for i in eachindex(result[2])
        result[2][i] = evalNaiveDerivative(spline, i, 1, newsample)
    end
    return result
end

"""
    increase(p::A) where {A<:AbstractBspline}

Return the same B-spline family with more basis functions.
Accept and ignore `protrusion`: plain B-splines have no extra domain buffer.
"""
function increase(p::A; kwargs...) where {A<:AbstractBspline}
    return A(polynomialorders(p)[1], increasednumpolynomials(p))
end
