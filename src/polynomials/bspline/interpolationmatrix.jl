function evalbsplines(
    bspline, evalpoints; basisfunctions=1:numBasisFunctions(bspline), extrapolate=false
)
    return [
        (_checkbsplinedomain(bspline, x); evalNaive(bspline, i, x)) for
        x in evalpoints, i in basisfunctions
    ]
end

"""
    interpolationmatrix(spline::AbstractBspline)

Return the banded interpolation matrix for B-spline interpolation.
"""
function interpolationmatrix(spline::AbstractBspline)
    return BandedMatrix{eltype(spline)}(
        evalbsplines(bspline(spline, 1), MultilevelKernelInterpolation.range(spline, 1)),
        (degree(bspline(spline, 1)) - 1, degree(bspline(spline, 1)) - 1),
    )
end

function interpolationmatrix(spline::BufferedBsplinePolynomialEquidistant)
    p = degree(bspline(spline, 1))

    if length(range(spline, 1)) != numBasisFunctions(bspline(spline, 1))
        _knotstoremove = knotstoremove(
            DEFAULT_BUFFERED_BSPLINE_STABILITY_POLICY,
            p,
            originalnumpolynomials(bspline(spline, 1)),
        )
        U2 = deepcopy(bspline(spline, 1).bspline.knotVec)
        for _ in 1:_knotstoremove
            popat!(U2, p + 2)
            popat!(U2, length(U2) - p - 1)
        end

        newspline = Bspline(polynomialorders(spline)[1], U2)
        x = LinRange(0, 1, numBasisFunctions(newspline))
        return BandedMatrix{eltype(spline)}(
            evalbsplines(newspline, x),
            (degree(bspline(spline, 1)) - 1, degree(bspline(spline, 1)) - 1),
        )
    end

    s = spline.bsplines[1].bspline
    u = LinRange(0, 1, numBasisFunctions(s))
    return BandedMatrix(
        [
            evalNaive(s, i, u[j]) for
            j in 1:numBasisFunctions(s), i in 1:numBasisFunctions(s)
        ],
        (degree(s) - 1, degree(s) - 1),
    )
end

"""
    transposeinterpolationmatrix(spline::AbstractBspline)

Return the transpose of `interpolationmatrix(spline)`.
"""
function transposeinterpolationmatrix(spline::AbstractBspline)
    return BandedMatrix(transpose(interpolationmatrix(spline)))
end
