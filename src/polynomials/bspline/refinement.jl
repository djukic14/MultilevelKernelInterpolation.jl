function onedimi2itranslations(poly::A, polyparent::A) where {A<:AbstractBspline}
    @assert polynomialorders(poly) == polynomialorders(polyparent)

    if numpolynomials(poly)[1] == numpolynomials(polyparent)[1]
        return i2itranslationsconstant(poly)
    else
        return i2itranslationsincreasing(poly, polyparent)
    end
end

function _mirrormatrix(res0)
    res1 = zeros(eltype(res0), size(res0))
    for i in axes(res0, 1)
        for j in axes(res0, 2)
            res1[i, j] = res0[size(res0, 1) - i + 1, size(res0, 2) - j + 1]
        end
    end
    return res1
end

# method using knot insertion is numerically stable
function i2itranslationsconstant(poly::A) where {A<:BsplinePolynomialEquidistant}
    b = numpolynomials(poly)[1]
    p = polynomialorders(poly)[1]
    bspline = poly.bsplines[1]

    U = bspline.knotVec[(p + 2):b]
    innerknotsparent = [U ./ 2; one(eltype(U)) / 2; (U .+ 1) ./ 2]

    res0 = zeros(Float64, b, b)
    for ib in 1:b
        controlpoints = [SVector(0, 0, 0.0) for _ in 1:b]
        controlpoints[ib] = SVector(0, 0, 1.0)

        bsplinecurve = BsplineCurve(bspline, controlpoints)

        for (i, innerknot) in enumerate(innerknotsparent)
            iseven(i) && continue
            bsplinecurve = insertKnot(bsplinecurve, innerknot)
        end

        if p - 1 > 0
            bsplinecurve = insertKnot(bsplinecurve, 0.5, p - 1)
        end
        newweights = [bsplinecurve.controlPoints[i][3] for i in 1:b]
        res0[ib, :] = newweights
    end

    res1 = _mirrormatrix(res0)

    return sparse(res0), sparse(res1)
end

function i2itranslationsincreasing(
    poly::B, polyparent::B
) where {B<:BsplinePolynomialEquidistant}
    @assert numpolynomials(polyparent)[1] == increasednumpolynomials(poly)
    b = numpolynomials(poly)[1]
    p = polynomialorders(poly)[1]

    res0 = zeros(Float64, 2 * b - p, b)
    parentbspline = Bspline(p, generateKnotVec(2 * b - p, p))
    ib = 1
    for ib in 1:numBasisFunctions(parentbspline)
        controlpoints = [SVector(0, 0, 0.0) for _ in 1:numBasisFunctions(parentbspline)]
        controlpoints[ib] = SVector(0, 0, 1.0)

        bsplinecurve = BsplineCurve(parentbspline, controlpoints)

        if p - 1 > 0
            bsplinecurve = insertKnot(bsplinecurve, 0.5, p - 1)
        end

        newweights = [bsplinecurve.controlPoints[i][3] for i in 1:b]
        res0[ib, :] = newweights
    end

    res1 = _mirrormatrix(res0)

    return sparse(res0), sparse(res1)
end

function i2itranslationsconstant(poly::A) where {A<:BufferedBsplinePolynomialEquidistant}
    return i2iknotinsertion(poly, poly)
end

function i2itranslationsincreasing(
    poly::B, polyparent::B
) where {B<:BufferedBsplinePolynomialEquidistant}
    return i2iknotinsertion(poly, polyparent)
end

function suboptimali2iweights(
    poly::BufferedBsplinePolynomialEquidistant,
    polyparent::BufferedBsplinePolynomialEquidistant,
)
    return suboptimali2iweights(bspline(poly, 1), bspline(polyparent, 1))
end

function suboptimali2iweights(bspl::BufferedBspline, bsplparent::BufferedBspline)
    T = eltype(bspl)
    b = numBasisFunctions(bspl)
    leftboundary =
        numbufferpoints(bspl) / (originalnumpolynomials(bspl) - degree(bspl)) / 2 *
        (1 - 1e4 * eps(T))

    u = T.(collect(LinRange(-leftboundary, 0.5 + leftboundary, b)))
    u0 = u[1:b] .* 2

    T0 = evalbsplines(bspl, u0)
    Q = evalbsplines(bsplparent, u)
    Q0 = Q[1:b, :]

    res0 = T.(T0 \ Q0)
    res0[abs.(res0) .< eps(eltype(res0)) * 1000] .= 0

    res1 = _mirrormatrix(res0)
    return sparse(transpose(res0)), sparse(transpose(res1))
end

function i2iknotinsertion(
    poly::B, polyparent::B; tol=1e-10
) where {B<:BufferedBsplinePolynomialEquidistant}
    b = numpolynomials(poly)[1]
    bparent = numpolynomials(polyparent)[1]
    p = polynomialorders(poly)[1]
    bspline = BufferedBspline(Bspline(p, generateKnotVec(b, p)), numbufferpoints(poly))
    bsplineparent = BufferedBspline(
        Bspline(p, generateKnotVec(bparent, p)), numbufferpoints(polyparent)
    )

    Uchild =
        (
            (bspline.bspline.knotVec) .*
            (1 + 2 * MultilevelKernelInterpolation._leftboundary(bspline)) .-
            MultilevelKernelInterpolation._leftboundary(bspline)
        ) ./ 2
    Uchildinparent =
        (Uchild .+ MultilevelKernelInterpolation._leftboundary(bsplineparent)) ./
        (1 + 2 * MultilevelKernelInterpolation._leftboundary(bsplineparent))

    Uparent = deepcopy(bsplineparent.bspline.knotVec)
    knotstoadd = eltype(Uchildinparent)[]

    res0 = zeros(Float64, bparent, b)

    for knot in Uchildinparent
        knot <= 0 && continue
        knotinparent = false
        for (i, parentknot) in enumerate(Uparent)
            if abs(knot - parentknot) < tol
                knotinparent = true
                popat!(Uparent, i)
                break
            end
        end

        if !knotinparent
            push!(knotstoadd, knot)
        end
    end

    for ib in 1:bparent
        controlpoints = [SVector(0, 0, 0.0) for _ in 1:bparent]
        controlpoints[ib] = SVector(0, 0, 1.0)

        bsplinecurve = BsplineCurve(bsplineparent.bspline, controlpoints)

        lengthbsplinecurve = numBasisFunctions(bsplinecurve.basis)
        for innerknot in knotstoadd
            multiplicity = length(searchsorted(bsplinecurve.basis.knotVec, innerknot))
            # At full multiplicity, split directly instead of inserting another knot.
            if multiplicity < p
                bsplinecurve = insertKnot(bsplinecurve, innerknot)
            else
                bsplinecurve = deepcopy(bsplinecurve)
            end
            if numBasisFunctions(bsplinecurve.basis) == lengthbsplinecurve
                oldMultIndices = searchsorted(bsplinecurve.basis.knotVec, innerknot)
                insert!(bsplinecurve.basis.knotVec, oldMultIndices[end], innerknot)
                insert!(
                    bsplinecurve.controlPoints,
                    oldMultIndices[end] - p,
                    bsplinecurve.controlPoints[oldMultIndices[end] - p],
                )
            end
            lengthbsplinecurve = numBasisFunctions(bsplinecurve.basis)
        end

        offset = findfirst(
            isapprox(Uchildinparent[1]; atol=1e-3), bsplinecurve.basis.knotVec
        )

        newweights = [bsplinecurve.controlPoints[i][3] for i in (offset):(b + offset - 1)]

        res0[ib, :] = newweights
    end

    res1 = _mirrormatrix(res0)

    return eltype(poly).(sparse(res0)), eltype(poly).(sparse(res1))
end

function i2icheapaveraging(poly)
    b = numBasisFunctions(poly)
    p = degree(poly)

    U = poly.knotVec
    U2 = copy(U)

    newnodes = []

    _knotstoremove = knotstoremove(p, b)

    for _ in 1:_knotstoremove
        push!(newnodes, U2[p + 2])
        push!(newnodes, U2[end - p - 1])
        popat!(U2, p + 2)
        popat!(U2, length(U2) - p - 1)
    end

    newspline = Bspline(p, U2)

    res = zeros(eltype(U), numBasisFunctions(newspline), b)

    for ib in 1:numBasisFunctions(newspline)
        controlpoints = [
            SVector(0, 0, zero(eltype(U))) for _ in 1:numBasisFunctions(newspline)
        ]
        controlpoints[ib] = SVector(0, 0, one(eltype(U)))

        bsplinecurve = BsplineCurve(newspline, controlpoints)

        for newnode in newnodes
            bsplinecurve = insertKnot(bsplinecurve, newnode)
        end

        newweights = [bsplinecurve.controlPoints[i][3] for i in 1:b]
        res[ib, :] = newweights
    end

    return sparse(res)
end
