"""
    childranges(poly)

Return the left and right child interpolation points in parent coordinates.
"""
function childranges(p::AbstractTensorProductPolynomial{D,T}) where {D,T}
    n = first(MultilevelKernelInterpolation.numpolynomials(p))
    range0 = Vector{eltype(T)}(undef, n)
    range1 = Vector{eltype(T)}(undef, n)

    rangep = range(p, 1)

    range0 .= rangep ./ 2
    range1 .= (rangep .+ 1)
    range1 ./= 2

    return range0, range1
end

"""
    onedimi2itranslations(child, parent)

Compute the one-dimensional parent-child interpolation matrices for Lagrange spaces.
"""
function onedimi2itranslations(
    poly::A, parentpoly::B
) where {
    DA,
    DB,
    A<:Union{AbstractLagrange,ScaledShiftedPolynomial{DA,<:AbstractLagrange}},
    B<:Union{AbstractLagrange,ScaledShiftedPolynomial{DB,<:AbstractLagrange}},
}
    range0, range1 = MultilevelKernelInterpolation.childranges(poly)

    ppoly = MultilevelKernelInterpolation.numpolynomials(poly)[1]
    pparentpoly = MultilevelKernelInterpolation.numpolynomials(parentpoly)[1]

    res0 = Matrix{eltype(poly)}(undef, pparentpoly, ppoly)
    res1 = Matrix{eltype(poly)}(undef, pparentpoly, ppoly)

    for i0 in eachindex(range0)
        MultilevelKernelInterpolation.evaluateall1dpolynomial!(
            view(res0, :, i0), parentpoly, range0[i0], 1
        )
    end

    for i1 in eachindex(range1)
        MultilevelKernelInterpolation.evaluateall1dpolynomial!(
            view(res1, :, i1), parentpoly, range1[i1], 1
        )
    end
    return sparse(res0), sparse(res1)
end
