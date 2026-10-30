using MultilevelKernelInterpolation, LinearAlgebra
include(
    joinpath(
        pkgdir(MultilevelKernelInterpolation), "docs", "src", "assets", "code", "figures.jl"
    ),
)
using .GuideFigures

lagrange = MultilevelKernelInterpolation.BarycentricLagrangePolynomialEquidistant(;
    dim=1, order=4
)
bspline = MultilevelKernelInterpolation.BsplinePolynomialEquidistant(;
    dim=1, degree=3, nbasis=9
)
for poly in (lagrange, bspline)
    left, right = MultilevelKernelInterpolation.onedimi2itranslations(poly, poly)
    @assert MultilevelKernelInterpolation.testonedimensionali2i(poly, poly, left, right)
end

left, right = MultilevelKernelInterpolation.onedimi2itranslations(lagrange, lagrange)
u = range(0.0, 1.0; length=201)
parent = basisvalues(lagrange, u ./ 2)
transferred = left * basisvalues(lagrange, u)
@assert maximum(abs, parent - transferred) < 1e-12
curvefigure(
    u ./ 2,
    ["Parent L3" => parent[3, :], "From left child" => transferred[3, :]];
    title="The same parent function, represented on its left child",
)
