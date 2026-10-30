using MultilevelKernelInterpolation, LinearAlgebra
include(
    joinpath(
        pkgdir(MultilevelKernelInterpolation), "docs", "src", "assets", "code", "figures.jl"
    ),
)
using .GuideFigures
poly = MultilevelKernelInterpolation.BarycentricLagrangePolynomialEquidistant(;
    dim=1, order=4
)
left, right = MultilevelKernelInterpolation.onedimi2itranslations(poly, poly)
a = [1.0, -0.5, 0.25, 0.75, 0.0]
u = range(0.0, 1.0; length=201)
parentfield = transpose(basisvalues(poly, u ./ 2)) * a
childfield = transpose(basisvalues(poly, u)) * (transpose(left) * a)
@assert maximum(abs, parentfield - childfield) < 1e-12
curvefigure(
    u ./ 2,
    ["Parent representation" => parentfield, "Child representation" => childfield];
    title="O2O preserves the represented field",
    ylabel="Field value",
)
