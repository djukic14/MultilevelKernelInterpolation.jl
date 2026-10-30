using MultilevelKernelInterpolation, PlotlyJS
include(
    joinpath(
        pkgdir(MultilevelKernelInterpolation), "docs", "src", "assets", "code", "figures.jl"
    ),
)
using .GuideFigures

xs = range(0.0, 1.0; length=401)
lagrange = MultilevelKernelInterpolation.BarycentricLagrangePolynomialEquidistant(;
    dim=1, order=4
)
bspline = MultilevelKernelInterpolation.BsplinePolynomialEquidistant(;
    dim=1, degree=3, nbasis=9
)
lagrangevalues = basisvalues(lagrange, xs)
bsplinevalues = basisvalues(bspline, xs)
nothing # hide
curvefigure(
    xs,
    ["B$i" => bsplinevalues[i, :] for i in axes(bsplinevalues, 1)];
    title="Cubic B-splines: nine basis functions",
)
