using MultilevelKernelInterpolation, LinearAlgebra
poly = MultilevelKernelInterpolation.BsplinePolynomialEquidistant(;
    dim=1, degree=3, nbasis=9
)
S = MultilevelKernelInterpolation.interpolationmatrix(poly)
nodes = MultilevelKernelInterpolation.range(poly, 1)
values = @. 1 + nodes + nodes^2
coefficients = S \ values
@assert S * coefficients ≈ values
nothing # hide
include(
    joinpath(
        pkgdir(MultilevelKernelInterpolation), "docs", "src", "assets", "code", "figures.jl"
    ),
)
using .GuideFigures
xs = range(0.0, 1.0; length=301)
reconstructed = transpose(basisvalues(poly, xs)) * coefficients
@assert maximum(abs, reconstructed .- (1 .+ xs .+ xs .^ 2)) < 1e-12
curvefigure(
    xs,
    ["Spline reconstruction" => reconstructed, "1 + x² + x" => 1 .+ xs .+ xs .^ 2];
    title="Sampling solve reproduces a quadratic",
    ylabel="Value",
)
