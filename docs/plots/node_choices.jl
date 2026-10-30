using MultilevelKernelInterpolation, PlotlyJS
families = [
    "Equidistant" =>
        MultilevelKernelInterpolation.BarycentricLagrangePolynomialEquidistant(;
            dim=1, order=6
        ),
    "Chebyshev roots" =>
        MultilevelKernelInterpolation.BarycentricLagrangePolynomialChebyshev(;
            dim=1, order=6
        ),
    "Chebyshev endpoints" =>
        MultilevelKernelInterpolation.BarycentricLagrangePolynomialChebyshev2(;
            dim=1, order=6
        ),
]
figure = Plot(
    [
        scatter(;
            x=collect(MultilevelKernelInterpolation.range(p, 1)),
            y=fill(k, 7),
            mode="markers",
            name=label,
            marker=attr(; size=10),
        ) for (k, (label, p)) in enumerate(families)
    ],
    Layout(;
        template="plotly_white",
        height=280,
        autosize=true,
        xaxis=attr(; title="Reference coordinate"),
        yaxis=attr(; tickvals=[1, 2, 3], ticktext=first.(families)),
        showlegend=false,
        margin=attr(; l=155, r=20, t=20, b=50),
    ),
)
include(
    joinpath(
        pkgdir(MultilevelKernelInterpolation), "docs", "src", "assets", "code", "figures.jl"
    ),
)
figure
