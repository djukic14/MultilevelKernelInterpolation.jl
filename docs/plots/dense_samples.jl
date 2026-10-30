using PlotlyJS
using MultilevelKernelInterpolation: MultilevelKernelInterpolation
x = collect(range(0.0, 1.0; length=7))
y = collect(range(2.0, 3.0; length=7))
G = [1 / abs(xi - yj) for xi in x, yj in y]
figure = Plot(
    heatmap(; z=G, colorscale="Viridis"),
    Layout(;
        template="plotly_white",
        autosize=true,
        height=360,
        xaxis=attr(; title="Source index j"),
        yaxis=attr(; title="Receiving index i"),
        title="Kernel samples for two separated boxes",
    ),
)
include(
    joinpath(
        pkgdir(MultilevelKernelInterpolation), "docs", "src", "assets", "code", "figures.jl"
    ),
)
figure
