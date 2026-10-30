module GuideFigures

using PlotlyJS
using MultilevelKernelInterpolation: MultilevelKernelInterpolation

export basisvalues, curvefigure

function basisvalues(poly, xs)
    values = zeros(MultilevelKernelInterpolation.numpolynomials(poly)[1], length(xs))
    for (j, x) in enumerate(xs)
        MultilevelKernelInterpolation.evaluateall1dpolynomial!(
            view(values, :, j), poly, x, 1
        )
    end
    return values
end

function curvefigure(xs, series; title, xlabel="x", ylabel="Basis value")
    colors = ("#4063d8", "#389826", "#9558b2", "#cb3c33")
    traces = [
        scatter(;
            x=collect(xs),
            y=collect(values),
            name,
            mode="lines",
            line=attr(;
                color=colors[mod1(i, length(colors))],
                dash=i > length(colors) ? "dash" : "solid",
            ),
        ) for (i, (name, values)) in enumerate(series)
    ]
    figure = Plot(
        traces,
        Layout(;
            title=attr(; text=title, x=0.02, font=attr(; size=16)),
            template="plotly_white",
            xaxis=attr(; title=xlabel),
            yaxis=attr(; title=ylabel),
            font=attr(; family="Arial, sans-serif", size=14),
            margin=attr(; l=60, r=20, t=55, b=60),
            legend=attr(; orientation="h", y=-0.22),
            height=420,
            autosize=true,
        ),
    )
    return figure
end

end
