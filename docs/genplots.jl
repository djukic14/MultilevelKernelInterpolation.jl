# Generate named PlotlyJS assets before Documenter copies docs/src/assets.
using MultilevelKernelInterpolation, PlotlyJS

isdefined(@__MODULE__, :PLOT_NAMES) || include(joinpath(@__DIR__, "plotutils.jl"))

function generateplots()
    output = joinpath(@__DIR__, "src", "assets", "plots")
    mkpath(output)
    for name in PLOT_NAMES
        sandbox = Core.eval(@__MODULE__, :(module $(gensym(:DocPlot)) end))
        figure = Base.include(sandbox, joinpath(@__DIR__, "plots", "$name.jl"))
        # Keep hover and drag zoom without a toolbar covering mobile titles.
        figure.config.displayModeBar = false
        Base.invokelatest(PlotlyJS.savefig, figure, joinpath(output, "$name.html"))
    end
    write(joinpath(@__DIR__, ".plots_hash"), sourcehash(dirname(@__DIR__)))
    @info "Generated $(length(PLOT_NAMES)) interactive documentation figures"
end

generateplots()
