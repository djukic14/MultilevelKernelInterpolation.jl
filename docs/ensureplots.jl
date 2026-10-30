include(joinpath(@__DIR__, "plotutils.jl"))

let output = joinpath(@__DIR__, "src", "assets", "plots"),
    marker = joinpath(@__DIR__, ".plots_hash")

    current = sourcehash(dirname(@__DIR__))
    complete = all(name -> isfile(joinpath(output, "$name.html")), PLOT_NAMES)
    if complete && isfile(marker) && read(marker, String) == current
        @info "Documentation plots are up to date; skipping generation"
    else
        sandbox = Core.eval(@__MODULE__, :(module $(gensym(:DocPlots)) end))
        Base.include(sandbox, joinpath(@__DIR__, "genplots.jl"))
    end
end
