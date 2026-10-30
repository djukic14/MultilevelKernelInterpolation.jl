module TestMaps

using Test

@testset verbose=true "Top-level MultilevelKernelInterpolation" begin
    for file in (
        "mapoperations.jl",
        "adjointleafmul.jl",
        "endtoend_smoke.jl",
        "petrov_endtoend_smoke.jl",
        "petrov_sideownership.jl",
        "petrov_toeplitz.jl",
        "petrov_rectangular_moments.jl",
        "helmholtz2d.jl",
        "workspaces.jl",
    )
        include(file)
        GC.gc()
    end
end

end
