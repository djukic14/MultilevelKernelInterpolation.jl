module TestMaps

using Test

@testset verbose=true "Top-level MultilevelKernelInterpolation" begin
    include("mapoperations.jl")
    include("adjointleafmul.jl")
    include("endtoend_smoke.jl")
    include("petrov_endtoend_smoke.jl")
    include("petrov_sideownership.jl")
    include("petrov_toeplitz.jl")
    include("petrov_rectangular_moments.jl")
    include("helmholtz2d.jl")
    include("workspaces.jl")
end

end
