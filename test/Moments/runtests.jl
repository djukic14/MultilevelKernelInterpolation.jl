module TestMoments

using Test

@testset verbose=true "Moments" begin
    include("assemblemoments.jl")
    include("interpolatemoments.jl")
end

end
