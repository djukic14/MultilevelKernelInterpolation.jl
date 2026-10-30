module TestOperators

using Test

@testset verbose=true "Operators" begin
    include("beastoperators.jl")
end

end
