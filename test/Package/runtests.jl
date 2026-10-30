module TestPackageGroup

using Test

@testset verbose = true "Package" begin
    include("h2factoryinterface.jl")
end

end # module TestPackageGroup
