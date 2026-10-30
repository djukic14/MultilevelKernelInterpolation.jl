module TestPolynomials

using Test

@testset verbose=true "Polynomials" begin
    include("barycentric_lagrange.jl")
    include("scaled_shifted.jl")
    include("bsplines.jl")
    include("bspline_domains.jl")
    include("levelpolynomials.jl")
    include("parent_child_interpolation.jl")
end

end
