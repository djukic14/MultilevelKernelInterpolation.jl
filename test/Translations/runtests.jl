module TestTranslations

using Test

@testset verbose=true "Translations" begin
    include("kernelcomputation.jl")
    include("interpolation.jl")
    include("i2i.jl")
    include("i2i_onedim.jl")
    include("o2o.jl")
    include("i2ocollection.jl")
    include("i2ocollection_toeplitz.jl")
    include("i2ocollection_bspline_toeplitz.jl")
    include("i2ocollection_symmetryreduced.jl")
    include("i2ocollection_symmetryreduced_dense.jl")
    include("bspline_symmetryguard.jl")
    include("toeplitz_equivalence.jl")
    include("bspline_pinv_solve.jl")
    include("bspline_i2o_shapes.jl")
    include("farfieldaccuracy.jl")
    include("checkprotrusionsafety.jl")
    include("checkprotrusionsafety_blocktree.jl")
end

end
