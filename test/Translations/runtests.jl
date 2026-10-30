module TestTranslations

using Test

@testset verbose=true "Translations" begin
    for file in (
        "kernelcomputation.jl",
        "interpolation.jl",
        "i2i.jl",
        "i2i_onedim.jl",
        "o2o.jl",
        "i2ocollection.jl",
        "i2ocollection_toeplitz.jl",
        "i2ocollection_bspline_toeplitz.jl",
        "i2ocollection_symmetryreduced.jl",
        "i2ocollection_symmetryreduced_dense.jl",
        "bspline_symmetryguard.jl",
        "toeplitz_equivalence.jl",
        "bspline_pinv_solve.jl",
        "bspline_i2o_shapes.jl",
        "farfieldaccuracy.jl",
        "checkprotrusionsafety.jl",
        "checkprotrusionsafety_blocktree.jl",
    )
        include(file)
        GC.gc()
    end
end

end
