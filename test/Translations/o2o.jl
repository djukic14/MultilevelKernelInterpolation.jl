module TestO2O

using Test
using MultilevelKernelInterpolation
using OhMyThreads
using CompScienceMeshes, BEAST
using H2Trees, H2Factory
using LinearAlgebra

# `O2OTranslator` is a DEMOTED CACHE since the level-transfer refactor -- disaggregation no longer
# applies it; it reconstructs the O2O action as `transpose(I2I)` through the level-transfer protocol. So
# this file tests two things: (1) the cache's construction invariant `transpose(O2O) == I2I` still holds,
# and (2) the ACTUAL disaggregation path -- applying the I2I translator in the `TransposeTranslation`
# direction via `leveltranslation!`/`applytransfer!` -- produces exactly the materialized O2O action.
@testset verbose=true "O2OTranslator (demoted cache) + transpose(I2I) disaggregation" begin
    poly = MultilevelKernelInterpolation.BarycentricLagrangePolynomialEquidistant3D(4)

    i2i = MultilevelKernelInterpolation.I2ITranslator([poly], 1:1)
    @test isnothing(i2i.transfers)
    o2o = MultilevelKernelInterpolation.O2OTranslator(i2i)
    @test isnothing(o2o.translationmatrices)

    i2i = MultilevelKernelInterpolation.I2ITranslator([poly, poly], 2:3)
    @test !isnothing(i2i.transfers)
    o2o = MultilevelKernelInterpolation.O2OTranslator(i2i)
    @test !isnothing(o2o.translationmatrices)
    # Cache-construction invariant: the materialized O2O is the transpose of the I2I.
    for i in eachindex(i2i.transfers[3])
        @test transpose(o2o.translationmatrices[3][i]) == i2i.transfers[3][i]
    end

    childmoment = MultilevelKernelInterpolation.PolynomialMoment{1,Float64}(
        size(LinearIndices(poly))
    )
    parentmoment = MultilevelKernelInterpolation.PolynomialMoment{1,Float64}(
        size(LinearIndices(poly))
    )

    λ = 1.0
    k = 2π / λ
    m = readmesh(
        joinpath(
            pkgdir(MultilevelKernelInterpolation), "test", "assets", "in", "16spheres.in"
        ),
    )

    X = lagrangec0d1(m)
    tree = buildtree(X; builder=TwoNTreeBuilder(; minhalfsize=λ / 4, minvalues=0))

    view(childmoment[1], :) .= 1:length(childmoment[1])

    ctx() = H2Factory.TranslationContext(SerialScheduler(), nothing)
    for node in H2Trees.leaves(tree)
        # Disaggregation applies the I2I translator TRANSPOSED (the O2O action) via the protocol; it must
        # equal the materialized O2O matrix (= transpose(I2I)) applied to the moment.
        H2Factory.leveltranslation!(
            parentmoment,
            childmoment,
            i2i,
            tree,
            0,
            node,
            H2Factory.TransposeTranslation(),
            H2Factory.Overwrite(),
            ctx(),
        )
        @test parentmoment[1][:] ≈
            o2o.translationmatrices[3][H2Trees.sector(tree, node) + 1] * childmoment[1][:]

        # Accumulate form (β=true) into the just-computed value -> doubles it.
        H2Factory.leveltranslation!(
            parentmoment,
            childmoment,
            i2i,
            tree,
            0,
            node,
            H2Factory.TransposeTranslation(),
            H2Factory.Accumulate(),
            ctx(),
        )
        @test parentmoment[1][:] ≈
            2 * (
            o2o.translationmatrices[3][H2Trees.sector(tree, node) + 1] * childmoment[1][:]
        )
    end
end

end
