module TestLevelPolynomials

using Test
using CompScienceMeshes
using H2Trees
using MultilevelKernelInterpolation

@testset verbose=true "Level polynomials" begin
    base = MultilevelKernelInterpolation.BarycentricLagrangePolynomialEquidistant{3,Float64}(
        2
    )

    @testset verbose=true "generic level sequence" begin
        levels = 1:4
        polys = MultilevelKernelInterpolation.levelpolynomials(
            base, levels, level -> level < 3
        )

        @test length(polys) == length(levels)
        @test all(poly -> typeof(poly) == typeof(base), polys)
        @test MultilevelKernelInterpolation.numpolynomials(polys[1]) == (9, 9, 9)
        @test MultilevelKernelInterpolation.numpolynomials(polys[2]) == (5, 5, 5)
        @test MultilevelKernelInterpolation.numpolynomials(polys[3]) == (3, 3, 3)
        @test MultilevelKernelInterpolation.numpolynomials(polys[4]) == (3, 3, 3)
    end

    @testset verbose=true "protrusion is passed to shifted polynomials" begin
        shifted = MultilevelKernelInterpolation.ScaledShiftedPolynomial(base, 1.0, zeros(3))
        polys = MultilevelKernelInterpolation.levelpolynomials(
            shifted, 1:3, level -> false; protrusion=[0.0, 0.1, 0.2]
        )

        @test all(poly -> poly isa typeof(shifted), polys)
        @test all(MultilevelKernelInterpolation.scaling(poly) >= 1.0 for poly in polys)
        @test MultilevelKernelInterpolation.shift(polys[end]) == [-0.2, -0.2, -0.2]
    end

    @testset verbose=true "HF level condition" begin
        mesh = readmesh(
            joinpath(
                pkgdir(MultilevelKernelInterpolation), "test", "assets", "in", "sphere.in"
            ),
        )
        tree = buildtree(
            vertices(mesh); builder=TwoNTreeBuilder(; minhalfsize=0.1, minvalues=0)
        )

        condition = MultilevelKernelInterpolation.HFConditionFunctor(tree, 0.3)
        @test condition(1)
        @test condition(2)
        @test condition(3)
        @test !condition(4)
        @test !condition(5)

        condition = MultilevelKernelInterpolation.HFConditionFunctor(tree, 0)
        @test condition(1)
        @test condition(2)
        @test condition(3)
        @test condition(4)
        @test condition(5)

        condition = MultilevelKernelInterpolation.HFConditionFunctor(tree, 10.0)
        @test !condition(1)
        @test !condition(2)
        @test !condition(3)
        @test !condition(4)
        @test !condition(5)
    end
end

end
