module TestAssembleMoments

using Test
using SparseArrays, StaticArrays
using CompScienceMeshes, BEAST
using H2Trees
using MultilevelKernelInterpolation

@testset verbose=true "Moment Assembly" begin
    @testset "Testmoments: Helmholtz3D" begin
        poly = MultilevelKernelInterpolation.BarycentricLagrangePolynomialEquidistant3D(4)

        m = readmesh(
            joinpath(
                pkgdir(MultilevelKernelInterpolation), "test", "assets", "in", "sphere.in"
            ),
        )

        X = lagrangec0d1(m)
        tree = buildtree(X; builder=TwoNTreeBuilder(; minhalfsize=0.1, minvalues=0))
        leafpolys = MultilevelKernelInterpolation.leafpolynomials(
            tree, [poly for _ in H2Trees.levels(tree)]
        )

        for op in [
            Helmholtz3D.singlelayer(; wavenumber=1.0),
            Helmholtz3D.doublelayer_transposed(; wavenumber=1.0),
            Helmholtz3D.hypersingular(; wavenumber=1.0),
            MultilevelKernelInterpolation.HH3DDoubleLayerTransposedFDBIOGradientI2O(;
                wavenumber=1.0
            ),
        ]
            moments = MultilevelKernelInterpolation.TestMomentCollections(
                op, tree, leafpolys, X; quadstrat=BEAST.SingleNumQStrat(2)
            )

            for i in eachindex(moments)
                @test length(moments[i].moments) == MultilevelKernelInterpolation.dims(
                    MultilevelKernelInterpolation.testmoment(op, poly)
                )

                for j in eachindex(moments[i].moments)
                    @test size(moments[i].moments[j], 2) ==
                        length(H2Trees.values(tree, H2Trees.leaves(tree)[i]))
                end
            end
        end
    end

    @testset "Trialmoments: Helmholtz3D" begin
        poly = MultilevelKernelInterpolation.BarycentricLagrangePolynomialEquidistant3D(4)
        m = readmesh(
            joinpath(
                pkgdir(MultilevelKernelInterpolation), "test", "assets", "in", "sphere.in"
            ),
        )

        X = lagrangec0d1(m)
        tree = buildtree(X; builder=TwoNTreeBuilder(; minhalfsize=0.1, minvalues=0))
        leafpolys = MultilevelKernelInterpolation.leafpolynomials(
            tree, [poly for _ in H2Trees.levels(tree)]
        )

        for op in [
            Helmholtz3D.singlelayer(; wavenumber=1.0),
            Helmholtz3D.doublelayer_transposed(; wavenumber=1.0),
            Helmholtz3D.hypersingular(; wavenumber=1.0),
            MultilevelKernelInterpolation.HH3DDoubleLayerTransposedFDBIOGradientI2O(;
                wavenumber=1.0
            ),
        ]
            moments = MultilevelKernelInterpolation.TrialMomentCollections(
                op, tree, leafpolys, X; quadstrat=BEAST.SingleNumQStrat(2)
            )

            for i in eachindex(moments)
                @test length(moments[i].moments) == MultilevelKernelInterpolation.dims(
                    MultilevelKernelInterpolation.trialmoment(op, poly)
                )

                for j in eachindex(moments[i].moments)
                    @test size(moments[i].moments[j], 2) ==
                        length(H2Trees.values(tree, H2Trees.leaves(tree)[i]))
                end
            end
        end
    end

    @testset "Testmoments: Maxwell3D" begin
        poly = MultilevelKernelInterpolation.BarycentricLagrangePolynomialEquidistant3D(4)
        m = readmesh(
            joinpath(
                pkgdir(MultilevelKernelInterpolation), "test", "assets", "in", "sphere.in"
            ),
        )

        X = raviartthomas(m)
        tree = buildtree(X; builder=TwoNTreeBuilder(; minhalfsize=0.1, minvalues=0))
        leafpolys = MultilevelKernelInterpolation.leafpolynomials(
            tree, [poly for _ in H2Trees.levels(tree)]
        )

        for op in [
            Maxwell3D.singlelayer(; wavenumber=1.0),
            Maxwell3D.doublelayer(; wavenumber=1.0),
            MultilevelKernelInterpolation.efie(; wavenumber=1.0),
            BEAST.DoubleLayerRotatedMW3D(1.0, im * 1.0),
        ]
            moments = MultilevelKernelInterpolation.TestMomentCollections(
                op, tree, leafpolys, X; quadstrat=BEAST.SingleNumQStrat(2)
            )

            for i in eachindex(moments)
                @test length(moments[i].moments) == MultilevelKernelInterpolation.dims(
                    MultilevelKernelInterpolation.testmoment(op, poly)
                )

                for j in eachindex(moments[i].moments)
                    @test size(moments[i].moments[j], 2) ==
                        length(H2Trees.values(tree, H2Trees.leaves(tree)[i]))
                end
            end
        end
    end

    @testset "Trialmoments: Maxwell3D" begin
        poly = MultilevelKernelInterpolation.BarycentricLagrangePolynomialEquidistant3D(4)
        m = readmesh(
            joinpath(
                pkgdir(MultilevelKernelInterpolation), "test", "assets", "in", "sphere.in"
            ),
        )

        X = raviartthomas(m)
        tree = buildtree(X; builder=TwoNTreeBuilder(; minhalfsize=0.1, minvalues=0))
        leafpolys = MultilevelKernelInterpolation.leafpolynomials(
            tree, [poly for _ in H2Trees.levels(tree)]
        )

        for op in [
            Maxwell3D.singlelayer(; wavenumber=1.0),
            Maxwell3D.doublelayer(; wavenumber=1.0),
            MultilevelKernelInterpolation.efie(; wavenumber=1.0),
            BEAST.DoubleLayerRotatedMW3D(1.0, im * 1.0),
        ]
            # `deprecated2/moments/test_assemblemoments.jl` called `TestMomentCollections`
            # here too (a copy-paste from the testset above it) -- this testset is titled
            # "Trialmoments" and asserts against `MultilevelKernelInterpolation.trialmoment`, so it must call
            # `TrialMomentCollections` to actually exercise what it claims to.
            moments = MultilevelKernelInterpolation.TrialMomentCollections(
                op, tree, leafpolys, X; quadstrat=BEAST.SingleNumQStrat(2)
            )

            for i in eachindex(moments)
                @test length(moments[i].moments) == MultilevelKernelInterpolation.dims(
                    MultilevelKernelInterpolation.trialmoment(op, poly)
                )

                for j in eachindex(moments[i].moments)
                    @test size(moments[i].moments[j], 2) ==
                        length(H2Trees.values(tree, H2Trees.leaves(tree)[i]))
                end
            end
        end
    end

    @testset "defaultmomenttype and postprocessmoment!" begin
        @test MultilevelKernelInterpolation.defaultmomenttype(nothing) ==
            MultilevelKernelInterpolation.PolynomialMoment

        @test typeof(
            MultilevelKernelInterpolation.postprocessmoment!(
                [zeros(Float64, 10, 10) for _ in 1:2], nothing
            ),
        ) == Vector{Matrix{Float64}}
        @test typeof(
            MultilevelKernelInterpolation.postprocessmoment!(
                [zeros(Float64, 10, 10) for _ in 1:2],
                MultilevelKernelInterpolation.BsplinePolynomialEquidistant3D{Float64}(2, 5),
            ),
        ) == SVector{2,SparseMatrixCSC{Float64,Int64}}
    end
end

end
