module TestI2I

using Test
using CompScienceMeshes, BEAST
using H2Trees, H2Factory
using MultilevelKernelInterpolation
using LinearAlgebra
using OhMyThreads

# Apply the dense I2I translator's FORWARD (aggregation) level transfer through the level-transfer
# protocol (`leveltranslation!` -> `leveltransfer` -> `applytransfer!`), replacing the removed callable
# form. `parent` is unused by the sector lookup, so a dummy `0` is fine.
function _applyi2i!(out, in, i2i, tree, child; accumulate::Bool=false)
    return H2Factory.leveltranslation!(
        out,
        in,
        i2i,
        tree,
        0,
        child,
        H2Factory.ForwardTranslation(),
        accumulate ? H2Factory.Accumulate() : H2Factory.Overwrite(),
        H2Factory.TranslationContext(SerialScheduler(), nothing),
    )
end

@testset verbose=true "I2ITranslator" begin
    @testset "1D oracle check (testonedimensionali2i)" begin
        # `onedimi2itranslations` is the function `threedimi2itranslations`/`I2ITranslator`
        # actually use to build the raw 1D parent-child matrices. `testonedimensionali2i` is
        # an independent, from-scratch oracle for it (direct polynomial evaluation, no
        # I2ITranslator/moments/tree/mesh machinery involved) -- exercised directly here so
        # a regression in `onedimi2itranslations` for any polynomial type shows up as a test
        # failure. This used to be reachable only through a `test=true` keyword that nothing
        # in the codebase ever set.
        lagrangepoly = MultilevelKernelInterpolation.BarycentricLagrangePolynomialEquidistant3D(
            4
        )
        res0, res1 = MultilevelKernelInterpolation.onedimi2itranslations(
            lagrangepoly, lagrangepoly
        )
        @test MultilevelKernelInterpolation.testonedimensionali2i(
            lagrangepoly, lagrangepoly, res0, res1
        )

        bsplinepoly = MultilevelKernelInterpolation.BsplinePolynomialEquidistant3D{Float64}(
            2, 5
        )
        res0, res1 = MultilevelKernelInterpolation.onedimi2itranslations(
            bsplinepoly, bsplinepoly
        )
        @test MultilevelKernelInterpolation.testonedimensionali2i(
            bsplinepoly, bsplinepoly, res0, res1
        )

        parentbspline = MultilevelKernelInterpolation.increase(bsplinepoly)
        res0, res1 = MultilevelKernelInterpolation.onedimi2itranslations(
            bsplinepoly, parentbspline
        )
        @test MultilevelKernelInterpolation.testonedimensionali2i(
            bsplinepoly, parentbspline, res0, res1
        )

        bufferedpoly = MultilevelKernelInterpolation.BufferedBsplinePolynomialEquidistant3D{
            Float64
        }(
            2, 7, 2
        )
        parentbuffered = MultilevelKernelInterpolation.increase(bufferedpoly)
        res0, res1 = MultilevelKernelInterpolation.onedimi2itranslations(
            bufferedpoly, parentbuffered
        )
        @test MultilevelKernelInterpolation.testonedimensionali2i(
            bufferedpoly, parentbuffered, res0, res1
        )
    end

    @testset "Lagrange const" begin
        poly = MultilevelKernelInterpolation.BarycentricLagrangePolynomialEquidistant3D(4)

        i2i = MultilevelKernelInterpolation.I2ITranslator([poly], 1:1)
        @test isnothing(i2i.transfers)

        i2i = MultilevelKernelInterpolation.I2ITranslator([poly, poly], 2:3)
        @test !isnothing(i2i.transfers)

        λ = 1.0
        k = 2π / λ
        m = readmesh(
            joinpath(
                pkgdir(MultilevelKernelInterpolation),
                "test",
                "assets",
                "in",
                "16spheres.in",
            ),
        )

        X = lagrangec0d1(m)
        tree = buildtree(X; builder=TwoNTreeBuilder(; minhalfsize=λ / 4, minvalues=0))
        treecoarse = buildtree(X; builder=TwoNTreeBuilder(; minhalfsize=λ / 2, minvalues=0))

        for op in [
            Helmholtz3D.singlelayer(; wavenumber=k),
            Helmholtz3D.doublelayer_transposed(; wavenumber=k),
            Helmholtz3D.hypersingular(; wavenumber=k),
            MultilevelKernelInterpolation.HH3DDoubleLayerTransposedFDBIOGradientI2O(;
                wavenumber=k
            ),
        ]
            moments = Dict(
                zip(
                    H2Trees.leaves(tree),
                    MultilevelKernelInterpolation.TestMomentCollections(
                        op,
                        tree,
                        MultilevelKernelInterpolation.leafpolynomials(
                            tree,
                            MultilevelKernelInterpolation.levelpolynomials(
                                poly, H2Trees.levels(tree), (level) -> false
                            ),
                        ),
                        X;
                        quadstrat=BEAST.SingleNumQStrat(6),
                    ),
                ),
            )
            parentmoments = Dict(
                zip(
                    H2Trees.leaves(treecoarse),
                    MultilevelKernelInterpolation.TestMomentCollections(
                        op,
                        treecoarse,
                        MultilevelKernelInterpolation.leafpolynomials(
                            treecoarse,
                            MultilevelKernelInterpolation.levelpolynomials(
                                poly, H2Trees.levels(treecoarse), (level) -> false
                            ),
                        ),
                        X;
                        quadstrat=BEAST.SingleNumQStrat(6),
                    ),
                ),
            )

            childmoment = MultilevelKernelInterpolation.PolynomialMoment{
                MultilevelKernelInterpolation.dims(
                    MultilevelKernelInterpolation.testmoment(op, poly)
                ),
                Float64,
            }(
                size(LinearIndices(poly))
            )
            parentmoment = MultilevelKernelInterpolation.PolynomialMoment{
                MultilevelKernelInterpolation.dims(
                    MultilevelKernelInterpolation.testmoment(op, poly)
                ),
                Float64,
            }(
                size(LinearIndices(poly))
            )
            iparentmoment = MultilevelKernelInterpolation.PolynomialMoment{
                MultilevelKernelInterpolation.dims(
                    MultilevelKernelInterpolation.testmoment(op, poly)
                ),
                Float64,
            }(
                size(LinearIndices(poly))
            )

            for coarseparent in H2Trees.leaves(treecoarse)
                amplitudes = collect(1:length(H2Trees.values(treecoarse, coarseparent)))

                LinearAlgebra.mul!(parentmoment, parentmoments[coarseparent], amplitudes)

                coarseleafchild = 0
                for leaf in H2Trees.leaves(tree)
                    if H2Trees.center(tree, H2Trees.parent(tree, leaf)) ≈
                        H2Trees.center(treecoarse, coarseparent)
                        coarseleafchild = leaf
                        break
                    end
                end
                children = collect(
                    H2Trees.children(tree, H2Trees.parent(tree, coarseleafchild))
                )

                for (i, child) in enumerate(children)
                    ii = [
                        findfirst(isequal(point), H2Trees.values(treecoarse, coarseparent))
                        for point in H2Trees.values(tree, child)
                    ]

                    LinearAlgebra.mul!(childmoment, moments[child], amplitudes[ii])
                    if i == 1
                        _applyi2i!(iparentmoment, childmoment, i2i, tree, child)
                    else
                        _applyi2i!(
                            iparentmoment, childmoment, i2i, tree, child; accumulate=true
                        )
                    end
                end

                for i in eachindex(parentmoment)
                    @test maximum(abs, parentmoment[i] - iparentmoment[i]) /
                          maximum(abs, parentmoment[i]) < 1e-10
                end
            end
        end
    end

    @testset "B-splines increasing" begin
        poly = MultilevelKernelInterpolation.BsplinePolynomialEquidistant3D{Float64}(2, 5)

        i2i = MultilevelKernelInterpolation.I2ITranslator(
            [MultilevelKernelInterpolation.increase(poly), poly], 2:3
        )
        @test !isnothing(i2i.transfers)

        λ = 1.0
        k = 2π / λ
        m = readmesh(
            joinpath(
                pkgdir(MultilevelKernelInterpolation),
                "test",
                "assets",
                "in",
                "16spheres.in",
            ),
        )

        X = lagrangec0d1(m)
        tree = buildtree(X; builder=TwoNTreeBuilder(; minhalfsize=λ / 4, minvalues=0))
        treecoarse = buildtree(X; builder=TwoNTreeBuilder(; minhalfsize=λ / 2, minvalues=0))

        for op in [
            Helmholtz3D.singlelayer(; wavenumber=k),
            Helmholtz3D.doublelayer_transposed(; wavenumber=k),
            Helmholtz3D.hypersingular(; wavenumber=k),
            MultilevelKernelInterpolation.HH3DDoubleLayerTransposedFDBIOGradientI2O(;
                wavenumber=k
            ),
        ]
            moments = Dict(
                zip(
                    H2Trees.leaves(tree),
                    MultilevelKernelInterpolation.TestMomentCollections(
                        op,
                        tree,
                        MultilevelKernelInterpolation.leafpolynomials(
                            tree,
                            MultilevelKernelInterpolation.levelpolynomials(
                                poly, H2Trees.levels(tree), (level) -> false
                            ),
                        ),
                        X;
                        quadstrat=BEAST.SingleNumQStrat(6),
                    ),
                ),
            )
            parentmoments = Dict(
                zip(
                    H2Trees.leaves(treecoarse),
                    MultilevelKernelInterpolation.TestMomentCollections(
                        op,
                        treecoarse,
                        MultilevelKernelInterpolation.leafpolynomials(
                            treecoarse,
                            MultilevelKernelInterpolation.levelpolynomials(
                                MultilevelKernelInterpolation.increase(poly),
                                H2Trees.levels(treecoarse),
                                (level) -> false,
                            ),
                        ),
                        X;
                        quadstrat=BEAST.SingleNumQStrat(6),
                    ),
                ),
            )

            childmoment = MultilevelKernelInterpolation.PolynomialMoment{
                MultilevelKernelInterpolation.dims(
                    MultilevelKernelInterpolation.testmoment(op, poly)
                ),
                Float64,
            }(
                size(LinearIndices(poly))
            )
            parentmoment = MultilevelKernelInterpolation.PolynomialMoment{
                MultilevelKernelInterpolation.dims(
                    MultilevelKernelInterpolation.testmoment(op, poly)
                ),
                Float64,
            }(
                size(LinearIndices(MultilevelKernelInterpolation.increase(poly)))
            )
            iparentmoment = MultilevelKernelInterpolation.PolynomialMoment{
                MultilevelKernelInterpolation.dims(
                    MultilevelKernelInterpolation.testmoment(op, poly)
                ),
                Float64,
            }(
                size(LinearIndices(MultilevelKernelInterpolation.increase(poly)))
            )

            for coarseparent in H2Trees.leaves(treecoarse)
                amplitudes = collect(1:length(H2Trees.values(treecoarse, coarseparent)))

                LinearAlgebra.mul!(parentmoment, parentmoments[coarseparent], amplitudes)

                coarseleafchild = 0
                for leaf in H2Trees.leaves(tree)
                    if H2Trees.center(tree, H2Trees.parent(tree, leaf)) ≈
                        H2Trees.center(treecoarse, coarseparent)
                        coarseleafchild = leaf
                        break
                    end
                end
                children = collect(
                    H2Trees.children(tree, H2Trees.parent(tree, coarseleafchild))
                )

                for (i, child) in enumerate(children)
                    ii = [
                        findfirst(isequal(point), H2Trees.values(treecoarse, coarseparent))
                        for point in H2Trees.values(tree, child)
                    ]

                    LinearAlgebra.mul!(childmoment, moments[child], amplitudes[ii])
                    if i == 1
                        _applyi2i!(iparentmoment, childmoment, i2i, tree, child)
                    else
                        _applyi2i!(
                            iparentmoment, childmoment, i2i, tree, child; accumulate=true
                        )
                    end
                end

                for i in eachindex(parentmoment)
                    @test maximum(abs, parentmoment[i] - iparentmoment[i]) /
                          maximum(abs, parentmoment[i]) < 1e-10
                end
            end
        end
    end

    @testset "B-splines increasing II" begin
        poly = MultilevelKernelInterpolation.BsplinePolynomialEquidistant3D{Float64}(2, 5)

        i2i = MultilevelKernelInterpolation.I2ITranslator(
            [MultilevelKernelInterpolation.increase(poly), poly], 2:3
        )
        @test !isnothing(i2i.transfers)

        λ = 1.0
        k = 2π / λ
        m = readmesh(
            joinpath(
                pkgdir(MultilevelKernelInterpolation),
                "test",
                "assets",
                "in",
                "16spheres.in",
            ),
        )

        X = raviartthomas(m)
        tree = buildtree(X; builder=TwoNTreeBuilder(; minhalfsize=λ / 4, minvalues=0))
        treecoarse = buildtree(X; builder=TwoNTreeBuilder(; minhalfsize=λ / 2, minvalues=0))
        for op in [
            Maxwell3D.singlelayer(; wavenumber=k),
            Maxwell3D.doublelayer(; wavenumber=k),
            MultilevelKernelInterpolation.efie(; wavenumber=k),
            BEAST.DoubleLayerRotatedMW3D(1.0, im * 1.0),
        ]
            moments = Dict(
                zip(
                    H2Trees.leaves(tree),
                    MultilevelKernelInterpolation.TestMomentCollections(
                        op,
                        tree,
                        MultilevelKernelInterpolation.leafpolynomials(
                            tree,
                            MultilevelKernelInterpolation.levelpolynomials(
                                poly, H2Trees.levels(tree), (level) -> false
                            ),
                        ),
                        X;
                        quadstrat=BEAST.SingleNumQStrat(6),
                    ),
                ),
            )
            parentmoments = Dict(
                zip(
                    H2Trees.leaves(treecoarse),
                    MultilevelKernelInterpolation.TestMomentCollections(
                        op,
                        treecoarse,
                        MultilevelKernelInterpolation.leafpolynomials(
                            treecoarse,
                            MultilevelKernelInterpolation.levelpolynomials(
                                MultilevelKernelInterpolation.increase(poly),
                                H2Trees.levels(treecoarse),
                                (level) -> false,
                            ),
                        ),
                        X;
                        quadstrat=BEAST.SingleNumQStrat(6),
                    ),
                ),
            )

            childmoment = MultilevelKernelInterpolation.PolynomialMoment{
                MultilevelKernelInterpolation.dims(
                    MultilevelKernelInterpolation.testmoment(op, poly)
                ),
                Float64,
            }(
                size(LinearIndices(poly))
            )
            parentmoment = MultilevelKernelInterpolation.PolynomialMoment{
                MultilevelKernelInterpolation.dims(
                    MultilevelKernelInterpolation.testmoment(op, poly)
                ),
                Float64,
            }(
                size(LinearIndices(MultilevelKernelInterpolation.increase(poly)))
            )
            iparentmoment = MultilevelKernelInterpolation.PolynomialMoment{
                MultilevelKernelInterpolation.dims(
                    MultilevelKernelInterpolation.testmoment(op, poly)
                ),
                Float64,
            }(
                size(LinearIndices(MultilevelKernelInterpolation.increase(poly)))
            )

            for coarseparent in H2Trees.leaves(treecoarse)
                amplitudes = collect(1:length(H2Trees.values(treecoarse, coarseparent)))

                LinearAlgebra.mul!(parentmoment, parentmoments[coarseparent], amplitudes)

                coarseleafchild = 0
                for leaf in H2Trees.leaves(tree)
                    if H2Trees.center(tree, H2Trees.parent(tree, leaf)) ≈
                        H2Trees.center(treecoarse, coarseparent)
                        coarseleafchild = leaf
                        break
                    end
                end
                children = collect(
                    H2Trees.children(tree, H2Trees.parent(tree, coarseleafchild))
                )

                for (i, child) in enumerate(children)
                    ii = [
                        findfirst(isequal(point), H2Trees.values(treecoarse, coarseparent))
                        for point in H2Trees.values(tree, child)
                    ]

                    LinearAlgebra.mul!(childmoment, moments[child], amplitudes[ii])
                    if i == 1
                        _applyi2i!(iparentmoment, childmoment, i2i, tree, child)
                    else
                        _applyi2i!(
                            iparentmoment, childmoment, i2i, tree, child; accumulate=true
                        )
                    end
                end

                for i in eachindex(parentmoment)
                    @test maximum(abs, parentmoment[i] - iparentmoment[i]) /
                          maximum(abs, parentmoment[i]) < 1e-10
                end
            end
        end
    end
end

end
