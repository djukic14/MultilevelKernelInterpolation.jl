module TestI2OCollectionBsplineToeplitz

using Test
using CompScienceMeshes, BEAST
using H2Trees
using MultilevelKernelInterpolation
using H2Factory
using LinearAlgebra
using H2FactoryTestUtils

import CompScienceMeshes: translate

function _withplanchunks(f, plan)
    schedule = H2Factory.planschedule(plan, Base.Threads.nthreads())
    return H2Factory.withnchunks(H2Factory.schedulechunkcounts(schedule)) do
        return f()
    end
end

# THE DIRECTION IS NO LONGER INERT. `AbstractKernelTranslator`'s third argument used to be discarded --
# the geometry was recomputed from the node pair -- so these tests passed `[0, 0, 0]` as a placeholder.
# It is now load-bearing: `computetranslations` hands the builder the translation ID's direction, which
# under `H2Trees.SymmetryDirectionInvariancePerLevel` is the ORBIT'S CANONICAL one and NOT the
# representative pair's own displacement. A builder that recomputes stores the wrong operator for the
# orbit. Passing `[0, 0, 0]` now places the translating grid on top of the receiving box, so the kernel
# is singular and the result is `Inf`.
#
# These tests want the pair's TRUE displacement, which is what the unreduced pipeline passes.
_pairdirection(tree, receiving, translating) =
    H2Trees.center(tree, receiving) - H2Trees.center(tree, translating)
function _pairdirection(tree::H2Trees.BlockTree, receiving, translating)
    return H2Trees.center(H2Trees.testtree(tree), receiving) -
           H2Trees.center(H2Trees.trialtree(tree), translating)
end

@testset verbose=true "Computation Toeplitz I2OCollection" begin
    @testset "Galerkin" begin
        λ = 1.0
        k = 2π / λ
        m = readmesh(
            joinpath(
                pkgdir(MultilevelKernelInterpolation), "test", "assets", "in", "sphere.in"
            ),
        )

        X = lagrangec0d1(m)
        tree = buildtree(X; builder=TwoNTreeBuilder(; minhalfsize=λ / 5, minvalues=0))

        plan = H2Trees.buildplans(
            tree;
            builder=H2Trees.PlanBuilder(; aggregatenode=H2Trees.AggregateAllNodesFunctor()),
        ).testdisaggregationplan

        for poly in [
            MultilevelKernelInterpolation.BsplinePolynomialEquidistant3D{Float64}(1, 5),
            MultilevelKernelInterpolation.BsplinePolynomialEquidistant3D{Float64}(2, 5),
            MultilevelKernelInterpolation.BsplinePolynomialEquidistant3D{Float64}(2, 7),
        ]
            treepolys = MultilevelKernelInterpolation.levelpolynomials(
                poly, H2Trees.levels(tree), (level) -> false
            )

            for op in [
                Maxwell3D.singlelayer(; wavenumber=k),
                MultilevelKernelInterpolation.efie(; wavenumber=k),
                MultilevelKernelInterpolation.HH3DDoubleLayerFDBIOGradientI2O(;
                    wavenumber=k
                ),
                MultilevelKernelInterpolation.MWDoubleLayer3D(; wavenumber=k),
                MultilevelKernelInterpolation.MWHyperSingular3D(; wavenumber=k),
            ]
                i2o, i2ochannel = _withplanchunks(plan) do
                    MultilevelKernelInterpolation.ToeplitzI2OTranslationCollection(
                        op, tree, treepolys, plan, X; verbose=false
                    )
                end

                # Demoted: the B-spline Toeplitz map is an `I2OTranslation` whose coupling is the plain
                # `H2Factory.ToeplitzI2OTranslationCollection` (identical to the non-B-spline path) and
                # whose modulation is the extracted `BsplineToeplitzModulation` policy.
                @test i2o isa H2Factory.I2OTranslation
                @test H2Factory.coupling(i2o) isa H2Factory.ToeplitzI2OTranslationCollection
                @test H2Factory.modulation(i2o) isa
                    MultilevelKernelInterpolation.BsplineToeplitzModulation
                @test typeof(i2ochannel) <:
                    MultilevelKernelInterpolation.BsplineToeplitzI2OChannel
                toeplitzcoupling = H2Factory.coupling(i2o)
                @test H2Factory.i2otrait(toeplitzcoupling) == ToeplitzI2OTranslationTrait()
                @test H2Trees.tree(toeplitzcoupling) == tree

                i2ocomputer = _withplanchunks(plan) do
                    MultilevelKernelInterpolation.ToeplitzTranslator(
                        op, treepolys, tree, plan
                    )
                end

                for tfs in plan.translatingnodes
                    for combination in collect(tfs)
                        @test H2Trees.level(toeplitzcoupling, combination[1]) ==
                            H2Trees.level(tree, combination[1])

                        for receivingnode in combination[2]
                            if toeplitzcoupling[(combination[1], receivingnode)] isa
                                H2Factory.ToeplitzMatrixRepresentation
                                er = maximum(
                                    norm,
                                    i2ocomputer(
                                        combination[1],
                                        receivingnode,
                                        _pairdirection(tree, combination[1], receivingnode),
                                    ).toeplitz -
                                    toeplitzcoupling[(combination[1], receivingnode)].toeplitz,
                                )
                                @test er < 1e-11
                            else
                                comparison = i2ocomputer(
                                    combination[1],
                                    receivingnode,
                                    _pairdirection(tree, combination[1], receivingnode),
                                )

                                for i in eachindex(comparison)
                                    er = maximum(
                                        norm,
                                        comparison[i].toeplitz -
                                        toeplitzcoupling[(combination[1], receivingnode)][i].toeplitz,
                                    )
                                    @test er < 1e-11
                                end
                            end
                        end
                    end
                end
            end
        end
    end

    @testset "Petrov" begin
        λ = 1.0
        k = 2π / λ
        m = readmesh(
            joinpath(
                pkgdir(MultilevelKernelInterpolation), "test", "assets", "in", "sphere.in"
            ),
        )

        X = lagrangec0d1(m)
        Y = lagrangec0d1(translate(m, [1.0, 0.0, 0.0]))
        tree = buildtree(
            X,
            Y;
            builder=BlockTreeBuilder(;
                test=TwoNTreeBuilder(; minhalfsize=λ / 5, minvalues=0),
                trial=TwoNTreeBuilder(; minhalfsize=λ / 5, minvalues=0),
            ),
        )

        plan = H2Trees.buildplans(
            tree;
            builder=H2Trees.PlanBuilder(; aggregatenode=H2Trees.AggregateAllNodesFunctor()),
        ).testdisaggregationplan

        for poly in [
            MultilevelKernelInterpolation.BsplinePolynomialEquidistant3D{Float64}(1, 5),
            MultilevelKernelInterpolation.BsplinePolynomialEquidistant3D{Float64}(2, 5),
        ]
            treepolys = MultilevelKernelInterpolation.levelpolynomials(
                poly, H2Trees.levels(tree.testcluster), (level) -> false
            )

            for op in [
                Maxwell3D.singlelayer(; wavenumber=k),
                MultilevelKernelInterpolation.efie(; wavenumber=k),
                MultilevelKernelInterpolation.HH3DDoubleLayerFDBIOGradientI2O(;
                    wavenumber=k
                ),
                MultilevelKernelInterpolation.MWDoubleLayer3D(; wavenumber=k),
                MultilevelKernelInterpolation.MWHyperSingular3D(; wavenumber=k),
            ]
                i2o, i2ochannel = _withplanchunks(plan) do
                    MultilevelKernelInterpolation.ToeplitzI2OTranslationCollection(
                        op, tree, treepolys, plan, X, X; verbose=false
                    )
                end

                @test i2o isa H2Factory.I2OTranslation
                @test H2Factory.coupling(i2o) isa H2Factory.ToeplitzI2OTranslationCollection
                @test H2Factory.modulation(i2o) isa
                    MultilevelKernelInterpolation.BsplineToeplitzModulation
                @test typeof(i2ochannel) <:
                    MultilevelKernelInterpolation.BsplineToeplitzI2OChannel
                toeplitzcoupling = H2Factory.coupling(i2o)
                @test H2Factory.i2otrait(toeplitzcoupling) == ToeplitzI2OTranslationTrait()

                i2ocomputer = _withplanchunks(plan) do
                    MultilevelKernelInterpolation.ToeplitzTranslator(
                        op, treepolys, tree, plan
                    )
                end

                for tfs in plan.translatingnodes
                    for combination in collect(tfs)
                        for receivingnode in combination[2]
                            if toeplitzcoupling[(combination[1], receivingnode)] isa
                                H2Factory.ToeplitzMatrixRepresentation
                                er = maximum(
                                    norm,
                                    i2ocomputer(
                                        combination[1],
                                        receivingnode,
                                        _pairdirection(tree, combination[1], receivingnode),
                                    ).toeplitz -
                                    toeplitzcoupling[(combination[1], receivingnode)].toeplitz,
                                )
                                @test er < 1e-11
                            else
                                comparison = i2ocomputer(
                                    combination[1],
                                    receivingnode,
                                    _pairdirection(tree, combination[1], receivingnode),
                                )

                                for i in eachindex(comparison)
                                    er = maximum(
                                        norm,
                                        comparison[i].toeplitz -
                                        toeplitzcoupling[(combination[1], receivingnode)][i].toeplitz,
                                    )
                                    @test er < 1e-11
                                end
                            end
                        end
                    end
                end
            end
        end
    end
end

@testset verbose=true "Compare Toeplitz Translations" begin
    @testset "Galerkin" begin
        λ = 1.0
        k = 2π / λ
        m = readmesh(
            joinpath(
                pkgdir(MultilevelKernelInterpolation), "test", "assets", "in", "sphere.in"
            ),
        )

        X = lagrangec0d1(m)
        tree = buildtree(X; builder=TwoNTreeBuilder(; minhalfsize=λ / 5, minvalues=0))

        plan = H2Trees.buildplans(
            tree;
            builder=H2Trees.PlanBuilder(; aggregatenode=H2Trees.AggregateAllNodesFunctor()),
        ).testdisaggregationplan

        for poly in [
            MultilevelKernelInterpolation.BsplinePolynomialEquidistant3D{Float64}(1, 5),
            MultilevelKernelInterpolation.BsplinePolynomialEquidistant3D{Float64}(3, 5),
        ]
            comparisonpoly = MultilevelKernelInterpolation.BarycentricLagrangePolynomialEquidistant3D(
                4
            )
            treepolys = MultilevelKernelInterpolation.levelpolynomials(
                poly, H2Trees.levels(tree), (level) -> false
            )
            treecomparisonpolys = MultilevelKernelInterpolation.levelpolynomials(
                comparisonpoly, H2Trees.levels(tree), (level) -> false
            )

            for op in [
                Maxwell3D.singlelayer(; wavenumber=k),
                MultilevelKernelInterpolation.efie(; wavenumber=k),
                MultilevelKernelInterpolation.HH3DDoubleLayerFDBIOGradientI2O(;
                    wavenumber=k
                ),
                MultilevelKernelInterpolation.MWDoubleLayer3D(; wavenumber=k),
                MultilevelKernelInterpolation.MWHyperSingular3D(; wavenumber=k),
            ]
                i2o, _ = MultilevelKernelInterpolation.I2OTranslationCollection(
                    op, tree, treecomparisonpolys, plan, X; verbose=false
                )
                toeplitzi2o, i2ochannel = _withplanchunks(plan) do
                    MultilevelKernelInterpolation.ToeplitzI2OTranslationCollection(
                        op, tree, treepolys, plan, X; verbose=false
                    )
                end
                # Demoted: unwrap the `I2OTranslation` to its plain Toeplitz coupling for per-pair indexing.
                toeplitzcoupling = H2Factory.coupling(toeplitzi2o)

                @test_nowarn MultilevelKernelInterpolation.takebuffer!(
                    i2ochannel, H2Trees.levels(plan)[begin], 1
                )

                ifftplan = H2Factory.takeifftplan!(i2ochannel, 4, 1, H2Factory.isForward())
                fftplan = H2Factory.takefftplan!(i2ochannel, 4, 1, H2Factory.isForward())

                xpad = zeros(
                    Float64, 2 .* MultilevelKernelInterpolation.numpolynomials(poly) .- 1
                )
                # ONE rng for the key loop, ADVANCED per key. `checkrng()` returns a fresh RNG on
                # the same fixed seed, so calling it inside would give every coupling key of a given
                # size the identical `x` -- many keys checked against one vector, and nothing to
                # notice because each assertion still passes.
                keyrng = H2FactoryTestUtils.checkrng()
                for key in keys(i2o.dict)
                    x = randn(
                        keyrng, Float64, MultilevelKernelInterpolation.numpolynomials(poly)
                    )

                    if toeplitzcoupling[key] isa H2Factory.ToeplitzMatrixRepresentation
                        y = i2o[key] * view(x, :)

                        xpad .= 0.0
                        xpad[axes(x)...] .= x

                        yt = (ifftplan * (toeplitzcoupling[key].toeplitz .* (fftplan * xpad)))[axes(
                            x
                        )...][:]
                        er = maximum(abs, y - yt)
                        @test er < 1e-10
                    else
                        for i in eachindex(i2o[key])
                            y = i2o[key][i] * view(x, :)

                            xpad .= 0.0
                            xpad[axes(x)...] .= x

                            yt = (ifftplan * (toeplitzcoupling[key][i].toeplitz .* (fftplan * xpad)))[axes(
                                x
                            )...][:]
                            er = maximum(abs, y - yt)
                            @test er < 1e-10
                        end
                    end
                end
            end
        end
    end

    @testset "Petrov" begin
        λ = 1.0
        k = 2π / λ
        m = readmesh(
            joinpath(
                pkgdir(MultilevelKernelInterpolation), "test", "assets", "in", "sphere.in"
            ),
        )

        X = lagrangec0d1(m)
        Y = lagrangec0d1(translate(m, [1.0, 0.0, 0.0]))
        tree = buildtree(
            X,
            Y;
            builder=BlockTreeBuilder(;
                test=TwoNTreeBuilder(; minhalfsize=λ / 5, minvalues=0),
                trial=TwoNTreeBuilder(; minhalfsize=λ / 5, minvalues=0),
            ),
        )

        plan = H2Trees.buildplans(
            tree;
            builder=H2Trees.PlanBuilder(; aggregatenode=H2Trees.AggregateAllNodesFunctor()),
        ).testdisaggregationplan

        for poly in [
            MultilevelKernelInterpolation.BsplinePolynomialEquidistant3D{Float64}(1, 5),
            MultilevelKernelInterpolation.BsplinePolynomialEquidistant3D{Float64}(3, 5),
        ]
            comparisonpoly = MultilevelKernelInterpolation.BarycentricLagrangePolynomialEquidistant3D(
                4
            )
            treepolys = MultilevelKernelInterpolation.levelpolynomials(
                poly, H2Trees.levels(tree.testcluster), (level) -> false
            )
            treecomparisonpolys = MultilevelKernelInterpolation.levelpolynomials(
                comparisonpoly, H2Trees.levels(tree.testcluster), (level) -> false
            )
            for op in [
                Maxwell3D.singlelayer(; wavenumber=k),
                MultilevelKernelInterpolation.efie(; wavenumber=k),
                MultilevelKernelInterpolation.HH3DDoubleLayerFDBIOGradientI2O(;
                    wavenumber=k
                ),
                MultilevelKernelInterpolation.MWDoubleLayer3D(; wavenumber=k),
                MultilevelKernelInterpolation.MWHyperSingular3D(; wavenumber=k),
            ]
                i2o, _ = MultilevelKernelInterpolation.I2OTranslationCollection(
                    op, tree, treecomparisonpolys, plan, X, Y; verbose=false
                )
                toeplitzi2o, i2ochannel = _withplanchunks(plan) do
                    MultilevelKernelInterpolation.ToeplitzI2OTranslationCollection(
                        op, tree, treepolys, plan, X, Y; verbose=false
                    )
                end
                # Demoted: unwrap the `I2OTranslation` to its plain Toeplitz coupling for per-pair indexing.
                toeplitzcoupling = H2Factory.coupling(toeplitzi2o)

                @test_nowarn MultilevelKernelInterpolation.takebuffer!(
                    i2ochannel, H2Trees.levels(plan)[begin], 1
                )

                ifftplan = H2Factory.takeifftplan!(i2ochannel, 4, 1, H2Factory.isForward())
                fftplan = H2Factory.takefftplan!(i2ochannel, 4, 1, H2Factory.isForward())

                xpad = zeros(
                    Float64, 2 .* MultilevelKernelInterpolation.numpolynomials(poly) .- 1
                )
                # ONE rng for the key loop, ADVANCED per key. `checkrng()` returns a fresh RNG on
                # the same fixed seed, so calling it inside would give every coupling key of a given
                # size the identical `x` -- many keys checked against one vector, and nothing to
                # notice because each assertion still passes.
                keyrng = H2FactoryTestUtils.checkrng()
                for key in keys(i2o.dict)
                    x = randn(
                        keyrng, Float64, MultilevelKernelInterpolation.numpolynomials(poly)
                    )
                    if toeplitzcoupling[key] isa H2Factory.ToeplitzMatrixRepresentation
                        y = i2o[key] * view(x, :)

                        xpad .= 0.0
                        xpad[axes(x)...] .= x

                        yt = (ifftplan * (toeplitzcoupling[key].toeplitz .* (fftplan * xpad)))[axes(
                            x
                        )...][:]
                        er = maximum(abs, y - yt)
                        @test er < 1e-10
                    else
                        for i in eachindex(i2o[key])
                            y = i2o[key][i] * view(x, :)

                            xpad .= 0.0
                            xpad[axes(x)...] .= x

                            yt = (ifftplan * (toeplitzcoupling[key][i].toeplitz .* (fftplan * xpad)))[axes(
                                x
                            )...][:]
                            er = maximum(abs, y - yt)
                            @test er < 1e-10
                        end
                    end
                end
            end
        end
    end
end

end
