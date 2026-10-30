module TestI2OCollectionToeplitz

using Test
using CompScienceMeshes, BEAST
using H2Trees
using MultilevelKernelInterpolation
using H2Factory
using LinearAlgebra

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
    @test MultilevelKernelInterpolation.defaultmomenttype(
        MultilevelKernelInterpolation.ToeplitzI2OStrategy()
    ) == MultilevelKernelInterpolation.ToeplitzPolynomialMoment

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

        poly = MultilevelKernelInterpolation.BarycentricLagrangePolynomialEquidistant3D(3)
        treepolys = MultilevelKernelInterpolation.levelpolynomials(
            poly, H2Trees.levels(tree), (level) -> false
        )

        for op in [
            Maxwell3D.singlelayer(; wavenumber=k),
            MultilevelKernelInterpolation.efie(; wavenumber=k),
            MultilevelKernelInterpolation.HH3DDoubleLayerFDBIOGradientI2O(; wavenumber=k),
            MultilevelKernelInterpolation.MWDoubleLayer3D(; wavenumber=k),
            MultilevelKernelInterpolation.MWHyperSingular3D(; wavenumber=k),
        ]
            i2o, i2ochannel = _withplanchunks(plan) do
                MultilevelKernelInterpolation.ToeplitzI2OTranslationCollection(
                    op, tree, treepolys, plan, X; verbose=false
                )
            end

            @test typeof(i2ochannel) <: H2Factory.ToeplitzI2OChannel
            @test H2Factory.i2otrait(i2o) == ToeplitzI2OTranslationTrait()

            i2ocomputer = _withplanchunks(plan) do
                MultilevelKernelInterpolation.ToeplitzTranslator(op, treepolys, tree, plan)
            end

            for tfs in plan.translatingnodes
                for combination in collect(tfs)
                    for receivingnode in combination[2]
                        if i2o[(combination[1], receivingnode)] isa
                            H2Factory.ToeplitzMatrixRepresentation
                            er = maximum(
                                norm,
                                i2ocomputer(
                                    combination[1],
                                    receivingnode,
                                    _pairdirection(tree, combination[1], receivingnode),
                                ).toeplitz - i2o[(combination[1], receivingnode)].toeplitz,
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
                                    i2o[(combination[1], receivingnode)][i].toeplitz,
                                )
                                @test er < 1e-11
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

        poly = MultilevelKernelInterpolation.BarycentricLagrangePolynomialEquidistant3D(3)
        treepolys = MultilevelKernelInterpolation.levelpolynomials(
            poly, H2Trees.levels(tree.testcluster), (level) -> false
        )

        for op in [
            Maxwell3D.singlelayer(; wavenumber=k),
            MultilevelKernelInterpolation.efie(; wavenumber=k),
            MultilevelKernelInterpolation.HH3DDoubleLayerFDBIOGradientI2O(; wavenumber=k),
            MultilevelKernelInterpolation.MWDoubleLayer3D(; wavenumber=k),
            MultilevelKernelInterpolation.MWHyperSingular3D(; wavenumber=k),
        ]
            i2o, i2ochannel = _withplanchunks(plan) do
                MultilevelKernelInterpolation.ToeplitzI2OTranslationCollection(
                    op, tree, treepolys, plan, X, X; verbose=false
                )
            end

            @test typeof(i2ochannel) <: H2Factory.ToeplitzI2OChannel
            @test H2Factory.i2otrait(i2o) == ToeplitzI2OTranslationTrait()

            i2ocomputer = _withplanchunks(plan) do
                MultilevelKernelInterpolation.ToeplitzTranslator(op, treepolys, tree, plan)
            end

            for tfs in plan.translatingnodes
                for combination in collect(tfs)
                    for receivingnode in combination[2]
                        if i2o[(combination[1], receivingnode)] isa
                            H2Factory.ToeplitzMatrixRepresentation
                            er = maximum(
                                norm,
                                i2ocomputer(
                                    combination[1],
                                    receivingnode,
                                    _pairdirection(tree, combination[1], receivingnode),
                                ).toeplitz - i2o[(combination[1], receivingnode)].toeplitz,
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
                                    i2o[(combination[1], receivingnode)][i].toeplitz,
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
