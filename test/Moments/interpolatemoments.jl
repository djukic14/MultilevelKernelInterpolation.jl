module TestInterpolateMoments

using Test
using CompScienceMeshes, BEAST
using H2Trees
using MultilevelKernelInterpolation

# Checks that I2I-translating child-level moments (computed directly from BEAST assembly on a
# fine tree) up to the parent level reproduces the parent-level moment computed directly from
# BEAST assembly on a coarser tree. The Helmholtz/Lagrange-constant-order portion here
# genuinely overlaps `Translations/i2i.jl`'s "Lagrange const" testset (same mesh/tree-pair/
# operators, same real-moment-consistency scenario) -- deliberately kept, not an oversight:
# this file applies the raw `i2itranslator.transfers[sector]` directly, while
# `i2i.jl` drives translation through the production `i2i(...)` callable `GalerkinMap`
# actually uses, so the overlap cross-checks that both interpolation code paths agree (same
# spirit as `toeplitz_equivalence.jl`'s deliberate dense-vs-Toeplitz agreement check). This
# file also covers Trial-side moments (`i2i.jl` only exercises Test) and plain constant-order
# Lagrange for Maxwell3D operators (`i2i.jl`'s only Maxwell coverage is inside its
# B-spline-order-increasing testset) -- neither of those is covered elsewhere.
@testset verbose=true "Lagrange Interpolation of Moments" begin
    @testset "Helmholtz3D" begin
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

        poly = MultilevelKernelInterpolation.BarycentricLagrangePolynomialEquidistant3D(7)
        finepolys = MultilevelKernelInterpolation.leafpolynomials(
            tree, [poly for _ in H2Trees.levels(tree)]
        )
        coarsepolys = MultilevelKernelInterpolation.leafpolynomials(
            treecoarse, [poly for _ in H2Trees.levels(treecoarse)]
        )

        for op in [
            Helmholtz3D.singlelayer(; wavenumber=k),
            Helmholtz3D.doublelayer_transposed(; wavenumber=k),
            Helmholtz3D.hypersingular(; wavenumber=k),
            MultilevelKernelInterpolation.HH3DDoubleLayerTransposedFDBIOGradientI2O(;
                wavenumber=k
            ),
        ]
            for collection in [
                MultilevelKernelInterpolation.TestMomentCollections,
                MultilevelKernelInterpolation.TrialMomentCollections,
            ]
                moments = Dict(
                    zip(
                        H2Trees.leaves(tree),
                        collection(
                            op, tree, finepolys, X; quadstrat=BEAST.SingleNumQStrat(6)
                        ),
                    ),
                )
                parentmoments = Dict(
                    zip(
                        H2Trees.leaves(treecoarse),
                        collection(
                            op,
                            treecoarse,
                            coarsepolys,
                            X;
                            quadstrat=BEAST.SingleNumQStrat(6),
                        ),
                    ),
                )

                i2itranslator = MultilevelKernelInterpolation.I2ITranslator(
                    [poly, poly], 1:2
                )
                for coarseparent in H2Trees.leaves(treecoarse)
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
                    @assert length(children) == 8

                    interpolationmatrices = i2itranslator.transfers[2]

                    for i in eachindex(parentmoments[coarseparent].moments)
                        interpolatedmoment = zeros(
                            size(parentmoments[coarseparent].moments[1])
                        )

                        for child in children
                            ii = [
                                findfirst(
                                    isequal(point), H2Trees.values(treecoarse, coarseparent)
                                ) for point in H2Trees.values(tree, child)
                            ]

                            interpolatedmoment[:, ii] .+=
                                interpolationmatrices[H2Trees.sector(tree, child) + 1] *
                                moments[child].moments[i]
                        end
                        er =
                            abs.(interpolatedmoment - parentmoments[coarseparent][i]) /
                            maximum(abs, parentmoments[coarseparent][i])
                        @test maximum(abs, er) < 1e-10
                    end
                end
            end
        end
    end

    @testset "Maxwell3D" begin
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

        poly = MultilevelKernelInterpolation.BarycentricLagrangePolynomialEquidistant3D(7)
        finepolys = MultilevelKernelInterpolation.leafpolynomials(
            tree, [poly for _ in H2Trees.levels(tree)]
        )
        coarsepolys = MultilevelKernelInterpolation.leafpolynomials(
            treecoarse, [poly for _ in H2Trees.levels(treecoarse)]
        )

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
                        op, tree, finepolys, X; quadstrat=BEAST.SingleNumQStrat(6)
                    ),
                ),
            )
            parentmoments = Dict(
                zip(
                    H2Trees.leaves(treecoarse),
                    MultilevelKernelInterpolation.TestMomentCollections(
                        op, treecoarse, coarsepolys, X; quadstrat=BEAST.SingleNumQStrat(6)
                    ),
                ),
            )

            i2itranslator = MultilevelKernelInterpolation.I2ITranslator([poly, poly], 1:2)
            for coarseparent in H2Trees.leaves(treecoarse)
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
                @assert length(children) == 8

                interpolationmatrices = i2itranslator.transfers[2]

                for i in eachindex(parentmoments[coarseparent].moments)
                    interpolatedmoment = zeros(size(parentmoments[coarseparent].moments[1]))

                    for child in children
                        ii = [
                            findfirst(
                                isequal(point), H2Trees.values(treecoarse, coarseparent)
                            ) for point in H2Trees.values(tree, child)
                        ]

                        interpolatedmoment[:, ii] .+=
                            interpolationmatrices[H2Trees.sector(tree, child) + 1] *
                            moments[child].moments[i]
                    end
                    er =
                        abs.(interpolatedmoment - parentmoments[coarseparent][i]) /
                        maximum(abs, parentmoments[coarseparent][i])
                    @test maximum(abs, er) < 1e-10
                end
            end
        end
    end
end

end
