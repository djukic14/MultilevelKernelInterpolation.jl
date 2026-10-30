module TestCheckProtrusionSafety

using Test
using CompScienceMeshes, BEAST, H2Trees, MultilevelKernelInterpolation

# `checkprotrusionsafety(tree, treepolynomials, plan)` checks the *real* geometric gap between
# every translating (receiving, translating) node pair `plan` actually drives an I2O
# translation for -- not an analytical per-level margin -- so exercising it needs a real
# mesh/tree/plan, not just synthetic polynomials. `plan` here always comes from
# `H2Trees.translatingplan(trialaggregationplan, testdisaggregationplan)`, matching exactly
# what `GalerkinKernelInterpolation` itself passes (see `mapoperations.jl`/`GalerkinKernelInterpolation.jl`), since which of
# the two plans actually carries translation data depends on `aggregationmode`.
@testset verbose=true "checkprotrusionsafety" begin
    m = readmesh(
        joinpath(pkgdir(MultilevelKernelInterpolation), "test", "assets", "in", "sphere.in")
    )
    X = lagrangecxd0(m)
    tree = buildtree(
        X;
        builder=TwoNTreeBuilder(;
            minvalues=120,
            protrusion=H2Trees.ProtrusionCheck(;
                max=0.25, compute=H2Trees.BEASTProtrusionFunctor(X)
            ),
        ),
    )

    plans = H2Trees.buildplans(
        tree;
        builder=H2Trees.PlanBuilder(; aggregatenode=H2Trees.AggregateAllNodesFunctor()),
    )
    plan = H2Trees.translatingplan(plans.trialaggregationplan, plans.testdisaggregationplan)

    @testset verbose=true "safe: real mesh-derived protrusion produces no warnings/errors" begin
        # The real, mesh-derived protrusion, exactly as `GalerkinKernelInterpolation` computes it -- this is
        # the same configuration `farfieldaccuracy.jl` verifies is numerically accurate
        # (~1e-5), so it should also be comfortably clear of both the hard-overlap error and
        # the paper's own recommended half-a-box warning zone.
        protrusion = H2Trees.maxprotrusion(
            tree; computeprotrusion=H2Trees.BEASTProtrusionFunctor(X)
        )
        poly = MultilevelKernelInterpolation.BufferedBsplinePolynomialEquidistant3D{Float64}(
            4, 10, MultilevelKernelInterpolation.AutoBufferFromProtrusion(0)
        )
        treepolys = MultilevelKernelInterpolation.levelpolynomials(
            poly, H2Trees.levels(tree), (level) -> false; protrusion=protrusion
        )

        @test_logs MultilevelKernelInterpolation.checkprotrusionsafety(
            tree, treepolys, plan
        )
    end

    @testset verbose=true "warning zone: close but not overlapping" begin
        # A synthetic, deliberately-oversized buffer (well beyond what the real mesh
        # protrusion needs) that's still short of actual overlap -- lands inside the paper's
        # own recommended "half a box distance" margin, so this should warn without erroring.
        poly = MultilevelKernelInterpolation.BufferedBsplinePolynomialEquidistant3D{Float64}(
            4, 10, 2
        )
        treepolys = [poly for _ in H2Trees.levels(tree)]

        @test_logs (:warn, r"half-a-box margin") match_mode = :any MultilevelKernelInterpolation.checkprotrusionsafety(
            tree, treepolys, plan
        )
    end

    @testset verbose=true "unsafe: forced overlap errors" begin
        # An even larger synthetic buffer, large enough that the buffer-extended sampling
        # domains of two well-separated boxes actually overlap.
        poly = MultilevelKernelInterpolation.BufferedBsplinePolynomialEquidistant3D{Float64}(
            4, 10, 6
        )
        treepolys = [poly for _ in H2Trees.levels(tree)]

        @test_throws "overlap" MultilevelKernelInterpolation.checkprotrusionsafety(
            tree, treepolys, plan
        )
    end
end

end
