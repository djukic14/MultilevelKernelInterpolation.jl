module TestCheckProtrusionSafetyBlockTree

using Test
using StaticArrays
using CompScienceMeshes, BEAST, H2Trees, MultilevelKernelInterpolation

# Cross-tree (5-arg) form of `checkprotrusionsafety`, exercised directly (not just indirectly
# via full `PetrovGalerkinKernelInterpolation` construction) across genuinely separate test/trial `BlockTree`
# sides -- two spheres of different radii, at four different relative distance regimes (far
# apart, moderately separated, overlapping, and concentric/same-center). For each regime, this
# checks both a case that should work (real/minimal protrusion -- no error) and a case that
# should deliberately fail (an artificially inflated protrusion, large enough to force the
# buffer-extended sampling domains of some translating pair to actually overlap).
function buildtrees(
    testcenter, testradius, testh, trialcenter, trialradius, trialh; testminvalues=120
)
    testmesh = translate(meshsphere(testradius, testh), SVector(testcenter...))
    trialmesh = translate(meshsphere(trialradius, trialh), SVector(trialcenter...))
    testspace = lagrangecxd0(testmesh)
    trialspace = lagrangecxd0(trialmesh)
    # `protrusion=NoProtrusionCheck()` on the tree builder: this is a wholly different, tree-
    # construction-time check from what this file actually tests. `checkprotrusionsafety`/
    # `buildpolys` below feed their own explicit `protrusionval` into `levelpolynomials`;
    # nothing here should depend on the tree's own construction-time protrusion check.
    tree = buildtree(
        testspace,
        trialspace;
        builder=BlockTreeBuilder(;
            test=TwoNTreeBuilder(; minvalues=testminvalues, protrusion=NoProtrusionCheck()),
            trial=TwoNTreeBuilder(;
                minvalues=testminvalues, protrusion=NoProtrusionCheck()
            ),
        ),
    )
    thetesttree = H2Trees.testtree(tree)
    thetrialtree = H2Trees.trialtree(tree)

    # `MultilevelKernelInterpolation.InteractionStrategy()` here is all defaults (isnear/translatingnodesiterator/
    # aggregatenode/aggregationmode), which is exactly `H2Trees.PlanBuilder()`'s own default
    # chain, so `buildplans` needs no explicit builder.
    plans = H2Trees.buildplans(tree)
    plan = H2Trees.translatingplan(plans.trialaggregationplan, plans.testdisaggregationplan)
    return thetesttree, thetrialtree, plan
end

function buildpolys(thetesttree, thetrialtree, protrusionval)
    poly = MultilevelKernelInterpolation.BufferedBsplinePolynomialEquidistant3D{Float64}(
        4, 10, MultilevelKernelInterpolation.AutoBufferFromProtrusion(0)
    )
    testpolys = MultilevelKernelInterpolation.levelpolynomials(
        poly,
        H2Trees.levels(thetesttree),
        (level) -> false;
        protrusion=fill(protrusionval, length(H2Trees.levels(thetesttree))),
    )
    trialpolys = MultilevelKernelInterpolation.levelpolynomials(
        poly,
        H2Trees.levels(thetrialtree),
        (level) -> false;
        protrusion=fill(protrusionval, length(H2Trees.levels(thetrialtree))),
    )
    return testpolys, trialpolys
end

# Calibrated by direct search over `protrusion` (see session notes): each scenario's own
# geometry determines what counts as "real/minimal" vs. "deliberately excessive" -- these are
# not arbitrary, they were found by bisecting each scenario's own safe/error boundary.
const SCENARIOS = [
    (
        label="far apart (distance 15, radii 1.0/0.8)",
        args=((0.0, 0.0, 0.0), 1.0, 0.25, (15.0, 0.0, 0.0), 0.8, 0.2),
        testminvalues=120,
        safeprotrusion=0.0,
        failprotrusion=8.0,
    ),
    (
        label="moderate distance (distance 4, radii 1.0/0.8)",
        args=((0.0, 0.0, 0.0), 1.0, 0.25, (4.0, 0.0, 0.0), 0.8, 0.2),
        testminvalues=120,
        safeprotrusion=0.0,
        failprotrusion=1.0,
    ),
    (
        label="overlapping (distance 1.6, radii 1.0/0.7)",
        args=((0.0, 0.0, 0.0), 1.0, 0.25, (1.6, 0.0, 0.0), 0.7, 0.2),
        testminvalues=120,
        safeprotrusion=0.0,
        failprotrusion=0.2,
    ),
    (
        label="same center, different radii (concentric, radii 3.0/0.3)",
        args=((0.0, 0.0, 0.0), 3.0, 0.5, (0.0, 0.0, 0.0), 0.3, 0.08),
        testminvalues=10,
        safeprotrusion=0.0,
        failprotrusion=2.0,
    ),
]

@testset verbose=true "checkprotrusionsafety (BlockTree, cross-tree)" begin
    for scenario in SCENARIOS
        @testset "$(scenario.label)" begin
            thetesttree, thetrialtree, plan = buildtrees(
                scenario.args...; testminvalues=scenario.testminvalues
            )

            @testset "works: real/minimal protrusion does not error" begin
                testpolys, trialpolys = buildpolys(
                    thetesttree, thetrialtree, scenario.safeprotrusion
                )
                @test begin
                    MultilevelKernelInterpolation.checkprotrusionsafety(
                        thetesttree, testpolys, thetrialtree, trialpolys, plan
                    )
                    true
                end
            end

            @testset "deliberately fails: excessive protrusion forces an overlap error" begin
                testpolys, trialpolys = buildpolys(
                    thetesttree, thetrialtree, scenario.failprotrusion
                )
                @test_throws "overlap" MultilevelKernelInterpolation.checkprotrusionsafety(
                    thetesttree, testpolys, thetrialtree, trialpolys, plan
                )
            end
        end
    end

    @testset "warning zone: close but not overlapping (moderate distance)" begin
        # Representative check that the intermediate "warn, don't error" tier also works
        # correctly across trees, not just within the safe/unsafe boundary -- picked from the
        # "moderate distance" scenario, which has a clean, wide warning window (protrusion
        # 0.15-0.5 warns without erroring; see calibration above).
        thetesttree, thetrialtree, plan = buildtrees(
            (0.0, 0.0, 0.0), 1.0, 0.25, (4.0, 0.0, 0.0), 0.8, 0.2; testminvalues=120
        )
        testpolys, trialpolys = buildpolys(thetesttree, thetrialtree, 0.2)
        @test_logs (:warn, r"half-a-box margin") match_mode = :any MultilevelKernelInterpolation.checkprotrusionsafety(
            thetesttree, testpolys, thetrialtree, trialpolys, plan
        )
    end
end

end
