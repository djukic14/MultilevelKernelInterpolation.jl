module TestPetrovRectangularMoments

using Test
using LinearAlgebra
using BEAST
using CompScienceMeshes
using H2Trees
using H2Factory
using MultilevelKernelInterpolation

# Part E PHASE 5b slice 1 -- PER-SIDE (coefficient-rectangular) moment sizing.
#
# `MultilevelKernelInterpolation.moments` used to size BOTH sides from one merged `level => polynomial` family, silently
# imposing one side's node count on the other. It now takes one family per side.
#
# THIS TEST WORKS BELOW THE PUBLIC CONSTRUCTOR, DELIBERATELY.
#
# End-to-end rectangular support now exists on the dense route -- `petrov_toeplitz.jl` asserts its
# accuracy through the public constructor. What that test cannot isolate is the SIZING machinery
# itself: it runs a real buffered B-spline map, where a sizing error would surface as an accuracy
# number rather than as a wrong moment shape.
#
# So this file drives `MultilevelKernelInterpolation.moments` directly with two deliberately different Lagrange families. It
# pins the invariant -- each side's moments are sized from ITS OWN family -- independently of the
# polynomial type, the modulation and the translation, so a failure here localizes to sizing alone.

const MESH = readmesh(
    joinpath(pkgdir(MultilevelKernelInterpolation), "test", "assets", "in", "sphere.in")
)
const OP = Helmholtz3D.singlelayer(; wavenumber=1.0)

nodecount(poly) = size(LinearIndices(poly))

function petrovsetup()
    testspace, trialspace = lagrangec0d1(MESH), lagrangecxd0(MESH)
    tree = H2Trees.buildtree(
        testspace,
        trialspace;
        builder=H2Trees.BlockTreeBuilder(;
            test=H2Trees.TwoNTreeBuilder(; minvalues=120),
            trial=H2Trees.TwoNTreeBuilder(; minvalues=120),
        ),
    )
    plans = MultilevelKernelInterpolation.assembleplans(
        tree, MultilevelKernelInterpolation.InteractionStrategy()
    )
    return tree, plans
end

@testset "per-side moment sizing: each side sized from its OWN family" begin
    tree, plans = petrovsetup()

    # Two genuinely DIFFERENT families. Lagrange orders are used rather than buffered B-splines
    # because the subject here is the SIZING machinery, and Lagrange lets the two sides differ by
    # construction without dragging in the B-spline modulation that is not yet side-specific.
    testpoly = MultilevelKernelInterpolation.BarycentricLagrangePolynomialEquidistant3D(6)
    trialpoly = MultilevelKernelInterpolation.BarycentricLagrangePolynomialEquidistant3D(4)

    # NON-VACUITY: with equal node counts, a collapsed implementation would satisfy everything below.
    @test nodecount(testpoly) != nodecount(trialpoly)

    testdict = Dict(
        level => testpoly for level in H2Trees.levels(plans.testaggregationplan)
    )
    trialdict = Dict(
        level => trialpoly for level in H2Trees.levels(plans.trialaggregationplan)
    )

    _, testmoments, trialmoments = MultilevelKernelInterpolation.moments(
        complex(BEAST.gamma(OP)),
        MultilevelKernelInterpolation.testmoment(OP, testpoly),
        # Each side's moment type from ITS OWN polynomial. Only the operator DIMENSION is read here,
        # so passing `testpoly` was harmless -- but in a test whose whole subject is "each side owns
        # its own data", the argument should say what it means.
        MultilevelKernelInterpolation.trialmoment(OP, trialpoly),
        tree,
        testdict,
        trialdict,
        plans.testaggregationplan,
        plans.trialaggregationplan,
        H2Factory.Moment,
    )

    testsizes = unique(size(m[1]) for m in values(testmoments))
    trialsizes = unique(size(m[1]) for m in values(trialmoments))

    # Each side carries ONE size, and the two sides differ -- the invariant slice 1 introduced.
    # Derived from the families rather than hardcoded: node counts depend on mesh and polynomial
    # parameters, and a previously recorded pair did not reproduce when re-measured.
    @test length(testsizes) == 1
    @test length(trialsizes) == 1
    @test testsizes != trialsizes
end

@testset "single-family form still serves both sides (Galerkin path unchanged)" begin
    # The 8-argument form forwards one family to both sides, which is what Galerkin needs -- one
    # tree, one space, one family. Pinned so the per-side split cannot silently change it.
    tree, plans = petrovsetup()
    poly = MultilevelKernelInterpolation.BarycentricLagrangePolynomialEquidistant3D(6)
    onedict = Dict(
        level => poly for level in union(
            H2Trees.levels(plans.testaggregationplan),
            H2Trees.levels(plans.trialaggregationplan),
        )
    )

    _, testmoments, trialmoments = MultilevelKernelInterpolation.moments(
        complex(BEAST.gamma(OP)),
        MultilevelKernelInterpolation.testmoment(OP, poly),
        MultilevelKernelInterpolation.trialmoment(OP, poly),
        tree,
        onedict,
        plans.testaggregationplan,
        plans.trialaggregationplan,
        H2Factory.Moment,
    )

    sizes = unique(
        vcat(
            [size(m[1]) for m in values(testmoments)],
            [size(m[1]) for m in values(trialmoments)],
        ),
    )
    @test length(sizes) == 1
end

end # module TestPetrovRectangularMoments
