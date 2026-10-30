module TestAdjointLeafMul

using Test
using LinearAlgebra
using StaticArrays
using CompScienceMeshes, BEAST
using H2Trees, H2Factory
using MultilevelKernelInterpolation

# THE REALNESS INVARIANT, CHECKED RATHER THAN ASSERTED.
#
# H2Factory has no shared `leafmul` default for the ADJOINT orientation: the adjoint conjugates BOTH
# leaf collections on top of the coupling's transpose, which is a no-op only when those collections
# are REAL. MultilevelKernelInterpolation selects the plain matvec, and the reason it may is a property of the COLLECTIONS,
# not of the core type.
#
# That distinction is the whole point of this file. MultilevelKernelInterpolation's collection element type is inferred from
# whatever matrices the moment assembler returns (`moments/momentcollection.jl`), and the assembler is
# user-supplied as `MomentAssembler` in `MomentAssemblyStrategy`. A declaration written for the core
# TYPE would assert realness for configurations nobody has measured, reintroducing one layer down
# exactly the defect H2Factory's refusal removed. So `leafmul` reads the scalar off the collections
# the core is actually holding, and these tests pin both halves: that real configurations select the
# plain matvec, and that a complex one is refused.

@testset verbose = true "adjoint leaf multiply follows the collections, not the core type" begin
    m = readmesh(
        joinpath(pkgdir(MultilevelKernelInterpolation), "test", "assets", "in", "sphere.in")
    )

    @testset "Galerkin core: real collections select the plain matvec" begin
        op = Helmholtz3D.singlelayer(; wavenumber=1.0)
        X = lagrangecxd0(m)
        A = MultilevelKernelInterpolation.GalerkinMap(
            op,
            X,
            H2Trees.buildtree(X; builder=H2Trees.TwoNTreeBuilder(; minvalues=200));
            strategy=MultilevelKernelInterpolation.InterpolationStrategy(
                op,
                X;
                nearinteractionsstrategy=H2Factory.DummyNearInteractionsStrategy(
                    BEAST.defaultquadstrat(op, X, X)
                ),
            ),
        )
        core = H2Factory.core(A)

        # The premise, stated as an assertion rather than left implicit: the map is COMPLEX while its
        # leaf blocks are REAL. If this ever flips, the selection below must flip with it, and the
        # map's own element type is exactly what fails to say so.
        @test eltype(A) <: Complex
        collections = H2Factory.trialmomentcollections(core, H2Factory.isForward())
        @test MultilevelKernelInterpolation._leafcollectionscalar(collections) <: Real

        @test H2Factory.leafmul(core, H2Factory.isAdjoint()) isa
            H2Factory.LinAlgebraMulFunctor
    end

    @testset "Petrov core: both sides are read, and both are real here" begin
        op = Helmholtz3D.doublelayer(; wavenumber=1.0)
        testspace = lagrangec0d1(m)
        trialspace = lagrangecxd0(m)
        A = MultilevelKernelInterpolation.PetrovGalerkinMap(
            op,
            testspace,
            trialspace;
            strategy=MultilevelKernelInterpolation.InterpolationStrategy(
                op,
                testspace,
                trialspace;
                nearinteractionsstrategy=H2Factory.DummyNearInteractionsStrategy(
                    BEAST.defaultquadstrat(op, testspace, trialspace)
                ),
            ),
        )
        core = H2Factory.core(A)

        trial = H2Factory.trialmomentcollections(core, H2Factory.isForward())
        test = H2Factory.testmomentcollections(core, H2Factory.isForward())
        @test MultilevelKernelInterpolation._leafcollectionscalar(trial) <: Real
        @test MultilevelKernelInterpolation._leafcollectionscalar(test) <: Real

        @test H2Factory.leafmul(core, H2Factory.isAdjoint()) isa
            H2Factory.LinAlgebraMulFunctor
    end
end

# A core whose collections are COMPLEX, which is what a custom `MomentAssembler` returning complex
# moment matrices would produce. Nothing in MultilevelKernelInterpolation forbids that today.
struct ComplexCollectionsCore end

const COMPLEXCOLLECTIONS = [
    MultilevelKernelInterpolation.TrialMomentCollection{1,ComplexF64}((3, 3))
]
const REALCOLLECTIONS = [
    MultilevelKernelInterpolation.TrialMomentCollection{1,Float64}((3, 3))
]

H2Factory.trialmomentcollections(::ComplexCollectionsCore, ::H2Factory.isForward) =
    COMPLEXCOLLECTIONS
H2Factory.testmomentcollections(::ComplexCollectionsCore, ::H2Factory.isForward) =
    COMPLEXCOLLECTIONS

# One real side and one complex side, to pin that the two are PROMOTED rather than the first one
# winning. A Petrov map can genuinely differ between sides.
struct MixedCollectionsCore end

H2Factory.trialmomentcollections(::MixedCollectionsCore, ::H2Factory.isForward) =
    REALCOLLECTIONS
H2Factory.testmomentcollections(::MixedCollectionsCore, ::H2Factory.isForward) =
    COMPLEXCOLLECTIONS

@testset "a complex leaf collection is refused, not silently plain-multiplied" begin
    # NEGATIVE CONTROL. Without it the two testsets above are consistent with a helper that returns
    # `Real` unconditionally, which would pass while checking nothing.
    @test MultilevelKernelInterpolation._leafcollectionscalar(COMPLEXCOLLECTIONS) <: Complex
    @test MultilevelKernelInterpolation._leafcollectionscalar(REALCOLLECTIONS) <: Real

    @test_throws ErrorException MultilevelKernelInterpolation._adjointleafmul(
        ComplexCollectionsCore()
    )

    message = try
        MultilevelKernelInterpolation._adjointleafmul(ComplexCollectionsCore())
        ""
    catch e
        sprint(showerror, e)
    end
    @test occursin("ComplexF64", message)
    @test occursin("MomentAssembler", message)

    # PROMOTION, not first-wins: one complex side is enough.
    @test_throws ErrorException MultilevelKernelInterpolation._adjointleafmul(
        MixedCollectionsCore()
    )

    # And the container shape must not decide the answer. Production stores a
    # `Dictionaries.Dictionary`, whose `eltype` is already the value; a `Base.Dict`'s is a `Pair`,
    # which would walk into the key type and answer `Int`.
    @test MultilevelKernelInterpolation._leafcollectionscalar(
        Dict(1 => first(COMPLEXCOLLECTIONS))
    ) <: Complex
    @test MultilevelKernelInterpolation._leafcollectionscalar(
        Dict(1 => first(REALCOLLECTIONS))
    ) <: Real
end

# THE SAME REFUSAL, REACHED THE WAY A USER WOULD REACH IT.
#
# The controls above call `_adjointleafmul` directly, which tests the helper but NOT the public
# method: reverting `leafmul(::GalerkinCore, ::isAdjoint)` to an unconditional
# `LinAlgebraMulFunctor()` would leave every one of them green, because the real cores are real and
# the complex ones never go through `leafmul`. So this builds a REAL core whose collections are
# genuinely complex, and asks the public entry point.
#
# It covers three things the synthetic controls cannot: that `H2Factory.leafmul` dispatches to the
# computed selection at all, that the production `Dictionaries.Dictionary` container is walked
# correctly, and that the documented extension path -- a custom `MomentAssembler` -- is what produces
# the situation.

# Wraps the stock assembler and widens its moments to complex. Deliberately the smallest thing that
# makes the collections complex without changing anything else about the map.
struct ComplexifyingMomentAssembler{A}
    inner::A
end

function ComplexifyingMomentAssembler(operator, space, quadstrategy, dimensionality::Int)
    return ComplexifyingMomentAssembler(
        MultilevelKernelInterpolation.BEASTMomentAssembler(
            operator, space, quadstrategy, dimensionality
        ),
    )
end

function (assembler::ComplexifyingMomentAssembler)(
    tree, leafpolynomials; verbose::Bool=false
)
    Fs = assembler.inner(tree, leafpolynomials; verbose=verbose)
    return [map(F -> ComplexF64.(F), leaf) for leaf in Fs]
end

# Same storage contract as the assembler it wraps; without this the moments have no store rule.
H2Factory.storetrait(::Type{<:ComplexifyingMomentAssembler}) = H2Factory.StoreAsIs

@testset "a live core with complex collections is refused by the PUBLIC leafmul" begin
    m = readmesh(
        joinpath(pkgdir(MultilevelKernelInterpolation), "test", "assets", "in", "sphere.in")
    )
    op = Helmholtz3D.singlelayer(; wavenumber=1.0)
    X = lagrangecxd0(m)

    A = MultilevelKernelInterpolation.GalerkinMap(
        op,
        X,
        H2Trees.buildtree(X; builder=H2Trees.TwoNTreeBuilder(; minvalues=200));
        strategy=MultilevelKernelInterpolation.InterpolationStrategy(
            op,
            X;
            momentassemblystrategy=MultilevelKernelInterpolation.MomentAssemblyStrategy(;
                MomentAssembler=ComplexifyingMomentAssembler
            ),
            nearinteractionsstrategy=H2Factory.DummyNearInteractionsStrategy(
                BEAST.defaultquadstrat(op, X, X)
            ),
        ),
    )
    core = H2Factory.core(A)

    # The premise: this really is the production container, holding really complex collections.
    collections = H2Factory.trialmomentcollections(core, H2Factory.isForward())
    @test MultilevelKernelInterpolation._leafcollectionscalar(collections) <: Complex

    # THE ASSERTION THAT CANNOT BE SATISFIED BY AN UNCONDITIONAL DEFAULT.
    @test_throws ErrorException H2Factory.leafmul(core, H2Factory.isAdjoint())

    message = try
        H2Factory.leafmul(core, H2Factory.isAdjoint())
        ""
    catch e
        sprint(showerror, e)
    end
    @test occursin("ComplexF64", message)
    @test occursin("MomentAssemblyStrategy", message)

    # Forward and transpose are unaffected: the plain matvec is correct there for any collection, so
    # a complex map must still get one rather than being refused wholesale.
    @test H2Factory.leafmul(core, H2Factory.isForward()) isa H2Factory.LinAlgebraMulFunctor
    @test H2Factory.leafmul(core, H2Factory.isTranspose()) isa
        H2Factory.LinAlgebraMulFunctor
end

end # module TestAdjointLeafMul
