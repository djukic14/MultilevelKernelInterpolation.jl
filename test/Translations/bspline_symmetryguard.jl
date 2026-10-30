module TestBsplineSymmetryGuard

using Test
using StaticArrays
using OhMyThreads: SerialScheduler
using H2Trees
using H2Factory
using MultilevelKernelInterpolation

# `BsplineI2OTranslationCollection` FORWARDS THE SYMMETRY GUARD.
#
# `H2Factory.assertnosymmetry` refuses a collection whose symmetry tags have not been consumed:
# applying one uses each ORBIT's translation in place of the interaction's own, which matches in
# shape and produces a plausible field, so nothing downstream would notice. H2Factory deliberately
# has no permissive fallback for it, so a wrapper either declares or fails loudly.
#
# This type is a wrapper and had neither. It was safe only by accident of its call sites: every MultilevelKernelInterpolation
# constructor passes `DirectionInvariance()`, and `symmetryreduced=true` routes to collections whose
# tags are already consumed. That is a property of today's constructors, not of this type, and a
# later one gaining a trait argument would have opened the hole silently.
#
# TWO SEPARATE THINGS HAVE TO HOLD, which is why this file exists rather than one assertion:
#
#   1. the forward itself, tested below on both a tagged and an untagged inner collection;
#   2. that the guard is REACHED, which the forward alone does not give.
#
# HOW (2) IS SATISFIED CHANGED, and the change is the point of #180. This type used to define
# `i2otranslation!`/`o2itranslation!`/`adjointo2itranslation!` at the SAME ARITY as H2Factory's
# guarded entries, so its methods were strictly more specific and H2Factory's guard never ran for it
# -- which is why each of those six methods had to call `assertnosymmetry` by hand. It now supplies
# `applycouplingkernel!` methods instead, and a kernel cannot displace the entry, so the guard is
# reached structurally. The six calls below still run, because what must hold is that every direction
# and accumulation mode refuses; only the reason it holds has moved from convention to dispatch.

# Whether a method's defining module is MultilevelKernelInterpolation or anything nested inside it. Walked to the root rather
# than compared against `MultilevelKernelInterpolation` directly: a method defined in a SUBMODULE reports that submodule, so an
# identity test would read a bypass added there as "no MultilevelKernelInterpolation methods at all" and pass. Same predicate as
# `test/Package/h2factoryinterface.jl`, restated here because that one lives in its own test module.
function belongstopackage(mod::Module)
    mod === MultilevelKernelInterpolation && return true
    parent = parentmodule(mod)
    return parent !== mod && belongstopackage(parent)
end

struct DirectionRecorder end
Base.eltype(::DirectionRecorder) = SVector{3,Float64}
(::DirectionRecorder)(receiving, translating, direction) = direction

function collections()
    points = [
        SVector(Float64(i), Float64(j), Float64(k)) for i in 0:5 for j in 0:5 for k in 0:5
    ]
    tree = H2Trees.TwoNTree(
        points; builder=H2Trees.TwoNTreeBuilder(; minhalfsize=0.6, minvalues=0)
    )
    plans = H2Trees.buildplans(tree)
    plan = H2Trees.translatingplan(plans.trialaggregationplan, plans.testdisaggregationplan)

    plain, _ = H2Factory.I2OTranslationCollection(
        tree, plan, H2Trees.DirectionInvariancePerLevel(), DirectionRecorder()
    )
    tagged, _ = H2Factory.I2OTranslationCollection(
        tree,
        plan,
        H2Trees.SymmetryDirectionInvariancePerLevel(H2Trees.FullLatticeSymmetry()),
        DirectionRecorder(),
    )
    return plain, tagged
end

wrap(inner) = MultilevelKernelInterpolation.BsplineI2OTranslationCollection(
    inner,
    MultilevelKernelInterpolation.BsplineDenseModulation(
        nothing, nothing, nothing, nothing
    ),
)

@testset "BsplineI2OTranslationCollection forwards the symmetry guard" begin
    plain, tagged = collections()

    # Non-vacuity: the two inner collections really do differ in the property under test, so the
    # pair of assertions below is a comparison rather than two statements about one thing.
    @test H2Factory.translationsymmetrygroup(plain) === nothing
    @test H2Factory.translationsymmetrygroup(tagged) !== nothing

    # Forwarded, in both directions. An untagged inner has nothing to consume and passes; a tagged
    # one is refused with H2Factory's own message.
    @test H2Factory.assertnosymmetry(wrap(plain)) === nothing
    @test_throws "symmetry-reduced" H2Factory.assertnosymmetry(wrap(tagged))

    # DECLARED, NOT INHERITED. Asking `which` is what tells those apart: were H2Factory to regain a
    # permissive fallback on `AbstractI2OTranslationCollection`, calling it would still return
    # `nothing` and the assertion above would still pass while meaning nothing at all.
    resolved = which(H2Factory.assertnosymmetry, Tuple{typeof(wrap(plain))})
    @test resolved.module === MultilevelKernelInterpolation

    # And the guard is REACHED, in every direction and accumulation mode. Argument shapes are
    # irrelevant here because the refusal fires before anything is indexed.
    key = first(keys(tagged))
    context = H2Factory.TranslationContext(SerialScheduler(), nothing)
    for direction in (
        H2Factory.ForwardTranslation(),
        H2Factory.TransposeTranslation(),
        H2Factory.AdjointTranslation(),
    )
        for accumulation in (H2Factory.Overwrite(), H2Factory.Accumulate())
            @test_throws "symmetry-reduced" H2Factory.applycoupling!(
                nothing,
                nothing,
                wrap(tagged),
                key[1],
                key[2],
                direction,
                accumulation,
                context,
            )
        end
    end

    # MultilevelKernelInterpolation ADDS NO `applycoupling!` METHOD, which is the structural half of the claim above. Dispatch
    # cannot forbid one -- a method at a concrete collection type would shadow H2Factory's guarded
    # entry exactly as the old direction-named primitives did -- so what is enforceable is that none
    # exists, and that is pinned here rather than left to inspection. An edit that moved these
    # implementations back onto the entry would reopen exactly the hole #172 recorded.
    #
    # `belongstopackage` walks up to the root module rather than comparing against `MultilevelKernelInterpolation` directly: a
    # method defined inside a SUBMODULE has `m.module === MultilevelKernelInterpolation.SomeSubmodule`, so an identity test
    # would read a real bypass added there as "no MultilevelKernelInterpolation methods" and pass.
    @test !any(m -> belongstopackage(m.module), methods(H2Factory.applycoupling!))

    # Non-vacuity: MultilevelKernelInterpolation really does participate in this protocol, so the assertion above is a
    # statement about WHERE its methods are rather than about it having none.
    @test count(m -> belongstopackage(m.module), methods(H2Factory.applycouplingkernel!)) ==
        2
end

end # module TestBsplineSymmetryGuard
