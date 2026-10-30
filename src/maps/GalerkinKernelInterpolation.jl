"""
    GalerkinKernelInterpolationCore(operator, space; kwargs...)
    GalerkinKernelInterpolationCore(operator, space, tree; kwargs...)

Build the read-only data for a Galerkin map: tree, plans, near interactions,
moment collections, and translations. The core also stores metadata for workspace allocation.

Use `H2Workspace(core, scheduling)` to get an executable `LinearMap`:

```julia
core = GalerkinKernelInterpolationCore(operator, space)
ws = H2Workspace(core, scheduling)
mul!(y, ws, x)
```

For concurrent multiplications, give each task its own workspace over the same core.
[`GalerkinKernelInterpolation`](@ref) builds a core and one workspace in a single call.
"""
struct GalerkinKernelInterpolationCore{
    T,
    TreeType,
    NearInteractionsType,
    MomentCollectionDictType,
    LevelTranslatorsType,
    I2OTranslatorType,
    PlansType,
    TreePolynomialsType,
    I2OPolynomialsType,
    ISNEAR,
    αType,
    αTransposeType,
    FarMulType,
    γType,
    TestMomentOpType,
    TrialMomentOpType,
    MomentTypeType,
    SchedulerType,
} <: GalerkinH2Core{T}
    tree::TreeType
    nearinteractions::NearInteractionsType
    momentcollections::MomentCollectionDictType
    leveltranslators::LevelTranslatorsType
    i2otranslator::I2OTranslatorType
    plans::PlansType
    treepolynomials::TreePolynomialsType
    # Preserve the I2O builder's polynomial layout for workspace scratch allocation.
    i2opolynomials::I2OPolynomialsType
    verbose::Bool
    isnear::ISNEAR
    α::αType
    # Resolve the transpose coefficient once, without retaining the operator.
    αtranspose::αTransposeType
    # Precompute conj.(αtranspose), including vector coefficients, to avoid matvec allocations.
    αadjoint::αTransposeType
    farmulmode::FarMulType
    # This field lets the constructor infer T. Keep gamma too: moments use complex(gamma),
    # while modulation scratch uses eltype(gamma), which can differ for real gamma.
    scalartype::Type{T}
    γ::γType
    testmomentoperator::TestMomentOpType
    trialmomentoperator::TrialMomentOpType
    momenttype::MomentTypeType
    # Default scheduler for new workspaces; mutable scheduling state stays in the workspace.
    defaultscheduler::SchedulerType
end

"""
    GalerkinKernelInterpolation(operator, space; kwargs...)
    GalerkinKernelInterpolation(operator, space, tree; kwargs...)

Build a ready-to-use `LinearMap` with a [`GalerkinKernelInterpolationCore`](@ref) and one workspace.

Use each workspace for only one multiplication at a time. For concurrent calls, share the
core but allocate a workspace per task:

```julia
core = GalerkinKernelInterpolationCore(operator, space; strategy)
workspaces = [H2Workspace(core, scheduling) for _ in 1:4]
```
"""
function GalerkinKernelInterpolation(operator, space; kwargs...)
    return _coreandworkspace(GalerkinKernelInterpolationCore, (operator, space); kwargs...)
end

function GalerkinKernelInterpolation(operator, space, tree; kwargs...)
    return _coreandworkspace(
        GalerkinKernelInterpolationCore, (operator, space, tree); kwargs...
    )
end

function GalerkinKernelInterpolationCore(
    operator,
    space;
    tree=buildtree(space; builder=TwoNTreeBuilder(; minvalues=120)),
    kwargs...,
)
    return GalerkinKernelInterpolationCore(operator, space, tree; kwargs...)
end

function GalerkinKernelInterpolationCore(
    operator, space, tree; strategy=InterpolationStrategy(operator, space), verbose=false
)
    # Qualify the accessor because the local variable has the same name.
    polynomial = MultilevelKernelInterpolation.polynomial(polynomialstrategy(strategy))
    # The moment layout must match both the polynomial and the I2O strategy.
    Moment = defaultmomenttype(i2ostrategy(translatorstrategy(strategy)), polynomial)

    @assert numberoflevels(tree) > 2
    assertpolynomialdimension(polynomial, tree)
    @assert trialmoment(operator, polynomial) == testmoment(operator, polynomial)

    γ = operatorgamma(operator)

    verbose && @info "Assembling moments"
    # `assembleplans` returns an `H2Trees.PlanSet`; H2Factory consumes it directly (no re-wrap).
    plans = assembleplans(tree, nearfieldstrategy(strategy))

    # I2O pairs belong to the upward or downward plan, depending on aggregation mode.
    translating_plan = translatingplan(
        plans.trialaggregationplan, plans.testdisaggregationplan
    )

    levelpolynomials, leafpolynomials = assembletreepolynomials(
        tree, space, polynomialstrategy(strategy)
    )
    checkprotrusionsafety(tree, levelpolynomials, translating_plan)

    momentcollections, _ = assemblemomentcollections(
        operator,
        tree,
        leafpolynomials,
        space,
        tree,
        leafpolynomials,
        space,
        polynomialstrategy(strategy),
        momentassemblystrategy(strategy);
        verbose=verbose,
    )

    # This chunk scope holds temporary construction scratch only.
    # Persistent moment pools and channels are allocated by each workspace.
    leveltranslators, i2otranslator = withnchunks(
        _constructionchunks(1:numberoflevels(tree))
    ) do
        leveltranslators = assembleleveltranslators(
            tree, levelpolynomials, plans, translatorstrategy(strategy)
        )
        verbose && @info "Assembling translations"
        i2otranslator = assemblei2otranslator(
            operator,
            tree,
            levelpolynomials,
            translating_plan,
            space,
            space,
            translatorstrategy(strategy);
            verbose=verbose,
        )
        return leveltranslators, i2otranslator
    end

    verbose && @info "Assembling near interactions"
    nearinteractions = assemblenearinteractions(
        operator,
        space,
        space,
        tree,
        γ,
        nearinteractionsstrategy(strategy),
        nearfieldstrategy(strategy);
        verbose=verbose,
    )

    return GalerkinKernelInterpolationCore(
        tree,
        nearinteractions,
        momentcollections,
        leveltranslators,
        i2otranslator,
        plans,
        TreePolynomials(levelpolynomials, plans.relevantlevels[begin]),
        levelpolynomials,
        verbose,
        nearcriterion(nearfieldstrategy(strategy))(tree),
        α(operator, polynomial),
        αtranspose(operator, polynomial),
        conj.(αtranspose(operator, polynomial)),
        aggregationmode(nearfieldstrategy(strategy)),
        complex(typeof(γ)),
        γ,
        testmoment(operator, polynomial),
        trialmoment(operator, polynomial),
        Moment,
        scheduler(strategy),
    )
end

# --- Read-only accessors -----------------------------------------------------------------------
#
function momentcollection(A::GalerkinKernelInterpolationCore, node::Int)
    return A.momentcollections[node]
end

# Both sides share one transfer family. Leave orientation selection to H2Factory's accessors.
H2Factory.triali2itranslator(A::GalerkinKernelInterpolationCore) =
    A.leveltranslators.i2itranslator
H2Factory.testi2itranslator(A::GalerkinKernelInterpolationCore) =
    A.leveltranslators.i2itranslator
H2Factory.trialo2otranslator(A::GalerkinKernelInterpolationCore) =
    A.leveltranslators.o2otranslator
H2Factory.testo2otranslator(A::GalerkinKernelInterpolationCore) =
    A.leveltranslators.o2otranslator

# Reuse one I2O collection for all orientations. Concrete traits avoid accessor ambiguities.
H2Factory.i2otranslator(A::GalerkinKernelInterpolationCore, ::isForward) = A.i2otranslator
H2Factory.i2otranslator(A::GalerkinKernelInterpolationCore, ::isTranspose) = A.i2otranslator
H2Factory.i2otranslator(A::GalerkinKernelInterpolationCore, ::isAdjoint) = A.i2otranslator

# A plain adjoint leaf multiply is valid only when both collections are real.
# Check the stored scalar types: a user-supplied MomentAssembler may return complex data.
H2Factory.leafmul(A::GalerkinKernelInterpolationCore, ::isAdjoint) = _adjointleafmul(A)

# Read the container's value type, then the collection's array type and its scalar type.
# Base.Dict yields pairs from eltype; Dictionaries.Dictionary yields values.
_leafcollectiontype(::Type{C}) where {C} = eltype(C)
_leafcollectiontype(::Type{C}) where {C<:AbstractDict} = valtype(C)

function _leafcollectionscalar(collections)
    arraytype = eltype(_leafcollectiontype(typeof(collections)))
    return eltype(arraytype)
end

# Promote both sides so either complex collection requires a conjugating multiply.
function _adjointleafmul(A)
    scalar = promote_type(
        _leafcollectionscalar(H2Factory.trialmomentcollections(A, H2Factory.isForward())),
        _leafcollectionscalar(H2Factory.testmomentcollections(A, H2Factory.isForward())),
    )
    scalar <: Real && return H2Factory.LinAlgebraMulFunctor()
    return error(
        "MultilevelKernelInterpolation's leaf moment collections came out with scalar type $(scalar), which is not real, " *
        "so the plain adjoint leaf multiply would be wrong: the adjoint conjugates both leaf " *
        "collections and that is a no-op only for real ones. This normally means the " *
        "`MomentAssembler` in `MultilevelKernelInterpolation.MomentAssemblyStrategy` (passed as `InterpolationStrategy`'s " *
        "`momentassemblystrategy`) returned complex moment matrices. Supply a conjugating " *
        "`H2Factory.leafmul(::typeof(core), ::H2Factory.isAdjoint)` for that configuration, the " *
        "way MLFMA does for its complex leaf storage.",
    )
end

# Without this method, H2Trees' generic fallback would treat the core as a tree.
H2Trees.isnear(A::GalerkinKernelInterpolationCore, ::isForward) = A.isnear

# Galerkin uses the same leaf collections on both sides.
H2Factory.testmomentcollections(A::GalerkinKernelInterpolationCore, ::isForward) =
    A.momentcollections
H2Factory.trialmomentcollections(A::GalerkinKernelInterpolationCore, ::isForward) =
    A.momentcollections

H2Factory.α(A::GalerkinKernelInterpolationCore, ::isForward) = A.α
H2Factory.α(A::GalerkinKernelInterpolationCore, ::isTranspose) = A.αtranspose
H2Factory.α(A::GalerkinKernelInterpolationCore, ::isAdjoint) = A.αadjoint

H2Factory.verbose(A::GalerkinKernelInterpolationCore, ::H2Factory.AbstractMapTrait) =
    A.verbose

H2Factory.minrelevantlevel(A::GalerkinKernelInterpolationCore, ::isForward) =
    A.treepolynomials.minrelevantlevel

H2Factory.testaggregationplan(A::GalerkinKernelInterpolationCore, ::isForward) =
    A.plans.testaggregationplan
H2Factory.trialaggregationplan(A::GalerkinKernelInterpolationCore, ::isForward) =
    A.plans.trialaggregationplan
H2Factory.testdisaggregationplan(A::GalerkinKernelInterpolationCore, ::isForward) =
    A.plans.testdisaggregationplan
H2Factory.trialdisaggregationplan(A::GalerkinKernelInterpolationCore, ::isForward) =
    A.plans.trialdisaggregationplan

# --- Workspace construction hooks ----------------------------------------------------------------

# A Galerkin map's scheduling containers span its one tree's levels.
H2Factory.schedulinglevels(A::GalerkinKernelInterpolationCore) =
    1:H2Trees.numberoflevels(A.tree)

function H2Factory.buildmoments(A::GalerkinKernelInterpolationCore)
    momentbuffers, testmoments, trialmoments = MultilevelKernelInterpolation.moments(
        complex(A.γ),
        A.testmomentoperator,
        A.trialmomentoperator,
        A.tree,
        A.treepolynomials.polynomials,
        A.plans.testaggregationplan,
        A.plans.trialaggregationplan,
        A.momenttype,
    )
    return momentbuffers, Moments(testmoments, trialmoments)
end

# Both sides use the same per-node vector buffers.
H2Factory.buildvectorbuffers(A::GalerkinKernelInterpolationCore) =
    computevectorbuffers(A.tree, complex(typeof(A.γ)))

H2Factory.buildi2ichannels(A::GalerkinKernelInterpolationCore) =
    H2Factory.i2ichannels(A.leveltranslators.i2itranslator)
H2Factory.buildo2ochannels(A::GalerkinKernelInterpolationCore) =
    H2Factory.o2ochannels(A.leveltranslators.i2itranslator)

H2Factory.buildi2ochannel(A::GalerkinKernelInterpolationCore) = _buildi2ochannel(A)
