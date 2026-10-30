"""
    PetrovGalerkinKernelInterpolationCore(operator, testspace, trialspace; kwargs...)
    PetrovGalerkinKernelInterpolationCore(operator, testspace, trialspace, tree; kwargs...)

Build the read-only data for a Petrov-Galerkin map.
Each side has its own tree, polynomial family, and level transfers.

Use `H2Workspace(core, scheduling)` to get an executable map. Separate workspaces can
share one core for concurrent multiplications; see [`GalerkinKernelInterpolationCore`](@ref).
"""
struct PetrovGalerkinKernelInterpolationCore{
    T,
    TreeType,
    NearInteractionsType,
    MomentCollectionsType,
    LevelTranslatorsType,
    I2OTranslatorType,
    PlansType,
    TestPolynomialType,
    TrialPolynomialType,
    MomentPolynomialsType,
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
} <: PetrovGalerkinH2Core{T}
    tree::TreeType
    nearinteractions::NearInteractionsType
    momentcollections::MomentCollectionsType
    leveltranslators::LevelTranslatorsType
    i2otranslator::I2OTranslatorType
    plans::PlansType
    minrelevantlevel::Int
    testpolynomials::TestPolynomialType
    trialpolynomials::TrialPolynomialType
    # Keep per-side dictionaries for moment sizing and the I2O layout for scratch builders.
    momentpolynomials::MomentPolynomialsType
    i2opolynomials::I2OPolynomialsType
    verbose::Bool
    isnear::ISNEAR
    α::αType
    # Resolve the transpose coefficient once at construction.
    αtranspose::αTransposeType
    # Adjoint conjugates the transpose coefficient.
    αadjoint::αTransposeType
    farmulmode::FarMulType
    # Let the default constructor infer the moment/vector scalar type T.
    scalartype::Type{T}
    γ::γType
    testmomentoperator::TestMomentOpType
    trialmomentoperator::TrialMomentOpType
    momenttype::MomentTypeType
    defaultscheduler::SchedulerType
end

"""
    PetrovGalerkinKernelInterpolation(operator, testspace, trialspace; kwargs...)
    PetrovGalerkinKernelInterpolation(operator, testspace, trialspace, tree; kwargs...)

Build a ready-to-use `LinearMap` with a [`PetrovGalerkinKernelInterpolationCore`](@ref) and one workspace.
Use each workspace for only one multiplication at a time. Concurrent calls need separate
workspaces over the shared core; see [`GalerkinKernelInterpolation`](@ref).
"""
function PetrovGalerkinKernelInterpolation(operator, testspace, trialspace; kwargs...)
    return _coreandworkspace(
        PetrovGalerkinKernelInterpolationCore, (operator, testspace, trialspace); kwargs...
    )
end

function PetrovGalerkinKernelInterpolation(operator, testspace, trialspace, tree; kwargs...)
    return _coreandworkspace(
        PetrovGalerkinKernelInterpolationCore,
        (operator, testspace, trialspace, tree);
        kwargs...,
    )
end

# The shared assemble entry point in MultilevelKernelInterpolation.jl selects Galerkin or Petrov by space identity.

function PetrovGalerkinKernelInterpolationCore(
    operator,
    testspace,
    trialspace;
    tree=buildtree(
        testspace,
        trialspace;
        builder=BlockTreeBuilder(;
            test=TwoNTreeBuilder(; minvalues=120), trial=TwoNTreeBuilder(; minvalues=120)
        ),
    ),
    kwargs...,
)
    return PetrovGalerkinKernelInterpolationCore(
        operator, testspace, trialspace, tree; kwargs...
    )
end

function PetrovGalerkinKernelInterpolationCore(
    operator,
    testspace,
    trialspace,
    tree;
    # Default: equidistant Lagrange with Toeplitz I2O. Dense I2O allows independent grids.
    strategy=InterpolationStrategy(operator, testspace, trialspace),
    verbose=false,
)
    # Qualify the accessor because the local variable has the same name.
    polynomial = MultilevelKernelInterpolation.polynomial(polynomialstrategy(strategy))
    # Keep the moment layout consistent with the translator and polynomial.
    Moment = defaultmomenttype(i2ostrategy(translatorstrategy(strategy)), polynomial)

    # The two trees are independent, so validate both dimensions.
    assertpolynomialdimension(polynomial, testtree(tree))
    assertpolynomialdimension(polynomial, trialtree(tree))

    γ = operatorgamma(operator)

    verbose && @info "Assembling near interactions"
    nearinteractions = assemblenearinteractions(
        operator,
        testspace,
        trialspace,
        tree,
        γ,
        nearinteractionsstrategy(strategy),
        nearfieldstrategy(strategy);
        verbose=verbose,
    )

    verbose && @info "Assembling moments"
    thetesttree = testtree(tree)
    thetrialtree = trialtree(tree)

    # `assembleplans` returns an `H2Trees.PlanSet`; H2Factory consumes it directly (no re-wrap).
    plans = assembleplans(tree, nearfieldstrategy(strategy))

    # Select the plan carrying I2O pairs for this aggregation mode.
    translating_plan = translatingplan(
        plans.trialaggregationplan, plans.testdisaggregationplan
    )

    minrelevantlevel = plans.relevantlevels[begin]

    # Reconcile protrusion requirements before building either hierarchy. Patching grids
    # afterward could break the parent-child nesting needed by I2I/O2O.
    testprotrusions = assemblelevelprotrusions(
        thetesttree, testspace, polynomialstrategy(strategy)
    )
    trialprotrusions = assemblelevelprotrusions(
        thetrialtree, trialspace, polynomialstrategy(strategy)
    )

    # Shared protrusion requirements need not produce identical grids: non-shared levels
    # and refinement history also affect each hierarchy. Validate the result below.
    sharedgrids = _sharedi2ogrids(i2ostrategy(translatorstrategy(strategy)))
    if sharedgrids
        for level in intersect(levels(thetesttree), levels(thetrialtree))
            testlid = H2Trees.leveltolevelid(thetesttree, level)
            triallid = H2Trees.leveltolevelid(thetrialtree, level)
            p = max(testprotrusions[testlid], trialprotrusions[triallid])
            testprotrusions[testlid] = p
            trialprotrusions[triallid] = p
        end
    end

    testpolynomials, testleafpolynomials = assembletreepolynomials(
        thetesttree, testspace, polynomialstrategy(strategy); protrusion=testprotrusions
    )
    trialpolynomials, trialleafpolynomials = assembletreepolynomials(
        thetrialtree, trialspace, polynomialstrategy(strategy); protrusion=trialprotrusions
    )

    # Reject incompatible grids and direct the caller to dense I2O; do not patch them here.
    if sharedgrids
        for level in intersect(levels(thetesttree), levels(thetrialtree))
            testlid = H2Trees.leveltolevelid(thetesttree, level)
            triallid = H2Trees.leveltolevelid(thetrialtree, level)
            toeplitzcompatible(testpolynomials[testlid], trialpolynomials[triallid]) ||
                error(
                    "Toeplitz shared-grid reconciliation failed at level $level: reconciling the " *
                    "per-level protrusion requirement (max of the two sides) did not yield " *
                    "Toeplitz-compatible polynomials (test grid " *
                    "$(size(LinearIndices(testpolynomials[testlid]))), trial grid " *
                    "$(size(LinearIndices(trialpolynomials[triallid])))). Use the dense I2O strategy " *
                    "(`TranslatorStrategy(; i2ostrategy=DenseI2OStrategy())`) for this configuration.",
                )
        end
    end

    checkprotrusionsafety(
        thetesttree, testpolynomials, thetrialtree, trialpolynomials, translating_plan
    )

    testmomentcollections, trialmomentcollections = assemblemomentcollections(
        operator,
        thetesttree,
        testleafpolynomials,
        testspace,
        thetrialtree,
        trialleafpolynomials,
        trialspace,
        polynomialstrategy(strategy),
        momentassemblystrategy(strategy);
        verbose=verbose,
    )

    # Index each side by its own raw levels. Include every tree level: translating plans
    # can reference levels outside that side's aggregation/disaggregation plans.
    _sidedict(tree, treepolynomials) = Dict(
        level => treepolynomials[H2Trees.leveltolevelid(tree, level)] for
        level in levels(tree)
    )
    testpolynomialdict = _sidedict(thetesttree, testpolynomials)
    trialpolynomialdict = _sidedict(thetrialtree, trialpolynomials)
    sidedpolynomialdicts = SidedPolynomialDicts(testpolynomialdict, trialpolynomialdict)

    # Each side needs its own transfers, ordered leaf-to-root by its aggregation plan.
    triali2ilevels = reverse(levels(plans.trialaggregationplan))
    triali2ipolynomials = [trialpolynomialdict[level] for level in triali2ilevels]
    testi2ilevels = reverse(levels(plans.testaggregationplan))
    testi2ipolynomials = [testpolynomialdict[level] for level in testi2ilevels]

    # Allocate only temporary construction scratch here; workspace pools are built later.
    leveltranslators, i2otranslator = withnchunks(
        _constructionchunks(1:max(levels(thetesttree)[end], levels(thetrialtree)[end]))
    ) do
        verbose && @info "Assembling translations"
        # Run the same transfer builder once per side.
        leveltranslators = SidedLevelTranslators(
            assembleleveltranslators(
                testi2ipolynomials, testi2ilevels, translatorstrategy(strategy)
            ),
            assembleleveltranslators(
                triali2ipolynomials, triali2ilevels, translatorstrategy(strategy)
            ),
        )
        i2otranslator = assemblei2otranslator(
            operator,
            tree,
            sidedpolynomialdicts,
            translating_plan,
            testspace,
            trialspace,
            translatorstrategy(strategy);
            verbose=verbose,
        )
        return leveltranslators, i2otranslator
    end

    return PetrovGalerkinKernelInterpolationCore(
        tree,
        nearinteractions,
        MomentCollections(testmomentcollections, trialmomentcollections),
        leveltranslators,
        i2otranslator,
        plans,
        minrelevantlevel,
        testpolynomials,
        trialpolynomials,
        TestTrialPolynomials(testpolynomialdict, trialpolynomialdict),
        sidedpolynomialdicts,
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
# Expose each side's transfers separately. H2Factory selects the side and orientation;
# overriding the resolved one-argument accessors would bypass that selection.
H2Factory.triali2itranslator(A::PetrovGalerkinKernelInterpolationCore) =
    A.leveltranslators.trial.i2itranslator
H2Factory.testi2itranslator(A::PetrovGalerkinKernelInterpolationCore) =
    A.leveltranslators.test.i2itranslator
H2Factory.trialo2otranslator(A::PetrovGalerkinKernelInterpolationCore) =
    A.leveltranslators.trial.o2otranslator
H2Factory.testo2otranslator(A::PetrovGalerkinKernelInterpolationCore) =
    A.leveltranslators.test.o2otranslator

# Reuse one I2O collection for all orientations. Concrete traits avoid accessor ambiguities.
H2Factory.i2otranslator(A::PetrovGalerkinKernelInterpolationCore, ::isForward) =
    A.i2otranslator
H2Factory.i2otranslator(A::PetrovGalerkinKernelInterpolationCore, ::isTranspose) =
    A.i2otranslator
H2Factory.i2otranslator(A::PetrovGalerkinKernelInterpolationCore, ::isAdjoint) =
    A.i2otranslator

# Either complex leaf collection requires a conjugating adjoint multiply.
H2Factory.leafmul(A::PetrovGalerkinKernelInterpolationCore, ::isAdjoint) =
    _adjointleafmul(A)

# Without this method, H2Trees' generic fallback would treat the core as a tree.
H2Trees.isnear(A::PetrovGalerkinKernelInterpolationCore, ::isForward) = A.isnear

H2Factory.testmomentcollections(A::PetrovGalerkinKernelInterpolationCore, ::isForward) =
    A.momentcollections.test
H2Factory.trialmomentcollections(A::PetrovGalerkinKernelInterpolationCore, ::isForward) =
    A.momentcollections.trial

H2Factory.α(A::PetrovGalerkinKernelInterpolationCore, ::isForward) = A.α
H2Factory.α(A::PetrovGalerkinKernelInterpolationCore, ::isTranspose) = A.αtranspose
H2Factory.α(A::PetrovGalerkinKernelInterpolationCore, ::isAdjoint) = A.αadjoint

H2Factory.verbose(A::PetrovGalerkinKernelInterpolationCore, ::H2Factory.AbstractMapTrait) =
    A.verbose

H2Factory.minrelevantlevel(A::PetrovGalerkinKernelInterpolationCore, ::isForward) =
    A.minrelevantlevel

H2Factory.testaggregationplan(A::PetrovGalerkinKernelInterpolationCore, ::isForward) =
    A.plans.testaggregationplan
H2Factory.trialaggregationplan(A::PetrovGalerkinKernelInterpolationCore, ::isForward) =
    A.plans.trialaggregationplan
H2Factory.testdisaggregationplan(A::PetrovGalerkinKernelInterpolationCore, ::isForward) =
    A.plans.testdisaggregationplan
function H2Factory.trialdisaggregationplan(
    A::PetrovGalerkinKernelInterpolationCore, ::isForward
)
    return A.plans.trialdisaggregationplan
end

# --- Workspace construction hooks ----------------------------------------------------------------

# Schedules use raw levels, including levels below a side's minlevel.
function H2Factory.schedulinglevels(A::PetrovGalerkinKernelInterpolationCore)
    return 1:max(levels(testtree(A.tree))[end], levels(trialtree(A.tree))[end])
end

# Size each side's moments from its own polynomial family.
function H2Factory.buildmoments(A::PetrovGalerkinKernelInterpolationCore)
    momentbuffers, testmoments, trialmoments = MultilevelKernelInterpolation.moments(
        complex(A.γ),
        A.testmomentoperator,
        A.trialmomentoperator,
        A.tree,
        A.momentpolynomials.test,
        A.momentpolynomials.trial,
        A.plans.testaggregationplan,
        A.plans.trialaggregationplan,
        A.momenttype,
    )
    return momentbuffers, Moments(testmoments, trialmoments)
end

function H2Factory.buildvectorbuffers(A::PetrovGalerkinKernelInterpolationCore)
    testvectorbuffer, trialvectorbuffer = computevectorbuffers(A.tree, complex(typeof(A.γ)))
    return VectorBuffers(testvectorbuffer, trialvectorbuffer)
end

# Per side, from that side's own I2I collection, the same ownership the transfers have.
function H2Factory.buildi2ichannels(A::PetrovGalerkinKernelInterpolationCore)
    return SidedChannels(
        H2Factory.i2ichannels(A.leveltranslators.test.i2itranslator),
        H2Factory.i2ichannels(A.leveltranslators.trial.i2itranslator),
    )
end

function H2Factory.buildo2ochannels(A::PetrovGalerkinKernelInterpolationCore)
    return SidedChannels(
        H2Factory.o2ochannels(A.leveltranslators.test.i2itranslator),
        H2Factory.o2ochannels(A.leveltranslators.trial.i2itranslator),
    )
end

H2Factory.buildi2ochannel(A::PetrovGalerkinKernelInterpolationCore) = _buildi2ochannel(A)
