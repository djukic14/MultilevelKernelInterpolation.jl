"""
    assembleleveltranslators(tree, polynomials, rawplans, translatorstrategy)

Build I2I/O2O transfers for `rawplans.relevantlevels` and return
an `H2Factory.LevelTranslators` bundle.
"""
function assembleleveltranslators(
    tree, polynomials, rawplans, translatorstrategy::AbstractTranslatorStrategy
)
    relevantlevels = rawplans.relevantlevels
    relevantindices = [findfirst(isequal(level), levels(tree)) for level in relevantlevels]
    return assembleleveltranslators(
        polynomials[relevantindices], relevantlevels, translatorstrategy
    )
end

"""
    assembleleveltranslators(i2ipolynomials, i2ilevels, translatorstrategy)

Build I2I/O2O transfers from matching polynomial and level sequences.
Return a read-only `H2Factory.LevelTranslators` bundle.
Workspace scratch is allocated separately through `H2Factory.i2ichannels` and `o2ochannels`.
"""
function assembleleveltranslators(
    i2ipolynomials, i2ilevels, translatorstrategy::AbstractTranslatorStrategy
)
    i2itranslator = buildi2itranslator(
        i2ipolynomials, i2ilevels, i2istrategy(translatorstrategy)
    )
    o2otranslator = buildo2otranslator(i2itranslator, o2ostrategy(translatorstrategy))
    return LevelTranslators(i2itranslator, o2otranslator)
end

"""
    buildi2itranslator(i2ipolynomials, i2ilevels, strategy::AbstractI2IStrategy)

Build the child-to-parent translator for `strategy`.
`DefaultI2IStrategy` combines the tensor factors; `OneDimI2IStrategy` keeps them separate.
Define a method here for a custom I2I strategy. The result also feeds [`buildo2otranslator`](@ref).
"""
function buildi2itranslator(i2ipolynomials, i2ilevels, ::DefaultI2IStrategy)
    return I2ITranslator(i2ipolynomials, i2ilevels)
end

function buildi2itranslator(i2ipolynomials, i2ilevels, ::OneDimI2IStrategy)
    return I2ITranslatorOneDim(i2ipolynomials, i2ilevels)
end

"""
    buildo2otranslator(i2itranslator, strategy::AbstractO2OStrategy)

Build the parent-to-child translator from the I2I translator.
Using the same transfer data keeps the two directions consistent.
"""
function buildo2otranslator(i2itranslator, ::DefaultO2OStrategy)
    return O2OTranslator(i2itranslator)
end

"""
    assemblei2otranslator(operator, tree, polynomials, translating_plan, testspace, trialspace, translatorstrategy; verbose=false)

Build the read-only I2O coupling and compose it with its modulation policy.
`buildi2ocoupling` selects the coupling through `i2ostrategy(translatorstrategy)`.
Workspace FFT plans and basis-conversion buffers are allocated separately by
`H2Factory.buildi2ochannel`.
"""
function assemblei2otranslator(
    operator,
    tree,
    polynomials,
    translating_plan,
    testspace,
    trialspace,
    translatorstrategy::AbstractTranslatorStrategy;
    verbose=false,
)
    translation = _composei2otranslation(
        buildi2ocoupling(
            operator,
            tree,
            polynomials,
            translating_plan,
            testspace,
            trialspace,
            i2ostrategy(translatorstrategy);
            verbose=verbose,
        ),
    )
    _assertbsplinemodulation(translation, polynomials)
    return translation
end

"""
    _assertbsplinemodulation(translation, polynomials)

Reject a B-spline family paired with a non-B-spline modulation policy.
B-spline moments need raw-to-reduced conversion; a plain policy would skip it.
This catches builders that omit the space arguments used to select the B-spline route.
"""
function _assertbsplinemodulation(translation, polynomials)
    _anybspline(polynomials) || return nothing
    modulation = H2Factory.modulation(translation)
    modulation isa Union{BsplineDenseModulation,BsplineToeplitzModulation} && return nothing
    return error(
        "B-spline polynomials were composed with $(typeof(modulation)) instead of a B-spline " *
        "modulation, so the raw<->reduced basis conversion would be silently skipped and the " *
        "far field would be wrong. This means the I2O build was routed through a non-B-spline " *
        "entry point, the B-spline overloads live on the space-carrying " *
        "(`…, testspace, trialspace`) forms, so a builder must call those.",
    )
end

# Check both level vectors and per-side dictionaries. Qualify Base.values because
# H2Trees.values reads tree node data, not dictionary values.
_anybspline(polynomials::SidedPolynomialDicts) =
    _anybspline(Base.values(polynomials.test)) ||
    _anybspline(Base.values(polynomials.trial))
_anybspline(polynomials) = any(p -> p isa AbstractBspline, polynomials)

"""
    _composei2otranslation(translator)

Pair an I2O coupling with its modulation policy as an `H2Factory.I2OTranslation`.
A collection with its own basis conversion must declare `_i2oselfmodulates` and override
this method. The fallback rejects it rather than skipping the conversion.
Already-composed translations are returned unchanged.
"""
function _composei2otranslation(translator)
    _i2oselfmodulates(translator) && error(
        "$(typeof(translator)) is declared self-modulating (`_i2oselfmodulates`) but has no " *
        "`_composei2otranslation` override: composing it with `NoModulation` would silently skip its " *
        "raw<->reduced basis conversion. Add a `_composei2otranslation` override returning " *
        "`I2OTranslation(collection, <its modulation policy>)`.",
    )
    return H2Factory.I2OTranslation(translator)
end

# The Toeplitz B-spline builder already returns a composed translation.
# Qualify the type: MultilevelKernelInterpolation.I2OTranslation is a separate operator trait.
_composei2otranslation(translator::H2Factory.I2OTranslation) = translator

# Dense B-spline collections need an explicit basis-conversion policy.
_i2oselfmodulates(::Any) = false
_i2oselfmodulates(::BsplineI2OTranslationCollection) = true

# Pair dense B-spline coupling with its basis-conversion policy.
_composei2otranslation(translator::BsplineI2OTranslationCollection) =
    H2Factory.I2OTranslation(translator, translator.modulation)

"""
    buildi2ocoupling(operator, tree, polynomials, translating_plan, testspace, trialspace, i2ostrategy; verbose=false)

Build the read-only I2O coupling, without a scratch channel.
Use [`i2ochannelfor`](@ref) to allocate workspace scratch for the composed translation.
"""
function buildi2ocoupling(
    operator,
    tree,
    polynomials,
    translating_plan,
    testspace,
    trialspace,
    strategy::DenseI2OStrategy;
    verbose=false,
)
    return densei2ocoupling(
        operator,
        tree,
        polynomials,
        translating_plan,
        testspace,
        trialspace;
        verbose=verbose,
        symmetryreduced=strategy.symmetryreduced,
    )
end

function buildi2ocoupling(
    operator,
    tree,
    polynomials,
    translating_plan,
    testspace,
    trialspace,
    strategy::ToeplitzI2OStrategy;
    verbose=false,
)
    # Pass the spaces to select B-spline conversion. The four-argument builder must stay
    # plain: the B-spline builder uses it internally and an overload would recurse.
    return toeplitzi2ocoupling(
        operator,
        tree,
        polynomials,
        translating_plan,
        testspace,
        trialspace;
        verbose=verbose,
        symmetryreduced=strategy.symmetryreduced,
    )
end

# Workspace-local scratch, selected by the coupling and modulation rather than the strategy.

"""
    i2ochannelfor(translation, treepolynomials, translatingplan, T)

Allocate fresh workspace scratch for a built I2O translation.
`T` is the modulation scalar type, `eltype(operatorgamma(operator))`.
`treepolynomials` has the builder's layout: a level vector or `SidedPolynomialDicts`.
Run inside `withschedulechunks`; allocation reads `H2Factory.nchunks(level)`.
"""
function i2ochannelfor(t::H2Factory.I2OTranslation, treepolynomials, translatingplan, T)
    return i2ochannelfor(
        H2Factory.coupling(t), H2Factory.modulation(t), treepolynomials, translatingplan, T
    )
end

# No basis reduction: the coupling's own scratch is the whole channel. Dense and diagonal couplings
# use `EmptyChannels`; Toeplitz couplings use per-`(level, chunk)` FFT plan pools.
function i2ochannelfor(
    coupling,
    ::Union{H2Factory.NoModulation,H2Factory.ToeplitzModulation},
    treepolynomials,
    translatingplan,
    T,
)
    return H2Factory.i2ochannels(coupling, translatingplan)
end

# Dense B-spline: the coupling is an ordinary dense collection (no scratch of its own); everything
# staged is the modulation's raw<->reduced conversion.
function i2ochannelfor(
    coupling, ::BsplineDenseModulation, treepolynomials, translatingplan, T
)
    translatinglevels = levels(translatingplan)
    return BsplineI2OChannel(
        bsplinedensescratch(T, treepolynomials, translatinglevels), translatinglevels
    )
end

# Toeplitz B-spline: BOTH, the coupling's FFT plans and the modulation's conversion scratch.
function i2ochannelfor(
    coupling, ::BsplineToeplitzModulation, treepolynomials, translatingplan, T
)
    translatinglevels = levels(translatingplan)
    return BsplineToeplitzI2OChannel(
        H2Factory.i2ochannels(coupling, translatingplan),
        bsplinetoeplitzscratch(T, treepolynomials, translatinglevels),
        translatinglevels,
    )
end

function _buildi2ochannel(core)
    return i2ochannelfor(
        H2Factory.i2otranslator(core, isForward()),
        core.i2opolynomials,
        coretranslatingplan(core),
        eltype(core.γ),
    )
end

"""
    coretranslatingplan(core)

Return the core's plan carrying I2O pairs, selected by aggregation mode.
"""
function coretranslatingplan(core)
    return translatingplan(
        core.plans.trialaggregationplan, core.plans.testdisaggregationplan
    )
end

"""
    _constructionchunks(levels)

Return chunk counts for temporary translation-construction scratch, based on
`Threads.nthreads()`. These pools are discarded after construction.
Workspace pools use their own schedule chunk counts instead.
"""
_constructionchunks(levels) = Dict{Int,Int}(level => Threads.nthreads() for level in levels)

"""
    _coreandworkspace(Core, args; scheduling=nothing, checkaliasing=false, kwargs...)

Build a core and one workspace for `GalerkinKernelInterpolation` or `PetrovGalerkinKernelInterpolation`.
`scheduling` and `checkaliasing` apply to the workspace; other keywords go to the core.

With `scheduling=nothing`, use the core's default scheduler on its levels, serial inner
kernels, and one chunk per thread. `checkaliasing=true` runs the optional `O(n^2)`
pooled-moment check after workspace allocation; see `assertmomentaliasing`.
"""
function _coreandworkspace(
    Core, args; scheduling=nothing, checkaliasing::Bool=false, kwargs...
)
    core = Core(args...; kwargs...)
    ws = H2Workspace(core, isnothing(scheduling) ? core.defaultscheduler : scheduling)
    checkaliasing && assertmomentaliasing(ws)
    return ws
end
