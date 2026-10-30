
function KernelTranslator(
    operator, treepolynomials::A, tree, disaggregationplan
) where {A<:AbstractArray{<:AbstractBspline}}
    return BsplineKernelTranslator(operator, treepolynomials, tree, disaggregationplan)
end

# SidedPolynomialDicts needs its own B-spline route; it is not an AbstractArray.
function KernelTranslator(
    operator, treepolynomials::SidedPolynomialDicts, tree, disaggregationplan
)
    translatinglevels = collect(levels(disaggregationplan))
    isbspline =
        any(l -> treepolynomials.test[l] isa AbstractBspline, translatinglevels) ||
        any(l -> treepolynomials.trial[l] isa AbstractBspline, translatinglevels)

    # Select by polynomial family, not container type. Dense I2O supports separate grids.
    isbspline &&
        return BsplineKernelTranslator(operator, treepolynomials, tree, disaggregationplan)

    polynomialdict = leveltopolynomialdict(treepolynomials, levels(disaggregationplan))

    # Call the struct constructor: redispatch with the sided dictionary would recurse.
    return KernelTranslator{
        typeof(operator),typeof(polynomialdict),typeof(tree),typeof(disaggregationplan)
    }(
        operator, polynomialdict, tree, disaggregationplan
    )
end

"""
    BSPLINEREDUCTIONKRONECKERCROSSOVER

Switch between combined and separable B-spline reduction at this per-axis reduced size.
Below the threshold, use a precomputed Kronecker matrix on each side (`O(n^(3D))` work).
At or above it, apply one factor per axis (`O(n^(2D+1))`).
The threshold matches `interpolationinverse`, though it controls a different cost there.
"""
const BSPLINEREDUCTIONKRONECKERCROSSOVER = 7

# The per-axis interpolation inverse for one polynomial.
function _luT(p)
    n = size(LinearIndices(p), 1)
    return SMatrix{n,n}(pinv(collect(interpolationmatrix(p))))
end

# Precompute combined reduction below the crossover; otherwise keep the 1-D factors.
function _kroneckerreduction(l, D::Int)
    size(l, 1) < BSPLINEREDUCTIONKRONECKERCROSSOVER || return nothing
    return Matrix(computend(ntuple(_ -> Matrix(l), D)...))
end

# Reduction depends on the grid, so key by side and level, not node count.
"""
    BsplineKernelTranslator{O,P,T,D,TL,RL,TK,RK} <: AbstractKernelTranslator

Build dense B-spline couplings and convert kernel samples to interpolation coefficients.
Stores separate reduction matrices for each side and level.
"""
struct BsplineKernelTranslator{O,P,T,D,TL,RL,TK,RK} <: AbstractKernelTranslator
    operator::O
    polynomialdict::P
    tree::T
    disaggregationplan::D
    testluTs::TL
    trialluTs::RL
    testreductionmatrices::TK
    trialreductionmatrices::RK

    function BsplineKernelTranslator(
        operator::O, treepolynomials::P, tree::T, disaggregationplan::D
    ) where {O,P,T,D}
        return BsplineKernelTranslator(
            operator, treepolynomials, tree, levels(disaggregationplan), disaggregationplan
        )
    end

    function BsplineKernelTranslator(
        operator::O,
        treepolynomials::P,
        tree::T,
        disaggregationlevels,
        disaggregationplan::D,
    ) where {O,P,T,D}

        # Precompute factors once, using the same crossover as the apply routine.
        testpolys, trialpolys = _sidepolynomials(treepolynomials, disaggregationlevels)
        # Reuse these vectors to avoid recomputing pseudo-inverses.
        testluTvec = _luT.(testpolys)
        trialluTvec = _luT.(trialpolys)
        testluT = Dict(zip(disaggregationlevels, testluTvec))
        trialluT = Dict(zip(disaggregationlevels, trialluTvec))
        # Form each side's tensor product with its own spatial rank.
        testK = Dict(
            zip(disaggregationlevels, _kroneckerreduction.(testluTvec, dims.(testpolys)))
        )
        trialK = Dict(
            zip(disaggregationlevels, _kroneckerreduction.(trialluTvec, dims.(trialpolys)))
        )
        treepolynomials = leveltopolynomialdict(treepolynomials, disaggregationlevels)

        return new{
            O,
            typeof(treepolynomials),
            T,
            D,
            typeof(testluT),
            typeof(trialluT),
            typeof(testK),
            typeof(trialK),
        }(
            operator,
            treepolynomials,
            tree,
            disaggregationplan,
            testluT,
            trialluT,
            testK,
            trialK,
        )
    end
end

# Resolve reduction data while node levels are available; size does not identify a grid.
"""
    calculateI2Otranslation!(
        translator::BsplineKernelTranslator, i2o::I2OTranslation, receivingpolynomial, translatingpolynomial,
    )

Sample the kernel, then convert to B-spline interpolation coefficients.
Order-1 receiving polynomials need no reduction. Other orders use combined or separable
reduction; see `BSPLINEREDUCTIONKRONECKERCROSSOVER`.
The kernel trait selects a scalar matrix, a vector of matrices, or a matrix of matrices.
"""
function calculateI2Otranslation!(
    translator::BsplineKernelTranslator,
    ::isTwoNTree,
    receivingnode::Int,
    translatingnode::Int,
    direction,
)
    return _bsplinei2otranslation!(
        translator,
        shiftedpolynomial(
            translator.tree, polynomial(translator, receivingnode), receivingnode
        ),
        translatingpolynomialat(
            translator.tree,
            polynomial(translator, translatingnode),
            translatingnode,
            H2Trees.center(translator.tree, receivingnode) .- direction,
        ),
        level(translator.tree, receivingnode),
        level(translator.tree, translatingnode),
    )
end

function calculateI2Otranslation!(
    translator::BsplineKernelTranslator,
    ::isBlockTree,
    receivingnode::Int,
    translatingnode::Int,
    direction,
)
    return _bsplinei2otranslation!(
        translator,
        shiftedpolynomial(
            testtree(translator.tree),
            testpolynomial(translator, receivingnode),
            receivingnode,
        ),
        translatingpolynomialat(
            trialtree(translator.tree),
            trialpolynomial(translator, translatingnode),
            translatingnode,
            H2Trees.center(testtree(translator.tree), receivingnode) .- direction,
        ),
        level(testtree(translator.tree), receivingnode),
        level(trialtree(translator.tree), translatingnode),
    )
end

# Without node levels, derive reduction data from the supplied polynomials.
# This costs a pseudo-inverse per call; production uses cached node-level methods.
function calculateI2Otranslation!(
    translator::BsplineKernelTranslator,
    i2o::I2OTranslation,
    receivingpolynomial::ScaledShiftedPolynomial,
    translatingpolynomial::ScaledShiftedPolynomial,
)
    tf = _tf(translator, receivingpolynomial, translatingpolynomial, i2o)

    translation!(tf, translator.operator, i2o, receivingpolynomial, translatingpolynomial)

    MultilevelKernelInterpolation.polynomialorders(polynomial(receivingpolynomial))[1] ==
    1 &&
        MultilevelKernelInterpolation.polynomialorders(
            polynomial(translatingpolynomial)
        )[1] == 1 &&
        return tf

    luTrecv = _luT(polynomial(receivingpolynomial))
    luTtransl = _luT(polynomial(translatingpolynomial))
    Krecv = _kroneckerreduction(luTrecv, dims(receivingpolynomial))
    Ktransl = _kroneckerreduction(luTtransl, dims(translatingpolynomial))

    (isnothing(Krecv) || isnothing(Ktransl)) && return _reducetointerpolationbasis!(
        tf, luTrecv, luTtransl, dims(receivingpolynomial)
    )

    return _reducetointerpolationbasis!(tf, Krecv, Ktransl)
end

# Reduce test rows and trial columns with their own inverses; grids may differ.
function _bsplinei2otranslation!(
    translator::BsplineKernelTranslator,
    receivingpolynomial,
    translatingpolynomial,
    receivinglevel::Int,
    translatinglevel::Int,
)
    i2o = i2otranslation(translator.operator)
    tf = _tf(translator, receivingpolynomial, translatingpolynomial, i2o)

    translation!(tf, translator.operator, i2o, receivingpolynomial, translatingpolynomial)

    MultilevelKernelInterpolation.polynomialorders(polynomial(receivingpolynomial))[1] ==
    1 &&
        MultilevelKernelInterpolation.polynomialorders(
            polynomial(translatingpolynomial)
        )[1] == 1 &&
        return tf

    Krecv = translator.testreductionmatrices[receivinglevel]
    Ktransl = translator.trialreductionmatrices[translatinglevel]

    (isnothing(Krecv) || isnothing(Ktransl)) && return _reducetointerpolationbasis!(
        tf,
        translator.testluTs[receivinglevel],
        translator.trialluTs[translatinglevel],
        dims(receivingpolynomial),
    )

    return _reducetointerpolationbasis!(tf, Krecv, Ktransl)
end

# Low orders: one combined Kronecker multiplication per side.
function _reducetointerpolationbasis!(
    tf::AbstractMatrix{<:Number}, Krecv::Matrix, Ktransl::Matrix
)
    return Krecv * tf * transpose(Ktransl)
end

function _reducetointerpolationbasis!(
    tf::AbstractArray{<:AbstractMatrix}, Krecv::Matrix, Ktransl::Matrix
)
    return map(m -> Krecv * m * transpose(Ktransl), tf)
end

# High orders: apply each 1-D inverse along D spatial axes.
function _reducetointerpolationbasis!(
    tf::AbstractMatrix{<:Number}, luTrecv::SMatrix, luTtransl::SMatrix, D::Int
)
    return _reduceseparably!(tf, luTrecv, luTtransl, D)
end

function _reducetointerpolationbasis!(
    tf::AbstractArray{<:AbstractMatrix}, luTrecv::SMatrix, luTtransl::SMatrix, D::Int
)
    return map(m -> _reduceseparably!(m, luTrecv, luTtransl, D), tf)
end

# The first D axes are test nodes; the last D are trial nodes.
# For D >= 2, the last sweep reads scratch, so it can overwrite the input slice.
function _reduceseparably!(m, luTrecv::SMatrix, luTtransl::SMatrix, D::Int)
    nrecv = size(luTrecv, 1)
    ntransl = size(luTtransl, 1)
    tftensor = reshape(m, ntuple(_ -> nrecv, D)..., ntuple(_ -> ntransl, D)...)

    # Each intermediate has n^D entries; the sweep needs D-1 intermediates.
    recvscratch = separablescratch(eltype(m), nrecv, nrecv, D)
    translscratch = separablescratch(eltype(m), ntransl, ntransl, D)
    allrecv = ntuple(_ -> Colon(), D)
    alltransl = ntuple(_ -> Colon(), D)

    for j in CartesianIndices(ntuple(_ -> ntransl, D))
        slice = view(tftensor, allrecv..., j)
        separable1dinterpolations!(slice, luTrecv, slice, recvscratch)
    end
    for i in CartesianIndices(ntuple(_ -> nrecv, D))
        slice = view(tftensor, i, alltransl...)
        separable1dinterpolations!(slice, luTtransl, slice, translscratch)
    end

    return reshape(tftensor, nrecv^D, ntransl^D)
end

# Conversion matrices depend on side and level, not just node count.
# Forward reduces trial input and expands test output; transpose/adjoint swap sides.
struct BsplineDenseModulation{TR,RR,TT,RT} <: H2Factory.AbstractI2OModulation
    # raw -> reduced, applied to the INCOMING moment (`modulate_in!`)
    testrawtoreduced::TR
    trialrawtoreduced::RR
    # reduced -> raw, applied to the OUTGOING moment (`modulate_out!`)
    testreducedtoraw::TT
    trialreducedtoraw::RT
end

"""
    BsplineI2OTranslationCollection{I2O,M} <: AbstractI2OTranslationCollection

Pair dense B-spline couplings with raw/reduced basis conversion.
Moments and level transfers use the raw basis; I2O matrices act on reduced storage.
`modulate_in!` and `modulate_out!` apply the conversion without an FFT round trip.
"""
struct BsplineI2OTranslationCollection{I2O,M<:BsplineDenseModulation} <:
       AbstractI2OTranslationCollection
    i2ocollection::I2O
    modulation::M
end

# Expose the policy to direct callers as well as the composed traversal translation.
H2Factory.modulation(c::BsplineI2OTranslationCollection) = c.modulation

# Forward the guard: only the stored collection knows whether symmetry tags remain.
H2Factory.assertnosymmetry(c::BsplineI2OTranslationCollection) =
    H2Factory.assertnosymmetry(i2ocollection(c))

# Forward uses trial input scratch and test output scratch; transpose/adjoint swap them.
_incomingscratch(buffer, ::isForward) = buffer.trial
_incomingscratch(buffer, ::Union{isTranspose,isAdjoint}) = buffer.test
_outgoingscratch(buffer, ::isForward) = buffer.test
_outgoingscratch(buffer, ::Union{isTranspose,isAdjoint}) = buffer.trial

_incomingrawtoreduced(m::BsplineDenseModulation, ::isForward) = m.trialrawtoreduced
function _incomingrawtoreduced(m::BsplineDenseModulation, ::Union{isTranspose,isAdjoint})
    return m.testrawtoreduced
end
_outgoingreducedtoraw(m::BsplineDenseModulation, ::isForward) = m.testreducedtoraw
function _outgoingreducedtoraw(m::BsplineDenseModulation, ::Union{isTranspose,isAdjoint})
    return m.trialreducedtoraw
end

function i2ocollection(T::BsplineI2OTranslationCollection)
    return T.i2ocollection
end

function Base.getindex(T::BsplineI2OTranslationCollection, key)
    return Base.getindex(i2ocollection(T), key)
end

"""
    BsplineI2OTranslationTrait()

Select coupling on reduced moment storage rather than raw B-spline storage.
A separate trait also avoids ambiguity between this collection's kernel and H2Factory's
generic dense kernel.
"""
struct BsplineI2OTranslationTrait <: H2Factory.AbstractI2OTranslationTrait end

i2otrait(::BsplineI2OTranslationCollection) = BsplineI2OTranslationTrait()

function H2Trees.level(T::BsplineI2OTranslationCollection, translatingnode::Int)
    return level(i2ocollection(T), translatingnode)
end

function tree(T::BsplineI2OTranslationCollection)
    return tree(i2ocollection(T))
end

"""
    _rawtoreducedmodulation(polynomial)

Return the raw-to-reduced transfer `S` (the paper's stabilization matrix, Eq. 42),
or `I` when no knots are removed.

Dense kernel matrices already include the interpolation inverse from `BsplineKernelTranslator`.
Apply only the dimension change here; another inverse would apply the reduction twice.
"""
function _rawtoreducedmodulation(polynomial)
    length(range(polynomial, 1)) != numpolynomials(polynomial)[1] || return I
    S = i2icheapaveraging(MultilevelKernelInterpolation.bspline(polynomial, 1).bspline)
    return SMatrix{size(S, 1),size(S, 2)}(collect(S))
end

function _reducedtorawmodulation(polynomial)
    mlu = _rawtoreducedmodulation(polynomial)
    mlu == I && return I
    return SMatrix{size(mlu, 2),size(mlu, 1)}(collect(mlu)')
end

# Dense Petrov B-spline couplings support separate test/trial grids.
# Toeplitz requires matching grids, checked through SidedPolynomialDicts.
function I2OTranslationCollection(
    operator,
    tree,
    treepolynomials::SidedPolynomialDicts,
    translatingplan::AbstractPlan;
    verbose=false,
)
    translatinglevels = collect(levels(translatingplan))
    testpolys = [treepolynomials.test[level] for level in translatinglevels]
    trialpolys = [treepolynomials.trial[level] for level in translatinglevels]
    isbspline =
        any(p -> p isa AbstractBspline, testpolys) ||
        any(p -> p isa AbstractBspline, trialpolys)

    if isbspline
        # Separate modulation and scratch on each side support rectangular grids.
        return _bsplinei2ocollection(
            operator, tree, treepolynomials, translatingplan; verbose=verbose
        )
    end

    # Other polynomial families use the plain dense collection.
    return H2Factory.I2OTranslationCollection(
        tree,
        translatingplan,
        DirectionInvariance(),
        KernelTranslator(operator, treepolynomials, tree, translatingplan);
        i2otranslationtrait=i2otranslationtrait(i2otranslation(operator)),
        verbose=verbose,
    )
end

# Keep both routes: Galerkin arrays and Petrov SidedPolynomialDicts.
function densei2ocoupling(
    operator,
    tree,
    treepolynomials::SidedPolynomialDicts,
    translatingplan::AbstractPlan;
    verbose=false,
)
    translatinglevels = collect(levels(translatingplan))
    testpolys = [treepolynomials.test[level] for level in translatinglevels]
    trialpolys = [treepolynomials.trial[level] for level in translatinglevels]
    isbspline =
        any(p -> p isa AbstractBspline, testpolys) ||
        any(p -> p isa AbstractBspline, trialpolys)

    isbspline && return _bsplinei2ocoupling(
        operator, tree, treepolynomials, translatingplan; verbose=verbose
    )
    return first(
        H2Factory.I2OTranslationCollection(
            tree,
            translatingplan,
            DirectionInvariance(),
            KernelTranslator(operator, treepolynomials, tree, translatingplan);
            i2otranslationtrait=i2otranslationtrait(i2otranslation(operator)),
            verbose=verbose,
        ),
    )
end

function densei2ocoupling(
    operator, tree, treepolynomials::A, translatingplan::AbstractPlan; verbose=false
) where {A<:AbstractArray{<:AbstractBspline}}
    return _bsplinei2ocoupling(
        operator, tree, treepolynomials, translatingplan; verbose=verbose
    )
end

# Dense symmetry reduction is not implemented for B-splines or tested for Petrov maps.
# Reject both container forms instead of falling through to the generic reduced builder.
function symmetryreduceddensei2ocoupling(
    operator, tree, treepolynomials::A, translatingplan::AbstractPlan; verbose=false
) where {A<:AbstractArray{<:AbstractBspline}}
    return error(
        "symmetryreduceddensei2ocoupling: the dense B-spline route stores its couplings in a " *
        "`BsplineI2OTranslationCollection` carrying its own raw<->reduced modulation, which the " *
        "symmetry reduction does not compose with. Use `DenseI2OStrategy()` (unreduced), or " *
        "`ToeplitzI2OStrategy(; symmetryreduced=true)`, which does support B-splines.",
    )
end

function symmetryreduceddensei2ocoupling(
    operator,
    tree,
    treepolynomials::SidedPolynomialDicts,
    translatingplan::AbstractPlan;
    verbose=false,
)
    return error(
        "symmetryreduceddensei2ocoupling: the symmetry-reduced dense route is implemented for " *
        "GALERKIN maps only. A Petrov map resolves each node's grid through its own side's tree, " *
        "which is unexercised here and would be a silent wrong-grid permutation rather than an " *
        "error. Use `DenseI2OStrategy()` (unreduced) for a Petrov map.",
    )
end

# Galerkin entry: one family, already an array of B-splines.
function I2OTranslationCollection(
    operator, tree, treepolynomials::A, translatingplan::AbstractPlan; verbose=false
) where {A<:AbstractArray{<:AbstractBspline}}
    return _bsplinei2ocollection(
        operator, tree, treepolynomials, translatingplan; verbose=verbose
    )
end

# Share coupling and scratch builders with workspace construction.
function _bsplinei2ocollection(
    operator, tree, treepolynomials, translatingplan::AbstractPlan; verbose=false
)
    coupling = _bsplinei2ocoupling(
        operator, tree, treepolynomials, translatingplan; verbose=verbose
    )
    return coupling,
    i2ochannelfor(
        _composei2otranslation(coupling),
        treepolynomials,
        translatingplan,
        eltype(operatorgamma(operator)),
    )
end

# Both container forms support level indexing; keep per-side data separate below.
function _bsplinei2ocoupling(
    operator, tree, treepolynomials, translatingplan::AbstractPlan; verbose=false
)
    translator = KernelTranslator(operator, treepolynomials, tree, translatingplan)
    i2o, _ = H2Factory.I2OTranslationCollection(
        tree,
        translatingplan,
        DirectionInvariance(),
        translator;
        i2otranslationtrait=i2otranslationtrait(i2otranslation(operator)),
        verbose=verbose,
    )

    translatinglevels = levels(translatingplan)
    # Shared-level indexing would require matching grids, which dense I2O does not need.
    testpolys, trialpolys = _sidepolynomials(treepolynomials, translatinglevels)

    return BsplineI2OTranslationCollection(
        i2o,
        BsplineDenseModulation(
            Dict(zip(translatinglevels, _rawtoreducedmodulation.(testpolys))),
            Dict(zip(translatinglevels, _rawtoreducedmodulation.(trialpolys))),
            Dict(zip(translatinglevels, _reducedtorawmodulation.(testpolys))),
            Dict(zip(translatinglevels, _reducedtorawmodulation.(trialpolys))),
        ),
    )
end

"""
    bsplinedensescratch(T, treepolynomials, translatinglevels)

Allocate dense B-spline conversion buffers as a level-ordered vector of chunk buffers.
Each chunk holds separate `test` and `trial` sets with `.toreduced` and `.toraw` scratch.
Forward uses trial input and test output; transpose/adjoint swap sides.

Scratch is workspace-local. Run inside `withschedulechunks` because allocation reads
`H2Factory.nchunks(level)`.
"""
function bsplinedensescratch(T, treepolynomials, translatinglevels)
    # separablescratch supplies the D-1 intermediates needed in each direction.
    _scratch(poly) = (
        raw=numpolynomials(poly)[1];
        reduced=size(LinearIndices(poly), 1);
        D=dims(poly);
        (
            toreduced=separablescratch(T, raw, reduced, D),
            toraw=separablescratch(T, reduced, raw, D),
        )
    )

    testpolys, trialpolys = _sidepolynomials(treepolynomials, translatinglevels)
    buffers = []
    for (level, testpoly, trialpoly) in zip(translatinglevels, testpolys, trialpolys)
        push!(
            buffers,
            [
                deepcopy((test=_scratch(testpoly), trial=_scratch(trialpoly))) for
                _ in 1:(H2Factory.nchunks(level))
            ],
        )
    end
    return convert(Vector{typeof(buffers[1])}, buffers)
end

"""
    _sidepolynomials(treepolynomials, levels)

Return `(test, trial)` polynomial vectors for `levels`.
A Galerkin family is returned twice. Read sided dictionaries separately, without
requiring matching grids.
"""
function _sidepolynomials(treepolynomials::SidedPolynomialDicts, levels)
    return (
        [treepolynomials.test[level] for level in levels],
        [treepolynomials.trial[level] for level in levels],
    )
end

function _sidepolynomials(treepolynomials, levels)
    polynomials = treepolynomials[levels]
    return (polynomials, polynomials)
end

# Convert incoming raw moments to reduced storage and outgoing moments back to raw.
# mlu == I needs only a copy; scratch is selected by side and orientation.
function H2Factory.modulate_in!(
    in::M,
    level::Int,
    maptrait,
    modulation::BsplineDenseModulation,
    coupling,
    channel,
    ::Scheduler,
    buffer,
) where {M<:AbstractBsplineDenseMoment}
    mlu = _incomingrawtoreduced(modulation, maptrait)[level]

    if mlu == I
        for i in eachindex(in)
            reducedmoment(in, i) .= in[i]
        end
        return in
    end

    scratch = _incomingscratch(buffer, maptrait)
    for i in eachindex(in)
        separable1dinterpolations!(reducedmoment(in, i), mlu, in[i], scratch.toreduced)
    end
    return in
end

function H2Factory.modulate_out!(
    out::M,
    level::Int,
    maptrait,
    modulation::BsplineDenseModulation,
    coupling,
    channel,
    ::Scheduler,
    buffer,
) where {M<:AbstractBsplineDenseMoment}
    tmlu = _outgoingreducedtoraw(modulation, maptrait)[level]

    if tmlu == I
        for i in eachindex(out)
            out[i] .= reducedmoment(out, i)
        end
        return out
    end

    scratch = _outgoingscratch(buffer, maptrait)
    for i in eachindex(out)
        separable1dinterpolations!(out[i], tmlu, reducedmoment(out, i), scratch.toraw)
    end
    return out
end

# Traversal and direct callers resolve this collection's policy through `H2Factory.modulation`
# and apply it through `modulate_in!`/`modulate_out!`.

# Convert protocol directions to the map traits used by block application.
_bsplinemode(::H2Factory.ForwardTranslation) = H2Factory.isForward()
_bsplinemode(::H2Factory.TransposeTranslation) = H2Factory.isTranspose()
_bsplinemode(::H2Factory.AdjointTranslation) = H2Factory.isAdjoint()

"""
    applycouplingkernel!(out, in, tfs::BsplineI2OTranslationCollection, ::BsplineI2OTranslationTrait,
                         receivingnode, translatingnode, direction, accumulation, context)

Apply the stored coupling to the reduced arrays of `in` and `out`.
Modulation prepares these arrays; the generic dense kernel instead uses raw storage.

`_bsplinei2oapply!` handles a scalar matrix, independent component matrices, or a fully
coupled matrix of matrices. Transpose/adjoint transform each block lazily and swap outer
component indices where needed, without allocating a transformed container.

The guarded `H2Factory.applycoupling!` entry runs `assertnosymmetry` before this kernel.
"""
function applycouplingkernel!(
    out,
    in,
    tfs::BsplineI2OTranslationCollection,
    ::BsplineI2OTranslationTrait,
    receivingnode::Int,
    translatingnode::Int,
    direction::H2Factory.AbstractTranslationDirection,
    ::H2Factory.Overwrite,
    ::H2Factory.TranslationContext,
)
    green = i2ocollection(tfs)[H2Factory.couplingkey(
        receivingnode, translatingnode, direction
    )]
    _bsplinei2oapply!(out, green, in, _bsplinemode(direction))
    return out
end

function applycouplingkernel!(
    out,
    in,
    tfs::BsplineI2OTranslationCollection,
    ::BsplineI2OTranslationTrait,
    receivingnode::Int,
    translatingnode::Int,
    direction::H2Factory.AbstractTranslationDirection,
    ::H2Factory.Accumulate,
    ::H2Factory.TranslationContext,
)
    green = i2ocollection(tfs)[H2Factory.couplingkey(
        receivingnode, translatingnode, direction
    )]
    _bsplinei2oapply!(out, green, in, _bsplinemode(direction), true, true)
    return out
end

# Transform individual blocks lazily instead of copying the container.
_bsplineblock(g, ::H2Factory.isForward) = g
_bsplineblock(g, ::H2Factory.isTranspose) = transpose(g)
_bsplineblock(g, ::H2Factory.isAdjoint) = adjoint(g)

# Transpose/adjoint also swap outer component indices.
_bsplineindex(i, j, ::H2Factory.isForward) = (i, j)
_bsplineindex(i, j, ::Union{H2Factory.isTranspose,H2Factory.isAdjoint}) = (j, i)

# A scalar kernel acts on each component independently.
# Use cached reducedvector views to avoid allocating array headers.
function _bsplinei2oapply!(out, green::AbstractMatrix{<:Number}, in, mode)
    g = _bsplineblock(green, mode)
    for i in eachindex(in)
        LinearAlgebra.mul!(reducedvector(out, i), g, reducedvector(in, i))
    end
    return out
end

function _bsplinei2oapply!(out, green::AbstractMatrix{<:Number}, in, mode, α, β)
    g = _bsplineblock(green, mode)
    for i in eachindex(in)
        LinearAlgebra.mul!(reducedvector(out, i), g, reducedvector(in, i), α, β)
    end
    return out
end

# `∇Green3DTranslation`: `green` is a vector of 3 independent (uncoupled) matrices.
function _bsplinei2oapply!(out, green::AbstractVector{<:AbstractMatrix}, in, mode)
    for i in eachindex(green)
        LinearAlgebra.mul!(
            reducedvector(out, i), _bsplineblock(green[i], mode), reducedvector(in, i)
        )
    end
    return out
end

function _bsplinei2oapply!(out, green::AbstractVector{<:AbstractMatrix}, in, mode, α, β)
    for i in eachindex(green)
        LinearAlgebra.mul!(
            reducedvector(out, i), _bsplineblock(green[i], mode), reducedvector(in, i), α, β
        )
    end
    return out
end

# `∇∇Green3DTranslation`/`DyadicGreen3DTranslation`/`∇CrossGreen3DTranslation`: `green` is a
# 3x3, cross-component-coupled matrix of matrices.
function _bsplinei2oapply!(out, green::AbstractMatrix{<:AbstractMatrix}, in, mode)
    for i in axes(green, 1)
        for (k, j) in enumerate(axes(green, 2))
            gi, gj = _bsplineindex(i, j, mode)
            block = _bsplineblock(green[gi, gj], mode)
            if k == 1
                LinearAlgebra.mul!(reducedvector(out, i), block, reducedvector(in, j))
            else
                LinearAlgebra.mul!(
                    reducedvector(out, i), block, reducedvector(in, j), true, true
                )
            end
        end
    end
    return out
end

function _bsplinei2oapply!(out, green::AbstractMatrix{<:AbstractMatrix}, in, mode, α, β)
    for i in axes(green, 1)
        outvec = reducedvector(out, i)
        for (k, j) in enumerate(axes(green, 2))
            gi, gj = _bsplineindex(i, j, mode)
            block = _bsplineblock(green[gi, gj], mode)
            # Scale old output by beta once; alpha scales every term in the component sum.
            LinearAlgebra.mul!(outvec, block, reducedvector(in, j), α, k == 1 ? β : true)
        end
    end
    return out
end

"""
    BsplineI2OChannel{B}

Workspace-local buffers for dense B-spline raw/reduced conversion,
indexed by level and chunk, with `.toreduced` and `.toraw` scratch on each side.
"""
struct BsplineI2OChannel{B}
    buffers::B
end

function BsplineI2OChannel(buffers, nodelevels)
    @assert length(buffers) == length(nodelevels)
    isempty(buffers) && return BsplineI2OChannel(Dict{Int,Any}())

    bufferchannels = []
    for i in eachindex(buffers)
        push!(bufferchannels, createchannel(buffers[i]))
    end
    bufferchannels = convert(Vector{typeof(bufferchannels[1])}, bufferchannels)

    return BsplineI2OChannel(Dict(zip(nodelevels, bufferchannels)))
end

function H2Factory.borrowbuffer(
    channel::BsplineI2OChannel, level::Int, chunkid::Int, maptrait
)
    !haskey(channel.buffers, level) && return EmptyChannels()
    return takebuffer!(channel, level, chunkid)
end

function takebuffer!(channel::BsplineI2OChannel, level::Int, chunkid::Int)
    return H2Factory.borrowbuffer(channel.buffers[level], chunkid)
end
