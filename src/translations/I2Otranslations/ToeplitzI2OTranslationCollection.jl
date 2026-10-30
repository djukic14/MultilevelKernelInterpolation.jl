"""
    ToeplitzTranslator(operator, treepolynomials, tree, disaggregationplan)

Build Toeplitz I2O tensors from kernel samples on equidistant grids.
Stores level polynomials, the translating plan, and temporary FFT plan pools.
The type parameter `N` records the spatial tensor rank for `_toeplitzeltype`.
"""
struct ToeplitzTranslator{O,P,T,D,F,N} <: AbstractKernelTranslator
    operator::O
    polynomialdict::P
    tree::T
    disaggregationplan::D
    fftplans::F
    function ToeplitzTranslator(
        operator::O, treepolynomials::P, tree::T, disaggregationplan::D
    ) where {O,P,T,D}
        tensorsizes = []
        fftplans = []
        disaggregationlevels = levels(disaggregationplan)
        polynomialdict = leveltopolynomialdict(treepolynomials, disaggregationlevels)

        # All levels must have one spatial rank so stored tensors and FFT plans share a type.
        perlevelpolynomials = treepolynomials[disaggregationlevels]
        polynomialranks = [dims(polynomial) for polynomial in perlevelpolynomials]
        allequal(polynomialranks) || error(
            "The Toeplitz I2O route needs one spatial rank for the whole tree, but the " *
            "per-level polynomials are " *
            join(("$(r)-D" for r in polynomialranks), ", ") *
            " at levels $(collect(disaggregationlevels)) respectively.",
        )
        N = first(polynomialranks)

        for (level, polynomial) in zip(disaggregationlevels, perlevelpolynomials)
            tensorsize = 2 .* size(LinearIndices(polynomial)) .- 1
            tf = Array{scalartype(operator),N}(undef, tensorsize)
            threadfftplans = [
                plan_fft(tf; num_threads=nothing) for _ in 1:(H2Factory.nchunks(level))
            ]

            # Construction uses a locking Channel pool, not traversal's chunk-indexed buffers.
            fftplanpool = Channel{eltype(threadfftplans)}(length(threadfftplans))
            for p in threadfftplans
                put!(fftplanpool, p)
            end

            push!(tensorsizes, tensorsize)
            push!(fftplans, fftplanpool)
        end

        fftplans = convert(Vector{typeof(fftplans[1])}, fftplans)
        fftplans = Dict(zip(tensorsizes, fftplans))
        return new{O,typeof(polynomialdict),T,D,typeof(fftplans),N}(
            operator, polynomialdict, tree, disaggregationplan, fftplans
        )
    end
end

# Resolve the shared Petrov polynomials before choosing the B-spline route.
"""
    ToeplitzI2OTranslationCollection(
        operator,
        tree,
        polynomials,
        translatingplan::H2Trees.AbstractPlan,
        space;
        verbose=false,
    )

Build FFT-accelerated I2O couplings and a scratch channel for `translatingplan`.
The level `polynomials` must use equidistant grids. B-spline families also get basis
conversion. Set `verbose=true` to show progress.
"""
function ToeplitzI2OTranslationCollection(
    operator,
    tree,
    treepolynomials::SidedPolynomialDicts,
    translatingplan::AbstractPlan,
    testspace,
    trialspace;
    verbose=true,
    symmetryreduced=false,
)
    # Use resolved polynomials only for routing; retain raw-level indexing in the builder.
    agreed = treepolynomials[collect(levels(translatingplan))]
    couplingbuilder = _couplingbuilder(symmetryreduced)
    if all(p -> p isa AbstractBspline, agreed)
        return _bsplinetoeplitzi2otranslation(
            operator,
            tree,
            treepolynomials,
            translatingplan;
            verbose=verbose,
            couplingbuilder=couplingbuilder,
        )
    end
    return ToeplitzI2OTranslationCollection(
        operator,
        tree,
        treepolynomials,
        translatingplan;
        verbose=verbose,
        couplingbuilder=couplingbuilder,
    )
end

# Share reduced/unreduced routing across constructors.
_couplingbuilder(symmetryreduced::Bool) =
    symmetryreduced ? symmetryreducedtoeplitzi2ocoupling : toeplitzi2ocoupling

# Use the same routing for read-only construction.
function toeplitzi2ocoupling(
    operator,
    tree,
    treepolynomials::SidedPolynomialDicts,
    translatingplan::AbstractPlan,
    testspace,
    trialspace;
    verbose=true,
    symmetryreduced=false,
)
    agreed = treepolynomials[collect(levels(translatingplan))]
    couplingbuilder = _couplingbuilder(symmetryreduced)
    if all(p -> p isa AbstractBspline, agreed)
        return _bsplinetoeplitzcoupling(
            operator,
            tree,
            treepolynomials,
            translatingplan;
            verbose=verbose,
            couplingbuilder=couplingbuilder,
        )
    end
    return couplingbuilder(
        operator, tree, treepolynomials, translatingplan; verbose=verbose
    )
end

# Forward symmetryreduced through the Galerkin space-taking entry points too.
function toeplitzi2ocoupling(
    operator,
    tree,
    treepolynomials,
    translatingplan::AbstractPlan,
    space;
    verbose=true,
    symmetryreduced=false,
)
    return _couplingbuilder(symmetryreduced)(
        operator, tree, treepolynomials, translatingplan; verbose=verbose
    )
end

function toeplitzi2ocoupling(
    operator,
    tree,
    treepolynomials,
    translatingplan::AbstractPlan,
    testspace,
    trialspace;
    verbose=true,
    symmetryreduced=false,
)
    return _couplingbuilder(symmetryreduced)(
        operator, tree, treepolynomials, translatingplan; verbose=verbose
    )
end

function ToeplitzI2OTranslationCollection(
    operator,
    tree,
    treepolynomials,
    translatingplan::AbstractPlan,
    space;
    verbose=true,
    symmetryreduced=false,
)
    return ToeplitzI2OTranslationCollection(
        operator,
        tree,
        treepolynomials,
        translatingplan;
        verbose=verbose,
        couplingbuilder=_couplingbuilder(symmetryreduced),
    )
end

function ToeplitzI2OTranslationCollection(
    operator,
    tree,
    treepolynomials,
    translatingplan::AbstractPlan,
    testspace,
    trialspace;
    verbose=true,
    symmetryreduced=false,
)
    return ToeplitzI2OTranslationCollection(
        operator,
        tree,
        treepolynomials,
        translatingplan;
        verbose=verbose,
        couplingbuilder=_couplingbuilder(symmetryreduced),
    )
end

# Reuse the read-only builder and the workspace channel builder.
function ToeplitzI2OTranslationCollection(
    operator,
    tree,
    treepolynomials,
    translatingplan::AbstractPlan;
    verbose=true,
    couplingbuilder=toeplitzi2ocoupling,
)
    coupling = couplingbuilder(
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

"""
    toeplitzi2ocoupling(operator, tree, treepolynomials, translatingplan; verbose=true)

Build the read-only Toeplitz coupling without a workspace FFT channel.
B-spline families use the same modulation routing as [`ToeplitzI2OTranslationCollection`](@ref).
"""
function toeplitzi2ocoupling(
    operator, tree, treepolynomials, translatingplan::AbstractPlan; verbose=true
)
    @assert all(isequidistantpolynomial, treepolynomials)

    translator = ToeplitzTranslator(operator, treepolynomials, tree, translatingplan)
    return H2Factory.toeplitzi2ocollection(
        tree,
        translatingplan,
        DirectionInvariance(),
        translator;
        i2otranslationtrait=ToeplitzI2OTranslationTrait(),
        verbose=verbose,
    )
end

"""
    symmetryreducedtoeplitzi2ocoupling(operator, tree, treepolynomials, translatingplan; verbose=true)

Build Toeplitz couplings with one difference tensor per group of symmetry-related
lattice directions. Other directions use index permutations, without extra approximation.

Equal spacing and extent on every tensor axis allow the full signed-permutation group
(`D! * 2^D` elements, 48 in 3-D). [`symmetrycomponentaction`](@ref) selects the kernel's
field-component transformation. Non-lattice displacements use unreduced couplings.
"""
function symmetryreducedtoeplitzi2ocoupling(
    operator, tree, treepolynomials, translatingplan::AbstractPlan; verbose=true
)
    @assert all(isequidistantpolynomial, treepolynomials)

    translator = ToeplitzTranslator(operator, treepolynomials, tree, translatingplan)

    # Fall back only for non-lattice geometry; propagate other failures.
    canonical = try
        first(
            H2Factory.I2OTranslationCollection(
                tree,
                translatingplan,
                H2Trees.SymmetryDirectionInvariancePerLevel(H2Trees.FullLatticeSymmetry()),
                translator;
                i2otranslationtrait=ToeplitzI2OTranslationTrait(),
                verbose=verbose,
            ),
        )
    catch e
        e isa H2Trees.NonLatticeTranslationError || rethrow()
        verbose &&
            @info "MultilevelKernelInterpolation: displacements are not lattice aligned; building the unreduced coupling."
        return toeplitzi2ocoupling(
            operator, tree, treepolynomials, translatingplan; verbose=verbose
        )
    end

    reduced = H2Factory.symmetryreducedtoeplitzcollection(
        canonical; componentaction=symmetrycomponentaction(i2otranslation(operator))
    )
    return H2Factory.ToeplitzI2OTranslationCollection(
        reduced, H2Trees.tree(translatingplan)
    )
end

"""
    _toeplitzeltype(translator::ToeplitzTranslator)

Return the stored Toeplitz representation type, with an `N`-dimensional array of the
operator's scalar type. `N` comes from the translator's spatial-rank parameter.
"""
function _toeplitzeltype(translator::ToeplitzTranslator{O,P,T,D,F,N}) where {O,P,T,D,F,N}
    return ToeplitzMatrixRepresentation{Array{scalartype(translator.operator),N}}
end

function eltypei2otranslation(::ScalarI2OTranslation, translator::ToeplitzTranslator)
    return _toeplitzeltype(translator)
end

# Match the generic trait groups to avoid ambiguous element-type dispatch.
function eltypei2otranslation(::∇Green3DTranslation, translator::ToeplitzTranslator)
    return Vector{_toeplitzeltype(translator)}
end

function eltypei2otranslation(
    ::Union{∇∇Green3DTranslation,DyadicGreen3DTranslation,∇CrossGreen3DTranslation},
    translator::ToeplitzTranslator,
)
    return Matrix{_toeplitzeltype(translator)}
end

function _convertdifference(diff)
    diff > 0 && return diff + 1, 1
    return 1, 1 - diff
end

function _toeplitzdifferences(maxindex)
    return [0:(maxindex - 1); (-(maxindex - 1)):(-1)]
end

"""
    calculateI2Otranslation!(
        translator::ToeplitzTranslator, momentop::I2OTranslation, receivingpolynomial, translatingpolynomial,
    )

Build Toeplitz translation tensors with a borrowed FFT plan, returning it afterward.
The kernel trait selects one tensor, a vector of tensors, or a matrix of tensors.
"""
function calculateI2Otranslation!(
    translator::ToeplitzTranslator,
    momentop::I2OTranslation,
    receivingpolynomial,
    translatingpolynomial,
)
    tensorsize = 2 .* size(LinearIndices(receivingpolynomial)) .- 1
    T = toeplitzeltype(eltype(translator))
    tf = _toeplitztf(T, tensorsize, momentop)
    fftplan = take!(translator.fftplans[tensorsize])
    tf = toeplitztranslation!(
        tf,
        translator.operator,
        momentop,
        receivingpolynomial,
        translatingpolynomial,
        fftplan,
    )
    put!(translator.fftplans[tensorsize], fftplan)
    return tf
end

# Every scalar kernel stores one tensor, regardless of spatial dimension.
function _toeplitztf(T, tensorsize, ::ScalarI2OTranslation)
    return T(undef, tensorsize)
end

function _toeplitztf(T, tensorsize, i2o::I2OTranslation)
    tf = Array{T}(undef, _tfcontainershape(i2o)...)
    for i in eachindex(tf)
        tf[i] = T(undef, tensorsize)
    end
    return tf
end

"""
    function toeplitztranslation!(
        toeplitztensor, operator, i2otranslationtrait, testpolynomial, trialpolynomial, fftplan
    )

Fill `toeplitztensor` with kernel samples at test/trial node differences, then return
`storetoeplitz(fftplan, toeplitztensor)`. The trait selects the kernel and component layout.
"""
function toeplitztranslation!(
    toeplitztensor, operator, i2otranslationtrait, testpolynomial, trialpolynomial, fftplan
)
    # Each sampling buffer follows its polynomial's spatial dimension.
    buffertest = Vector{eltype(testpolynomial)}(undef, dims(testpolynomial))
    buffertrial = Vector{eltype(trialpolynomial)}(undef, dims(trialpolynomial))

    testlinearindices = LinearIndices(testpolynomial)
    triallinearindices = LinearIndices(trialpolynomial)

    # Sweep the Cartesian product of per-axis node differences.
    # Tensor sizing uses the receiving grid; this route requires matching grids.
    rank = ndims(testlinearindices)
    differences = ntuple(d -> _toeplitzdifferences(size(testlinearindices, d)), rank)

    # Resolve each signed difference to a test/trial node pair once per axis.
    testnodes = ntuple(d -> [first(_convertdifference(r)) for r in differences[d]], rank)
    trialnodes = ntuple(d -> [last(_convertdifference(r)) for r in differences[d]], rank)

    buffer = kernelbuffer(eltype(toeplitztensor), i2otranslationtrait)

    for tensorindex in CartesianIndices(map(length, differences))
        testnode = CartesianIndex(map(getindex, testnodes, Tuple(tensorindex)))
        trialnode = CartesianIndex(map(getindex, trialnodes, Tuple(tensorindex)))

        getsamplingpoint!(buffertest, testpolynomial, testlinearindices[testnode])
        getsamplingpoint!(buffertrial, trialpolynomial, triallinearindices[trialnode])

        computetoeplitzkernel!(
            toeplitztensor,
            tensorindex,
            operator,
            i2otranslationtrait,
            buffertest,
            buffertrial,
            buffer,
        )
    end

    return storetoeplitz(fftplan, toeplitztensor)
end

"""
    function computetoeplitzkernel!(
        toeplitztensor, tensorindex, operator, i2o, buffertest, buffertrial, kernelbuffer
    )

Evaluate the kernel at `buffertest` and `buffertrial` using `kernelbuffer` as scratch.
Write it at `tensorindex`, a `CartesianIndex`, in `toeplitztensor` or its component tensors.
"""
function computetoeplitzkernel!(
    toeplitztensor,
    tensorindex,
    operator,
    i2o::ScalarI2OTranslation,
    buffertest,
    buffertrial,
    ::Any,
)
    toeplitztensor[tensorindex] = kernel(operator, i2o, buffertest, buffertrial)
    return toeplitztensor
end

# Vector and dyadic kernels have 3 or 3-by-3 field components.
# Their tensor index rank still follows the spatial grid.
function computetoeplitzkernel!(
    toeplitztensor,
    tensorindex,
    operator,
    i2o::Union{∇Green3DTranslation,DyadicGreen3DTranslation,∇∇Green3DTranslation},
    buffertest,
    buffertrial,
    kernelbuffer,
)
    kernel!(kernelbuffer, operator, i2o, buffertest, buffertrial)
    for i in eachindex(kernelbuffer)
        toeplitztensor[i][tensorindex] = kernelbuffer[i]
    end
    return toeplitztensor
end

function computetoeplitzkernel!(
    toeplitztensor,
    tensorindex,
    operator,
    i2o::Union{∇CrossGreen3DTranslation},
    buffertest,
    buffertrial,
    kernelbuffer,
)
    kernel!(kernelbuffer, operator, i2o, buffertest, buffertrial)
    toeplitztensor[1, 1][tensorindex] = 0
    toeplitztensor[1, 2][tensorindex] = -kernelbuffer[3]
    toeplitztensor[1, 3][tensorindex] = kernelbuffer[2]
    toeplitztensor[2, 1][tensorindex] = kernelbuffer[3]
    toeplitztensor[2, 2][tensorindex] = 0
    toeplitztensor[2, 3][tensorindex] = -kernelbuffer[1]
    toeplitztensor[3, 1][tensorindex] = -kernelbuffer[2]
    toeplitztensor[3, 2][tensorindex] = kernelbuffer[1]
    toeplitztensor[3, 3][tensorindex] = 0
    return toeplitztensor
end

"""
    function storetoeplitz(fftplan, toeplitztensors)

Return a Toeplitz matrix representation of the tensors using `fftplan`.
"""
function storetoeplitz(fftplan, toeplitztensors::AbstractArray{<:AbstractArray})
    return map(t -> ToeplitzMatrixRepresentation(fftplan * t), toeplitztensors)
end

function storetoeplitz(fftplan, toeplitztensor)
    return ToeplitzMatrixRepresentation(fftplan * toeplitztensor)
end
