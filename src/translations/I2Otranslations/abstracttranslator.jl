
"""
    abstract type AbstractKernelTranslator

Base type for I2O translation builders.
"""
abstract type AbstractKernelTranslator end

"""
    leveltopolynomialdict(treepolynomials, disaggregationlevels)

Return a `level => polynomial` dictionary restricted to `disaggregationlevels`.
Used by dense, B-spline, and Toeplitz translators.
"""
function leveltopolynomialdict(treepolynomials, disaggregationlevels)
    return Dict(zip(disaggregationlevels, treepolynomials[disaggregationlevels]))
end

"""
    SidedPolynomialDicts(test, trial)

Keep separate test and trial `level => polynomial` dictionaries for a Petrov map.
Each side's space determines its protrusion and polynomial family.
I2O uses test polynomials at receiving nodes and trial polynomials at translating nodes.
A Galerkin translator needs only one dictionary.
"""
struct SidedPolynomialDicts{T,R}
    test::T
    trial::R
end

# Restrict each side separately; indexing the pair would also require matching grids.
function leveltopolynomialdict(treepolynomials::SidedPolynomialDicts, disaggregationlevels)
    return SidedPolynomialDicts(
        Dict(level => treepolynomials.test[level] for level in disaggregationlevels),
        Dict(level => treepolynomials.trial[level] for level in disaggregationlevels),
    )
end

# Iterate over polynomial values from both sides for whole-family predicates.
function Base.iterate(d::SidedPolynomialDicts, state...)
    return iterate(Iterators.flatten((Base.values(d.test), Base.values(d.trial))), state...)
end
Base.length(d::SidedPolynomialDicts) = length(d.test) + length(d.trial)
# Iteration can yield either side's polynomial type.
function Base.eltype(d::SidedPolynomialDicts)
    return Union{eltype(Base.values(d.test)),eltype(Base.values(d.trial))}
end

"""
    toeplitzcompatible(testpolynomial, trialpolynomial)

Check whether both polynomials use the same grid for Toeplitz coupling.
Compare type, dimension, node counts, and node positions along every axis.
If sampling matrices exist, both must have one and the matrices must agree.

Equal sizes alone do not establish equal spacing. Structural `==` is also insufficient:
some polynomial types compare unequal even when their grids match.
"""
function toeplitzcompatible(testpolynomial, trialpolynomial)
    testpolynomial === trialpolynomial && return true
    typeof(testpolynomial) === typeof(trialpolynomial) || return false
    dims(testpolynomial) == dims(trialpolynomial) || return false
    size(LinearIndices(testpolynomial)) == size(LinearIndices(trialpolynomial)) ||
        return false

    for dim in 1:dims(testpolynomial)
        range(testpolynomial, dim) == range(trialpolynomial, dim) || return false
    end

    hastestsampling = applicable(interpolationmatrix, testpolynomial)
    hastestsampling == applicable(interpolationmatrix, trialpolynomial) || return false

    if hastestsampling
        interpolationmatrix(testpolynomial) == interpolationmatrix(trialpolynomial) ||
            return false
    end

    return true
end

# Shared-level indexing requires matching grids. Petrov construction reconciles
# protrusion requirements first; this check validates the resulting grids.
function Base.getindex(d::SidedPolynomialDicts, level)
    testpolynomial = d.test[level]
    trialpolynomial = d.trial[level]
    toeplitzcompatible(testpolynomial, trialpolynomial) && return testpolynomial
    return error(
        "PetrovGalerkinKernelInterpolation with Toeplitz or B-spline I2O is not supported at level $(level): the " *
        "test and trial sides carry DIFFERENT polynomial grids there " *
        "(test $(typeof(testpolynomial)) with $(size(LinearIndices(testpolynomial))) nodes, " *
        "trial $(typeof(trialpolynomial)) with $(size(LinearIndices(trialpolynomial))) nodes), " *
        "so the cross-side translation is not Toeplitz-compatible -- the Toeplitz structure depends " *
        "only on node-position differences and therefore needs ONE shared grid. Levels where the " *
        "two sides agree are supported. Use the dense I2O strategy " *
        "(`TranslatorStrategy(; i2ostrategy=DenseI2OStrategy())`) for this configuration.",
    )
end

# Apply the same compatibility check to every requested level.
function Base.getindex(d::SidedPolynomialDicts, levels::AbstractVector)
    return [d[level] for level in levels]
end

# Plain dict (Galerkin, single tree): the same family serves both sides.
_testpolynomialdict(d) = d
_trialpolynomialdict(d) = d
_testpolynomialdict(d::SidedPolynomialDicts) = d.test
_trialpolynomialdict(d::SidedPolynomialDicts) = d.trial

function polynomial(translator::AbstractKernelTranslator, node)
    # A single tree uses the same family on both sides.
    return _testpolynomialdict(translator.polynomialdict)[level(translator.tree, node)]
end

function testpolynomial(translator::AbstractKernelTranslator, node)
    return _testpolynomialdict(translator.polynomialdict)[level(
        testtree(translator.tree), node
    )]
end

function trialpolynomial(translator::AbstractKernelTranslator, node)
    return _trialpolynomialdict(translator.polynomialdict)[level(
        trialtree(translator.tree), node
    )]
end

# Use the supplied direction: symmetry reduction may pass a canonical direction
# different from the representative node pair's displacement.
function (translator::AbstractKernelTranslator)(
    receivingnode::Int, translatingnode::Int, direction
)
    return calculateI2Otranslation!(
        translator, treetrait(translator.tree), receivingnode, translatingnode, direction
    )
end

function Base.eltype(translator::AbstractKernelTranslator)
    return eltypei2otranslation(i2otranslation(translator.operator), translator)
end
