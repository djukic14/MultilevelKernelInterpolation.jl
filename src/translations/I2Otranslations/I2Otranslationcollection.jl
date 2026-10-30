"""
    struct KernelTranslator{O,P,T,D} <: AbstractKernelTranslator

Build dense I2O matrices from an operator, level polynomials, a tree, and its translating plan.
"""
struct KernelTranslator{O,P,T,D} <: AbstractKernelTranslator
    operator::O
    polynomialdict::P
    tree::T
    disaggregationplan::D
end

function KernelTranslator(
    operator, treepolynomials::A, tree, disaggregationplan
) where {A<:AbstractArray{<:AbstractTensorProductPolynomial}}
    return KernelTranslator(
        operator,
        leveltopolynomialdict(treepolynomials, levels(disaggregationplan)),
        tree,
        disaggregationplan,
    )
end

"""
    I2OTranslationCollection(
        operator,
        tree,
        polynomials,
        translatingplan::H2Trees.AbstractPlan,
        space;
        verbose=false,
    )

Build dense I2O couplings for `translatingplan` from the level `polynomials`.
`space` selects any basis-specific conversion. Set `verbose=true` to show progress.
"""
function I2OTranslationCollection(
    operator, tree, polynomials, translatingplan::AbstractPlan, space; verbose=false
)
    return I2OTranslationCollection(
        operator, tree, polynomials, translatingplan; verbose=verbose
    )
end

function I2OTranslationCollection(
    operator,
    tree,
    polynomials,
    translatingplan::AbstractPlan,
    testspace,
    trialspace;
    verbose=false,
)
    return I2OTranslationCollection(
        operator, tree, polynomials, translatingplan; verbose=verbose
    )
end

"""
    densei2ocoupling(operator, tree, polynomials, translatingplan[, spaces...]; verbose=false)

Build the read-only dense I2O coupling without allocating workspace scratch.
Uses the same routing as [`I2OTranslationCollection`](@ref), including B-spline conversion.
Plain dense couplings need only an `EmptyChannels` channel.
"""
function densei2ocoupling(
    operator,
    tree,
    polynomials,
    translatingplan::AbstractPlan,
    space;
    verbose=false,
    symmetryreduced=false,
)
    return _densecouplingbuilder(symmetryreduced)(
        operator, tree, polynomials, translatingplan; verbose=verbose
    )
end

function densei2ocoupling(
    operator,
    tree,
    polynomials,
    translatingplan::AbstractPlan,
    testspace,
    trialspace;
    verbose=false,
    symmetryreduced=false,
)
    return _densecouplingbuilder(symmetryreduced)(
        operator, tree, polynomials, translatingplan; verbose=verbose
    )
end

# Share reduced/unreduced routing between the space-taking entry points.
_densecouplingbuilder(symmetryreduced::Bool) =
    symmetryreduced ? symmetryreduceddensei2ocoupling : densei2ocoupling

function densei2ocoupling(
    operator, tree, polynomials, translatingplan::AbstractPlan; verbose=false
)
    return first(
        I2OTranslationCollection(
            operator, tree, polynomials, translatingplan; verbose=verbose
        ),
    )
end

function I2OTranslationCollection(
    operator, tree, polynomials, translatingplan::AbstractPlan; verbose=false
)
    return H2Factory.I2OTranslationCollection(
        tree,
        translatingplan,
        DirectionInvariance(),
        KernelTranslator(operator, polynomials, tree, translatingplan);
        i2otranslationtrait=i2otranslationtrait(i2otranslation(operator)),
        verbose=verbose,
    )
end

"""
    symmetryreduceddensei2ocoupling(operator, tree, polynomials, translatingplan; verbose=false)

Build dense couplings with one matrix per group of lattice directions related by symmetry.
Other directions use grid permutations; reduction adds no approximation.
In 3-D, permutations need `O(b^3)` entries versus `O(b^6)` for a dense matrix.

Supports Galerkin maps with non-B-spline polynomials and scalar or true/pseudo-tensor
3-by-3 kernels. Unsupported families error; non-lattice trees use unreduced couplings.
"""
function symmetryreduceddensei2ocoupling(
    operator, tree, polynomials, translatingplan::AbstractPlan; verbose=false
)
    componentaction = symmetrycomponentaction(i2otranslation(operator))
    componentaction isa H2Factory.VectorField && error(
        "symmetryreduceddensei2ocoupling: $(typeof(operator)) has component action " *
        "$(componentaction), which the dense reduction does not serve. The operators declaring it " *
        "have `trialmoment != testmoment`, so they cannot form a Galerkin map at all, and the " *
        "Petrov dense route is refused separately. Use `DenseI2OStrategy()` (unreduced) or " *
        "`ToeplitzI2OStrategy(; symmetryreduced=true)` for this operator.",
    )

    translator = KernelTranslator(operator, polynomials, tree, translatingplan)

    # Fall back only for non-lattice geometry; propagate all other failures.
    canonical = try
        first(
            H2Factory.I2OTranslationCollection(
                tree,
                translatingplan,
                H2Trees.SymmetryDirectionInvariancePerLevel(H2Trees.FullLatticeSymmetry()),
                translator;
                i2otranslationtrait=H2Factory.DenseTranslation(),
                verbose=verbose,
            ),
        )
    catch e
        e isa H2Trees.NonLatticeTranslationError || rethrow()
        verbose &&
            @info "MultilevelKernelInterpolation: displacements are not lattice aligned; building the unreduced coupling."
        return densei2ocoupling(
            operator, tree, polynomials, translatingplan; verbose=verbose
        )
    end

    _assertlatticegrid(translator, translatingplan)

    gridsizefor(node) = size(LinearIndices(polynomial(translator, node)))
    return H2Factory.symmetryreduceddensecollection(
        canonical, gridsizefor; componentaction=componentaction
    )
end

# Check that the actual grid is closed under lattice symmetry. An index permutation alone
# cannot establish this for an asymmetric node family. Cost: |group| times node count per level.
function _assertlatticegrid(translator, translatingplan)
    # Choose the lattice group from the tree dimension.
    group = H2Trees.symmetrygroup(
        H2Trees.FullLatticeSymmetry(), Val(length(eltype(translator.tree)))
    )

    for level in levels(translatingplan)
        nodes = H2Trees.receivingnodes(translatingplan, level)
        isempty(nodes) && continue
        node = first(nodes)

        reference = polynomial(translator, node)
        gridsize = size(LinearIndices(reference))
        shifted = shiftedpolynomial(translator.tree, reference, node)
        center = H2Trees.center(translator.tree, node)

        # Sampling buffers follow the grid's spatial dimension.
        point = zeros(eltype(shifted), length(gridsize))
        points = map(1:prod(gridsize)) do i
            getsamplingpoint!(point, shifted, i)
            return Tuple(point .- center)
        end

        offender = latticeclosuredefect(points, gridsize, group)
        offender === nothing || error(
            "symmetryreduceddensei2ocoupling: the interpolation grid at level $(level) is NOT " *
            "closed under the lattice symmetry $(offender) about its box centre, so " *
            "`latticegridpermutation` would return a plausible permutation of the wrong grid and " *
            "the reduction would produce a wrong field rather than an error. This needs a node " *
            "family symmetric about the box centre on every axis (equidistant and Chebyshev both " *
            "are). Use `DenseI2OStrategy()` for this family.",
        )
    end
    return nothing
end

"""
    latticeclosuredefect(points, gridsize, group) -> symmetry or nothing

Return the first symmetry that does not preserve `points`, or `nothing` if all do.
Points are relative to the box center, in the linear order defined by `gridsize`.
Closure requires `points[π[i]] ≈ S * points[i]` for the lattice-grid permutation `π`.
"""
function latticeclosuredefect(points, gridsize, group)
    # Scale the tolerance to the grid extent so tiny and large boxes behave consistently.
    tolerance = 1e-9 * maximum(p -> maximum(abs, p), points)

    for symmetry in group
        permutation = H2Factory.latticegridpermutation(gridsize, symmetry)
        for i in eachindex(points)
            image = H2Trees.applysymmetry(symmetry, points[i])
            maximum(abs, image .- points[permutation[i]]) <= tolerance || return symmetry
        end
    end
    return nothing
end

"""
    calculateI2Otranslation!(
        translator, treetrait, receivingnode::Int, translatingnode::Int
    )

Build the I2O translation between `receivingnode` and `translatingnode`.
`treetrait` selects single-tree or block-tree geometry.
"""
function calculateI2Otranslation!(
    translator, ::isTwoNTree, receivingnode::Int, translatingnode::Int, direction
)
    receiving = shiftedpolynomial(
        translator.tree, polynomial(translator, receivingnode), receivingnode
    )
    return calculateI2Otranslation!(
        translator,
        i2otranslation(translator.operator),
        receiving,
        translatingpolynomialat(
            translator.tree,
            polynomial(translator, translatingnode),
            translatingnode,
            H2Trees.center(translator.tree, receivingnode) .- direction,
        ),
    )
end

function calculateI2Otranslation!(
    translator, ::isBlockTree, receivingnode::Int, translatingnode::Int, direction
)
    return calculateI2Otranslation!(
        translator,
        i2otranslation(translator.operator),
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
    )
end

function calculateI2Otranslation!(
    translator, i2o::I2OTranslation, receivingpolynomial, translatingpolynomial
)
    tf = _tf(translator, receivingpolynomial, i2o)
    return translation!(
        tf, translator.operator, i2o, receivingpolynomial, translatingpolynomial
    )
end

_tfcontainershape(::∇Green3DTranslation) = (3,)
function _tfcontainershape(
    ::Union{∇CrossGreen3DTranslation,DyadicGreen3DTranslation,∇∇Green3DTranslation}
)
    return (3, 3)
end

function _tf(
    translator,
    polynomial,
    i2o::Union{
        ∇Green3DTranslation,
        ∇CrossGreen3DTranslation,
        DyadicGreen3DTranslation,
        ∇∇Green3DTranslation,
    },
)
    numberrows = prod(size(LinearIndices(polynomial)))
    tf = eltype(translator)(undef, _tfcontainershape(i2o)...)
    for i in eachindex(tf)
        tf[i] = eltype(eltype(translator))(undef, numberrows, numberrows)
    end
    return tf
end

# All scalar kernels store one matrix, independent of spatial dimension.
function _tf(translator, polynomial, ::ScalarI2OTranslation)
    numberrows = prod(size(LinearIndices(polynomial)))
    return eltype(translator)(undef, numberrows, numberrows)
end

# Rectangular couplings have test rows and trial columns.
function _tf(translator, receivingpolynomial, translatingpolynomial, ::ScalarI2OTranslation)
    numberrows = prod(size(LinearIndices(receivingpolynomial)))
    numbercols = prod(size(LinearIndices(translatingpolynomial)))
    return eltype(translator)(undef, numberrows, numbercols)
end

function _tf(
    translator,
    receivingpolynomial,
    translatingpolynomial,
    i2o::Union{
        ∇Green3DTranslation,
        ∇CrossGreen3DTranslation,
        DyadicGreen3DTranslation,
        ∇∇Green3DTranslation,
    },
)
    numberrows = prod(size(LinearIndices(receivingpolynomial)))
    numbercols = prod(size(LinearIndices(translatingpolynomial)))
    tf = eltype(translator)(undef, _tfcontainershape(i2o)...)
    for i in eachindex(tf)
        tf[i] = eltype(eltype(translator))(undef, numberrows, numbercols)
    end
    return tf
end
