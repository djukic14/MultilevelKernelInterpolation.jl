const POLYNOMIALRECOM = BarycentricLagrangePolynomialChebyshev3D2(6)
const HFHALFSIZERECOM = 0.25
const MOMENTSQUADSTRATEGYRECOM = BEAST.SingleNumQStrat(7)

struct MomentSizeFunctor{P}
    polynomialdict::P
end

function (f::MomentSizeFunctor)(::Any, level)
    return numpolynomials(f.polynomialdict[level])
end

# Size each side from its own polynomial family. Buffered B-spline counts can differ.
# Use named storagemomentsplan arguments to keep moment types and sizes paired correctly.
"""
    moments(
        ::T,
        testoperator,
        trialoperator,
        tree,
        polynomials,
        testaggregationplan,
        trialaggregationplan,
        Moment,
    ) where {T}

Allocate pooled moments for aggregation and disaggregation. The scalar argument fixes
`T`; `testoperator` and `trialoperator` fix the component counts.

Pass separate test/trial polynomial families when their per-level sizes differ, or one
family to use on both sides. Moments that are never needed together may share storage.
"""
function moments(
    ::T,
    testoperator,
    trialoperator,
    tree,
    testpolynomials,
    trialpolynomials,
    testaggregationplan,
    trialaggregationplan,
    Moment,
) where {T}
    return storagemomentsplan(
        testaggregationplan,
        trialaggregationplan,
        tree;
        TestMomentType=(level) -> Moment{dims(testoperator),T},
        TrialMomentType=(level) -> Moment{dims(trialoperator),T},
        testmomentsize=MomentSizeFunctor(testpolynomials),
        trialmomentsize=MomentSizeFunctor(trialpolynomials),
    )
end

# One polynomial family serves both sides.
function moments(
    γ::T,
    testoperator,
    trialoperator,
    tree,
    polynomials,
    testaggregationplan,
    trialaggregationplan,
    Moment,
) where {T}
    return moments(
        γ,
        testoperator,
        trialoperator,
        tree,
        polynomials,
        polynomials,
        testaggregationplan,
        trialaggregationplan,
        Moment,
    )
end

#TODO: functors
function moments(
    ::T,
    testoperator,
    trialoperator,
    tree,
    testpolynomials,
    trialpolynomials,
    testaggregationplan,
    trialaggregationplan,
    ::Type{M},
) where {T,M<:ToeplitzPolynomialMoment}
    return storagemomentsplan(
        testaggregationplan,
        trialaggregationplan,
        tree;
        TestMomentType=(level) -> M{dims(testoperator),T},
        TrialMomentType=(level) -> M{dims(trialoperator),T},
        testmomentsize=(tree, level) -> (
            numpolynomials(testpolynomials[level]),
            H2Factory.standardtoeplitzsize(size(LinearIndices(testpolynomials[level]))),
        ),
        trialmomentsize=(tree, level) -> (
            numpolynomials(trialpolynomials[level]),
            H2Factory.standardtoeplitzsize(size(LinearIndices(trialpolynomials[level]))),
        ),
    )
end

#TODO: functors
function moments(
    ::T,
    testoperator,
    trialoperator,
    tree,
    testpolynomials,
    trialpolynomials,
    testaggregationplan,
    trialaggregationplan,
    ::Type{M},
) where {T,M<:BsplineDensePolynomialMoment}
    return storagemomentsplan(
        testaggregationplan,
        trialaggregationplan,
        tree;
        TestMomentType=(level) -> M{dims(testoperator),T},
        TrialMomentType=(level) -> M{dims(trialoperator),T},
        testmomentsize=(tree, level) -> (
            numpolynomials(testpolynomials[level]),
            size(LinearIndices(testpolynomials[level])),
        ),
        trialmomentsize=(tree, level) -> (
            numpolynomials(trialpolynomials[level]),
            size(LinearIndices(trialpolynomials[level])),
        ),
    )
end

"""
    defaultmomenttype(i2ostrategy, polynomial)

Choose a moment type for the I2O strategy and polynomial.
Dense B-spline couplings need `BsplineDensePolynomialMoment` for raw and reduced storage.
Other cases use `defaultmomenttype(i2ostrategy)`.
"""
function defaultmomenttype(i2ostrategy, ::Any)
    return defaultmomenttype(i2ostrategy)
end

function defaultmomenttype(::DenseI2OStrategy, ::AbstractBspline)
    return BsplineDensePolynomialMoment
end

# The strategy type is available here, before the translation implementation is included.
function defaultmomenttype(::ToeplitzI2OStrategy)
    return ToeplitzPolynomialMoment
end

"""
    assertpolynomialdimension(polynomial, tree)

Throw if the polynomial and tree have different spatial dimensions.
The default polynomial is 3-D; choose a matching basis for a 2-D tree.
"""
function assertpolynomialdimension(polynomial, tree)
    treedimension = length(H2Trees.center(tree))
    dims(polynomial) == treedimension && return nothing

    return error(
        "Dimension mismatch: the tree is $(treedimension)-D but the interpolating polynomial " *
        "$(typeof(polynomial)) is $(dims(polynomial))-D. Pass a polynomial of matching dimension " *
        "through the polynomial strategy, e.g. " *
        "`PolynomialStrategy(operator; polynomial=BarycentricLagrangePolynomialChebyshev{$(treedimension),Float64}(6))` " *
        "-- the recommended default is 3-D.",
    )
end

# Restrict wavelength dispatch to these operator families; read gamma through operatorgamma.
function λ(operator::BEAST.MaxwellOperator3D)
    return 2π / abs(operatorgamma(operator))
end

function λ(operator::BEAST.HelmholtzOperator2D)
    return 2π / abs(operatorgamma(operator))
end

"""
    checkprotrusionsafety(tree, treepolynomials, plan)

Check the extended sampling boxes for every I2O pair in `plan`.
Throw if two boxes overlap or touch: evaluating the kernel there can produce singular
couplings or a native crash. Warn when the gap is positive but smaller than the lesser
raw box half-size, the empirical margin in arXiv:2606.13482, Section VI-B.

Uses each node's actual center, half-size, and realized polynomial extension, without
assuming uniform tree geometry or a particular near/far criterion.
This form uses one tree and polynomial family for both sides.
"""
function checkprotrusionsafety(tree, treepolynomials, plan)
    return checkprotrusionsafety(tree, treepolynomials, tree, treepolynomials, plan)
end

"""
    checkprotrusionsafety(receivingtree, receivingpolynomials, translatingtree, translatingpolynomials, plan)

Check I2O sampling boxes on separate receiving and translating trees.
Each side uses its own polynomial extension, which can differ with the function space.
The warning margin is the smaller of the two raw box half-sizes.
"""
function checkprotrusionsafety(
    receivingtree, receivingpolynomials, translatingtree, translatingpolynomials, plan
)
    for level in H2Trees.levels(plan)
        # Some plan levels have no I2O pairs.
        receivingnodes = H2Trees.receivingnodes(plan, level)
        isempty(receivingnodes) && continue

        for receivingnode in receivingnodes
            receivingcenter = H2Trees.center(receivingtree, receivingnode)
            # Each side's space determines its own extension.
            receivinghalfsize = H2Trees.halfsize(receivingtree, receivingnode)
            receivingextension = _protrusionextension(
                receivingtree, receivingpolynomials, receivingnode
            )

            for translatingnode in H2Trees.translatingnodes(plan, receivingnode, level)
                translatingcenter = H2Trees.center(translatingtree, translatingnode)
                translatinghalfsize = H2Trees.halfsize(translatingtree, translatingnode)
                translatingextension = _protrusionextension(
                    translatingtree, translatingpolynomials, translatingnode
                )
                # Enlarge each box before measuring the gap, including diagonal separations.
                remaining = _protrusionboxgap(
                    receivingcenter,
                    receivinghalfsize + receivingextension,
                    translatingcenter,
                    translatinghalfsize + translatingextension,
                )
                remaining <= 0 && error(
                    "Buffer/scaled-shifted-extended sampling domains of translating node " *
                    "pair (receiving=$receivingnode, translating=$translatingnode) at level " *
                    "$level overlap: remaining gap between the protrusion-extended boxes " *
                    "(receiving halfsize $receivinghalfsize + extension $receivingextension; " *
                    "translating halfsize $translatinghalfsize + extension " *
                    "$translatingextension) is $remaining <= 0. This can produce near-singular " *
                    "I2O translation matrices and, in extreme cases, a process crash. Use a " *
                    "finer tree (larger `minvalues` split threshold) or a coarser polynomial " *
                    "(larger buffer/order margin) to reduce relative protrusion.",
                )

                # The half-box margin is an empirical recommendation, not a hard bound.
                halfsize = min(receivinghalfsize, translatinghalfsize)
                0 < remaining < halfsize && @warn(
                    "Buffer/scaled-shifted-extended sampling domains of translating node " *
                        "pair (receiving=$receivingnode, translating=$translatingnode) at level " *
                        "$level are closer than the recommended half-a-box margin (remaining gap " *
                        "$remaining < $halfsize; see arXiv:2606.13482, Section VI-B). Not an " *
                        "error, the kernel is not yet near-singular, but accuracy/robustness " *
                        "may be degraded relative to the paper's own validated configuration."
                )
            end
        end
    end
    return nothing
end

function _protrusionextension(tree, treepolynomials, node)
    poly = treepolynomials[H2Trees.leveltolevelid(tree, H2Trees.level(tree, node))]
    return realizedprotrusion(poly) * 2 * H2Trees.halfsize(tree, node)
end

function _protrusionboxgap(centera, halfsizea, centerb, halfsizeb)
    return sqrt(
        sum(
            max(zero(halfsizea), abs(centera[d] - centerb[d]) - halfsizea - halfsizeb)^2 for
            d in eachindex(centera)
        ),
    )
end
