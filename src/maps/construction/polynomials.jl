"""
    assemblelevelprotrusions(tree, space, polynomialstrategy)

Return the protrusion requirements for `space`, aligned with `levels(tree)`.
Uses `H2Trees.maxprotrusion` and `H2Trees.BEASTProtrusionFunctor`.

Petrov Toeplitz construction reconciles these requirements before building the two
polynomial hierarchies. This preserves the nesting needed by level transfers.
"""
function assemblelevelprotrusions(tree, space, ::AbstractPolynomialStrategy)
    return H2Trees.maxprotrusion(
        tree; computeprotrusion=H2Trees.BEASTProtrusionFunctor(space)
    )
end

"""
    assembletreepolynomials(tree, space, polynomialstrategy; protrusion)

Build the level and leaf polynomials for `space` on `tree`.
Return `(treepolynomials, leafpolynomials)`.
`protrusion` defaults to `assemblelevelprotrusions`; Petrov construction can supply
reconciled requirements instead.
"""
function assembletreepolynomials(
    tree,
    space,
    polynomialstrategy::AbstractPolynomialStrategy;
    protrusion=assemblelevelprotrusions(tree, space, polynomialstrategy),
)
    # Protrusions must align with levels; their maximum seeds the hierarchy.
    length(protrusion) == length(levels(tree)) || error(
        "protrusion vector (length $(length(protrusion))) must be aligned with levels(tree) " *
        "(length $(length(levels(tree)))).",
    )

    treepolynomials = levelpolynomials(
        polynomial(polynomialstrategy),
        levels(tree),
        HFConditionFunctor(tree, hfhalfsize(polynomialstrategy));
        increase=increase(polynomialstrategy),
        protrusion=protrusion,
    )

    treeleafpolynomials = MultilevelKernelInterpolation.leafpolynomials(
        tree, treepolynomials
    )

    return treepolynomials, treeleafpolynomials
end

"""
    assemblemomentcollections(operator, testtree, testleafpolynomials, testspace, trialtree, trialleafpolynomials, trialspace, polynomialstrategy, momentassemblystrategy; verbose=false)

Build and return `(testcollections, trialcollections)`, indexed by leaf.
If the tree, space, and moment operator coincide on both sides, assemble once and
return the same dictionary twice.
"""
function assemblemomentcollections(
    operator,
    testtree,
    testleafpolynomials,
    testspace,
    trialtree,
    trialleafpolynomials,
    trialspace,
    polynomialstrategy::AbstractPolynomialStrategy,
    momentassemblystrategy::AbstractMomentAssemblyStrategy;
    verbose=false,
)
    # Qualify accessors whose names are also used for local variables.
    polynomial = MultilevelKernelInterpolation.polynomial(polynomialstrategy)
    quadstrat = momentquadstrategy(momentassemblystrategy)
    MomentAssembler = MultilevelKernelInterpolation.MomentAssembler(momentassemblystrategy)

    # These moment operators are singleton structs, so equality identifies the same operator.
    if testtree === trialtree &&
        testspace === trialspace &&
        testmoment(operator, polynomial) == trialmoment(operator, polynomial)
        momentcollections = Dictionary(
            leaves(testtree),
            TrialMomentCollections(
                operator,
                testtree,
                testleafpolynomials,
                testspace;
                quadstrat=quadstrat,
                MomentAssembler=MomentAssembler,
                verbose=verbose,
            ),
        )
        return momentcollections, momentcollections
    end

    testmomentcollections = Dictionary(
        leaves(testtree),
        TestMomentCollections(
            operator,
            testtree,
            testleafpolynomials,
            testspace;
            quadstrat=quadstrat,
            MomentAssembler=MomentAssembler,
            verbose=verbose,
        ),
    )
    trialmomentcollections = Dictionary(
        leaves(trialtree),
        TrialMomentCollections(
            operator,
            trialtree,
            trialleafpolynomials,
            trialspace;
            quadstrat=quadstrat,
            MomentAssembler=MomentAssembler,
            verbose=verbose,
        ),
    )
    return testmomentcollections, trialmomentcollections
end
