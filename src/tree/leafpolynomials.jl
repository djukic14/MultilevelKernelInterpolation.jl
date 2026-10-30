"""
    leafpolynomials(tree, originalpolynomials; kwargs...)

Generate a vector of `ScaledShiftedPolynomial`s, each fitted to the box corresponding to a
leaf node in `tree`.
"""
function leafpolynomials(tree, originalpolynomials; kwargs...)
    return leafpolynomials(tree, originalpolynomials, treetrait(tree); kwargs...)
end

function leafpolynomials(tree, originalpolynomials, ::isTwoNTree; halfsizefactor=1)
    leaves = H2Trees.leaves(tree)
    hs = halfsize(tree, leaves[1]) * halfsizefactor
    firstshiftedp = ScaledShiftedPolynomial(
        originalpolynomials[end], hs, H2Trees.center(tree, leaves[1]) .- hs
    )
    leafpolynomials = Vector{typeof(firstshiftedp)}(undef, length(leaves))

    for (i, leaf) in enumerate(leaves)
        leafpolynomials[i] = shiftedpolynomial(
            tree,
            originalpolynomials[H2Trees.leveltolevelid(tree, H2Trees.level(tree, leaf))],
            leaf;
            halfsizefactor=halfsizefactor,
        )
    end
    return leafpolynomials
end

function shiftedpolynomial(data::BoxData, polynomial; halfsizefactor=1)
    hs = halfsize(data)
    return ScaledShiftedPolynomial(
        polynomial, hs * 2 * halfsizefactor, H2Trees.center(data) .- hs
    )
end

function shiftedpolynomial(tree, polynomial, nodeid; halfsizefactor=1)
    return shiftedpolynomial(data(tree, nodeid), polynomial; halfsizefactor=halfsizefactor)
end

# Place the translating grid at center(receiving) - direction, keeping its own box size.
# Under symmetry reduction, direction is canonical and may differ from the node-pair displacement.
function translatingpolynomialat(data::BoxData, polynomial, center; halfsizefactor=1)
    hs = halfsize(data)
    return ScaledShiftedPolynomial(polynomial, hs * 2 * halfsizefactor, center .- hs)
end

function translatingpolynomialat(tree, polynomial, nodeid, center; halfsizefactor=1)
    return translatingpolynomialat(
        data(tree, nodeid), polynomial, center; halfsizefactor=halfsizefactor
    )
end
