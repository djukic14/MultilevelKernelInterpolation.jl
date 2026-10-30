
"""
    levelpolynomials(poly, levels, cond; increase=MultilevelKernelInterpolation.increase)

Build one polynomial per level, returned in the order of `levels`.
Work from fine to coarse: use `increase` when `cond(level)` is true, except on the first
step, and `noincrease` otherwise. Both receive that level's `protrusion`.
The initial basis uses the maximum protrusion across levels.
"""
function levelpolynomials(
    polynomial,
    levels,
    cond;
    increase=MultilevelKernelInterpolation.increase,
    noincrease=MultilevelKernelInterpolation.noincrease,
    protrusion=zeros(eltype(polynomial), length(levels)),
)
    polynomials = Vector{typeof(polynomial)}(undef, length(levels) + 1)
    polynomials[begin] = noincrease(polynomial; protrusion=maximum(protrusion))
    protrusion = reverse(protrusion)
    for (i, level) in enumerate(reverse(levels))
        if i == firstindex(levels) || !cond(level)
            polynomials[i + 1] = noincrease(polynomials[i]; protrusion=protrusion[i])
        else
            polynomials[i + 1] = increase(polynomials[i]; protrusion=protrusion[i])
        end
    end
    return reverse(polynomials[2:end])
end

"""
    noincrease(poly; kwargs...)

Return `poly` unchanged. The fallback accepts and ignores keywords such as `protrusion`;
bases with an extended domain override it.
"""
function noincrease(poly; kwargs...)
    return poly
end

struct HFConditionFunctor
    level::Int
end

"""
    HFConditionFunctor(tree, halfsize)

Create a predicate that is true through the finest level with box half-size at least
`halfsize`. If no level qualifies, the predicate is false for every tree level.
"""
function HFConditionFunctor(tree, hs)
    treelevels = levels(tree)
    # halfsizes is level-indexed; halfsize(tree, id) takes a node id, not a level.
    levelhalfsizes = halfsizes(tree)
    for i in reverse(eachindex(treelevels))
        if levelhalfsizes[i] >= hs
            return HFConditionFunctor(treelevels[i])
        end
    end
    return HFConditionFunctor(treelevels[begin] - 1)
end

function (f::HFConditionFunctor)(level)
    return level <= f.level
end
