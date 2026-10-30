"""
    BEASTMomentAssembler(operator, space, quadstrategy, dimensionality)

Assemble polynomial moments with BEAST. `operator` selects the moment integrand,
`space` supplies the basis functions, and `quadstrategy` selects quadrature.
`dimensionality` is the number of moment components.
"""
struct BEASTMomentAssembler{O,S,Q,D}
    operator::O
    space::S
    quadstrategy::Q
    dimensionality::Val{D}
    function BEASTMomentAssembler(
        operator::O, space::S, quadstrategy::Q, dimensionality::Int
    ) where {O,S,Q}
        return new{O,S,Q,dimensionality}(operator, space, quadstrategy, Val(dimensionality))
    end
end

# no post processing of moments
function storetrait(::Type{BEASTMomentAssembler{O,S,Q,D}}) where {O,S,Q,D}
    return StoreAsIs
end

"""
    AssembleMomentsFunctor(extrapolate)

Evaluate and integrate leaf polynomials. Set `extrapolate=true` to allow evaluation
outside their reference domain.
"""
struct AssembleMomentsFunctor
    extrapolate::Bool
end

"""
    (assembler::BEASTMomentAssembler{O,S,Q,D})(tree, leafpolynomials; verbose::Bool=false) where {O,S,Q,D}

Assemble moments in parallel, one entry per tree leaf.
Each entry contains `D` dense matrices: polynomial rows and leaf basis-function columns.
Set `verbose=true` to show progress.
"""
function (assembler::BEASTMomentAssembler{O,S,Q,D})(
    tree, leafpolynomials; verbose::Bool=false
) where {O,S,Q,D}
    T = BEAST.scalartype(operator(assembler))
    leaves = H2Trees.leaves(tree)
    Fs = Vector{SVector{D,Matrix{T}}}(undef, length(leaves))
    @tasks for i in eachindex(leaves)
        Fvector = Vector{Matrix{T}}(undef, D)
        for j in eachindex(Fvector)
            Fvector[j] = zeros(
                T, prod(numpolynomials(leafpolynomials[i])), length(values(tree, leaves[i]))
            )
        end
        Fs[i] = SVector{D,Matrix{T}}(Fvector)
    end

    clusters = leafclusters(tree)

    assemblemoments!(
        buildstorer(Fs, assembler, leafpolynomials, tree),
        operator(assembler),
        space(assembler),
        clusters,
        leafpolynomials,
        Val(valuetype(operator(assembler))),
        quadstrategy(assembler),
        AssembleMomentsFunctor(true),
        BufferFunctor();
        verbose=verbose,
    )

    return Fs
end

function operator(assembler::BEASTMomentAssembler)
    return assembler.operator
end

function space(assembler::BEASTMomentAssembler)
    return assembler.space
end

function quadstrategy(assembler::BEASTMomentAssembler)
    return assembler.quadstrategy
end

function (f::AssembleMomentsFunctor)(
    store,
    operator,
    trialspace,
    functionids,
    refspace,
    computationbuffer,
    elements,
    elementids,
    ad,
    qdata,
    qstrat,
    leafpolynomial,
    cluster;
)
    zlocal, polynomialbuffer, integrandbuffer = computationbuffer

    linearindex = LinearIndices(numpolynomials(leafpolynomial))

    for (e, elementid) in enumerate(elementids)
        @inbounds element = elements[elementid]
        qrule = quadrule(operator, refspace, e, element, qdata, qstrat)
        fill!(zlocal, zero(Base.eltype(zlocal)))

        momentlocal!(
            zlocal,
            operator,
            leafpolynomial,
            polynomialbuffer,
            integrandbuffer,
            qrule;
            extrapolate=f.extrapolate,
        )
        assemblelocalconstributions!(
            store, zlocal, linearindex, ad, functionids, elementid, cluster, leafpolynomial
        )
    end
    return nothing
end

function assemblelocalconstributions!(
    store,
    zlocal,
    linearindex,
    assemblydata,
    basisindices,
    elementid,
    cluster,
    ::AbstractTensorProductPolynomial{D},
) where {D}
    for idx in CartesianIndices(axes(zlocal)[2:end])
        p = linearindex[idx]
        for (r, zval) in enumerate(view(zlocal, :, idx))
            for (functionid, b) in assemblydata[elementid, r]
                functionid ∉ basisindices && continue

                store(zval * b, cluster, p, functionid)
            end
        end
    end
end

"""
    momentlocal!(
        zlocal,
        operator,
        polynomial,
        polynomialbuffer,
        quadrule;
        extrapolate::Bool=false,
    )

Accumulate a box's polynomial moments into `zlocal` using `quadrule`.
Reuse `polynomialbuffer` for basis evaluation. `operator` selects the integrand;
`extrapolate` controls evaluation outside the reference domain.
"""

# Generate D nested loops with axis 1 innermost to match array storage.
# Skip whole subtrees when a basis factor is zero (important for compact-support bases).
@generated function _accumulatemoment!(
    zlocal, operator, polynomial, polynomialbuffer, integrandbuffer, F, point, dx, ::Val{D}
) where {D}
    idxsyms = [Symbol(:i_, d) for d in 1:D]
    vsyms = [Symbol(:v_, d) for d in 1:D]

    body = quote
        for r in eachindex(axes(zlocal, 1))
            MultilevelKernelInterpolation.integrand!(
                integrandbuffer,
                operator,
                polynomial,
                $(Expr(:call, :*, vsyms...)),
                F[r],
                point,
            )
            integrandbuffer .*= dx
            zlocal[r, $(idxsyms...)] += integrandbuffer
        end
    end
    for d in 1:D
        body = quote
            for $(idxsyms[d]) in eachindex(polynomialbuffer[$d])
                $(vsyms[d]) = polynomialbuffer[$d][$(idxsyms[d])]
                iszero($(vsyms[d])) && continue
                $body
            end
        end
    end
    return body
end

function momentlocal!(
    zlocal,
    operator::AbstractMomentOperator,
    polynomial::AbstractTensorProductPolynomial{D,T},
    polynomialbuffer,
    integrandbuffer,
    quadrule;
    extrapolate::Bool=false,
) where {D,T}
    for q in quadrule
        point = q.point
        F = q.value
        dx = q.weight
        xcart = CompScienceMeshes.cartesian(point)

        for d in 1:D
            evaluateall1dpolynomial!(
                polynomialbuffer[d], polynomial, xcart[d], d; extrapolate=extrapolate
            )
        end

        _accumulatemoment!(
            zlocal,
            operator,
            polynomial,
            polynomialbuffer,
            integrandbuffer,
            F,
            point,
            dx,
            Val(D),
        )
    end
end

# These gradient operators use 3-D cross products.
# Do not skip a zero basis value: its derivative can still contribute.
@generated function _accumulategradientmoment!(
    zlocal, operator, polynomial, polynomialbuffer, integrandbuffer, F, point, dx, ::Val{D}
) where {D}
    idxsyms = [Symbol(:i_, d) for d in 1:D]
    valsyms = [Symbol(:val_, d) for d in 1:D]
    derivsyms = [Symbol(:deriv_, d) for d in 1:D]

    # ∇polynomial[k] = deriv_k * ∏_{d≠k} val_d
    gradcomponents = [
        Expr(:call, :*, derivsyms[k], (valsyms[d] for d in 1:D if d != k)...) for k in 1:D
    ]

    body = quote
        ∇polynomial = SVector{$D}($(gradcomponents...))
        for r in eachindex(axes(zlocal, 1))
            MultilevelKernelInterpolation.integrand!(
                integrandbuffer, operator, polynomial, ∇polynomial, F[r], point
            )
            integrandbuffer .*= dx
            zlocal[r, $(idxsyms...)] += integrandbuffer
        end
    end
    for d in 1:D
        body = quote
            for $(idxsyms[d]) in eachindex(polynomialbuffer[$d][1])
                $(valsyms[d]) = polynomialbuffer[$d][1][$(idxsyms[d])]
                $(derivsyms[d]) = polynomialbuffer[$d][2][$(idxsyms[d])]
                $body
            end
        end
    end
    return body
end

function momentlocal!(
    zlocal,
    operator::Union{GradLaplaceMoment,NormalGradLaplaceMoment},
    polynomial::AbstractTensorProductPolynomial{D,T},
    polynomialbuffer,
    integrandbuffer,
    qr;
    extrapolate::Bool=false,
) where {D,T}
    for q in qr
        point = q.point
        F = q.value
        dx = q.weight
        xcart = CompScienceMeshes.cartesian(point)

        for d in 1:D
            evaluateall1dpolynomialderivative!(
                polynomialbuffer[d], polynomial, xcart[d], d; extrapolate=extrapolate
            )
        end

        _accumulategradientmoment!(
            zlocal,
            operator,
            polynomial,
            polynomialbuffer,
            integrandbuffer,
            F,
            point,
            dx,
            Val(D),
        )
    end
end
