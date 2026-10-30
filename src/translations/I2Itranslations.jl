"""
    abstract type AbstractSectorLevelTranslator

Base type for O2O reference caches indexed by sector and level.
Traversal uses the I2I collection through H2Factory's level-transfer protocol instead.
"""
abstract type AbstractSectorLevelTranslator end

"""
    function ndimi2itranslations(poly, parentpoly)

Return the tensor-product child-to-parent matrices, one per sector (`2^D`),
where `D = dims(poly)`.

Check the underlying 1-D matrices with [`testonedimensionali2i`](@ref).
B-spline knot insertion is exact even when basis orders differ. Lagrange interpolation
is exact for a fixed order, but need not be when the order increases.
"""
function ndimi2itranslations(poly, parentpoly)
    res0, res1 = onedimi2itranslations(poly, parentpoly)
    D = dims(poly)

    tfmatrices = []

    for sector in 0:(2 ^ D - 1)
        sectormatrices = sectorordering(res0, res1, sector, D)
        push!(tfmatrices, computend(sectormatrices...))
    end

    return convert(Vector{typeof(tfmatrices[1])}, tfmatrices)
end

"""
    function I2ITranslator(
        polynomials::V, levels
    ) where {A<:AbstractTensorProductPolynomial,V<:AbstractArray{A}}

Build a read-only `H2Factory.I2ITranslationCollection` with the `DenseI2I` trait.
`polynomials` and `levels` are matching sequences. Each transfer is a combined
Kronecker matrix; this trait needs no workspace scratch.
"""
function I2ITranslator(
    polynomials::V, levels
) where {A<:AbstractTensorProductPolynomial,V<:AbstractArray{A}}
    @assert length(polynomials) == length(levels)
    selector = H2Factory.SectorSelector{dims(polynomials[begin])}()
    if length(polynomials) <= 1
        return H2Factory.I2ITranslationCollection(nothing, selector, H2Factory.DenseI2I())
    end

    i2itranslations = []

    for i in eachindex(polynomials)
        i == lastindex(polynomials) && continue
        push!(i2itranslations, ndimi2itranslations(polynomials[i + 1], polynomials[i]))
    end
    i2itranslations = convert(Vector{typeof(i2itranslations[1])}, i2itranslations)
    return H2Factory.I2ITranslationCollection(
        Dict(zip(levels[(begin + 1):end], i2itranslations)), selector, H2Factory.DenseI2I()
    )
end

"""
    computend(P::Vararg{Matrix,D}) where {D}
    computend(P::Vararg{SparseMatrixCSC,D}) where {D}

Form the tensor product of `D` one-dimensional matrices, with axis 1 varying fastest.
Generated loops support any positive `D` for dense or sparse matrices.
The zero-argument method rejects an empty product and resolves the two vararg methods' ambiguity.
"""
computend() = throw(ArgumentError("computend requires at least one matrix"))

@generated function computend(P::Vararg{Matrix,D}) where {D}
    isyms = [Symbol(:i_, d) for d in 1:D]
    jsyms = [Symbol(:j_, d) for d in 1:D]
    psyms = [Symbol(:p_, d) for d in 1:D]

    body = quote
        Pout[$(isyms...), $(jsyms...)] = $(Expr(:call, :*, psyms...))
    end
    for d in 1:D
        body = quote
            for $(jsyms[d]) in axes(P[$d], 2)
                for $(isyms[d]) in axes(P[$d], 1)
                    $(psyms[d]) = P[$d][$(isyms[d]), $(jsyms[d])]
                    iszero($(psyms[d])) && continue
                    $body
                end
            end
        end
    end

    sizeexpr = Expr(
        :tuple, [:(size(P[$d], 1)) for d in 1:D]..., [:(size(P[$d], 2)) for d in 1:D]...
    )
    return quote
        Pout = zeros(promote_type(map(eltype, P)...), $sizeexpr)
        $body
        rowlen = prod(size(Pout)[1:($D)])
        collen = prod(size(Pout)[($D + 1):end])
        return reshape(Pout, rowlen, collen)
    end
end

@generated function computend(P::Vararg{SparseMatrixCSC,D}) where {D}
    isyms = [Symbol(:i_, d) for d in 1:D]
    jsyms = [Symbol(:j_, d) for d in 1:D]
    rawisyms = [Symbol(:_i_, d) for d in 1:D]
    psyms = [Symbol(:p_, d) for d in 1:D]
    rowssyms = [Symbol(:rows_, d) for d in 1:D]
    valssyms = [Symbol(:values_, d) for d in 1:D]

    body = quote
        I[i] = is[$(isyms...)]
        J[i] = js[$(jsyms...)]
        V[i] = $(Expr(:call, :*, psyms...))
        i += 1
    end
    for d in 1:D
        body = quote
            for $(jsyms[d]) in axes(P[$d], 2)
                for $(rawisyms[d]) in nzrange(P[$d], $(jsyms[d]))
                    $(isyms[d]) = $(rowssyms[d])[$(rawisyms[d])]
                    $(psyms[d]) = $(valssyms[d])[$(rawisyms[d])]
                    $body
                end
            end
        end
    end

    rowsizeexpr = Expr(:tuple, [:(size(P[$d], 1)) for d in 1:D]...)
    colsizeexpr = Expr(:tuple, [:(size(P[$d], 2)) for d in 1:D]...)
    nnzexpr = Expr(:call, :*, [:(nnz(P[$d])) for d in 1:D]...)
    rowvalsassign = [:($(rowssyms[d]) = rowvals(P[$d])) for d in 1:D]
    nonzerosassign = [:($(valssyms[d]) = nonzeros(P[$d])) for d in 1:D]

    return quote
        is = LinearIndices($rowsizeexpr)
        js = LinearIndices($colsizeexpr)
        $(rowvalsassign...)
        $(nonzerosassign...)
        _nnz = $nnzexpr
        I = zeros(Int, _nnz)
        J = zeros(Int, _nnz)
        V = zeros(promote_type(map(eltype, P)...), _nnz)
        i = 1
        $body
        return sparse(I, J, V, length(is), length(js))
    end
end

function sectorordering(array0::T, array1::T, sectorID::Int, dim::Int) where {T}
    bits = digits(sectorID; base=2, pad=dim)
    return T[ifelse(Bool(bits[i]), array1, array0) for i in 1:dim]
end

"""
    testonedimensionali2i(poly, parentpoly, i2i0, i2i1; n=numpolynomials(poly)[1]*10, verbose=false)

Check a pair of 1-D child-to-parent transfer matrices by evaluating both bases directly.
Sample each child interval at `n` points and compare the transferred values with the
parent basis at tolerance `1e-10`. Return `true` or `false`.

This checks `onedimi2itranslations` without moments, trees, or a mesh. Lagrange transfers
with increasing order need not pass; see [`ndimi2itranslations`](@ref).
"""
function testonedimensionali2i(
    poly,
    parentpoly,
    i2i0,
    i2i1;
    n=MultilevelKernelInterpolation.numpolynomials(poly)[1] * 10,
    verbose=false,
)
    evaluatedparent = zeros(MultilevelKernelInterpolation.numpolynomials(parentpoly)[1], n)
    evaluatedchild = zeros(MultilevelKernelInterpolation.numpolynomials(poly)[1], n)

    childrange1 = MultilevelKernelInterpolation.range(poly, 1) ./ 2
    childrange1[1] += eps(eltype(poly)) * 1e3
    childrange1[end] -= eps(eltype(poly)) * 1e3

    childrange2 = (MultilevelKernelInterpolation.range(poly, 1) .+ 1) ./ 2
    childrange2[1] += eps(eltype(poly)) * 1e3
    childrange2[end] -= eps(eltype(poly)) * 1e3

    evalpoints1 = LinRange(childrange1[1], childrange1[end], n)
    evalpoints2 = LinRange(childrange2[1], childrange2[end], n)

    for i in 1:n
        MultilevelKernelInterpolation.evaluateall1dpolynomial!(
            view(evaluatedparent, :, i), parentpoly, evalpoints1[i], 1; extrapolate=false
        )

        MultilevelKernelInterpolation.evaluateall1dpolynomial!(
            view(evaluatedchild, :, i), poly, evalpoints1[i] * 2, 1; extrapolate=false
        )
    end

    er0 = evaluatedparent - i2i0 * evaluatedchild
    verbose && println("I2I translation error (child 0): ", maximum(abs, er0))

    for i in 1:n
        MultilevelKernelInterpolation.evaluateall1dpolynomial!(
            view(evaluatedparent, :, i), parentpoly, evalpoints2[i], 1; extrapolate=false
        )

        MultilevelKernelInterpolation.evaluateall1dpolynomial!(
            view(evaluatedchild, :, i), poly, evalpoints2[i] * 2 - 1, 1; extrapolate=false
        )
    end

    er1 = evaluatedparent - i2i1 * evaluatedchild
    verbose && println("I2I translation error (child 1): ", maximum(abs, er1))

    success = max(maximum(abs, er0), maximum(abs, er1)) <= 1e-10
    verbose && println("I2I translation ", success ? "successful" : "failed", " for poly!")

    return success
end
