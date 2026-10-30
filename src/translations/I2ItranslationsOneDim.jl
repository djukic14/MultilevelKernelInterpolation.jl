"""
    I2ITranslatorOneDim(polynomials, levels)

Build a read-only `H2Factory.I2ITranslationCollection` with the `SeparableI2I` trait.
`polynomials` and `levels` are matching sequences. Each transfer stores an `NTuple{D}`
of one-dimensional factors, applied as an `H2Factory.SeparableTransfer`.

For `N` nodes per axis, storage is `O(D*N^2)` rather than the dense product's
`O(N^(2D))`. Select this with `TranslatorStrategy(; i2istrategy=OneDimI2IStrategy())`.
Allocate workspace scratch through `H2Factory.i2ichannels` and `o2ochannels`.
"""
function I2ITranslatorOneDim(
    polynomials::V, levels
) where {A<:AbstractTensorProductPolynomial,V<:AbstractArray{A}}
    @assert length(polynomials) == length(levels)
    selector = H2Factory.SectorSelector{dims(polynomials[begin])}()
    if length(polynomials) <= 1
        return H2Factory.I2ITranslationCollection(
            nothing, selector, H2Factory.SeparableI2I()
        )
    end

    i2itranslations = []

    for i in eachindex(polynomials)
        i == lastindex(polynomials) && continue

        D = dims(polynomials[i])
        res0, res1 = onedimi2itranslations(polynomials[i + 1], polynomials[i])
        tfmatrices = NTuple{D,typeof(res0)}[]

        for sector in 0:(2 ^ D - 1)
            push!(tfmatrices, Tuple(sectorordering(res0, res1, sector, D)))
        end
        push!(i2itranslations, SVector(tfmatrices...))
    end

    i2itranslations = convert(Vector{typeof(i2itranslations[1])}, i2itranslations)
    return H2Factory.I2ITranslationCollection(
        Dict(zip(levels[(begin + 1):end], i2itranslations)),
        selector,
        H2Factory.SeparableI2I(),
    )
end
