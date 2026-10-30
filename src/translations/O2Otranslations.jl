"""
    struct O2OTranslator{T}

Read-only reference cache of transposed I2I matrices, indexed by level and sector.
`translationmatrices` is a dictionary, or `nothing` when there are no transfers.

Construct with `O2OTranslator(c::H2Factory.I2ITranslationCollection)`.
The dense trait returns this type; the separable trait returns [`O2OTranslatorOneDim`](@ref).
Traversal does not use this cache: it applies transpose/adjoint transfers from the I2I
collection. Workspace scratch also comes from that collection via `H2Factory.o2ochannels`.
"""
struct O2OTranslator{T} <: AbstractSectorLevelTranslator
    translationmatrices::T
end

# Build from the same stored transfers as I2I, selecting the representation by trait.
function O2OTranslator(c::H2Factory.I2ITranslationCollection)
    return O2OTranslator(c, H2Factory.i2itrait(c))
end

function O2OTranslator(c::H2Factory.I2ITranslationCollection, ::H2Factory.DenseI2I)
    isnothing(c.transfers) && return O2OTranslator(nothing)

    o2omatrices = []
    levels = Int[]

    for (key, tfmatrices) in c.transfers
        o2otfmatrices = []
        for tfmatrix in tfmatrices
            push!(o2otfmatrices, permutedims(tfmatrix))
        end
        o2otfmatrices = convert(Vector{typeof(o2otfmatrices[1])}, o2otfmatrices)
        push!(levels, key)
        push!(o2omatrices, o2otfmatrices)
    end
    o2omatrices = convert(Vector{typeof(o2omatrices[1])}, o2omatrices)

    translationmatrices = Dict(zip(levels, o2omatrices))
    return O2OTranslator(translationmatrices)
end
