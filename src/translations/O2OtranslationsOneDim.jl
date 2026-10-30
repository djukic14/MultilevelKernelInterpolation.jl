"""
    struct O2OTranslatorOneDim{T}

Read-only O2O reference cache of transposed one-dimensional factors.
Built by `O2OTranslator` for a `SeparableI2I` collection.
Traversal instead applies `transpose`/`adjoint` to the original separable I2I transfer.
No separate O2O strategy is needed; see [`I2ITranslatorOneDim`](@ref).
"""
struct O2OTranslatorOneDim{T} <: AbstractSectorLevelTranslator
    translationmatrices::T
end

function O2OTranslator(c::H2Factory.I2ITranslationCollection, ::H2Factory.SeparableI2I)
    isnothing(c.transfers) && return O2OTranslatorOneDim(nothing)

    o2omatrices = []

    for (level, tfmatrices) in c.transfers
        o2otfmatrices = []
        for tfmatrix in tfmatrices
            push!(o2otfmatrices, Tuple(permutedims.(tfmatrix)))
        end
        convert(Vector{typeof(o2otfmatrices[1])}, o2otfmatrices)
        push!(o2omatrices, SVector(o2otfmatrices...))
    end
    o2omatrices = convert(Vector{typeof(o2omatrices[1])}, o2omatrices)

    # Workspace scratch comes from H2Factory.o2ochannels(c), not these cached transposes.
    translationmatrices = Dict(zip(keys(c.transfers), o2omatrices))
    return O2OTranslatorOneDim(translationmatrices)
end
