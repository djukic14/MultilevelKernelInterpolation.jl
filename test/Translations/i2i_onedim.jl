module TestI2IOneDim

using Test
using CompScienceMeshes, BEAST
using H2Trees, H2Factory
using MultilevelKernelInterpolation
using LinearAlgebra
using OhMyThreads

_withchunks(f, levels) = H2Factory.withnchunks(Dict(level => 1 for level in levels)) do
    return f()
end

@testset verbose=true "I2ITranslatorOneDim single-level tree (no translation needed)" begin
    # Regression test: the single-level early return used to build the wrong I2I flavour. Now both
    # flavours build ONE `H2Factory.I2ITranslationCollection` distinguished by its trait, and downstream
    # `O2OTranslator(...)` dispatches on that trait (`SeparableI2I` here) -- pin the trait + that the edge
    # case still constructs.
    poly = MultilevelKernelInterpolation.BarycentricLagrangePolynomialEquidistant3D(7)

    i2i = MultilevelKernelInterpolation.I2ITranslatorOneDim([poly], 1:1)
    @test i2i isa H2Factory.I2ITranslationCollection
    @test H2Factory.i2itrait(i2i) isa H2Factory.SeparableI2I
    @test isnothing(i2i.transfers)
    # Scratch is workspace-local now and is derived from the finished collection, so the
    # transfer-less edge case must still answer -- with the no-op stand-in, not a `KeyError`.
    @test H2Factory.i2ichannels(i2i) isa H2Factory.EmptyChannels

    o2o = MultilevelKernelInterpolation.O2OTranslator(i2i)
    @test o2o isa MultilevelKernelInterpolation.O2OTranslatorOneDim
    @test isnothing(o2o.translationmatrices)
    @test H2Factory.o2ochannels(i2i) isa H2Factory.EmptyChannels
end

@testset verbose=true "I2ITranslatorOneDim matches I2ITranslator" begin
    poly = MultilevelKernelInterpolation.BarycentricLagrangePolynomialEquidistant3D(7)

    # Interpolation order INCREASES up the tree (parent = increase(poly)), so each per-axis 1D transfer
    # is NON-SQUARE (parent order != child order). This is the case that exercises direction-dependent
    # separable-buffer shapes: aggregation contracts child->parent, disaggregation (transpose) contracts
    # parent->child, and the two need differently-shaped scratch buffers. Assert non-squareness so the
    # transpose/o2o coverage below can't silently degrade to the (easy) square case.
    polys = [MultilevelKernelInterpolation.increase(poly), poly]
    i2i = MultilevelKernelInterpolation.I2ITranslator(polys, 2:3)

    i2i2 = MultilevelKernelInterpolation.I2ITranslatorOneDim(polys, 2:3)
    i2ichannel2 = _withchunks(2:3) do
        H2Factory.i2ichannels(i2i2)
    end
    # Level 3, not 2: both level-transfer channel dicts are keyed by the TRANSFER's own (child) level
    # -- the same key `i2i2.transfers` uses, and the same one the `o2ochannels` borrow below already
    # used. The traversal offsets to it (an aggregation loops parents, so it borrows `level + 1`);
    # here the transfer is picked by hand, so the key is used directly.
    buffer2 = H2Factory.borrowbuffer(i2ichannel2, 3, 1, H2Factory.isForward())

    let firstaxis = i2i2.transfers[3][1][1]
        @test size(firstaxis, 1) != size(firstaxis, 2)   # genuinely non-square per-axis transfer
    end

    op = Maxwell3D.singlelayer(; wavenumber=1.0)

    childmoment = MultilevelKernelInterpolation.PolynomialMoment{
        MultilevelKernelInterpolation.dims(
            MultilevelKernelInterpolation.testmoment(op, poly)
        ),
        Float64,
    }(
        size(LinearIndices(polys[2]))
    )
    for i in eachindex(childmoment)
        childmoment.moments[i] .= i
    end

    parentmoment = MultilevelKernelInterpolation.PolynomialMoment{
        MultilevelKernelInterpolation.dims(
            MultilevelKernelInterpolation.testmoment(op, poly)
        ),
        Float64,
    }(
        size(LinearIndices(polys[1]))
    )
    parentmoment2 = MultilevelKernelInterpolation.PolynomialMoment{
        MultilevelKernelInterpolation.dims(
            MultilevelKernelInterpolation.testmoment(op, poly)
        ),
        Float64,
    }(
        size(LinearIndices(polys[1]))
    )

    # Dense and OneDim now go through the SAME level-transfer framework: `leveltransfer` resolves the
    # transfer (a dense matrix vs the per-axis 1D-matrix tuple), and `applytransfer!` -- dispatched on
    # the transfer's type -- applies it in the requested direction. Equivalence is asserted at that
    # `applytransfer!` seam, per (level, sector). The dense transfer needs no buffer; the OneDim one
    # takes its separable scratch from the context.
    ctx(buf) = H2Factory.TranslationContext(OhMyThreads.SerialScheduler(), buf)
    Tdense(sector) = i2i.transfers[3][sector + 1]
    Tonedim(sector) = H2Factory.SeparableTransfer(i2i2.transfers[3][sector + 1])

    # Forward = aggregation (child -> parent).
    @testset "i2i agreement (forward), sector $sector" for sector in 0:7
        H2Factory.applytransfer!(
            parentmoment,
            childmoment,
            Tdense(sector),
            H2Factory.ForwardTranslation(),
            H2Factory.Overwrite(),
            ctx(nothing),
        )
        H2Factory.applytransfer!(
            parentmoment2,
            childmoment,
            Tonedim(sector),
            H2Factory.ForwardTranslation(),
            H2Factory.Overwrite(),
            ctx(buffer2),
        )
        for i in eachindex(parentmoment)
            @test maximum(abs, parentmoment.moments[i] .- parentmoment2.moments[i]) < 1e-12
        end
    end

    # Transpose = disaggregation (parent -> child): this is the O2O action, now reconstructed as
    # transpose(I2I) rather than a separate O2O translator. The OneDim separable contraction needs a
    # buffer shaped for the parent->child direction -- the o2o channel provides exactly that.
    o2ochannel2 = _withchunks(2:3) do
        H2Factory.o2ochannels(i2i2)
    end
    obuffer2 = H2Factory.borrowbuffer(o2ochannel2, 3, 1, H2Factory.isForward())

    childmomentout = MultilevelKernelInterpolation.PolynomialMoment{
        MultilevelKernelInterpolation.dims(
            MultilevelKernelInterpolation.testmoment(op, poly)
        ),
        Float64,
    }(
        size(LinearIndices(polys[2]))
    )
    childmomentout2 = MultilevelKernelInterpolation.PolynomialMoment{
        MultilevelKernelInterpolation.dims(
            MultilevelKernelInterpolation.testmoment(op, poly)
        ),
        Float64,
    }(
        size(LinearIndices(polys[2]))
    )

    @testset "o2o agreement (transpose(I2I)), sector $sector" for sector in 0:7
        H2Factory.applytransfer!(
            childmomentout,
            parentmoment,
            Tdense(sector),
            H2Factory.TransposeTranslation(),
            H2Factory.Overwrite(),
            ctx(nothing),
        )
        H2Factory.applytransfer!(
            childmomentout2,
            parentmoment2,
            Tonedim(sector),
            H2Factory.TransposeTranslation(),
            H2Factory.Overwrite(),
            ctx(obuffer2),
        )
        for i in eachindex(childmomentout)
            @test maximum(abs, childmomentout.moments[i] .- childmomentout2.moments[i]) <
                1e-12
        end
    end

    # Allocation regression for the separable path (`H2Factory.SeparableTransfer`): the transpose/adjoint
    # apply must stay allocation-free on the hot path -- the wrapper and the per-axis
    # `map(transpose/adjoint, ...)` in `_directedtransfer` stay off-heap by escape analysis, and the
    # contraction runs entirely in the preallocated o2o scratch. Warm up, then pin `@allocated == 0` so a
    # future change can't reintroduce a per-apply allocation. Non-square transfers (order increasing up
    # the tree, asserted above), so this also covers the shape-sensitive scratch path.
    @testset "separable $dir apply is allocation-free" for dir in (
        H2Factory.TransposeTranslation(), H2Factory.AdjointTranslation()
    )
        actx = ctx(obuffer2)
        T0 = Tonedim(0)
        apply() = H2Factory.applytransfer!(
            childmomentout2, parentmoment2, T0, dir, H2Factory.Overwrite(), actx
        )
        apply()
        apply()
        @test (@allocated apply()) == 0
    end
end

end
