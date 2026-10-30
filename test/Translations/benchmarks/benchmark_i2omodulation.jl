module BenchmarkI2OModulation

using BenchmarkTools
using MultilevelKernelInterpolation
using H2Trees
using H2Factory
using OhMyThreads
using LinearAlgebra

include("benchmarkutils.jl")
using .BenchmarkUtils

# `modulate_in!`/`modulate_out!` bridge the raw B-spline basis (I2I/O2O aggregation)
# and the reduced collocation/interpolation basis translation matrices use -- a no-op
# passthrough for Lagrange and non-buffered B-splines (raw == reduced), so only B-spline
# fixtures are meaningful here. Covers both the dense (`BsplineI2OTranslationCollection`, an
# `S`-matrix multiply) and Toeplitz (`BsplineToeplitzModulation` on a composed `I2OTranslation`, a
# `B^-1`-composed multiply) modulation schemes, at both sides of the
# Kronecker-vs-separable dense-translation-reduction crossover (`benchmarkbsplinepoly_lowerorder`/
# `benchmarkbsplinepoly_higherorder`) -- since that crossover only affects the underlying
# translation matrices, not modulation itself, this doubles as a check that modulation cost
# stays flat across it, and a regression guard should that change.
function _add_group!(group, label, collectiontype, poly, X, tree, plan)
    op = op1d()
    dims = MultilevelKernelInterpolation.dims(
        MultilevelKernelInterpolation.testmoment(op, poly)
    )
    treepolys = benchmarktreepolys(tree, poly)

    i2o, i2ochannel = collectiontype(op, tree, treepolys, plan, X; verbose=false)
    (testnode, _) = first(keys(benchmarki2odict(i2o)))
    level = H2Trees.level(tree, testnode)

    # The Toeplitz path's modulation reads/writes the FFT-domain (`AbstractToeplitzMoment`)
    # storage, not the plain/`BsplineDensePolynomialMoment` storage the dense path uses --
    # needs `benchmarktoeplitzmoment`'s correct raw/reduced sizing (see its docstring), not
    # `benchmarkinmoment`.
    moment =
        if collectiontype === MultilevelKernelInterpolation.ToeplitzI2OTranslationCollection
            benchmarktoeplitzmoment(poly, dims)
        else
            benchmarkinmoment(poly, dims)
        end
    buffer = H2Factory.borrowbuffer(i2ochannel, level, 1, H2Factory.isForward())

    modulation = H2Factory.modulation(i2o)
    coupling = H2Factory.coupling(i2o)
    group["modulate_in!"][label] = @benchmarkable(
        H2Factory.modulate_in!(
            $moment,
            $level,
            $(H2Factory.isForward()),
            $modulation,
            $coupling,
            $i2ochannel,
            $(SerialScheduler()),
            $buffer,
        )
    )
    return group["modulate_out!"][label] = @benchmarkable(
        H2Factory.modulate_out!(
            $moment,
            $level,
            $(H2Factory.isForward()),
            $modulation,
            $coupling,
            $i2ochannel,
            $(SerialScheduler()),
            $buffer,
        )
    )
end

function suite()
    group = BenchmarkGroup()
    group["modulate_in!"] = BenchmarkGroup()
    group["modulate_out!"] = BenchmarkGroup()

    X, tree, _ = benchmarktree()
    plan = benchmarkplan(tree)

    for (label, poly) in (
        ("dense, lower order", benchmarkbsplinepoly_lowerorder()),
        ("dense, higher order", benchmarkbsplinepoly_higherorder()),
    )
        _add_group!(
            group,
            label,
            MultilevelKernelInterpolation.I2OTranslationCollection,
            poly,
            X,
            tree,
            plan,
        )
    end

    for (label, poly) in (
        ("Toeplitz, lower order", benchmarkbsplinepoly_lowerorder()),
        ("Toeplitz, higher order", benchmarkbsplinepoly_higherorder()),
    )
        _add_group!(
            group,
            label,
            MultilevelKernelInterpolation.ToeplitzI2OTranslationCollection,
            poly,
            X,
            tree,
            plan,
        )
    end

    return group
end

end
