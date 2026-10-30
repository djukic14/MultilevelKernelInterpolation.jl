module BenchmarkI2ODense

using BenchmarkTools
using MultilevelKernelInterpolation
using H2Trees
using H2Factory
using OhMyThreads
using LinearAlgebra

include("benchmarkutils.jl")
using .BenchmarkUtils

# Dense (non-Toeplitz) counterpart to `benchmark_i2o_toeplitz.jl`: `MultilevelKernelInterpolation.I2OTranslationCollection`
# dispatches to `BsplineI2OTranslationCollection` for B-spline polynomials (exercising the
# Kronecker-vs-separable reduction crossover, see `benchmarkbsplinepoly_lowerorder`/
# `benchmarkbsplinepoly_higherorder`) and to the plain dense path otherwise.
function _add_group!(group, label, poly, X, tree, plan)
    treepolys = benchmarktreepolys(tree, poly)

    group["setup"][label] = BenchmarkGroup()
    for (oplabel, op) in (("1D", op1d()), ("3D", op3d()))
        group["setup"][label][oplabel] = @benchmarkable(
            MultilevelKernelInterpolation.I2OTranslationCollection(
                $op, $tree, $treepolys, $plan, $X; verbose=false
            )
        )
    end

    group["application"][label] = BenchmarkGroup()
    for (oplabel, op) in (("1D", op1d()), ("3D", op3d()))
        i2o, _ = MultilevelKernelInterpolation.I2OTranslationCollection(
            op, tree, treepolys, plan, X; verbose=false
        )
        testnode, trialnode = first(keys(benchmarki2odict(i2o)))

        dims = MultilevelKernelInterpolation.dims(
            MultilevelKernelInterpolation.testmoment(op, poly)
        )
        inmoment = benchmarkinmoment(poly, dims)
        outmoment = benchmarkinmoment(poly, dims)

        group["application"][label]["$oplabel forward"] = @benchmarkable(
            H2Factory.applyi2o!(
                $outmoment,
                $inmoment,
                $i2o,
                $testnode,
                $trialnode,
                $(H2Factory.ForwardTranslation()),
                $(H2Factory.Overwrite()),
                $(H2Factory.TranslationContext(SerialScheduler(), nothing)),
            )
        )
        group["application"][label]["$oplabel transpose"] = @benchmarkable(
            H2Factory.applyi2o!(
                $outmoment,
                $inmoment,
                $i2o,
                $trialnode,
                $testnode,
                $(H2Factory.TransposeTranslation()),
                $(H2Factory.Overwrite()),
                $(H2Factory.TranslationContext(SerialScheduler(), nothing)),
            )
        )
    end
end

function suite()
    group = BenchmarkGroup()
    group["setup"] = BenchmarkGroup()
    group["application"] = BenchmarkGroup()

    X, tree, _ = benchmarktree()
    plan = benchmarkplan(tree)

    _add_group!(group, "Lagrange", benchmarkpoly(), X, tree, plan)
    _add_group!(
        group, "B-spline (lower order)", benchmarkbsplinepoly_lowerorder(), X, tree, plan
    )
    _add_group!(
        group, "B-spline (higher order)", benchmarkbsplinepoly_higherorder(), X, tree, plan
    )

    return group
end

end
