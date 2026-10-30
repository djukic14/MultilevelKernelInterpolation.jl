module BenchmarkI2OToeplitz

using BenchmarkTools
using MultilevelKernelInterpolation
using H2Trees
using H2Factory
using OhMyThreads
using LinearAlgebra

include("benchmarkutils.jl")
using .BenchmarkUtils

function _add_group!(group, label, poly, X, tree, plan)
    treepolys = benchmarktreepolys(tree, poly)

    group["setup"][label] = BenchmarkGroup()
    for (oplabel, op) in (("1D", op1d()), ("3D", op3d()))
        group["setup"][label][oplabel] = @benchmarkable(
            MultilevelKernelInterpolation.ToeplitzI2OTranslationCollection(
                $op, $tree, $treepolys, $plan, $X; verbose=false
            )
        )
    end

    group["application"][label] = BenchmarkGroup()
    for (oplabel, op) in (("1D", op1d()), ("3D", op3d()))
        toeplitzi2o, _ = MultilevelKernelInterpolation.ToeplitzI2OTranslationCollection(
            op, tree, treepolys, plan, X; verbose=false
        )
        testnode, trialnode = first(keys(benchmarki2odict(toeplitzi2o)))

        dims = MultilevelKernelInterpolation.dims(
            MultilevelKernelInterpolation.testmoment(op, poly)
        )
        inmoment = benchmarktoeplitzmoment(poly, dims)
        outmoment = benchmarktoeplitzmoment(poly, dims)

        group["application"][label]["$oplabel forward"] = @benchmarkable(
            H2Factory.applyi2o!(
                $outmoment,
                $inmoment,
                $toeplitzi2o,
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
                $toeplitzi2o,
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
