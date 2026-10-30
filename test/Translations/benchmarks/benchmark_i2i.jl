module BenchmarkI2I

using BenchmarkTools
using MultilevelKernelInterpolation
using H2Trees, H2Factory
using OhMyThreads
using LinearAlgebra

include("benchmarkutils.jl")
using .BenchmarkUtils

function suite()
    group = BenchmarkGroup()

    poly = benchmarkpoly()
    polys = [poly, poly]

    group["setup"] = BenchmarkGroup()
    group["setup"]["I2ITranslator"] = @benchmarkable MultilevelKernelInterpolation.I2ITranslator(
        $polys, 2:3
    )

    i2i = MultilevelKernelInterpolation.I2ITranslator(polys, 2:3)

    group["application"] = BenchmarkGroup()
    for (label, op) in (("1D", op1d()), ("3D", op3d()))
        dims = MultilevelKernelInterpolation.dims(
            MultilevelKernelInterpolation.testmoment(op, poly)
        )
        childmoment = MultilevelKernelInterpolation.PolynomialMoment{dims,Float64}(
            size(LinearIndices(poly))
        )
        parentmoment = MultilevelKernelInterpolation.PolynomialMoment{dims,Float64}(
            size(LinearIndices(poly))
        )
        for i in eachindex(childmoment)
            childmoment.moments[i] .= 1.0
        end

        # Forward (aggregation) application through the level-transfer protocol: resolve the sector-0,
        # level-3 transfer, then apply it in the forward direction.
        transfer = i2i.transfers[3][0 + 1]
        context = H2Factory.TranslationContext(SerialScheduler(), nothing)
        group["application"][label] = @benchmarkable(
            H2Factory.applytransfer!(
                $parentmoment,
                $childmoment,
                $transfer,
                H2Factory.ForwardTranslation(),
                H2Factory.Overwrite(),
                $context,
            )
        )
    end

    return group
end

end
