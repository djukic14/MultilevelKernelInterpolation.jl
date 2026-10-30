module BenchmarkO2O

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

    i2i = MultilevelKernelInterpolation.I2ITranslator(polys, 2:3)

    group["setup"] = BenchmarkGroup()
    group["setup"]["O2OTranslator"] = @benchmarkable MultilevelKernelInterpolation.O2OTranslator(
        $i2i
    )

    o2o = MultilevelKernelInterpolation.O2OTranslator(i2i)

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
        for i in eachindex(parentmoment)
            parentmoment.moments[i] .= 1.0
        end

        # The O2O (disaggregation) application is now the I2I transfer applied TRANSPOSED through the
        # level-transfer protocol (O2O == transpose(I2I)); the materialized `o2o` is only benchmarked for
        # construction above. Resolve the sector-0, level-3 I2I transfer and apply it transposed.
        transfer = i2i.transfers[3][0 + 1]
        context = H2Factory.TranslationContext(SerialScheduler(), nothing)
        group["application"][label] = @benchmarkable(
            H2Factory.applytransfer!(
                $childmoment,
                $parentmoment,
                $transfer,
                H2Factory.TransposeTranslation(),
                H2Factory.Overwrite(),
                $context,
            )
        )
    end

    return group
end

end
