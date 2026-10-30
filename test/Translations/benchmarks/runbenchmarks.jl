using BenchmarkTools
using Statistics

include(joinpath(@__DIR__, "benchmark_i2i.jl"))
include(joinpath(@__DIR__, "benchmark_i2o_dense.jl"))
include(joinpath(@__DIR__, "benchmark_i2o_toeplitz.jl"))
include(joinpath(@__DIR__, "benchmark_i2omodulation.jl"))
include(joinpath(@__DIR__, "benchmark_o2o.jl"))

function printsummary(name, group)
    println("\n=== $name ===")
    return printgroup(group, "  ")
end

# Recurses through arbitrarily nested `BenchmarkGroup`s (suites nest to different depths --
# e.g. i2i/o2o are `phase -> label -> benchmarkable`, while i2o/modulation are
# `phase -> config -> label -> benchmarkable`) instead of assuming a fixed depth.
function printgroup(group, indent)
    for (key, value) in group
        if value isa BenchmarkTools.BenchmarkGroup
            println(indent, key, ":")
            printgroup(value, indent * "  ")
        else
            trial = run(value)
            println(
                indent,
                key,
                ": ",
                "median=",
                BenchmarkTools.prettytime(median(trial).time),
                ", mean=",
                BenchmarkTools.prettytime(mean(trial).time),
                ", allocs=",
                trial.allocs,
                ", memory=",
                BenchmarkTools.prettymemory(trial.memory),
            )
        end
    end
end

results = Dict(
    "i2i" => BenchmarkI2I.suite(),
    "i2o (dense)" => BenchmarkI2ODense.suite(),
    "i2o (Toeplitz)" => BenchmarkI2OToeplitz.suite(),
    "i2omodulation" => BenchmarkI2OModulation.suite(),
    "o2o" => BenchmarkO2O.suite(),
)

for (name, group) in results
    printsummary(name, group)
end
