"""
    TreePolynomials(polynomials, minrelevantlevel)

Store the level polynomials and the coarsest relevant I2O level for a Galerkin map.
See [`TestTrialPolynomials`](@ref) for separate test and trial families.
"""
struct TreePolynomials{P}
    polynomials::P
    minrelevantlevel::Int
end

"""
    TestTrialPolynomials(test, trial)

Store separate level-polynomial dictionaries for a Petrov-Galerkin map.
The coarsest relevant I2O level is stored separately on the core.
"""
struct TestTrialPolynomials{Test,Trial}
    test::Test
    trial::Trial
end

"""
    assertmomentaliasing(A)

Check a map's pooled moments with `H2Factory.checkmomentaliasing`.
Return `A` if no error-severity findings exist; otherwise throw with the findings.

Enabled by the constructors' `checkaliasing=true` option. The overlap scan costs
`O(n^2)` in the moment count, so it runs at construction, not during multiplication.
"""
function assertmomentaliasing(A)
    report = H2Factory.checkmomentaliasing(A)
    report.ok && return A

    errs = filter(f -> f.severity === :error, report.findings)
    io = IOBuffer()
    println(
        io,
        "checkmomentaliasing found $(length(errs)) error-severity finding(s) in the pooled ",
        "moment layout of $(nameof(typeof(A))):",
    )
    for f in errs
        println(io, "  [", f.code, "] ", f.message)
    end
    return error(String(take!(io)))
end
