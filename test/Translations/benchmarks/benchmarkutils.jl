module BenchmarkUtils

using CompScienceMeshes, BEAST
using H2Trees
using H2Factory
using MultilevelKernelInterpolation
using OhMyThreads
using LinearAlgebra

export benchmarktree, benchmarkpoly, benchmarkplan, benchmarktreepolys
export benchmarkbsplinepoly_lowerorder, benchmarkbsplinepoly_higherorder
export benchmarkinmoment, benchmarktoeplitzmoment, benchmarki2odict
export op1d, op3d

const λ = 1.0
const k = 2π / λ

"""
    benchmarktree()

Small, fixed mesh/tree used as the common fixture for all translation benchmarks.
"""
function benchmarktree()
    m = readmesh(
        joinpath(
            pkgdir(MultilevelKernelInterpolation), "test", "assets", "in", "16spheres.in"
        ),
    )
    X = lagrangec0d1(m)
    tree = buildtree(X; builder=TwoNTreeBuilder(; minhalfsize=λ / 4, minvalues=0))
    treecoarse = buildtree(X; builder=TwoNTreeBuilder(; minhalfsize=λ / 2, minvalues=0))
    return X, tree, treecoarse
end

"""
    benchmarkpoly()

Order-4 Lagrange polynomial, matching the order used in the ported I2ITranslator tests.
"""
function benchmarkpoly()
    return MultilevelKernelInterpolation.BarycentricLagrangePolynomialEquidistant3D(4)
end

function benchmarktreepolys(tree, poly)
    return MultilevelKernelInterpolation.levelpolynomials(
        poly, H2Trees.levels(tree), (level) -> false
    )
end

function benchmarktreepolys(tree)
    return benchmarktreepolys(tree, benchmarkpoly())
end

"""
    benchmarkbsplinepoly_lowerorder()

Buffered B-spline with active knot removal, reduced per-axis size below
`MultilevelKernelInterpolation.BSPLINEREDUCTIONKRONECKERCROSSOVER` -- exercises the raw<->reduced modulation
(`modulate_in!`/`modulate_out!`) with a nontrivial (non-identity) `S` matrix, on
the cheap, dense-Kronecker side of the dense-translation-reduction crossover.
"""
function benchmarkbsplinepoly_lowerorder()
    return MultilevelKernelInterpolation.BufferedBsplinePolynomialEquidistant3D{Float64}(
        3, 7, 1
    )
end

"""
    benchmarkbsplinepoly_higherorder()

Buffered B-spline with no active knot removal, reduced per-axis size at/above
`MultilevelKernelInterpolation.BSPLINEREDUCTIONKRONECKERCROSSOVER` -- the same shape (order 5, b=9) that revealed a
severe (16m46s -> 1m57s) performance regression in the dense translation-reduction path
during this session; kept as a fixture so a future regression there, or in its modulation,
shows up directly in these benchmarks instead of only as an unexpectedly slow test run.
"""
function benchmarkbsplinepoly_higherorder()
    return MultilevelKernelInterpolation.BufferedBsplinePolynomialEquidistant3D{Float64}(
        5, 9, 0
    )
end

"""
    benchmarkinmoment(poly, dims)

Builds a moment matching `poly`'s storage scheme -- `BsplineDensePolynomialMoment` (raw and
reduced sizes) for B-splines, plain `PolynomialMoment` (reduced size only, since raw==reduced
for non-B-spline polynomials) otherwise -- filled with dummy data. Shared by the dense I2O
and modulation benchmarks; also usable as an output buffer, since its contents don't matter
for benchmark purposes.
"""
function benchmarkinmoment(poly, dims)
    m = if poly isa MultilevelKernelInterpolation.AbstractBspline
        MultilevelKernelInterpolation.BsplineDensePolynomialMoment{dims,ComplexF64}(
            MultilevelKernelInterpolation.numpolynomials(poly),
            size(LinearIndices(poly)),
        )
    else
        MultilevelKernelInterpolation.PolynomialMoment{dims,ComplexF64}(
            size(LinearIndices(poly))
        )
    end
    for i in eachindex(m)
        m.moments[i] .= 1.0
    end
    return m
end

"""
    benchmarktoeplitzmoment(poly, dims)

Builds a `ToeplitzPolynomialMoment` matching `poly`'s raw/reduced sizes, filled with dummy
data. Explicitly sizes the FFT-domain storage from the *reduced* size
(`H2Factory.standardtoeplitzsize(reduced)`) rather than relying on the single-argument
constructor's default (which derives it from the *primary* (raw) size instead) -- that
default is wrong whenever raw != reduced (active B-spline knot removal), a footgun found and
fixed in `test/Translations/toeplitz_equivalence.jl` earlier this session.
"""
function benchmarktoeplitzmoment(poly, dims)
    raw = MultilevelKernelInterpolation.numpolynomials(poly)
    reduced = size(LinearIndices(poly))
    m = MultilevelKernelInterpolation.ToeplitzPolynomialMoment{dims,ComplexF64}(
        raw, H2Factory.standardtoeplitzsize(reduced)
    )
    for i in eachindex(m)
        m.moments[i] .= 1.0
    end
    return m
end

"""
    benchmarki2odict(i2o)

The underlying `(receivingnode, translatingnode) => translation` `Dict` for an I2O
collection, unwrapping through nested `.i2ocollection` fields as needed -- both the dense
B-spline path (`BsplineI2OTranslationCollection`) and the Toeplitz path
(`H2Factory.ToeplitzI2OTranslationCollection`, wrapping a plain `H2Factory.I2OTranslationCollection`)
nest one level; the plain dense path doesn't nest at all. Structural (`hasfield`) rather than
type-enumerated so it doesn't need updating for every wrapper type.
"""
function benchmarki2odict(i2o)
    hasfield(typeof(i2o), :dict) && return i2o.dict
    return benchmarki2odict(i2o.i2ocollection)
end

# Demoted Toeplitz B-spline path: the map is a composed `H2Factory.I2OTranslation` (coupling +
# modulation), not a `.i2ocollection`-nesting wrapper -- unwrap to its plain Toeplitz coupling.
benchmarki2odict(i2o::H2Factory.I2OTranslation) = benchmarki2odict(H2Factory.coupling(i2o))

function benchmarkplan(tree)
    return H2Trees.buildplans(
        tree;
        builder=H2Trees.PlanBuilder(; aggregatenode=H2Trees.AggregateAllNodesFunctor()),
    ).testdisaggregationplan
end

"""
    op1d()

Scalar operator producing a 1-dimensional `PolynomialMoment` (Helmholtz single layer).
"""
function op1d()
    return Helmholtz3D.singlelayer(; wavenumber=k)
end

"""
    op3d()

Vector operator producing a 3-dimensional `PolynomialMoment` (Maxwell EFIE).
"""
function op3d()
    return MultilevelKernelInterpolation.efie(; wavenumber=k)
end

end
