module TestToeplitzEquivalence

using Test
using OhMyThreads, LinearAlgebra
using CompScienceMeshes, BEAST
using H2Trees, H2Factory, MultilevelKernelInterpolation
using H2FactoryTestUtils

# Only B-spline polynomials go through `BsplineI2OTranslationCollection`'s raw<->reduced
# modulation (dual-storage `BsplineDensePolynomialMoment`, sized `(raw, reduced)`); every
# other polynomial type (e.g. equidistant Lagrange) uses the plain, unwrapped dense
# `H2Factory.I2OTranslationCollection` and single-storage `PolynomialMoment` (sized just
# `raw`, since raw == reduced always for it), exactly as before this session's B-spline fix.
# These helpers let the granular test below stay polynomial-type-agnostic.
function _builddenseinmoment(
    ::MultilevelKernelInterpolation.AbstractBspline, momentcollection, x, raw, reduced
)
    m = MultilevelKernelInterpolation.BsplineDensePolynomialMoment{1,ComplexF64}(
        raw, reduced
    )
    LinearAlgebra.mul!(m, momentcollection, x)
    return m
end
function _builddenseinmoment(::Any, momentcollection, x, raw, reduced)
    m = MultilevelKernelInterpolation.PolynomialMoment{1,ComplexF64}(raw)
    LinearAlgebra.mul!(m, momentcollection, x)
    return m
end

_builddenseoutmoment(::MultilevelKernelInterpolation.AbstractBspline, raw, reduced) =
    MultilevelKernelInterpolation.BsplineDensePolynomialMoment{1,ComplexF64}(raw, reduced)
_builddenseoutmoment(::Any, raw, reduced) =
    MultilevelKernelInterpolation.PolynomialMoment{1,ComplexF64}(raw)

_i2odict(i2o::MultilevelKernelInterpolation.BsplineI2OTranslationCollection) =
    i2o.i2ocollection.dict
_i2odict(i2o) = i2o.dict

function _withplanchunks(f, plan)
    schedule = H2Factory.planschedule(plan, Base.Threads.nthreads())
    return H2Factory.withnchunks(H2Factory.schedulechunkcounts(schedule)) do
        return f()
    end
end

# Verifies that the dense (`BsplineI2OTranslationCollection`) and Toeplitz (a composed
# `I2OTranslation` with a `BsplineToeplitzModulation`) B-spline I2O translation paths agree, and that
# each is internally correct, in all three `LinearMaps` operator modes (forward, transpose,
# adjoint). Both paths route through the same H2Factory `aggregatetree!`/`disaggregatetree!`
# machinery, which for each mode does: build a moment -> `modulate_in!` -> `applycoupling!` with the
# mode's direction (transpose/adjoint swap the receiving/translating node) -> `modulate_out!`. Testing through real
# `GalerkinMap` objects (rather than hand-driving a single node pair) is the most faithful
# way to exercise that exact scheme.
#
# Two kinds of checks, neither of which needs an external dense-matrix reference:
#   - Self-consistency: the algebraic identities `sum(z .* (transpose(A)*x)) == sum(x .* (A*z))`
#     and `dot(z, A*x) == dot(A'*z, x)` must hold for ANY correctly-implemented transpose/
#     adjoint, regardless of whether `A*x` itself is an accurate physical approximation.
#   - Cross-agreement: dense and Toeplitz must produce the same result in every mode, since
#     they represent the same underlying translation operator via two different numerical
#     schemes.
#
# Previously disabled: this testset (every case, individually, verified passing at machine
# precision in isolated single-purpose processes) reproducibly segfaulted specifically when run
# as part of the full `Pkg.test("MultilevelKernelInterpolation")` suite (native crash inside the dense path's
# `_six1dinterpolations!`, `I2Obsplinetranslationcollection.jl` -- deeply nested `SMatrix`-into-
# strided-view `mul!` calls, a known StaticArrays codegen risk at larger sizes). That function
# has since been deleted and replaced with a single dense `mul!` against a precomputed 3D
# Kronecker-product reduction matrix (see `BsplineKernelTranslator`'s constructor and
# `calculateI2Otranslation!`), which should remove the crash site entirely. Re-enabled to
# verify against the full suite.
@testset verbose=true "B-spline dense vs Toeplitz I2O translation (forward/transpose/adjoint)" begin
    λ = 1.0
    k = 2π / λ
    op = Helmholtz3D.singlelayer(; wavenumber=k)

    m = readmesh(
        joinpath(pkgdir(MultilevelKernelInterpolation), "test", "assets", "in", "sphere.in")
    )
    X = lagrangecxd0(m)
    tree = buildtree(X; builder=TwoNTreeBuilder(; minvalues=120))

    # `s - p` is kept generously large (matching the paper's recommended `s = p + 4`) to
    # avoid the unrelated, separately-tracked numerical edge case where a too-small `s - p`
    # margin makes the buffer-knot granularity coarser than the actual protrusion needs (see
    # project memory: this can make even a small relative protrusion force a full-box-halfsize
    # buffer overextension on a coarse tree/mesh like this test fixture's).
    for (label, poly) in [
        (
            "equidistant Lagrange (no B-spline modulation at all)",
            MultilevelKernelInterpolation.BarycentricLagrangePolynomialEquidistant3D(7),
        ),
        (
            "plain B-spline (no reduction mechanism)",
            MultilevelKernelInterpolation.BsplinePolynomialEquidistant3D{Float64}(4, 5),
        ),
        (
            "buffered, active knot removal",
            MultilevelKernelInterpolation.BufferedBsplinePolynomialEquidistant3D{Float64}(
                4, 8, 0
            ),
        ),
        (
            "buffered, active knot removal",
            MultilevelKernelInterpolation.BufferedBsplinePolynomialEquidistant3D{Float64}(
                3, 7, 1
            ),
        ),
        (
            "buffered, no active knot removal, higher order",
            MultilevelKernelInterpolation.BufferedBsplinePolynomialEquidistant3D{Float64}(
                5, 9, 0
            ),
        ),
    ]
        @testset "$label" begin
            Adense = MultilevelKernelInterpolation.GalerkinMap(
                op,
                X;
                tree=tree,
                verbose=false,
                strategy=MultilevelKernelInterpolation.InterpolationStrategy(
                    op,
                    X;
                    polynomialstrategy=MultilevelKernelInterpolation.PolynomialStrategy(
                        op; polynomial=poly
                    ),
                    nearinteractionsstrategy=H2Factory.DummyNearInteractionsStrategy(
                        BEAST.defaultquadstrat(op, X, X)
                    ),
                ),
            )
            Atoep = MultilevelKernelInterpolation.GalerkinMap(
                op,
                X;
                tree=tree,
                verbose=false,
                strategy=MultilevelKernelInterpolation.InterpolationStrategy(
                    op,
                    X;
                    polynomialstrategy=MultilevelKernelInterpolation.PolynomialStrategy(
                        op; polynomial=poly
                    ),
                    nearinteractionsstrategy=H2Factory.DummyNearInteractionsStrategy(
                        BEAST.defaultquadstrat(op, X, X)
                    ),
                    translatorstrategy=MultilevelKernelInterpolation.TranslatorStrategy(;
                        i2ostrategy=MultilevelKernelInterpolation.ToeplitzI2OStrategy()
                    ),
                ),
            )

            rng = H2FactoryTestUtils.checkrng()
            x = randn(rng, ComplexF64, size(Adense, 2))
            z = randn(rng, ComplexF64, size(Adense, 2))

            yd, yt = Adense * x, Atoep * x
            ydT, ytT = transpose(Adense) * x, transpose(Atoep) * x
            ydAz, ytAz = Adense' * z, Atoep' * z

            @testset "cross-agreement (dense vs Toeplitz)" begin
                @test yd ≈ yt rtol = 1e-10
                @test ydT ≈ ytT rtol = 1e-10
                @test (Adense' * x) ≈ (Atoep' * x) rtol = 1e-10
            end

            @testset "self-consistency" begin
                # transpose identity: sum(z .* (A^T x)) == sum(x .* (A z))
                lhsdt = sum(z .* ydT)
                rhsdt = sum(x .* (Adense * z))
                @test lhsdt ≈ rhsdt rtol = 1e-10

                lhstt = sum(z .* ytT)
                rhstt = sum(x .* (Atoep * z))
                @test lhstt ≈ rhstt rtol = 1e-10

                # adjoint identity: dot(z, A x) == dot(A' z, x)
                lhsda = dot(z, yd)
                rhsda = dot(ydAz, x)
                @test lhsda ≈ rhsda rtol = 1e-10

                lhsta = dot(z, yt)
                rhsta = dot(ytAz, x)
                @test lhsta ≈ rhsta rtol = 1e-10
            end
        end
    end
end

# Regression test for a real bug found via the `test/Translations/benchmarks/benchmark_i2o_dense.jl`
# benchmark this session: `BsplineI2OTranslationCollection`'s forward and transpose coupling
# looped over each of a 3-component moment's components independently, applying the *whole*
# `green` to each -- correct only for the scalar (Helmholtz) case tested above, where `green`
# is a single dense matrix. For a dyadic/cross-component-coupled operator like EFIE (`green`
# is a 3x3 matrix of matrices), that loop doesn't even have matching dimensions and crashed
# outright the first time anything exercised it (nothing previously did -- every other test
# and benchmark in this file uses only Helmholtz/`ScalarGreen3DTranslation`). Fixed by
# `_bsplinei2oapply!` dispatching on `green`'s shape (plus the `isForward`/`isTranspose`/
# `isAdjoint` direction mode, which selects the per-block transform inside that same dispatch)
# instead of assuming scalar. Kept deliberately small (2 configs, not the full sweep above)
# since EFIE/RWG setup is more expensive than the scalar Helmholtz case.
@testset verbose=true "B-spline dense vs Toeplitz I2O translation, dyadic operator (EFIE)" begin
    λ = 1.0
    k = 2π / λ
    op = MultilevelKernelInterpolation.efie(; wavenumber=k)

    m = readmesh(
        joinpath(pkgdir(MultilevelKernelInterpolation), "test", "assets", "in", "sphere.in")
    )
    X = raviartthomas(m)
    tree = buildtree(X; builder=TwoNTreeBuilder(; minvalues=120))

    for (label, poly) in [
        (
            "plain B-spline (no reduction mechanism)",
            MultilevelKernelInterpolation.BsplinePolynomialEquidistant3D{Float64}(4, 5),
        ),
        (
            "buffered, active knot removal",
            MultilevelKernelInterpolation.BufferedBsplinePolynomialEquidistant3D{Float64}(
                3, 7, 1
            ),
        ),
    ]
        @testset "$label" begin
            Adense = MultilevelKernelInterpolation.GalerkinMap(
                op,
                X;
                tree=tree,
                verbose=false,
                strategy=MultilevelKernelInterpolation.InterpolationStrategy(
                    op,
                    X;
                    polynomialstrategy=MultilevelKernelInterpolation.PolynomialStrategy(
                        op; polynomial=poly
                    ),
                    nearinteractionsstrategy=H2Factory.DummyNearInteractionsStrategy(
                        BEAST.defaultquadstrat(op, X, X)
                    ),
                ),
            )
            Atoep = MultilevelKernelInterpolation.GalerkinMap(
                op,
                X;
                tree=tree,
                verbose=false,
                strategy=MultilevelKernelInterpolation.InterpolationStrategy(
                    op,
                    X;
                    polynomialstrategy=MultilevelKernelInterpolation.PolynomialStrategy(
                        op; polynomial=poly
                    ),
                    nearinteractionsstrategy=H2Factory.DummyNearInteractionsStrategy(
                        BEAST.defaultquadstrat(op, X, X)
                    ),
                    translatorstrategy=MultilevelKernelInterpolation.TranslatorStrategy(;
                        i2ostrategy=MultilevelKernelInterpolation.ToeplitzI2OStrategy()
                    ),
                ),
            )

            rng = H2FactoryTestUtils.checkrng()
            x = randn(rng, ComplexF64, size(Adense, 2))
            z = randn(rng, ComplexF64, size(Adense, 2))

            yd, yt = Adense * x, Atoep * x
            ydT, ytT = transpose(Adense) * x, transpose(Atoep) * x
            ydAz = Adense' * z

            @testset "cross-agreement (dense vs Toeplitz)" begin
                @test yd ≈ yt rtol = 1e-10
                @test ydT ≈ ytT rtol = 1e-10
            end

            @testset "self-consistency" begin
                lhsdt = sum(z .* ydT)
                rhsdt = sum(x .* (Adense * z))
                @test lhsdt ≈ rhsdt rtol = 1e-10

                lhsda = dot(z, yd)
                rhsda = dot(ydAz, x)
                @test lhsda ≈ rhsda rtol = 1e-10
            end
        end
    end
end

# Granular version of the check above: instead of a full GalerkinMap matrix-vector
# product, this drives exactly one translating node pair through the same sequence
# `aggregatetree!`/`disaggregatetree!` uses -- build a moment, `modulate_in!`,
# `applycoupling!` with the mode's direction (transpose swaps receiving/translating),
# `modulate_out!` --
# comparing dense against Toeplitz at the raw-basis moment level directly. Only forward and
# transpose are driven here; the adjoint (which since Part E conjugates the coupling OPERATOR
# at its boundary via `AdjointTranslation`, rather than bracketing a translation step with
# any per-node moment conjugation) is exercised at the full-map level above instead. This
# testset does NOT reproduce the native crash tracked above.
@testset verbose = true "moment -> modulate_in! -> applycoupling! -> modulate_out!" begin
    λ = 1.0
    k = 2π / λ
    op = Helmholtz3D.singlelayer(; wavenumber=k)

    m = readmesh(
        joinpath(pkgdir(MultilevelKernelInterpolation), "test", "assets", "in", "sphere.in")
    )
    X = lagrangecxd0(m)
    tree = buildtree(X; builder=TwoNTreeBuilder(; minvalues=120))

    plan = H2Trees.buildplans(
        tree;
        builder=H2Trees.PlanBuilder(; aggregatenode=H2Trees.AggregateAllNodesFunctor()),
    ).testdisaggregationplan

    for (label, poly) in [
        (
            "equidistant Lagrange",
            MultilevelKernelInterpolation.BarycentricLagrangePolynomialEquidistant3D(7),
        ),
        (
            "plain B-spline",
            MultilevelKernelInterpolation.BsplinePolynomialEquidistant3D{Float64}(4, 5),
        ),
        (
            "buffered, active knot removal",
            MultilevelKernelInterpolation.BufferedBsplinePolynomialEquidistant3D{Float64}(
                4, 8, 0
            ),
        ),
        (
            "buffered, active knot removal, order 3",
            MultilevelKernelInterpolation.BufferedBsplinePolynomialEquidistant3D{Float64}(
                3, 7, 1
            ),
        ),
        (
            "buffered, no active knot removal",
            MultilevelKernelInterpolation.BufferedBsplinePolynomialEquidistant3D{Float64}(
                5, 9, 0
            ),
        ),
    ]
        @testset "$label" begin
            treepolys = MultilevelKernelInterpolation.levelpolynomials(
                poly, H2Trees.levels(tree), (level) -> false
            )

            momentcollections = Dict(
                zip(
                    H2Trees.leaves(tree),
                    MultilevelKernelInterpolation.TrialMomentCollections(
                        op,
                        tree,
                        MultilevelKernelInterpolation.leafpolynomials(tree, treepolys),
                        X,
                    ),
                ),
            )

            i2odense, i2ochanneldense = _withplanchunks(plan) do
                MultilevelKernelInterpolation.I2OTranslationCollection(
                    op, tree, treepolys, plan, X; verbose=false
                )
            end
            i2otoep, i2ochanneltoep = _withplanchunks(plan) do
                MultilevelKernelInterpolation.ToeplitzI2OTranslationCollection(
                    op, tree, treepolys, plan, X; verbose=false
                )
            end

            function _buildmoment(MomentType, node, x, sizes...)
                m = MomentType{1,ComplexF64}(sizes...)
                LinearAlgebra.mul!(m, momentcollections[node], x)
                return m
            end

            # ONE rng for the pair loop, ADVANCED per pair. `checkrng()` returns a fresh RNG on the
            # same fixed seed, so calling it inside would give every node pair of a given size the
            # identical `x`: many pairs checked against one vector, with nothing to notice because
            # each assertion still passes.
            pairrng = H2FactoryTestUtils.checkrng()
            for (testnode, trialnode) in collect(keys(_i2odict(i2odense)))
                testpoly = treepolys[H2Trees.leveltolevelid(
                    tree, H2Trees.level(tree, testnode)
                )]
                raw = MultilevelKernelInterpolation.numpolynomials(testpoly)
                reduced = size(LinearIndices(testpoly))

                x = randn(pairrng, ComplexF64, length(H2Trees.values(tree, testnode)))

                @testset "forward, pair $((testnode, trialnode))" begin
                    indense = _builddenseinmoment(
                        poly, momentcollections[testnode], x, raw, reduced
                    )
                    intoep = _buildmoment(
                        MultilevelKernelInterpolation.ToeplitzPolynomialMoment,
                        testnode,
                        x,
                        raw,
                        H2Factory.standardtoeplitzsize(reduced),
                    )
                    outdense = _builddenseoutmoment(poly, raw, reduced)
                    outtoep = MultilevelKernelInterpolation.ToeplitzPolynomialMoment{
                        1,ComplexF64
                    }(
                        raw, H2Factory.standardtoeplitzsize(reduced)
                    )

                    level = H2Trees.level(tree, testnode)

                    for (in_, out, i2o, i2ochannel) in (
                        (indense, outdense, i2odense, i2ochanneldense),
                        (intoep, outtoep, i2otoep, i2ochanneltoep),
                    )
                        testbuffer = H2Factory.borrowbuffer(
                            i2ochannel, level, 1, H2Factory.isForward()
                        )
                        H2Factory.modulate_in!(
                            in_,
                            level,
                            H2Factory.isForward(),
                            H2Factory.modulation(i2o),
                            H2Factory.coupling(i2o),
                            i2ochannel,
                            SerialScheduler(),
                            testbuffer,
                        )
                        H2Factory.applyi2o!(
                            out,
                            in_,
                            i2o,
                            testnode,
                            trialnode,
                            H2Factory.ForwardTranslation(),
                            H2Factory.Overwrite(),
                            H2Factory.TranslationContext(SerialScheduler(), nothing),
                        )
                        trialbuffer = H2Factory.borrowbuffer(
                            i2ochannel, level, 1, H2Factory.isForward()
                        )
                        H2Factory.modulate_out!(
                            out,
                            level,
                            H2Factory.isForward(),
                            H2Factory.modulation(i2o),
                            H2Factory.coupling(i2o),
                            i2ochannel,
                            SerialScheduler(),
                            trialbuffer,
                        )
                    end

                    er = maximum(abs, outdense[1] - outtoep[1]) / maximum(abs, outtoep[1])
                    @test er < 1e-8
                end

                @testset "transpose, pair $((testnode, trialnode))" begin
                    indense = _builddenseinmoment(
                        poly, momentcollections[testnode], x, raw, reduced
                    )
                    intoep = _buildmoment(
                        MultilevelKernelInterpolation.ToeplitzPolynomialMoment,
                        testnode,
                        x,
                        raw,
                        H2Factory.standardtoeplitzsize(reduced),
                    )
                    outdense = _builddenseoutmoment(poly, raw, reduced)
                    outtoep = MultilevelKernelInterpolation.ToeplitzPolynomialMoment{
                        1,ComplexF64
                    }(
                        raw, H2Factory.standardtoeplitzsize(reduced)
                    )

                    level = H2Trees.level(tree, testnode)

                    for (in_, out, i2o, i2ochannel) in (
                        (indense, outdense, i2odense, i2ochanneldense),
                        (intoep, outtoep, i2otoep, i2ochanneltoep),
                    )
                        testbuffer = H2Factory.borrowbuffer(
                            i2ochannel, level, 1, H2Factory.isTranspose()
                        )
                        H2Factory.modulate_in!(
                            in_,
                            level,
                            H2Factory.isTranspose(),
                            H2Factory.modulation(i2o),
                            H2Factory.coupling(i2o),
                            i2ochannel,
                            SerialScheduler(),
                            testbuffer,
                        )
                        # transpose swaps receiving/translating (`couplingkey` owns that
                        # convention)
                        H2Factory.applyi2o!(
                            out,
                            in_,
                            i2o,
                            trialnode,
                            testnode,
                            H2Factory.TransposeTranslation(),
                            H2Factory.Overwrite(),
                            H2Factory.TranslationContext(SerialScheduler(), nothing),
                        )
                        trialbuffer = H2Factory.borrowbuffer(
                            i2ochannel, level, 1, H2Factory.isTranspose()
                        )
                        H2Factory.modulate_out!(
                            out,
                            level,
                            H2Factory.isTranspose(),
                            H2Factory.modulation(i2o),
                            H2Factory.coupling(i2o),
                            i2ochannel,
                            SerialScheduler(),
                            trialbuffer,
                        )
                    end

                    er = maximum(abs, outdense[1] - outtoep[1]) / maximum(abs, outtoep[1])
                    @test er < 1e-8
                end
            end
        end
    end
end
end
