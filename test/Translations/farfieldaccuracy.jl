module TestFarFieldAccuracy

using Test
using LinearAlgebra, OhMyThreads
using CompScienceMeshes, BEAST
using H2Trees, H2Factory, MultilevelKernelInterpolation
using H2FactoryTestUtils

# Neither `interpolation.jl` nor `toeplitz_equivalence.jl` checks this: `interpolation.jl`
# compares the raw translation *matrix* against a high-order reference matrix, never
# touching real moments/DOFs; `toeplitz_equivalence.jl`'s granular testset drives real
# moments through `modulate_in!`/`applycoupling!`/`modulate_out!`
# but only checks dense vs Toeplitz *agree with each other* -- two implementations that
# agree could still both be wrong. This testset closes that gap: build a moment from actual
# DOF amplitudes, push it through the real pipeline (via `H2FactoryTestUtils.checki2otranslation`,
# which owns the correct forward/transpose/adjoint call sequence/role-swap -- H2Factory
# defines and enforces that ordering, so the check for it belongs there, not here),
# reconstruct a DOF-space output vector, and compare against a BEAST dense assembly for the
# same DOF pair (no multilevel interpolation/interpolation machinery involved at all) -- i.e. does the I2O
# translation actually reproduce the true far-field interaction, not just agree with a
# second implementation of itself.
function _i2odict(i2o)
    hasfield(typeof(i2o), :dict) && return i2o.dict
    return _i2odict(i2o.i2ocollection)
end
# Demoted Toeplitz B-spline path: the map is a composed `H2Factory.I2OTranslation` (coupling +
# modulation), not a `.i2ocollection`-nesting wrapper -- unwrap to its plain Toeplitz coupling.
_i2odict(i2o::H2Factory.I2OTranslation) = _i2odict(H2Factory.coupling(i2o))

function _withplanchunks(f, plan)
    schedule = H2Factory.planschedule(plan, Base.Threads.nthreads())
    return H2Factory.withnchunks(H2Factory.schedulechunkcounts(schedule)) do
        return f()
    end
end

_buildtoeplitzmomentbuilder(raw, reduced, dims) =
    () -> MultilevelKernelInterpolation.ToeplitzPolynomialMoment{dims,ComplexF64}(
        raw, H2Factory.standardtoeplitzsize(reduced)
    )

# `MultilevelKernelInterpolation.α(operator, polynomial)` is mode-independent for every operator here except the
# Maxwell double layer, which sign-flips between forward and transpose/adjoint -- exactly
# mirroring `MultilevelKernelInterpolation.α(A::AbstractInterpolationMap{...,<:MultilevelKernelInterpolation.MWDoubleLayer3D},
# ::Union{isTranspose,isAdjoint}) = -α(A)` (`mapoperations.jl`), just computed from
# `(operator, polynomial)` directly since this test has no full `GalerkinKernelInterpolation`/
# `AbstractInterpolationMap` object to dispatch on. This was found via this test: a raw (α=1)
# forward/transpose/adjoint comparison shows the ∇Cross (MWDoubleLayer3D) translation
# reproduces exactly `-truereference` in transpose/adjoint mode, needing the negation; the
# ∇Green gradient operators (HH3DDoubleLayerFDBIOGradientI2O and its "Transposed" counterpart)
# looked superficially similar (also a first-derivative kernel) and had an *identical* sign
# flip applied here previously, but empirically their raw transpose/adjoint ratio already
# equals `+truereference`/`MultilevelKernelInterpolation.α(op,poly)` unchanged with NO flip needed -- that pre-existing
# flip for the gradient operators was itself wrong (a hypothesis never validated against a
# true dense reference until this test existed) and has been removed both here and in
# `mapoperations.jl`.
_αformode(op, poly, ::H2Factory.isForward) = MultilevelKernelInterpolation.α(op, poly)
_αformode(op, poly, ::H2Factory.isTranspose) = _αmodeflipped(op, poly)
# Adjoint conjugates the coefficient: `adjoint(α·M) = conj(α)·adjoint(M)` (Part E adjoint-
# factorization slice -- mirrors the map's `MultilevelKernelInterpolation.α(A, ::isAdjoint) = conj.(αtranspose)`). The old
# convention got this conjugation from a trailing `conj!(y)`, now removed.
_αformode(op, poly, ::H2Factory.isAdjoint) = conj.(_αmodeflipped(op, poly))

_αmodeflipped(op, poly) = MultilevelKernelInterpolation.α(op, poly)
_αmodeflipped(op::MultilevelKernelInterpolation.MWDoubleLayer3D, poly) =
    -MultilevelKernelInterpolation.α(op, poly)

@testset verbose=true "I2O translation reproduces the true dense far-field interaction" begin
    λ = 1.0
    k = 2π / λ

    m = readmesh(
        joinpath(pkgdir(MultilevelKernelInterpolation), "test", "assets", "in", "sphere.in")
    )

    # Covers all six `i2otranslation(operator)` shapes in `beastoperators.jl`:
    #   - a plain scalar kernel with a scalar (1-component) moment (Helmholtz single layer);
    #   - a scalar kernel with *multi*-component, *symmetric* (same on test and trial side)
    #     moments and a nontrivial per-component `α`/`β` prefactor applied at reconstruction:
    #     `Maxwell3D.singlelayer`'s `LaplaceMomentDiv{4,...}` (`[x,y,z,div]`, `α=[α,α,α,β]` --
    #     this is exactly the case that exposed a real bug this session: the dense B-spline
    #     path's scalar-translation dispatch only ever touched moment component 1, silently
    #     dropping the other 3) and `Helmholtz3D.hypersingular`'s `HyperSingularMoment{6,...}`
    #     (`α=[α,α,α,β,β,β]`);
    #   - three genuinely cross-component-coupled (3x3 matrix of matrices) kernels sharing the
    #     same dispatch shape and symmetric dims=3 moment (`DyadicGreen3DTranslation` for EFIE,
    #     `∇∇Green3DTranslation` for the Maxwell hypersingular operator,
    #     `∇CrossGreen3DTranslation` for the Maxwell double layer) -- the first of these is the
    #     one whose forward/transpose coupling handling was fixed earlier this
    #     session;
    #   - `∇Green3DTranslation` (the gradient double-layer operators), the one shape with
    #     *asymmetric* test/trial moment dims (1 vs 3, swapped between the plain and
    #     "Transposed" variant) -- these have a mode-*independent* `α` (unlike
    #     `∇CrossGreen3DTranslation` above); the reason `checki2otranslation` takes separate
    #     `buildtestmoment`/`buildtrialmoment` instead of one shared builder.
    # Per-operator tolerance `ϵ`, tailored rather than one blanket bound: the true accuracy at
    # this order/mesh/sample size genuinely differs by kernel (scalar kernels are markedly
    # more accurate than dyadic/derivative-heavy ones), so a single loose bound sized for the
    # worst operator would leave the more accurate ones with far less regression sensitivity
    # than they could have.
    #
    # THE CONTRACT. These are a deterministic regression check on the EXACT probe vectors
    # `H2FactoryTestUtils.checkrng()` produces below. They are NOT a bound on the error for a
    # random input: a 192-seed sweep of this file's exact nesting found samples up to 1.32x the
    # bound, which is why `Helmholtz3D.singlelayer` keeps `ϵ=6e-4` although that sweep saw
    # 7.85e-4. CHANGING THE SEED, THE MESH, THE TREE, OR THE POLYNOMIAL ORDER INVALIDATES ALL
    # EIGHT NUMBERS and requires re-deriving them the same way. Stated because an earlier
    # version of this comment claimed distribution coverage instead; the measurement that
    # withdrew that claim is recorded in the #193 commit, not here.
    #
    # SIZING. Five bounds are sized at 3.0x to 3.4x their seeded probe maximum, rounded up:
    # Helmholtz3D.hypersingular, Maxwell3D.singlelayer, MultilevelKernelInterpolation.efie, MultilevelKernelInterpolation.MWDoubleLayer3D and
    # MultilevelKernelInterpolation.HH3DDoubleLayerFDBIOGradientI2O. The other three KEEP WIDER MARGINS, 3.3x for
    # Helmholtz3D.singlelayer, 4.1x for MultilevelKernelInterpolation.HH3DDoubleLayerTransposedFDBIOGradientI2O and 4.8x
    # for MultilevelKernelInterpolation.MWHyperSingular3D, because sizing them from their probes would tighten them by at
    # most 1.2x and the churn is not worth it. The platform component of every margin is
    # measured: across `JULIA_NUM_THREADS` of 1, 8 and 16 the 432 probe errors are BITWISE
    # identical, and eight BLAS threads move 14 of them by at most 1.2e-11 relative. So nearly
    # all of the margin is insurance against a different CPU or BLAS build, which a single
    # machine cannot sample, rather than a measured requirement.
    #
    # CALIBRATION, FOR A UNIFORM RECONSTRUCTION GAIN ERROR ONLY. The one mutation class measured
    # here is scaling `α` by `(1 + δ)`, over 15 values of |δ| from 1e-4 to 1.2e-2 in both signs,
    # which agree to within 17%. For that class the injected error appears as δ in this metric,
    # so a bound rejects δ only once δ exceeds roughly `ϵ`, and every factor of headroom is paid
    # one for one in detection. A regression with different spatial or per-component structure
    # has its OWN threshold and is NOT characterised by this table.
    #
    #   operator                                        probe     ϵ        rejects |δ| >=
    #   Helmholtz3D.singlelayer                         1.82e-4   6e-4     6e-4
    #   Helmholtz3D.hypersingular                       9.06e-4   3e-3     3e-3
    #   Maxwell3D.singlelayer                           1.55e-3   5e-3     5e-3
    #   MultilevelKernelInterpolation.efie                                       1.33e-3   4e-3     5e-3
    #   MultilevelKernelInterpolation.MWHyperSingular3D                          1.04e-3   5e-3     5e-3
    #   MultilevelKernelInterpolation.MWDoubleLayer3D                            7.92e-4   2.5e-3   2.5e-3
    #   MultilevelKernelInterpolation.HH3DDoubleLayerFDBIOGradientI2O            1.48e-4   5e-4     5e-4
    #   MultilevelKernelInterpolation.HH3DDoubleLayerTransposedFDBIOGradientI2O  1.46e-4   6e-4     6e-4
    #
    # Helmholtz3D.hypersingular at its old `ϵ=2e-2` rejected NO gain error up to 1.2e-2.
    #
    # THE METRIC WAS KEPT DELIBERATELY. `checki2otranslation` reports
    # `maximum(abs, y - ytrue) / maximum(abs, ytrue)`, which has a heavy tail near a cancelling
    # output. Three replacements were measured and rejected: an RMS denominator widens the
    # spread, pure operator scaling costs about 14x median sensitivity for almost no stability
    # gain, and a mixed `rtol`/`atol` criterion is provably optimised at `atol = 0` for
    # worst-case detection. Numbers and derivation are in the #193 commit.
    for (oplabel, op, X, ϵ) in [
        (
            "Helmholtz3D.singlelayer (scalar kernel, scalar moment)",
            Helmholtz3D.singlelayer(; wavenumber=k),
            lagrangecxd0(m),
            6e-4,
        ),
        (
            "Helmholtz3D.hypersingular (scalar kernel, 6-component moment, α/β prefactor)",
            Helmholtz3D.hypersingular(; wavenumber=k),
            lagrangec0d1(m),
            3e-3,
        ),
        (
            "Maxwell3D.singlelayer (scalar kernel, 4-component moment, α/β prefactor)",
            Maxwell3D.singlelayer(; wavenumber=k),
            raviartthomas(m),
            5e-3,
        ),
        (
            "MultilevelKernelInterpolation.efie (dyadic kernel, 3-component moment)",
            MultilevelKernelInterpolation.efie(; wavenumber=k),
            raviartthomas(m),
            4e-3,
        ),
        (
            "MultilevelKernelInterpolation.MWHyperSingular3D (∇∇, 3-component moment)",
            MultilevelKernelInterpolation.MWHyperSingular3D(; wavenumber=k),
            raviartthomas(m),
            5e-3,
        ),
        (
            "MultilevelKernelInterpolation.MWDoubleLayer3D (∇×, 3-component moment)",
            MultilevelKernelInterpolation.MWDoubleLayer3D(; wavenumber=k),
            raviartthomas(m),
            2.5e-3,
        ),
        (
            "MultilevelKernelInterpolation.HH3DDoubleLayerFDBIOGradientI2O (∇, asymmetric 1/3-component moment)",
            MultilevelKernelInterpolation.HH3DDoubleLayerFDBIOGradientI2O(; wavenumber=k),
            lagrangecxd0(m),
            5e-4,
        ),
        (
            "MultilevelKernelInterpolation.HH3DDoubleLayerTransposedFDBIOGradientI2O (∇, asymmetric 3/1-component moment)",
            MultilevelKernelInterpolation.HH3DDoubleLayerTransposedFDBIOGradientI2O(;
                wavenumber=k
            ),
            lagrangecxd0(m),
            6e-4,
        ),
    ]
        # `maxprotrusion=0.25` constrains tree construction itself (via the default,
        # point-based `ComputeProtrusionFunctor`) to stop subdividing once boxes are large
        # enough to keep protrusion under control -- matching the standard usage pattern in
        # TAP-MultilevelKernelInterpolation (`parameters.jl`'s `maxprotrusion = 0.25`), rather than leaving it
        # unconstrained (`TwoNTree`'s own default is `NaN`, i.e. no cap at all) and hoping
        # the mesh/tree combination happens to stay geometrically safe (see
        # `MultilevelKernelInterpolation.checkprotrusionsafety`, which now checks the real per-pair box gap rather than
        # an analytical per-level margin). The separate, more precise
        # `H2Trees.BEASTProtrusionFunctor(X)`-based measurement below (accounting for actual
        # mesh-element extent, not just point positions) is still what actually sizes the
        # B-spline buffer via `levelpolynomials(...; protrusion=...)`.
        tree = buildtree(
            X;
            builder=TwoNTreeBuilder(;
                minvalues=120,
                protrusion=H2Trees.ProtrusionCheck(;
                    max=0.25, compute=H2Trees.BEASTProtrusionFunctor(X)
                ),
            ),
        )

        # Ground truth: a plain dense BEAST assembly for the whole (small, fixture) mesh,
        # with no multilevel interpolation/interpolation machinery involved whatsoever.
        Adense = assemble(op, X, X)

        plans = H2Trees.buildplans(
            tree;
            builder=H2Trees.PlanBuilder(; aggregatenode=H2Trees.AggregateAllNodesFunctor()),
        )
        plan = H2Trees.translatingplan(
            plans.trialaggregationplan, plans.testdisaggregationplan
        )

        # The real, mesh/space-derived relative protrusion -- exactly what `GalerkinKernelInterpolation`
        # computes and feeds into `levelpolynomials`. This matters a lot here (unlike in this
        # file's sibling tests, which only check internal dense-vs-Toeplitz agreement):
        # `noincrease`/`increase` for `BufferedBsplinePolynomialEquidistant` resolve the buffer
        # from this value via `MultilevelKernelInterpolation.resolvebuffer(MultilevelKernelInterpolation.AutoBufferFromProtrusion(0), p, b,
        # protrusion)` (see `bspline_buffer.md`), and unlike Lagrange (global polynomial
        # support, tolerates a DOF's quadrature points sitting slightly outside the nominal box),
        # a B-spline has *compact* support -- any protrusion past its domain isn't approximated
        # inaccurately, it's silently zeroed out entirely. Passing `protrusion=0` (this file's
        # sibling tests' implicit default, harmless there) understates the needed buffer and was
        # first found, via exactly this testset, to produce 15-58% relative error against the
        # true dense reference -- entirely a test-configuration bug (insufficient buffering), not
        # a translation-pipeline bug: with the real protrusion below, error drops to ~1e-5,
        # matching Lagrange.
        protrusion = H2Trees.maxprotrusion(
            tree; computeprotrusion=H2Trees.BEASTProtrusionFunctor(X)
        )

        for (polylabel, poly) in [
            (
                "equidistant Lagrange",
                MultilevelKernelInterpolation.BarycentricLagrangePolynomialEquidistant3D(7),
            ),
            (
                "buffered B-spline (buffer sized from the real protrusion)",
                MultilevelKernelInterpolation.BufferedBsplinePolynomialEquidistant3D{
                    Float64
                }(
                    4, 10, MultilevelKernelInterpolation.AutoBufferFromProtrusion(0)
                ),
            ),
        ]
            @testset "$oplabel / $polylabel" begin
                treepolys = MultilevelKernelInterpolation.levelpolynomials(
                    poly, H2Trees.levels(tree), (level) -> false; protrusion=protrusion
                )
                MultilevelKernelInterpolation.checkprotrusionsafety(tree, treepolys, plan)
                leafpolys = MultilevelKernelInterpolation.leafpolynomials(tree, treepolys)

                trialmomentcollections = Dict(
                    zip(
                        H2Trees.leaves(tree),
                        MultilevelKernelInterpolation.TrialMomentCollections(
                            op, tree, leafpolys, X
                        ),
                    ),
                )
                testmomentcollections = Dict(
                    zip(
                        H2Trees.leaves(tree),
                        MultilevelKernelInterpolation.TestMomentCollections(
                            op, tree, leafpolys, X
                        ),
                    ),
                )

                # Toeplitz-only: dense-vs-Toeplitz agreement is already thoroughly verified
                # elsewhere (`toeplitz_equivalence.jl`, including its own dedicated EFIE
                # regression test), and the dense (non-Toeplitz) path is *much* more
                # expensive to construct here -- for the dyadic-shaped operators
                # (EFIE/MWHyperSingular3D/MWDoubleLayer3D) combined with the buffered
                # B-spline config, `MultilevelKernelInterpolation.I2OTranslationCollection`'s dense construction
                # builds the reduction for every node pair in the whole tree (not just the
                # 3 pairs sampled below) via 9 separate reductions per pair, which measured
                # tens of GB of memory and many minutes per (operator, polynomial)
                # combination earlier this session -- prohibitive across 8 operators.
                i2otoep, i2ochanneltoep = _withplanchunks(plan) do
                    MultilevelKernelInterpolation.ToeplitzI2OTranslationCollection(
                        op, tree, treepolys, plan, X; verbose=false
                    )
                end

                pairs = collect(keys(_i2odict(i2otoep)))
                @test !isempty(pairs)

                testdims = MultilevelKernelInterpolation.dims(
                    MultilevelKernelInterpolation.testmoment(op, poly)
                )
                trialdims = MultilevelKernelInterpolation.dims(
                    MultilevelKernelInterpolation.trialmoment(op, poly)
                )

                # Sampling a handful of pairs (rather than every pair) keeps runtime
                # reasonable given the multiplicative cost of 8 operators x 2 polynomial
                # types x 3 modes.
                for (testnode, trialnode) in first(pairs, min(3, length(pairs)))
                    @test testnode in H2Trees.leaves(tree)
                    @test trialnode in H2Trees.leaves(tree)

                    level = H2Trees.level(tree, testnode)
                    testpoly = treepolys[H2Trees.leveltolevelid(tree, level)]
                    raw = MultilevelKernelInterpolation.numpolynomials(testpoly)
                    reduced = size(LinearIndices(testpoly))

                    testdofs = H2Trees.values(tree, testnode)
                    trialdofs = H2Trees.values(tree, trialnode)
                    Adenseblock = Adense[testdofs, trialdofs]

                    for (implabel, i2o, i2ochannel, buildtestmoment, buildtrialmoment) in ((
                        "Toeplitz",
                        i2otoep,
                        i2ochanneltoep,
                        _buildtoeplitzmomentbuilder(raw, reduced, testdims),
                        _buildtoeplitzmomentbuilder(raw, reduced, trialdims),
                    ),)
                        for (modelabel, mode, truereference, xsize) in (
                            (
                                "forward",
                                H2Factory.isForward(),
                                Adenseblock,
                                length(trialdofs),
                            ),
                            (
                                "transpose",
                                H2Factory.isTranspose(),
                                transpose(Adenseblock),
                                length(testdofs),
                            ),
                            (
                                "adjoint",
                                H2Factory.isAdjoint(),
                                Adenseblock',
                                length(testdofs),
                            ),
                        )
                            @testset "$implabel $modelabel" begin
                                H2FactoryTestUtils.checki2otranslation(
                                    testmomentcollections[testnode],
                                    trialmomentcollections[trialnode],
                                    i2o,
                                    i2ochannel,
                                    testnode,
                                    trialnode,
                                    level,
                                    buildtestmoment,
                                    buildtrialmoment,
                                    # SEEDED, and not decoratively. Same flake class the accuracy
                                    # helpers were seeded for: `n` random vectors against a FIXED
                                    # `ϵ`, so an unseeded draw turns a tolerance that sits near the
                                    # error into a coin flip. Measured here, not hypothetical: the
                                    # `Toeplitz adjoint` case reported `er = 6.1977e-4` against
                                    # `ϵ = 6e-4`, a 3.3% overshoot, on a run whose immediate
                                    # predecessor passed with behaviourally identical code.
                                    # A fresh RNG per call, so adding or removing a case upstream
                                    # does not silently change the vectors every later case sees.
                                    let rng = H2FactoryTestUtils.checkrng()
                                        () -> randn(rng, ComplexF64, xsize)
                                    end,
                                    truereference;
                                    mode=mode,
                                    α=_αformode(op, poly, mode),
                                    scheduler=SerialScheduler(),
                                    n=3,
                                    # Per-operator `ϵ`, tailored above in the operator list
                                    # (not a blanket bound) -- see that comment for the
                                    # measured max relative error this headroom is based on.
                                    ϵ=ϵ,
                                    verbose=false,
                                )
                            end
                        end
                    end
                end
            end
        end
    end
end

# Complementary to the granular (single node-pair, forward-only) testset above: builds a
# real `GalerkinKernelInterpolation` (both dense and Toeplitz I2O translators), so this exercises the
# `conjmomentTransposeAdjoint!` per-node bracketing that transpose/adjoint mode needs around
# aggregation/disaggregation -- the granular testset above deliberately doesn't attempt that
# (see `toeplitz_equivalence.jl`'s granular testset, which punts on adjoint for the same
# reason) -- and compares against a whole-mesh dense BEAST reference for all three
# `LinearMaps` modes at once.
#
# Deferred (end-to-end/full-GalerkinKernelInterpolation-level testing, as opposed to the node-pair-level
# testset above, explicitly deprioritized for now): also, this specific operator/space
# combination's *default* `NearInteractionsAssembler` hits an unrelated `BEAST.blockassembler`
# `MethodError` (a pre-existing gap this session didn't touch), so it currently needs
# `NearInteractionsAssembler=H2Factory.DummyNearInteractionsAssembler` plus comparing against
# `H2Factory.farinteractions(Adense, A)` rather than `Adense` directly -- not yet verified to
# pass. Re-enable and finish debugging when end-to-end testing is picked back up.
# RE-ENABLED. This testset spent an unknown stretch behind a bare `if false`, which the suite
# walks past without reporting a skip, so the gate stayed green while the only end-to-end
# forward/transpose/adjoint-versus-dense check for a `GalerkinKernelInterpolation` never ran. It had also gone
# stale: it configured the map through `polynomialstrategy=`/`assemblystrategy=`/`tree=` keywords
# and an `MultilevelKernelInterpolation.AssemblyStrategy` that no longer exist, so it could not have run even if enabled.
# Ported to the current surface: everything goes through `strategy=InterpolationStrategy(...)`, the tree is
# positional, and the near-interactions assembler is selected by a STRATEGY
# (`H2Factory.DummyNearInteractionsStrategy`) rather than by an assembler type.
#
# It matters more than an ordinary revived test. MultilevelKernelInterpolation selects the plain adjoint leaf multiply
# because its leaf collections are real, and `MultilevelKernelInterpolation/adjointleafmul.jl` checks that the SELECTION
# follows the collections. Nothing there checks that the selected multiply is numerically right
# end to end; this does, in all three orientations.
@testset verbose=true "Full GalerkinKernelInterpolation reproduces the true dense matrix (forward/transpose/adjoint)" begin
    λ = 1.0
    k = 2π / λ
    op = Helmholtz3D.singlelayer(; wavenumber=k)

    m = readmesh(
        joinpath(pkgdir(MultilevelKernelInterpolation), "test", "assets", "in", "sphere.in")
    )
    X = lagrangecxd0(m)
    tree = buildtree(
        X;
        builder=TwoNTreeBuilder(;
            minvalues=120,
            protrusion=H2Trees.ProtrusionCheck(;
                max=0.25, compute=H2Trees.BEASTProtrusionFunctor(X)
            ),
        ),
    )

    Adense = assemble(op, X, X)

    for (label, poly) in [
        (
            "equidistant Lagrange",
            MultilevelKernelInterpolation.BarycentricLagrangePolynomialEquidistant3D(7),
        ),
        (
            "buffered B-spline (buffer sized from the real protrusion)",
            MultilevelKernelInterpolation.BufferedBsplinePolynomialEquidistant3D{Float64}(
                4, 10, MultilevelKernelInterpolation.AutoBufferFromProtrusion(0)
            ),
        ),
    ]
        @testset "$label" begin
            # `NearInteractionsAssembler=H2Factory.DummyNearInteractionsAssembler` matches
            # every other `GalerkinKernelInterpolation` construction in this test suite -- the *default*
            # near-interactions assembler hits an unrelated `BEAST.blockassembler`
            # `MethodError` for this operator/space combination (a pre-existing gap, not
            # touched this session), which is presumably why no existing test constructs a
            # real one. `A` therefore only represents the *far* interaction, so the
            # comparison target is `H2Factory.farinteractions(Adense, A)` (the same utility
            # `H2FactoryTestUtils.checkinteractions` uses), not `Adense` itself.
            _buildmap(i2ostrategy) =
                MultilevelKernelInterpolation.GalerkinKernelInterpolation(
                    op,
                    X,
                    tree;
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
                            i2ostrategy=i2ostrategy
                        ),
                    ),
                )

            # SEEDED, and hoisted out of the loop so it ADVANCES. A newly enabled test with an
            # unseeded draw against a fixed relative bound is a flake waiting to be blamed on
            # whatever lands next; the same class already turned this file's `Toeplitz adjoint` case
            # red. But `checkrng()` inside the loop would be the opposite mistake: a fresh RNG on the
            # same fixed seed hands the dense and Toeplitz passes the IDENTICAL `x`, halving the
            # coverage without failing anything.
            implrng = H2FactoryTestUtils.checkrng()
            for (implabel, A) in (
                ("dense", _buildmap(MultilevelKernelInterpolation.DenseI2OStrategy())),
                (
                    "Toeplitz",
                    _buildmap(MultilevelKernelInterpolation.ToeplitzI2OStrategy()),
                ),
            )
                @testset "$implabel" begin
                    Adensefar = H2Factory.farinteractions(Adense, A)
                    x = randn(implrng, ComplexF64, size(A, 2))

                    yforward = A * x
                    ytrueforward = Adensefar * x
                    @test maximum(abs, yforward - ytrueforward) /
                          maximum(abs, ytrueforward) < 1e-3

                    ytranspose = transpose(A) * x
                    ytruetranspose = transpose(Adensefar) * x
                    @test maximum(abs, ytranspose - ytruetranspose) /
                          maximum(abs, ytruetranspose) < 1e-3

                    yadjoint = A' * x
                    ytrueadjoint = Adensefar' * x
                    @test maximum(abs, yadjoint - ytrueadjoint) /
                          maximum(abs, ytrueadjoint) < 1e-3
                end
            end
        end
    end
end

end
