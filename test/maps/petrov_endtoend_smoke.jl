module TestPetrovEndToEndSmoke

using Test
using LinearAlgebra
using CompScienceMeshes, BEAST
using H2Trees, H2Factory
using H2FactoryTestUtils
using MultilevelKernelInterpolation

# Mirrors `endtoend_smoke.jl`, but for `PetrovGalerkinKernelInterpolation` -- genuinely distinct test/trial spaces
# (not the same space passed twice, which would only re-exercise `GalerkinKernelInterpolation`'s single-tree fast
# path). No test in the repo called `PetrovGalerkinKernelInterpolation` directly before the Milestone 6 refactor,
# so this was the first real exercise of `matvecscheduler`, `checkprotrusionsafety`, the
# side-specific I2O polynomial dicts, and the `aggregationmode`-aware `translatingplan` dispatch for
# the Petrov (`BlockTree`) case.
#
# HISTORY -- worth keeping, because the original diagnosis here was wrong twice over.
#
# A `lagrangec0d1` test side paired with `lagrangecxd0` trial used to produce wildly wrong output
# (relative errors of order 1, i.e. no correct digits). That was attributed to `lagrangec0d1`'s
# shared-support basis functions interacting badly with per-node tree-position assignment, and
# worked around by using `lagrangecxd0` on both sides. Both halves of that were false:
#
#   - The failure was NOT about shared support, and NOT about ~30-45% of rows: measurement showed
#     only 5 of 1609 rows wrong (0.3%), all confined to shallow level-3 leaves.
#   - The real cause was ADMISSIBILITY. `H2Trees`' legacy centre-distance near predicate placed the
#     near/far boundary at `sqrt(12)*halfsize`, which in 3D is EXACTLY the corner-neighbour centre
#     distance. Two independently built `BlockTree` sides do not share a box grid (each side fits
#     its own root box), so corner neighbours were offset a hair past that knife-edge boundary and
#     admitted into the FAR field -- and the far-field interpolation was then asked to interpolate
#     across two boxes that physically touch. Fixed in `H2Trees` by `isneargap`, a box-gap
#     predicate with an explicit one-half-size admissibility margin (see `DEFAULTNEARGAPBOXES`).
#
# The old workaround is therefore obsolete, and all three space pairings are exercised below.
#
# DEFAULT CONFIGURATION. This test passes only `nearinteractionsstrategy`, so it exercises whatever
# `PetrovGalerkinKernelInterpolation`'s DEFAULT is -- and that default is now order-6 equidistant Lagrange + Toeplitz
# I2O (it was Chebyshev + dense when the measurements below were taken). The tolerances are retained
# from that earlier characterization and RE-VERIFIED green under the new default; they hold with MORE
# headroom now, not less (forward measured ~5e-4 under Toeplitz vs the 8e-3 bound). The specific
# per-mode/40-trial magnitudes in this comment are therefore prior-default provenance -- the exact
# numbers differ under Toeplitz -- but the STRUCTURAL claims (transpose == transpose of the forward
# factors; mode-independent approximation error) are I2O-strategy-independent and still hold. Pin the
# dense route explicitly (`TranslatorStrategy(; i2ostrategy=DenseI2OStrategy())`) to test it instead.
#
# TOLERANCES. Every number below was measured, 40 trials per configuration, using `checkfarmul`'s own
# metric -- `_maxerror = max|y - yH2| / max|y|`, NOT a relative 2-norm, which is several times
# smaller and would size these bounds far too tightly. Observed maxima (prior default; see above):
#
#     cxd0/cxd0   fwd 2.3e-3    transpose 3.05e-2   adjoint 2.64e-2
#     c0d1/cxd0   fwd 2.44e-3   transpose 2.66e-2   adjoint 3.56e-2
#     cxd0/c0d1   fwd 3.7e-4    transpose 5.6e-4    adjoint 7.4e-4
#
# The old `3e-1` transpose/adjoint bound was about 10x above the real maximum -- loose, and its
# stated justification (a "heavy tail" at "near-margin translating pairs already flagged by
# `checkprotrusionsafety`") was describing the corner-neighbour pairs, i.e. the admissibility bug.
#
# WHY TRANSPOSE/ADJOINT NEED A LOOSER BOUND THAN FORWARD -- a measurement effect, NOT a transpose
# bug. It looks alarming (roughly 10x worse whenever `lagrangecxd0` is the trial space), so the
# operator itself was checked directly, by extracting the H2 far operator as a dense matrix in each
# mode. ESTABLISHED:
#
#   - the transposed operator IS exactly the transpose of the forward one, to ~1e-14
#     (`||M_transp - transpose(M_fwd)|| / ||M_fwd|| = 1.1e-14`), and likewise for the adjoint. So
#     `transpose(A)` applies the transpose of the forward factors, not a second approximation;
#   - the approximation error against the dense reference is MODE-INDEPENDENT: relative Frobenius
#     6.604e-4 for forward, transpose and adjoint alike, agreeing to ~14 digits;
#   - measured on the MATRICES with `_maxerror`'s own max-norm, all three modes agree too (0.0337).
#
# The gap therefore appears only when that metric is fed a random VECTOR, and it is a property of the
# metric rather than of the approximation: `_maxerror` is not transpose-invariant. Decomposed on the
# `cxd0/cxd0` configuration (40 draws), the denominator `max|Fx|` is unchanged between modes (ratio
# 0.996) and the entire effect sits in the numerator `max|Ex|` (ratio 17.9).
#
# LIKELY EXPLANATION, NOT YET CONFIRMED: the error matrix `E` looks column-concentrated on that same
# configuration -- median column-max 2.6e-10 against median row-max 2.2e-8 (~90x skew), though both
# reach the same global max 8.6e-8. If a few trial DOFs are approximated worse with their error
# spread thinly across many rows, then `Ex` scatters those entries over many output components while
# `E'x` collapses each such column into ONE component where they accumulate, which a max-norm would
# find. That story predicts low skew AND ratio ~1 for `cxd0/c0d1`, which shows almost no asymmetry --
# but that control has NOT been run, so the mechanism is a hypothesis consistent with one
# configuration, not an established cause. Do not cite it as settled without running the control.
#
# The bounds below were sized for ~2-3x headroom over the prior default's observed maxima; under the
# current Toeplitz default the forward headroom is larger (~16x, forward ~5e-4 vs 8e-3). Each
# configuration keeps its own bound so a regression in the accurate pairing cannot hide behind a
# bound sized for the coarser ones.
#
# THE DRAWS ARE NOW SEEDED. This comment used to add "the RNG here is unseeded, so tail draws
# happen", and that was the whole problem: the tightest bound here (`lagrangecxd0 / lagrangec0d1` at
# 2e-3, against 8e-3/8e-2 for its siblings) sat inside the draw-to-draw spread and failed
# intermittently at 2.2275e-3, which three standalone re-runs then could not reproduce. A tolerance
# that only usually holds cannot distinguish a real regression from a tail draw. The accuracy helpers
# now draw from `H2FactoryTestUtils.CHECKSEED` via `checkrng()`, so these bounds either hold or do
# not, reproducibly. Measured on the fixed fixture, this config's forward error maxes at 7.185e-4
# over its ten draws, 2.78x inside the 2e-3 bound. That is headroom for THIS fixture and it removes
# the run-to-run sampling variation; it is not a measurement of the old unseeded distribution, so it
# does not by itself establish that 2e-3 was a well-sized bound over that distribution. If a future
# change needs that question answered, sweep it deliberately by passing your own `rng`.
@testset verbose = true "Petrov end-to-end BEM solve smoke tests" begin
    m = readmesh(
        joinpath(pkgdir(MultilevelKernelInterpolation), "test", "assets", "in", "sphere.in")
    )

    dummynear(op, X, Y) =
        H2Factory.DummyNearInteractionsStrategy(BEAST.defaultquadstrat(op, X, Y))

    # (label, testspace, trialspace, forward ϵ, transpose/adjoint ϵ)
    configs = (
        ("lagrangecxd0 / lagrangecxd0", lagrangecxd0, lagrangecxd0, 8e-3, 8e-2),
        ("lagrangec0d1 / lagrangecxd0", lagrangec0d1, lagrangecxd0, 8e-3, 8e-2),
        ("lagrangecxd0 / lagrangec0d1", lagrangecxd0, lagrangec0d1, 2e-3, 2e-3),
    )

    @testset "Helmholtz3D.doublelayer (Petrov): $label" for (
            label, testfn, trialfn, ϵforward, ϵtransposed
        ) in configs

        op = Helmholtz3D.doublelayer(; wavenumber=1.0)
        testspace = testfn(m)
        trialspace = trialfn(m)

        Adense = assemble(op, testspace, trialspace)
        A = MultilevelKernelInterpolation.PetrovGalerkinKernelInterpolation(
            op,
            testspace,
            trialspace;
            strategy=MultilevelKernelInterpolation.InterpolationStrategy(
                op,
                testspace,
                trialspace;
                nearinteractionsstrategy=dummynear(op, testspace, trialspace),
            ),
        )

        # The assembled translation scheme must be geometrically admissible in the first place.
        # This is the check that would have caught the corner-neighbour bug directly, at
        # construction, instead of it surfacing as wrong matrix entries.
        report = H2Factory.Diagnostics.admissibilityreport(A; coverage=false, throw=false)
        @test report.ok

        farinteractions = H2Factory.farinteractions(Adense, A)
        H2FactoryTestUtils.checkfarmul(farinteractions, A; n=10, ϵ=ϵforward, verbose=false)

        @testset "transpose/adjoint mul!" begin
            transposefarinteractions = H2Factory.farinteractions(
                transpose(Adense), transpose(A)
            )
            H2FactoryTestUtils.checkfarmul(
                transposefarinteractions, transpose(A); n=10, ϵ=ϵtransposed, verbose=false
            )

            adjointfarinteractions = H2Factory.farinteractions(adjoint(Adense), adjoint(A))
            H2FactoryTestUtils.checkfarmul(
                adjointfarinteractions, adjoint(A); n=10, ϵ=ϵtransposed, verbose=false
            )
        end

        @testset "mul! allocation baseline" begin
            x = randn(H2FactoryTestUtils.checkrng(), ComplexF64, size(A, 2))
            y = Vector{ComplexF64}(undef, size(A, 1))
            mul!(y, A, x)  # warm up (compilation, buffer setup)
            # Fresh baseline, not assumed to match `GalerkinKernelInterpolation`'s -- see that file's identical
            # comment for why this isn't literally allocation-free. Measured (SERIAL) 10048 bytes for the
            # cxd0/cxd0 pairing and 20480 for the two mixed pairings.
            #
            # The budget SCALES WITH THREAD COUNT: the multithreaded traversal (`@tasks` in
            # `_aggregatelevel!`/`_disaggregatetranslatelevel!`) allocates per-task closure/scheduler
            # state -- pre-existing infrastructure, NOT the level transfers, which allocate nothing at the
            # `applytransfer!` boundary (profiled). Measured worst case (the mixed pairings) ~20k serial,
            # ~211k at 4 threads; `100_000 * nthreads` gives headroom over that at every thread count. The
            # old flat serial threshold silently assumed single-thread runs.
            @test (@allocated mul!(y, A, x)) < 100_000 * Threads.nthreads()
        end
    end
end

end
