module TestEndToEndSmoke

using Test
using LinearAlgebra
using CompScienceMeshes, BEAST
using H2Trees, H2Factory
using H2FactoryTestUtils
using MultilevelKernelInterpolation

# Real, end-to-end BEM solve smoke coverage: build a `GalerkinMap` for an actual operator on
# an actual mesh and verify its far-field approximation reproduces the true dense far-field
# interaction (not just the translation-level checks `Translations/farfieldaccuracy.jl`
# already does). Uses `H2Factory.DummyNearInteractionsAssembler` so construction skips real
# (expensive) near-interaction quadrature -- `H2Factory.farinteractions(Adense, A)` still
# recovers the *true* far block cheaply by indexing into the already-assembled `Adense` using
# `A`'s own near/far pattern, no re-quadrature needed, so this doesn't weaken the check.
@testset verbose=true "End-to-end BEM solve smoke tests" begin
    m = readmesh(
        joinpath(pkgdir(MultilevelKernelInterpolation), "test", "assets", "in", "sphere.in")
    )

    @testset "Helmholtz3D.singlelayer (Galerkin)" begin
        op = Helmholtz3D.singlelayer(; wavenumber=1.0)
        X = lagrangecxd0(m)

        Adense = assemble(op, X, X)
        A = MultilevelKernelInterpolation.GalerkinMap(
            op,
            X,
            H2Trees.buildtree(X; builder=H2Trees.TwoNTreeBuilder(; minvalues=120));
            strategy=MultilevelKernelInterpolation.InterpolationStrategy(
                op,
                X;
                nearinteractionsstrategy=H2Factory.DummyNearInteractionsStrategy(
                    BEAST.defaultquadstrat(op, X, X)
                ),
            ),
        )

        farinteractions = H2Factory.farinteractions(Adense, A)
        H2FactoryTestUtils.checkfarmul(farinteractions, A; n=5, ϵ=1e-4, verbose=false)

        @testset "transpose/adjoint mul!" begin
            # Untested until now: several `H2Factory` accessors are two-step (a 1-arg
            # entrypoint unwraps `transpose`/`adjoint` via `forwardmap`/`maptrait`, then
            # dispatches 2-arg on `isForward`/`isTranspose`/`isAdjoint`) -- an override written
            # at the wrong dispatch level is silently skipped under `transpose(A)`/`adjoint(A)`
            # rather than erroring, so this is the only thing that actually catches it.
            transposefarinteractions = H2Factory.farinteractions(
                transpose(Adense), transpose(A)
            )
            H2FactoryTestUtils.checkfarmul(
                transposefarinteractions, transpose(A); n=5, ϵ=1e-4, verbose=false
            )

            adjointfarinteractions = H2Factory.farinteractions(adjoint(Adense), adjoint(A))
            H2FactoryTestUtils.checkfarmul(
                adjointfarinteractions, adjoint(A); n=5, ϵ=1e-4, verbose=false
            )
        end

        @testset "mul! allocation baseline" begin
            x = randn(H2FactoryTestUtils.checkrng(), ComplexF64, size(A, 2))
            y = Vector{ComplexF64}(undef, size(A, 1))
            mul!(y, A, x)  # warm up (compilation, buffer setup)
            # `matvecbuffers`/`vectorbuffer` exist so `mul!` avoids *large* (O(N) or
            # per-node/level) allocations, but it is not literally allocation-free today.
            # `InterpolationStrategy`'s default `translatorstrategy` is Toeplitz (FFT-accelerated I2O,
            # much faster than the dense default) -- that trades a real per-`mul!` allocation
            # cost for it (measured ~940-980KB here, vs. 13-35KB with the dense I2O translator),
            # presumably padded FFT input/output buffers not fully pooled through the existing
            # `takefftplan!`/`putfftplan!` channel machinery. This bound is a regression
            # baseline (catch a *new, unrelated* large allocation), not an assertion that
            # today's allocation profile is optimal -- reducing it is a real, open follow-up.
            @test (@allocated mul!(y, A, x)) < 1_500_000
        end

        @testset "user-settable nchunks (real translator buffer pools)" begin
            # The five MultilevelKernelInterpolation translator scratch pools (i2i/o2o OneDim, the Toeplitz/Bspline I2O
            # FFT/buffer channels) are all sized to `nchunks`, as is the chunk schedule. Building at
            # `nchunks != Threads.nthreads()` and still reproducing the *true* far block confirms every
            # buffer site tracks `nchunks` -- a site left on `Threads.nthreads()` would size its pool
            # differently from the schedule and OOB (or corrupt) at `buffer[chunkid]`. Covers below,
            # at, and above the thread count.
            for nc in (1, 2, 16)
                nctree = H2Trees.buildtree(
                    X; builder=H2Trees.TwoNTreeBuilder(; minvalues=120)
                )
                ncstrategy = MultilevelKernelInterpolation.InterpolationStrategy(
                    op,
                    X;
                    nearinteractionsstrategy=H2Factory.DummyNearInteractionsStrategy(
                        BEAST.defaultquadstrat(op, X, X)
                    ),
                )
                Anc = MultilevelKernelInterpolation.GalerkinMap(
                    op,
                    X,
                    nctree;
                    strategy=ncstrategy,
                    # `nchunks` is a property of the execution scheduling, not a standalone kwarg:
                    # it must agree with the chunk schedule the buffer pools are sized from, so it
                    # is set where that agreement is established. Mirrors the constructor's own
                    # default, with only `nchunks` overridden.
                    scheduling=H2Factory.executionscheduling(
                        MultilevelKernelInterpolation.scheduler(ncstrategy),
                        1:H2Trees.numberoflevels(nctree);
                        nchunks=nc,
                    ),
                )
                @test H2Factory.aggregationschedule(Anc).nchunks == nc
                H2FactoryTestUtils.checkfarmul(
                    farinteractions, Anc; n=3, ϵ=1e-4, verbose=false
                )
            end
        end

        @testset "checkaliasing=true validates the pooled moment layout at construction" begin
            # The opt-in self-check: `GalerkinMap`'s moments are `unsafe_wrap`s into shared pools that
            # deliberately alias, and `checkmomentaliasing` confirms the aliasing is sound. Building
            # with the flag must not throw on a real layout, and must leave the constructed map's
            # own check clean (no error-severity finding). Default construction does not run it (it is
            # an O(n^2) diagnostic), so this is the only thing exercising the flag end-to-end.
            Achk = MultilevelKernelInterpolation.GalerkinMap(
                op,
                X,
                H2Trees.buildtree(X; builder=H2Trees.TwoNTreeBuilder(; minvalues=120));
                strategy=MultilevelKernelInterpolation.InterpolationStrategy(
                    op,
                    X;
                    nearinteractionsstrategy=H2Factory.DummyNearInteractionsStrategy(
                        BEAST.defaultquadstrat(op, X, X)
                    ),
                ),
                checkaliasing=true,
            )
            @test checkmomentaliasing(Achk).ok
        end
    end

    @testset "MultilevelKernelInterpolation.efie (Maxwell3D, Galerkin)" begin
        op = MultilevelKernelInterpolation.efie(; wavenumber=1.0)
        X = raviartthomas(m)

        Adense = assemble(op, X, X)
        A = MultilevelKernelInterpolation.GalerkinMap(
            op,
            X,
            H2Trees.buildtree(X; builder=H2Trees.TwoNTreeBuilder(; minvalues=120));
            strategy=MultilevelKernelInterpolation.InterpolationStrategy(
                op,
                X;
                nearinteractionsstrategy=H2Factory.DummyNearInteractionsStrategy(
                    BEAST.defaultquadstrat(op, X, X)
                ),
            ),
        )

        farinteractions = H2Factory.farinteractions(Adense, A)
        # `checkfarmul` compares against a *random* `x` each run (unseeded `randn`) and reports a
        # max-norm relative error, so the worst of its `n` draws can spike well above any single
        # lucky measurement -- observed up to ~6.3e-3 for EFIE's dyadic kernel (which is
        # measurably less accurate than Helmholtz's scalar kernel at the same order, same pattern
        # as `Translations/farfieldaccuracy.jl`). The tolerance is a *ceiling* on the H2 far-field
        # approximation error (fundamentally bounded by the polynomial order), so 1e-2 gives robust
        # headroom over the random-draw spread instead of the earlier 3e-3 that sat right at its edge
        # and failed intermittently.
        H2FactoryTestUtils.checkfarmul(farinteractions, A; n=5, ϵ=1e-2, verbose=false)

        @testset "transpose/adjoint mul!" begin
            # Same dyadic-kernel translation machinery as forward mode, reached through the
            # `isTranspose`/`isAdjoint` accessor branches; the same 1e-2 ceiling applies.
            transposefarinteractions = H2Factory.farinteractions(
                transpose(Adense), transpose(A)
            )
            H2FactoryTestUtils.checkfarmul(
                transposefarinteractions, transpose(A); n=5, ϵ=1e-2, verbose=false
            )

            adjointfarinteractions = H2Factory.farinteractions(adjoint(Adense), adjoint(A))
            H2FactoryTestUtils.checkfarmul(
                adjointfarinteractions, adjoint(A); n=5, ϵ=1e-2, verbose=false
            )
        end

        @testset "mul! allocation baseline" begin
            x = randn(H2FactoryTestUtils.checkrng(), ComplexF64, size(A, 2))
            y = Vector{ComplexF64}(undef, size(A, 1))
            mul!(y, A, x)  # warm up
            # See the identical comment on the Helmholtz case above -- the Toeplitz I2O default
            # trades allocation-free `mul!` for FFT-accelerated speed.
            @test (@allocated mul!(y, A, x)) < 1_500_000
        end
    end
end

end
