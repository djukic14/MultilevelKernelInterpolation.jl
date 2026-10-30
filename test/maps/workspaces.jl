module TestWorkspaces

using Test
using LinearAlgebra
using Base.Threads: @spawn, nthreads
using CompScienceMeshes, BEAST
using H2Trees, H2Factory
using H2FactoryTestUtils
using MultilevelKernelInterpolation

# The core/workspace split: one read-only `…InterpolationCore` shared by several `H2Workspace`s, each owning
# every mutable buffer one in-flight `mul!` touches (moments + their pooled backing, leaf vector
# buffers, level-transfer scratch, the I2O channel with its FFT plans and B-spline modulation
# scratch).
#
# WHAT MAKES THIS TESTABLE AT ALL. A workspace's answer is a deterministic function of the core and
# the input -- chunks write disjoint nodes, so even a threaded traversal is bit-reproducible. So the
# oracle here is EXACT EQUALITY against a serial single-workspace run, not a tolerance: a shared
# mutable buffer shows up as a differing bit pattern, and a tolerance would hide exactly the small
# cross-contamination a race actually produces. Every `@test` below that compares two workspaces uses
# `==`, deliberately.
#
# The accuracy of the underlying approximation is NOT retested here -- `endtoend_smoke.jl` and
# `petrov_endtoend_smoke.jl` own that. These tests only assert that splitting construction from
# execution changed nothing, and that the split actually delivers isolation.

const MESH = readmesh(
    joinpath(pkgdir(MultilevelKernelInterpolation), "test", "assets", "in", "sphere.in")
)

dummynear(op, testspace, trialspace) = H2Factory.DummyNearInteractionsStrategy(
    BEAST.defaultquadstrat(op, testspace, trialspace)
)

# Order 4 / 10 knots with a protrusion-derived buffer -- the same buffered B-spline the Toeplitz and
# dense B-spline routes are exercised with elsewhere. Kept small: these tests care about buffer
# ownership, not approximation quality.
bsplinepolynomial() =
    MultilevelKernelInterpolation.BufferedBsplinePolynomialEquidistant3D{Float64}(
        4, 10, MultilevelKernelInterpolation.AutoBufferFromProtrusion(1)
    )

serialscheduling(core) =
    H2Factory.serialexecutionscheduling(H2Factory.schedulinglevels(core))
chunkedscheduling(core, nchunks) =
    H2Factory.defaultexecutionscheduling(H2Factory.schedulinglevels(core); nchunks=nchunks)

# --- the configurations under test ----------------------------------------------------------------
#
# All four I2O routes x {Galerkin, Petrov}. The routes differ in exactly what scratch a workspace has
# to allocate for itself, which is the thing being tested:
#
#   dense            -- no I2O scratch at all (`EmptyChannels`)
#   Toeplitz         -- per-(level, chunk) FFT/iFFT plan pools
#   B-spline dense   -- per-(level, chunk, side) raw<->reduced modulation scratch
#   B-spline Toeplitz-- both of the above
#
# `OneDimI2IStrategy` is included separately because it is the only I2I trait with level-transfer
# scratch (the separable contraction's intermediate buffers); the default dense trait has none, so
# without it the i2i/o2o channel builders would be exercised only in their `EmptyChannels` branch.
#
# EACH CONFIGURATION DECLARES THE SCRATCH IT MUST HAVE (`expected`), and the privacy testset asserts
# that scratch is genuinely present before asserting it is private. Without that, the whole testset
# degrades silently into a tautology: `EmptyChannels` is a stateless singleton, so two workspaces
# legitimately share ONE of them, and a configuration that accidentally allocated no scratch at all
# would pass every "buffers are private" check while testing nothing.
#
# TREE DEPTH IS PART OF THE FIXTURE. `minvalues=120` on this mesh yields a single relevant level, so
# there are NO I2I level transfers and therefore no separable scratch to be private about -- the
# separable configuration below would be vacuous. `minvalues=40` gives two relevant levels, which is
# the shallowest tree that produces real level transfers, and `minvalues=8` gives a third, so the
# stored transfers span more than one level (see the second separable configuration).
# `minvalues` is per configuration so the heavier B-spline routes keep the shallow tree (their I2O
# modulation scratch exists at any depth), while the level-transfer routes get the deeper one.
buildtree(space, minvalues) =
    H2Trees.buildtree(space; builder=H2Trees.TwoNTreeBuilder(; minvalues=minvalues))

function galerkinconfigurations()
    op = Helmholtz3D.singlelayer(; wavenumber=1.0)
    X = lagrangecxd0(MESH)
    near = dummynear(op, X, X)
    bspline = bsplinepolynomial()
    base(; kwargs...) = MultilevelKernelInterpolation.InterpolationStrategy(
        op, X; nearinteractionsstrategy=near, kwargs...
    )
    return [
        (
            name="dense I2O",
            op=op,
            space=X,
            minvalues=40,
            expected=(),
            strategy=base(;
                translatorstrategy=MultilevelKernelInterpolation.TranslatorStrategy(;
                    i2ostrategy=MultilevelKernelInterpolation.DenseI2OStrategy()
                ),
            ),
        ),
        (
            name="Toeplitz I2O",
            op=op,
            space=X,
            minvalues=40,
            expected=(:i2o,),
            strategy=base(),
        ),
        (
            # The configurations with LEVEL-TRANSFER scratch: a separable transfer stages its
            # contraction through per-(level, chunk) buffers, while the default dense trait is a
            # single `mul!` and needs none.
            name="separable I2I + dense I2O",
            op=op,
            space=X,
            minvalues=40,
            expected=(:i2i,),
            strategy=base(;
                translatorstrategy=MultilevelKernelInterpolation.TranslatorStrategy(;
                    i2istrategy=MultilevelKernelInterpolation.OneDimI2IStrategy(),
                    i2ostrategy=MultilevelKernelInterpolation.DenseI2OStrategy(),
                ),
            ),
        ),
        (
            # The same separable route on a DEEPER tree, so the stored transfers span TWO levels.
            # This is the strictly stronger check on how the level-transfer scratch is keyed: with a
            # single transfer level a wrong borrow level finds NO entry and fails loudly, whereas
            # with two it can find the neighbouring level's pool and quietly compute with buffers
            # belonging to a different level.
            name="separable I2I, two transfer levels",
            op=op,
            space=X,
            minvalues=8,
            expected=(:i2i,),
            strategy=base(;
                translatorstrategy=MultilevelKernelInterpolation.TranslatorStrategy(;
                    i2istrategy=MultilevelKernelInterpolation.OneDimI2IStrategy(),
                    i2ostrategy=MultilevelKernelInterpolation.DenseI2OStrategy(),
                ),
            ),
        ),
        (
            name="B-spline dense I2O",
            op=op,
            space=X,
            minvalues=120,
            expected=(:i2o,),
            strategy=base(;
                polynomialstrategy=MultilevelKernelInterpolation.PolynomialStrategy(
                    op; polynomial=bspline
                ),
                translatorstrategy=MultilevelKernelInterpolation.TranslatorStrategy(;
                    i2ostrategy=MultilevelKernelInterpolation.DenseI2OStrategy()
                ),
            ),
        ),
        (
            name="B-spline Toeplitz I2O",
            op=op,
            space=X,
            minvalues=120,
            expected=(:i2o,),
            strategy=base(;
                polynomialstrategy=MultilevelKernelInterpolation.PolynomialStrategy(
                    op; polynomial=bspline
                ),
            ),
        ),
    ]
end

function petrovconfigurations()
    op = Helmholtz3D.singlelayer(; wavenumber=1.0)
    testspace = lagrangecxd0(MESH)
    trialspace = lagrangec0d1(MESH)
    near = dummynear(op, testspace, trialspace)
    base(; kwargs...) = MultilevelKernelInterpolation.InterpolationStrategy(
        op, testspace, trialspace; nearinteractionsstrategy=near, kwargs...
    )
    return [
        (
            name="dense I2O",
            op=op,
            testspace=testspace,
            trialspace=trialspace,
            minvalues=120,
            expected=(),
            strategy=base(;
                translatorstrategy=MultilevelKernelInterpolation.TranslatorStrategy(;
                    i2ostrategy=MultilevelKernelInterpolation.DenseI2OStrategy()
                ),
            ),
        ),
        (
            name="Toeplitz I2O",
            op=op,
            testspace=testspace,
            trialspace=trialspace,
            minvalues=120,
            expected=(:i2o,),
            strategy=base(),
        ),
    ]
end

buildblocktree(testspace, trialspace, minvalues) = H2Trees.buildtree(
    testspace,
    trialspace;
    builder=H2Trees.BlockTreeBuilder(;
        test=H2Trees.TwoNTreeBuilder(; minvalues=minvalues),
        trial=H2Trees.TwoNTreeBuilder(; minvalues=minvalues),
    ),
)

# The scratch a configuration declared it must have, resolved to the workspace objects that hold it.
# `EmptyChannels` here means the builders produced nothing where the configuration says there should
# be something -- so this is what stops the privacy assertions below from passing vacuously.
function requiredscratch(ws, expected)
    buffers = H2Factory.workspacebuffers(ws)
    scratch = Any[]
    :i2i in expected && push!(scratch, buffers.i2ichannels, buffers.o2ochannels)
    :i2o in expected && push!(scratch, buffers.i2ochannel)
    return scratch
end

# Distinct objects, recursively: a `SidedChannels` pair is itself fresh per workspace, but so must be
# each side inside it.
function assertdistinct(a, b)
    @test a !== b
    if a isa H2Factory.SidedChannels && b isa H2Factory.SidedChannels
        @test a.test !== b.test
        @test a.trial !== b.trial
    end
    return nothing
end

# `mul!` in all three orientations, as one vector of results, so a comparison covers the whole
# accessor surface at once. Transpose/adjoint take a TEST-sized input (the map is rectangular for a
# Petrov configuration), which is also what makes them a real check rather than an accidental
# re-run of the forward pass.
function allmodes(ws, x, u)
    return [ws * x, transpose(ws) * u, adjoint(ws) * u]
end

@testset verbose = true "Core/workspace split" begin
    @testset verbose = true "Galerkin" begin
        @testset "$(config.name)" for config in galerkinconfigurations()
            core = MultilevelKernelInterpolation.GalerkinKernelInterpolationCore(
                config.op,
                config.space,
                buildtree(config.space, config.minvalues);
                strategy=config.strategy,
            )

            reference = H2Workspace(core, serialscheduling(core))
            rng = H2FactoryTestUtils.checkrng()
            x = randn(rng, ComplexF64, size(reference, 2))
            u = randn(rng, ComplexF64, size(reference, 1))
            modes(ws) = allmodes(ws, x, u)
            expected = modes(reference)

            @testset "a second workspace over the same core agrees exactly" begin
                other = H2Workspace(core, serialscheduling(core))
                @test modes(other) == expected
                # ... and the core really is shared, not rebuilt.
                @test H2Factory.core(other) === core
                @test H2Factory.core(reference) === core
                @test H2Factory.core(transpose(other)) === core
            end

            @testset "workspace buffers are private" begin
                other = H2Workspace(core, serialscheduling(core))
                a, b = H2Factory.workspacebuffers(reference),
                H2Factory.workspacebuffers(other)

                # The scratch this configuration exists to exercise must actually BE there. Without
                # this, every assertion below would pass on a pair of shared `EmptyChannels`.
                for scratch in requiredscratch(reference, config.expected)
                    @test !(scratch isa H2Factory.EmptyChannels)
                end

                # Moments and their pooled backing are what a matvec writes on every route.
                assertdistinct(a.moments, b.moments)
                assertdistinct(a.momentbuffers, b.momentbuffers)
                assertdistinct(a.vectorbuffers, b.vectorbuffers)
                @test H2Factory.testmoments(reference) !== H2Factory.testmoments(other)
                @test H2Factory.trialmoments(reference) !== H2Factory.trialmoments(other)

                # Channels only where this configuration has any. A route with no scratch shares one
                # stateless `EmptyChannels`, which is correct, not a leak.
                :i2i in config.expected && assertdistinct(a.i2ichannels, b.i2ichannels)
                :i2i in config.expected && assertdistinct(a.o2ochannels, b.o2ochannels)
                :i2o in config.expected && assertdistinct(a.i2ochannel, b.i2ochannel)
            end

            @testset "internally threaded workspace agrees with the serial one" begin
                threaded = H2Workspace(core, chunkedscheduling(core, 4))
                @test modes(threaded) == expected
            end

            @testset "repeated matvecs on one workspace are stable" begin
                # Catches scratch left dirty by a previous call -- a buffer that is written but not
                # fully overwritten shows up on the SECOND run, never the first.
                @test modes(reference) == expected
                @test modes(reference) == expected
            end

            @testset "concurrent workspaces, independent inputs" begin
                # THE REAL TEST. Several workspaces over one core, running at once, each with its own
                # input -- so a shared buffer corrupts a NEIGHBOUR's result, not its own. Distinct
                # inputs matter: with one shared input a race can overwrite a buffer with the same
                # values it already held and stay invisible.
                nworkspaces = 4
                workspaces = [
                    H2Workspace(core, chunkedscheduling(core, 4)) for _ in 1:nworkspaces
                ]
                inputs = let rng = H2FactoryTestUtils.checkrng()
                    [randn(rng, ComplexF64, size(reference, 2)) for _ in 1:nworkspaces]
                end
                serials = [reference * xi for xi in inputs]

                for _ in 1:3   # repeat: a race that loses often enough to matter shows up quickly
                    tasks = map(zip(workspaces, inputs)) do (ws, xi)
                        return @spawn ws * xi
                    end
                    @test fetch.(tasks) == serials
                end
            end
        end
    end

    @testset verbose = true "Petrov" begin
        @testset "$(config.name)" for config in petrovconfigurations()
            core = MultilevelKernelInterpolation.PetrovGalerkinKernelInterpolationCore(
                config.op,
                config.testspace,
                config.trialspace,
                buildblocktree(config.testspace, config.trialspace, config.minvalues);
                strategy=config.strategy,
            )

            reference = H2Workspace(core, serialscheduling(core))
            rng = H2FactoryTestUtils.checkrng()
            x = randn(rng, ComplexF64, size(reference, 2))
            u = randn(rng, ComplexF64, size(reference, 1))
            expected = allmodes(reference, x, u)

            @testset "second workspace / threaded workspace agree exactly" begin
                @test allmodes(H2Workspace(core, serialscheduling(core)), x, u) == expected
                @test allmodes(H2Workspace(core, chunkedscheduling(core, 4)), x, u) ==
                    expected
            end

            @testset "per-side scratch stays per side, and is private" begin
                other = H2Workspace(core, serialscheduling(core))
                buffers = H2Factory.workspacebuffers(reference)
                # A Petrov workspace holds `SidedChannels`, not one shared object -- that container
                # shape is what routes a transposed aggregation to the TEST side's own scratch rather
                # than to the forward O2O one.
                @test buffers.i2ichannels isa H2Factory.SidedChannels
                @test buffers.o2ochannels isa H2Factory.SidedChannels
                @test buffers.vectorbuffers isa H2Factory.VectorBuffers

                for scratch in requiredscratch(reference, config.expected)
                    @test !(scratch isa H2Factory.EmptyChannels)
                end
                other_buffers = H2Factory.workspacebuffers(other)
                assertdistinct(buffers.moments, other_buffers.moments)
                assertdistinct(buffers.vectorbuffers, other_buffers.vectorbuffers)
                :i2i in config.expected &&
                    assertdistinct(buffers.i2ichannels, other_buffers.i2ichannels)
                :i2i in config.expected &&
                    assertdistinct(buffers.o2ochannels, other_buffers.o2ochannels)
                :i2o in config.expected &&
                    assertdistinct(buffers.i2ochannel, other_buffers.i2ochannel)
            end

            @testset "concurrent workspaces, independent inputs" begin
                workspaces = [H2Workspace(core, chunkedscheduling(core, 4)) for _ in 1:4]
                inputs = let rng = H2FactoryTestUtils.checkrng()
                    [randn(rng, ComplexF64, size(reference, 2)) for _ in 1:4]
                end
                serials = [reference * xi for xi in inputs]
                for _ in 1:3
                    tasks = map(zip(workspaces, inputs)) do (ws, xi)
                        return @spawn ws * xi
                    end
                    @test fetch.(tasks) == serials
                end
            end
        end
    end

    @testset verbose = true "ownership guard" begin
        op = Helmholtz3D.singlelayer(; wavenumber=1.0)
        X = lagrangecxd0(MESH)
        core = MultilevelKernelInterpolation.GalerkinKernelInterpolationCore(
            op,
            X;
            strategy=MultilevelKernelInterpolation.InterpolationStrategy(
                op, X; nearinteractionsstrategy=dummynear(op, X, X)
            ),
        )
        ws = H2Workspace(core, serialscheduling(core))
        x = randn(H2FactoryTestUtils.checkrng(), ComplexF64, size(ws, 2))
        y = similar(x)

        @testset "a normal operation runs and leaves the workspace free" begin
            @test H2Factory.busyflag(ws)[] == false
            expected = ws * x
            @test H2Factory.busyflag(ws)[] == false
            # ...and the same holds for the far-only public entry and the wrapped orientations,
            # each of which takes and releases ownership in its own right.
            H2Factory.farmul!(y, ws, x)
            @test H2Factory.busyflag(ws)[] == false
            mul!(y, transpose(ws), x)
            @test H2Factory.busyflag(ws)[] == false
            @test ws * x == expected
        end

        @testset "the workspace is released after an error" begin
            # Without the `finally`, a failed operation would leave the workspace locked forever and
            # every later matvec would report a phantom concurrent use -- so this is the assertion
            # that makes the guard usable at all. Two failure shapes: one raised deterministically
            # inside the owned region, and one raised by the traversal itself.
            @test_throws ErrorException H2Factory.withworkspace(ws) do
                return error("boom")
            end
            @test H2Factory.busyflag(ws)[] == false

            # A SHORT input: the leaf pass views `x` at each leaf's value indices, so this is an
            # out-of-bounds well past the acquire. (A long one would not fail -- the extra entries
            # are simply never viewed.)
            @test_throws Exception H2Factory.farmul!(
                similar(x, length(x) - 1),
                ws,
                randn(H2FactoryTestUtils.checkrng(), ComplexF64, length(x) - 1),
            )
            @test H2Factory.busyflag(ws)[] == false
            @test ws * x isa Vector{ComplexF64}   # still usable
        end

        @testset "a second concurrent operation on one workspace fails fast" begin
            # Held by hand rather than by racing two real matvecs: a race would only *sometimes*
            # overlap, so a passing test would prove nothing. This pins the actual contract --
            # entering while owned throws, and throws a distinguishable type.
            H2Factory.acquireworkspace!(ws)
            try
                @test_throws H2Factory.ConcurrentWorkspaceError H2Factory.acquireworkspace!(
                    ws
                )
                @test_throws H2Factory.ConcurrentWorkspaceError H2Factory.farmul!(y, ws, x)
                @test_throws H2Factory.ConcurrentWorkspaceError mul!(y, ws, x)
                # A wrapped view is the same buffers, so it contends for the same flag.
                @test_throws H2Factory.ConcurrentWorkspaceError mul!(y, transpose(ws), x)
                @test_throws H2Factory.ConcurrentWorkspaceError mul!(y, adjoint(ws), x)
            finally
                H2Factory.releaseworkspace!(ws)
            end
            @test H2Factory.busyflag(ws)[] == false
            # It must FAIL, not serialize: a caller sharing one workspace across tasks has a bug,
            # and silently taking a lock would turn that into a mysterious loss of parallelism.
            @test ws * x isa Vector{ComplexF64}
        end

        @testset "a second TASK is rejected while the first is inside" begin
            # The cross-task counterpart of the hand-held test above, and deterministic where a
            # plain race is not: the contender is not spawned until the holder is provably inside
            # the owned region, and the holder does not leave it until the contender has finished,
            # so the overlap is guaranteed rather than hoped for. This is what the same-task
            # acquire above cannot show -- that the flag is shared between tasks.
            entered = Base.Event()
            release = Base.Event()
            holder = @spawn H2Factory.withworkspace(ws) do
                notify(entered)
                wait(release)
                return :held
            end
            wait(entered)
            contender = @spawn try
                ws * x
                :ok
            catch e
                e isa H2Factory.ConcurrentWorkspaceError ? :rejected : rethrow()
            end
            @test fetch(contender) == :rejected
            notify(release)
            @test fetch(holder) == :held
            @test H2Factory.busyflag(ws)[] == false
        end

        @testset "racing two real matvecs corrupts neither (opportunistic)" begin
            # Opportunistic on purpose, unlike the two tests above: whether the two matvecs
            # actually overlap is the scheduler's business, so this must NOT assert that a
            # rejection happens -- with no overlap both legitimately succeed, and an assertion on
            # the rejection count would be a flake. The deterministic tests own that half of the
            # contract. What only a real race can reach is the other half, asserted here: an
            # operation the guard let through returns the exact serial answer, and one it turned
            # away leaves nothing behind for the next.
            reference = ws * x
            for _ in 1:20
                results = map(1:2) do _
                    return @spawn try
                        ws * x
                    catch e
                        e isa H2Factory.ConcurrentWorkspaceError ? e : rethrow()
                    end
                end
                for outcome in fetch.(results)
                    outcome isa H2Factory.ConcurrentWorkspaceError && continue
                    @test outcome == reference
                end
            end
            @test H2Factory.busyflag(ws)[] == false
            @test ws * x == reference
        end

        @testset "separate workspaces over one core run concurrently" begin
            # The positive half of the contract, and the reason the guard fails loudly instead of
            # serializing: this is what a caller is meant to do, and it is unaffected.
            workspaces = [H2Workspace(core, chunkedscheduling(core, 4)) for _ in 1:4]
            inputs = let rng = H2FactoryTestUtils.checkrng()
                [randn(rng, ComplexF64, size(ws, 2)) for _ in 1:4]
            end
            serials = [ws * xi for xi in inputs]
            for _ in 1:5
                tasks = map(zip(workspaces, inputs)) do (w, xi)
                    return @spawn w * xi
                end
                @test fetch.(tasks) == serials
            end
        end

        @testset "internal threading inside one operation still works" begin
            # A single owner may use as many tasks as its scheduling asks for -- the chunked
            # traversal is exactly that, and the guard must not interfere with it.
            threaded = H2Workspace(core, chunkedscheduling(core, 4))
            @test threaded * x == ws * x
            @test H2Factory.busyflag(threaded)[] == false
        end

        @testset "the flag lookup falls back to nothing off the workspace types" begin
            # A map owning no shared scratch has no flag, so the generic entry points can call
            # `acquireworkspace!` unconditionally. (Exercised end-to-end on a real non-workspace map
            # in H2Factory's own `test/maps/test_workspace.jl`; here only the fallback is pinned,
            # since every map MultilevelKernelInterpolation builds IS a workspace.)
            @test H2Factory.busyflag(zeros(2, 2), H2Factory.isForward()) === nothing
        end
    end

    @testset verbose = true "workspace-level invariants" begin
        op = Helmholtz3D.singlelayer(; wavenumber=1.0)
        X = lagrangecxd0(MESH)
        core = MultilevelKernelInterpolation.GalerkinKernelInterpolationCore(
            op,
            X;
            strategy=MultilevelKernelInterpolation.InterpolationStrategy(
                op, X; nearinteractionsstrategy=dummynear(op, X, X)
            ),
        )

        @testset "scheduling and schedules belong to the workspace" begin
            serial = H2Workspace(core, serialscheduling(core))
            chunked = H2Workspace(core, chunkedscheduling(core, 4))
            # Two workspaces over one core may be scheduled differently -- that is the point of
            # holding `nchunks` here rather than on the core.
            @test H2Factory.nchunks(serial) == 1
            @test H2Factory.nchunks(chunked) == 4
            @test H2Factory.schedules(serial) !== H2Factory.schedules(chunked)
            # Orientation does not change either: they describe the workspace, not the view of it.
            @test H2Factory.scheduling(transpose(serial)) === H2Factory.scheduling(serial)
            @test H2Factory.schedules(adjoint(serial)) === H2Factory.schedules(serial)
        end

        @testset "buffer pools follow the workspace's own nchunks" begin
            # A pool sized by anything other than this workspace's schedule is an out-of-bounds at
            # `buffer[chunkid]`, so simply completing a matvec at each granularity is the assertion.
            # Covers below, at, and above the thread count.
            reference = H2Workspace(core, serialscheduling(core))
            x = randn(H2FactoryTestUtils.checkrng(), ComplexF64, size(reference, 2))
            expected = reference * x
            for nc in (1, 2, 16)
                @test H2Workspace(core, chunkedscheduling(core, nc)) * x == expected
            end
        end

        @testset "FFT plans are workspace-local and survive construct/destroy cycles" begin
            # The Toeplitz route plans FFTs per workspace. Repeatedly building and dropping
            # workspaces exercises that those plans are genuinely fresh (and that nothing in the core
            # is left pointing at a collected one).
            reference = H2Workspace(core, serialscheduling(core))
            x = randn(H2FactoryTestUtils.checkrng(), ComplexF64, size(reference, 2))
            expected = reference * x
            for _ in 1:5
                transient = H2Workspace(core, chunkedscheduling(core, 2))
                @test transient * x == expected
                GC.gc(false)
            end
            @test reference * x == expected
        end

        @testset "checkmomentaliasing runs on a real workspace" begin
            ws = H2Workspace(core, chunkedscheduling(core, 2))
            report = H2Factory.checkmomentaliasing(ws)
            @test report.ok
        end

        @testset "the convenience constructor returns a workspace over a fresh core" begin
            A = MultilevelKernelInterpolation.GalerkinKernelInterpolation(
                op,
                X;
                strategy=MultilevelKernelInterpolation.InterpolationStrategy(
                    op, X; nearinteractionsstrategy=dummynear(op, X, X)
                ),
            )
            @test A isa H2Factory.GalerkinH2Workspace
            @test H2Factory.core(A) isa
                MultilevelKernelInterpolation.GalerkinKernelInterpolationCore
            # It is a `LinearMap` and the core is not: only the workspace can execute.
            @test A isa H2Factory.AbstractH2Map
            @test !(H2Factory.core(A) isa H2Factory.AbstractH2Map)
        end
    end

    @testset verbose = true "separable I2I agrees with dense I2I in every orientation" begin
        # A CROSS-IMPLEMENTATION oracle, which everything above this point structurally cannot be.
        # The tests above compare one workspace against another workspace over the SAME core, so they
        # pin isolation, determinism and freedom from throws -- but a traversal that is uniformly
        # wrong is uniformly wrong in both, and compares equal. Two DIFFERENT I2I implementations over
        # identical geometry do not share that failure mode.
        #
        # `OneDimI2IStrategy` is a separable reformulation of the same transfer, so the two are
        # interchangeable to roundoff, not merely to multilevel interpolation accuracy -- hence the tight tolerance.
        #
        # This is the check that would have caught the separable transpose/adjoint defect as a WRONG
        # ANSWER rather than only as a `MethodError`: the scratch a transposed aggregation borrows
        # decides which buffers the contraction stages through, and borrowing the wrong pool can
        # equally well produce plausible numbers.
        op = Helmholtz3D.singlelayer(; wavenumber=1.0)
        X = lagrangecxd0(MESH)
        near = dummynear(op, X, X)
        strategy(i2i) = MultilevelKernelInterpolation.InterpolationStrategy(
            op,
            X;
            nearinteractionsstrategy=near,
            translatorstrategy=MultilevelKernelInterpolation.TranslatorStrategy(;
                i2istrategy=i2i,
                i2ostrategy=MultilevelKernelInterpolation.DenseI2OStrategy(),
            ),
        )
        core(i2i, minvalues) =
            MultilevelKernelInterpolation.GalerkinKernelInterpolationCore(
                op, X, buildtree(X, minvalues); strategy=strategy(i2i)
            )

        # Both depths: `minvalues=40` stores one transfer level, `minvalues=8` more than one. Only the
        # deeper one can catch a borrow that lands on a NEIGHBOURING level's pool rather than on none.
        @testset "minvalues=$minvalues" for minvalues in (40, 8)
            sepcore = core(MultilevelKernelInterpolation.OneDimI2IStrategy(), minvalues)
            dencore = core(MultilevelKernelInterpolation.DefaultI2IStrategy(), minvalues)
            # Serial and chunked: the pools are sized per level, so a chunked run indexes them with
            # chunk ids a serial run never produces.
            @testset "$schedname" for (schedname, sched) in (
                ("serial", serialscheduling), ("chunked", c -> chunkedscheduling(c, 4))
            )
                sep = H2Workspace(sepcore, sched(sepcore))
                den = H2Workspace(dencore, sched(dencore))
                rng = H2FactoryTestUtils.checkrng()
                x = randn(rng, ComplexF64, size(sep, 2))
                u = randn(rng, ComplexF64, size(sep, 1))

                @test sep * x ≈ den * x rtol = 1e-12
                @test transpose(sep) * u ≈ transpose(den) * u rtol = 1e-12
                @test adjoint(sep) * u ≈ adjoint(den) * u rtol = 1e-12

                # Duality, which needs no second implementation at all: it ties the transposed
                # traversal to the forward one through the map itself, so it still holds if BOTH
                # implementations were to drift together.
                @test dot(u, sep * x) ≈ dot(adjoint(sep) * u, x) rtol = 1e-10
            end
        end
    end
end

end
