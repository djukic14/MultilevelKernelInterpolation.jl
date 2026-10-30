module TestPetrovToeplitz

using Test
using LinearAlgebra
using Random
using CompScienceMeshes, BEAST
using H2Trees, H2Factory
using H2FactoryTestUtils
using MultilevelKernelInterpolation

# Part E Phase 4/5b: Petrov + Toeplitz I2O, which used to be a blanket error.
#
# A Toeplitz representation stores one kernel sample per node-position DIFFERENCE, so it exists only
# where both sides place their nodes on the SAME grid -- see `MultilevelKernelInterpolation.toeplitzcompatible`. Rather than
# accept each side's independent (protrusion-driven) grid and refuse the mismatch,
# `PetrovGalerkinMap` RECONCILES the two sides' per-level protrusion REQUIREMENTS before building
# either polynomial hierarchy, so the shared I2O levels come out on one grid by construction. Where
# that reconciliation forces a shared grid the map constructs and is accurate -- including genuinely
# mixed-space and rectangular buffered-B-spline pairings, asserted below. Where it provably cannot
# (the post-build `toeplitzcompatible` validation fails), construction errors naming the level and
# both grids and points at the DENSE route, which always accepts independent grids -- never a
# silently wrong grid.
@testset "Petrov + Toeplitz I2O" begin
    m = readmesh(
        joinpath(pkgdir(MultilevelKernelInterpolation), "test", "assets", "in", "sphere.in")
    )
    op = Helmholtz3D.singlelayer(; wavenumber=1.0)
    dummynear(o, X, Y) =
        H2Factory.DummyNearInteractionsStrategy(BEAST.defaultquadstrat(o, X, Y))

    # The polynomial + I2O-route strategy kwargs, as a NamedTuple for splatting into an `InterpolationStrategy`
    # (used by both the Galerkin `InterpolationStrategy(...)` calls below and by `petrovstrategy`). Toeplitz by
    # default; pass `translatorstrategy` to override the I2O route.
    toeplitz(
        poly;
        translatorstrategy=MultilevelKernelInterpolation.TranslatorStrategy(;
            i2ostrategy=MultilevelKernelInterpolation.ToeplitzI2OStrategy()
        ),
    ) = (
        polynomialstrategy=MultilevelKernelInterpolation.PolynomialStrategy(
            op; polynomial=poly
        ),
        translatorstrategy=translatorstrategy,
    )

    # A full `InterpolationStrategy` for a far-field-only (dummy near) Petrov map -- the single `strategy` object
    # `PetrovGalerkinMap` now takes in place of individual strategy kwargs. `kwargs...` forward to
    # `toeplitz` (e.g. `translatorstrategy=` to pick the dense route).
    petrovstrategy(testspace, trialspace, poly; kwargs...) =
        MultilevelKernelInterpolation.InterpolationStrategy(
            op,
            testspace,
            trialspace;
            nearinteractionsstrategy=dummynear(op, testspace, trialspace),
            toeplitz(poly; kwargs...)...,
        )

    @testset "equidistant Lagrange: $label" for (label, testfn, trialfn) in (
        ("cxd0 / cxd0", lagrangecxd0, lagrangecxd0),
        ("c0d1 / cxd0", lagrangec0d1, lagrangecxd0),
        ("cxd0 / c0d1", lagrangecxd0, lagrangec0d1),
    )
        testspace, trialspace = testfn(m), trialfn(m)
        A = MultilevelKernelInterpolation.PetrovGalerkinMap(
            op,
            testspace,
            trialspace;
            strategy=petrovstrategy(
                testspace,
                trialspace,
                MultilevelKernelInterpolation.BarycentricLagrangePolynomialEquidistant3D(6),
            ),
        )
        Adense = assemble(op, testspace, trialspace)
        # Measured ~2-4e-6 forward and transposed across these pairings; the bound leaves room for
        # the unseeded RNG rather than tracking the sample.
        H2FactoryTestUtils.checkfarmul(
            H2Factory.farinteractions(Adense, A), A; n=5, ϵ=5e-5, verbose=false
        )
        H2FactoryTestUtils.checkfarmul(
            H2Factory.farinteractions(transpose(Adense), transpose(A)),
            transpose(A);
            n=5,
            ϵ=5e-5,
            verbose=false,
        )
        H2FactoryTestUtils.checkfarmul(
            H2Factory.farinteractions(adjoint(Adense), adjoint(A)),
            adjoint(A);
            n=5,
            ϵ=5e-5,
            verbose=false,
        )
    end

    # B-spline: a STRUCTURAL assertion only. The B-spline builder dispatches on
    # `AbstractArray{<:AbstractBspline}`, which a `SidedPolynomialDicts` is not, so a Petrov B-spline
    # map used to fall through to the PLAIN Toeplitz path and silently skip the B-spline
    # interpolation-inverse machinery. Since the demotion, that machinery lives entirely in the
    # `modulation` of the composed `I2OTranslation` (the coupling is now the ordinary Toeplitz
    # collection either way), so B-spline-awareness is asserted on the MODULATION being a real
    # B-spline policy rather than the generic `NoModulation` fallback.
    #
    # Uses plain `BsplinePolynomialEquidistant` deliberately, as a low-level ROUTING diagnostic only.
    # That type lacks the buffered + knot-removal-stabilized construction the production method uses
    # and is inaccurate (a fixed ~0.17 that never improves under refinement), so no accuracy is
    # asserted here. The PRODUCTION path is accurate -- see the buffered Galerkin testset below, which
    # reaches ~1.3e-6 through both I2O strategies.
    @testset "B-spline routes to the B-spline-aware collection" begin
        testspace, trialspace = lagrangec0d1(m), lagrangecxd0(m)
        A = MultilevelKernelInterpolation.PetrovGalerkinMap(
            op,
            testspace,
            trialspace;
            strategy=petrovstrategy(
                testspace,
                trialspace,
                MultilevelKernelInterpolation.BsplinePolynomialEquidistant{3,Float64}(
                    4, 10
                ),
            ),
        )
        @test H2Factory.modulation(H2Factory.i2otranslator(A)) isa Union{
            MultilevelKernelInterpolation.BsplineDenseModulation,
            MultilevelKernelInterpolation.BsplineToeplitzModulation,
        }
    end

    # The B-spline MODULATION is the part a routing-only assertion does not reach.
    # `BsplineToeplitzModulation` looks its interpolation-inverse up by LEVEL
    # (`interpolationmatriceslu[level]`) -- so for a Petrov map it is correct because the precondition
    # `toeplitzcompatible` enforces both sides hold the same polynomial at that level (hence the same
    # shared grid), and the level key resolves to that grid's inverse. If that lapsed, the lookup would
    # be for the wrong grid.
    #
    # (An earlier same-size guard in `PetrovGalerkinMap` also enforced this, but Phase 5b removed it
    # -- moment sizing is now per side, so differing sizes are supported on the DENSE route. On the
    # TOEPLITZ route the shared grid instead comes from requirement-level reconciliation, and the
    # post-build `toeplitzcompatible` validation guarantees both sides reach this level-keyed lookup
    # with the SAME shared grid -- see the rectangular testset below, where a mixed-space buffered
    # B-spline Petrov + Toeplitz map reconciles to a shared grid and is accurate, rather than being
    # refused.)
    #
    # Driving a Petrov map with the SAME space on both sides makes it mathematically identical to the
    # Galerkin map, so the two must agree to roundoff -- see the assertion below for why "to the bit"
    # is one step too strong. Uses the BUFFERED (production) polynomial, so what is pinned is the
    # modulation wiring of the shipping path. The remaining plain-B-spline testset above is now purely
    # a low-level DISPATCH diagnostic and asserts nothing numerical.
    @testset "B-spline modulation matches Galerkin for a same-space Petrov map" begin
        space = lagrangecxd0(m)
        # PRODUCTION type. Same-space Petrov gives both sides the same protrusion, hence the same
        # buffer count and the same grid, so `toeplitzcompatible` is satisfied and the buffered
        # Toeplitz path is reachable here -- which makes this test pin the modulation wiring of the
        # path that actually ships, rather than of a diagnostic object.
        #
        # This testset is also what caught `toeplitzcompatible` being written as plain `==`: the two
        # sides build SEPARATE but identical buffered polynomials, and that type does not compare
        # structurally, so an equal grid was reported as incompatible and a legitimate configuration
        # was refused. Equidistant Lagrange could never have caught it -- there both sides resolve to
        # the same object. Keep this on a type whose two sides are distinct objects.
        polynomial = MultilevelKernelInterpolation.BufferedBsplinePolynomialEquidistant3D{
            Float64
        }(
            4, 10, MultilevelKernelInterpolation.AutoBufferFromProtrusion(1)
        )
        galerkin = MultilevelKernelInterpolation.GalerkinMap(
            op,
            space,
            H2Trees.buildtree(space; builder=H2Trees.TwoNTreeBuilder(; minvalues=120));
            strategy=MultilevelKernelInterpolation.InterpolationStrategy(
                op,
                space;
                nearinteractionsstrategy=dummynear(op, space, space),
                toeplitz(polynomial)...,
            ),
        )
        petrov = MultilevelKernelInterpolation.PetrovGalerkinMap(
            op, space, space; strategy=petrovstrategy(space, space, polynomial)
        )
        @test H2Factory.modulation(H2Factory.i2otranslator(galerkin)) isa
            MultilevelKernelInterpolation.BsplineToeplitzModulation
        @test H2Factory.modulation(H2Factory.i2otranslator(petrov)) isa
            MultilevelKernelInterpolation.BsplineToeplitzModulation

        x = randn(H2FactoryTestUtils.checkrng(), ComplexF64, size(galerkin, 2))
        ygalerkin = zeros(ComplexF64, size(galerkin, 1))
        ypetrov = zeros(ComplexF64, size(petrov, 1))
        H2Factory.farmul!(ygalerkin, galerkin, x)
        H2Factory.farmul!(ypetrov, petrov, x)
        # A TIGHT TOLERANCE, not exact equality. This assertion used to be `==`, on the reasoning that
        # both maps run the same traversal over the same translators with the same chunking and that
        # chunks write disjoint nodes, so nothing reorders. The traversal part of that is true and is
        # pinned just below -- but it is not what `==` here was resting on. CONSTRUCTION is threaded
        # too, and it is not bit-reproducible: two independently built maps can hold couplings that
        # differ in the last bit, and then agree with each other only to roundoff.
        #
        # Measured 2026-08-11 at JULIA_NUM_THREADS=4: two independent builds of the SAME Galerkin map,
        # in one process, differ on ~4% of entries by 4.25e-17 relative (about 0.2 ULP) in roughly one
        # run in five. So this was never a Petrov-vs-Galerkin property at all -- comparing two Galerkin
        # maps flakes identically, and the old `==` failed about one run in three.
        #
        # `1e-12` sits five orders above that noise floor and still catches what the test is for: a
        # Petrov path that genuinely diverged from the Galerkin one is wrong by O(1), or at worst by
        # the multilevel interpolation accuracy of ~1e-6, never by an ULP.
        @test ypetrov ≈ ygalerkin rtol = 1e-12

        # The determinism half of the original claim, asserted directly instead of being inferred.
        # Repeating a matvec on ONE map is bit-for-bit reproducible -- chunks really do write disjoint
        # nodes, so a threaded traversal reorders nothing. This is the assertion a shared mutable
        # buffer or a genuine race would break, and unlike the cross-map comparison above it holds
        # exactly.
        let g2 = zeros(ComplexF64, size(galerkin, 1)),
            p2 = zeros(ComplexF64, size(petrov, 1))

            H2Factory.farmul!(g2, galerkin, x)
            H2Factory.farmul!(p2, petrov, x)
            @test g2 == ygalerkin
            @test p2 == ypetrov
        end
    end

    # Negative regression: when the two sides genuinely hold different grids at a level, the
    # translation is not Toeplitz and construction must refuse -- naming the level and both grids
    # rather than failing obscurely deep inside the collection. Built directly on
    # `SidedPolynomialDicts` so the incompatibility is unambiguous, since the real constructors
    # currently produce agreeing sides for every configuration reachable here.
    @testset "incompatible sided grids are refused with a useful message" begin
        levelrange = 3:4
        lagrange = MultilevelKernelInterpolation.BarycentricLagrangePolynomialEquidistant3D(
            6
        )
        bspline = MultilevelKernelInterpolation.BsplinePolynomialEquidistant{3,Float64}(
            4, 10
        )
        mismatched = MultilevelKernelInterpolation.SidedPolynomialDicts(
            Dict(level => lagrange for level in levelrange),
            Dict(level => bspline for level in levelrange),
        )
        @test !MultilevelKernelInterpolation.toeplitzcompatible(lagrange, bspline)

        err = try
            mismatched[first(levelrange)]
            nothing
        catch e
            e
        end
        @test err isa ErrorException
        message = sprint(showerror, err)
        @test occursin(string(first(levelrange)), message)
        @test occursin("not Toeplitz-compatible", message)
        # The range form must refuse too -- that is the shape the constructors actually index with.
        @test_throws ErrorException mismatched[collect(levelrange)]
    end

    # BUFFERED B-splines are the production path (the paper's method: buffered knot vector +
    # knot-removal stabilization). Plain `BsplinePolynomialEquidistant` is a low-level diagnostic
    # object that lacks that construction and sits at a FIXED ~0.17 error which never improves under
    # refinement -- benchmarking it produced a false "B-splines are inaccurate" conclusion once, so
    # this testset uses the buffered type deliberately and says why.
    @testset "buffered B-spline Galerkin: dense and Toeplitz agree, and are accurate" begin
        space = lagrangecxd0(m)
        polynomial = MultilevelKernelInterpolation.BufferedBsplinePolynomialEquidistant3D{
            Float64
        }(
            4, 10, MultilevelKernelInterpolation.AutoBufferFromProtrusion(1)
        )
        # ONE tree object shared by both builds: the equivalence assertion below should test the two
        # I2O strategies, not incidental determinism of tree construction.
        sharedtree = H2Trees.buildtree(
            space; builder=H2Trees.TwoNTreeBuilder(; minvalues=120)
        )
        build(i2ostrategy) = MultilevelKernelInterpolation.GalerkinMap(
            op,
            space,
            sharedtree;
            strategy=MultilevelKernelInterpolation.InterpolationStrategy(
                op,
                space;
                nearinteractionsstrategy=dummynear(op, space, space),
                polynomialstrategy=MultilevelKernelInterpolation.PolynomialStrategy(
                    op; polynomial=polynomial
                ),
                translatorstrategy=MultilevelKernelInterpolation.TranslatorStrategy(;
                    i2ostrategy=i2ostrategy
                ),
            ),
        )
        densemap = build(MultilevelKernelInterpolation.DenseI2OStrategy())
        toeplitzmap = build(MultilevelKernelInterpolation.ToeplitzI2OStrategy())
        densereference = assemble(op, space, space)

        # Fixed RNG: this testset asserts absolute accuracy bounds, so a reproducible `x` keeps a
        # failure meaningful instead of a tail draw.
        x = randn(MersenneTwister(20260720), ComplexF64, size(densemap, 2))
        ydense = zeros(ComplexF64, size(densemap, 1))
        ytoeplitz = zeros(ComplexF64, size(toeplitzmap, 1))
        H2Factory.farmul!(ydense, densemap, x)
        H2Factory.farmul!(ytoeplitz, toeplitzmap, x)

        far = H2Factory.farinteractions(densereference, densemap)
        exact = far * x
        # Measured 1.264e-6 for both strategies on this fixed vector; the bound leaves room beyond
        # that measurement rather than tracking it. (Unlike the Lagrange testsets above, which go
        # through `checkfarmul` and its own unseeded `randn`, this one drives a seeded vector.)
        @test norm(ydense - exact) / norm(exact) < 1e-4
        @test norm(ytoeplitz - exact) / norm(exact) < 1e-4
        # The equivalence is the load-bearing part: a Toeplitz kernel or modulation inconsistent with
        # the dense reduction could not reproduce the dense result.
        @test norm(ytoeplitz - ydense) / norm(ydense) < 1e-10
    end

    # Petrov + B-spline I2O: the supported set, pinned NUMERICALLY.
    #
    # Two defects were found and fixed here (Part E Phase 5b slices 2-3), both instances of ONE bug
    # class -- `SidedPolynomialDicts` is not an `AbstractArray`, so B-spline-specific dispatch fell
    # through to a generic path:
    #
    #   slice 2: `I2OTranslationCollection`'s dense route -> `EmptyChannels`, crash on first matvec.
    #   slice 3: `KernelTranslator` -> generic instead of `BsplineKernelTranslator`, which SILENTLY
    #            skipped the reduction to the interpolation basis. Same-space measured 3.96e-2
    #            against the dense far reference where Galerkin reached 5.49e-6.
    #
    # The second is why these testsets assert ACCURACY, not construction: slice 2's version asserted
    # the channel type and that `farmul!` ran, and both passed while the answer was 4% wrong.
    @testset "Petrov + B-spline I2O: same-grid is supported and accurate" begin
        space = lagrangecxd0(m)
        bspline = MultilevelKernelInterpolation.BufferedBsplinePolynomialEquidistant3D{
            Float64
        }(
            4, 10, MultilevelKernelInterpolation.AutoBufferFromProtrusion(1)
        )
        Adense = assemble(op, space, space)

        @testset "$label" for (label, translator) in (
            (
                "dense",
                MultilevelKernelInterpolation.TranslatorStrategy(;
                    i2ostrategy=MultilevelKernelInterpolation.DenseI2OStrategy()
                ),
            ),
            (
                "toeplitz",
                MultilevelKernelInterpolation.TranslatorStrategy(;
                    i2ostrategy=MultilevelKernelInterpolation.ToeplitzI2OStrategy()
                ),
            ),
        )
            A = MultilevelKernelInterpolation.PetrovGalerkinMap(
                op,
                space,
                space;
                strategy=petrovstrategy(
                    space, space, bspline; translatorstrategy=translator
                ),
            )
            # Structure: the B-spline-aware modulation, not the generic `NoModulation` fallback. This
            # is the exact thing that was silently wrong, and it is invisible in the numbers alone.
            # Both routes now compose an `I2OTranslation` whose `modulation` carries the B-spline
            # reduction (dense `BsplineDenseModulation` or Toeplitz `BsplineToeplitzModulation`); the
            # generic-fallback bug would instead leave `NoModulation`.
            @test H2Factory.modulation(H2Factory.i2otranslator(A)) isa Union{
                MultilevelKernelInterpolation.BsplineDenseModulation,
                MultilevelKernelInterpolation.BsplineToeplitzModulation,
            }

            # Accuracy against the dense far reference. Measured 5.49e-6 on BOTH routes; the bound
            # sits far below the 3.96e-2 the broken path produced, so a regression to it fails loudly.
            x = randn(MersenneTwister(20260721), ComplexF64, size(A, 2))
            y = zeros(ComplexF64, size(A, 1))
            H2Factory.farmul!(y, A, x)
            F = H2Factory.farinteractions(Adense, A)
            @test maximum(abs.(y .- F * x)) / maximum(abs.(F * x)) < 1e-4
        end
    end

    # MIXED SPACES on the DENSE and TOEPLITZ routes, both supported now, by different mechanisms:
    #
    #  - DENSE keeps each side its own protrusion-driven grid (INDEPENDENT, genuinely different node
    #    counts) and reduces each side with its own inverse.
    #  - TOEPLITZ cannot represent two different grids (its structure is one kernel sample per
    #    node-position DIFFERENCE, a property of ONE grid). So Petrov construction RECONCILES the two
    #    sides to a shared conservative grid per level for the Toeplitz strategy -- the two sides'
    #    own grids are incommensurate (different counts -> different spacing), so a shared grid is
    #    chosen rather than accepting independent ones. The result is Toeplitz-compatible by design.
    @testset "Petrov + B-spline I2O: rectangular grids" begin
        testspace, trialspace = lagrangec0d1(m), lagrangecxd0(m)
        bspline = MultilevelKernelInterpolation.BufferedBsplinePolynomialEquidistant3D{
            Float64
        }(
            4, 10, MultilevelKernelInterpolation.AutoBufferFromProtrusion(1)
        )

        @testset "dense route is supported and accurate" begin
            A = MultilevelKernelInterpolation.PetrovGalerkinMap(
                op,
                testspace,
                trialspace;
                strategy=petrovstrategy(
                    testspace,
                    trialspace,
                    bspline;
                    translatorstrategy=MultilevelKernelInterpolation.TranslatorStrategy(;
                        i2ostrategy=MultilevelKernelInterpolation.DenseI2OStrategy()
                    ),
                ),
            )

            # NON-VACUITY: the two sides must genuinely differ, or this would be the same-grid case
            # wearing a mixed-space label and would pass without exercising any of the new code.
            testsizes = unique(size(mm[1]) for mm in values(H2Factory.testmoments(A)))
            trialsizes = unique(size(mm[1]) for mm in values(H2Factory.trialmoments(A)))
            @test Set(testsizes) != Set(trialsizes)
            @test size(A, 1) != size(A, 2)

            # ACCURACY in all three modes, against the dense far reference. Measured forward 7.78e-6,
            # transpose 2.54e-6, adjoint 2.69e-6 -- the same order as the same-grid case (5.49e-6 /
            # 2.28e-6 / 2.19e-6), which is the point: rectangular is not a degraded mode.
            Adense = assemble(op, testspace, trialspace)
            H2FactoryTestUtils.checkfarmul(
                H2Factory.farinteractions(Adense, A), A; n=3, ϵ=1e-4, verbose=false
            )
            H2FactoryTestUtils.checkfarmul(
                H2Factory.farinteractions(transpose(Adense), transpose(A)),
                transpose(A);
                n=3,
                ϵ=1e-4,
                verbose=false,
            )
            H2FactoryTestUtils.checkfarmul(
                H2Factory.farinteractions(adjoint(Adense), adjoint(A)),
                adjoint(A);
                n=3,
                ϵ=1e-4,
                verbose=false,
            )
        end

        @testset "toeplitz route reconciles to a shared grid and is accurate" begin
            # Previously refused. For the Toeplitz strategy, Petrov construction now reconciles the two
            # sides' per-level protrusion REQUIREMENTS (conservative `max` at each shared level) BEFORE
            # building either polynomial hierarchy, so both sides come out Toeplitz-compatible by
            # construction and the whole per-side pipeline (leaf polynomials, moment collections,
            # I2I/O2O) is built consistently from the reconciled grid. This is the asymmetric-depth
            # case: `lagrangec0d1`/`lagrangecxd0` give trees of different depth (a shared level is a
            # leaf on one side), which reconciling requirements handles because each side's hierarchy
            # is rebuilt from its own reconciled protrusion vector rather than grid-patched afterwards.
            A = MultilevelKernelInterpolation.PetrovGalerkinMap(
                op,
                testspace,
                trialspace;
                strategy=petrovstrategy(testspace, trialspace, bspline),
            )
            @test H2Factory.modulation(H2Factory.i2otranslator(A)) isa
                MultilevelKernelInterpolation.BsplineToeplitzModulation

            # THE SHARED-GRID PROPERTY, and the contrast with dense. Dense keeps independent grids, so
            # its two sides differ in moment size (asserted above); the reconciled Toeplitz map shares
            # the grid at every I2O level, so its two sides AGREE in moment size. That agreement is
            # precisely what makes it Toeplitz-representable.
            @test Set(size(mm[1]) for mm in values(H2Factory.testmoments(A))) ==
                Set(size(mm[1]) for mm in values(H2Factory.trialmoments(A)))

            # ACCURACY in all three modes vs the dense far reference. Measured forward 7.6e-6,
            # transpose 2.58e-6, adjoint 2.59e-6 -- the same order as the dense route (7.78/2.54/
            # 2.69e-6), so reconciling to the conservative grid is not a degraded mode.
            Adense = assemble(op, testspace, trialspace)
            H2FactoryTestUtils.checkfarmul(
                H2Factory.farinteractions(Adense, A), A; n=3, ϵ=1e-4, verbose=false
            )
            H2FactoryTestUtils.checkfarmul(
                H2Factory.farinteractions(transpose(Adense), transpose(A)),
                transpose(A);
                n=3,
                ϵ=1e-4,
                verbose=false,
            )
            H2FactoryTestUtils.checkfarmul(
                H2Factory.farinteractions(adjoint(Adense), adjoint(A)),
                adjoint(A);
                n=3,
                ϵ=1e-4,
                verbose=false,
            )
        end
    end

    # Dense + Lagrange stays correct, which is what makes the refusals above NARROW rather than a
    # blanket ban on Petrov dense I2O.
    @testset "Petrov + dense Lagrange remains correct" begin
        space = lagrangecxd0(m)
        A = MultilevelKernelInterpolation.PetrovGalerkinMap(
            op,
            space,
            space;
            strategy=petrovstrategy(
                space,
                space,
                MultilevelKernelInterpolation.BarycentricLagrangePolynomialEquidistant3D(6);
                translatorstrategy=MultilevelKernelInterpolation.TranslatorStrategy(;
                    i2ostrategy=MultilevelKernelInterpolation.DenseI2OStrategy()
                ),
            ),
        )
        Adense = assemble(op, space, space)
        H2FactoryTestUtils.checkfarmul(
            H2Factory.farinteractions(Adense, A), A; n=3, ϵ=5e-5, verbose=false
        )
    end

    @testset "toeplitzcompatible discriminates by node PLACEMENT, not size" begin
        # `toeplitzcompatible` is the predicate the constructor's requirement-level reconciliation is
        # validated against. Its load-bearing job is to REFUSE grids that share a node count but place
        # nodes differently -- those have different difference sets and no single Toeplitz kernel.
        # Size alone, or even a protrusion scalar, is insufficient: Toeplitz depends on the ACTUAL node
        # coordinates. (There is no post-hoc grid-picking helper anymore -- the shared grid is produced
        # by reconciling protrusion requirements before construction; see `PetrovGalerkinMap`.)

        # (a) SCALED LAGRANGE: same order (same count), different scaling/shift -> same size, different
        #     node range, genuinely Toeplitz-INCOMPATIBLE. The case a protrusion/count proxy passes
        #     only by luck; the per-dimension node-range comparison rejects it by construction.
        base = MultilevelKernelInterpolation.BarycentricLagrangePolynomialEquidistant3D(6)
        wide = MultilevelKernelInterpolation.ScaledShiftedPolynomial(
            base, 1 + 2 * 0.3, [-0.3 for _ in 1:3]
        )
        narrow = MultilevelKernelInterpolation.ScaledShiftedPolynomial(
            base, 1 + 2 * 0.1, [-0.1 for _ in 1:3]
        )
        @test size(LinearIndices(wide)) == size(LinearIndices(narrow))   # same count
        @test range(wide, 1) != range(narrow, 1)                         # different placement
        @test !MultilevelKernelInterpolation.toeplitzcompatible(wide, narrow)
        @test !MultilevelKernelInterpolation.toeplitzcompatible(narrow, wide)                     # order-independent

        # (b) BUFFERED B-SPLINE: same count, different realized protrusion, also incompatible.
        bwide = MultilevelKernelInterpolation.BufferedBsplinePolynomialEquidistant3D{
            Float64
        }(
            4, 6, 1
        )   # count 8
        bnarrow = MultilevelKernelInterpolation.BufferedBsplinePolynomialEquidistant3D{
            Float64
        }(
            4, 8, 1
        ) # count 8
        @test size(LinearIndices(bwide)) == size(LinearIndices(bnarrow))
        @test !MultilevelKernelInterpolation.toeplitzcompatible(bwide, bnarrow)
        @test !MultilevelKernelInterpolation.toeplitzcompatible(bnarrow, bwide)
    end

    @testset "toeplitzcompatible is the documented condition" begin
        p = MultilevelKernelInterpolation.BarycentricLagrangePolynomialEquidistant3D(6)
        q = MultilevelKernelInterpolation.BsplinePolynomialEquidistant{3,Float64}(4, 10)
        @test MultilevelKernelInterpolation.toeplitzcompatible(p, p)
        @test !MultilevelKernelInterpolation.toeplitzcompatible(p, q)

        # THE DISCRIMINATING CASES. The two above are passed just as well by a plain `==`, which is
        # what this predicate used to be -- so on their own they do not test anything. These do:
        #
        # Two SEPARATELY CONSTRUCTED but identical buffered B-splines must be compatible. They are
        # `!=` (the type does not compare structurally), so `==` refused a genuinely shared grid and
        # blocked a legitimate same-space Petrov + buffered B-spline map.
        buffered = MultilevelKernelInterpolation.BufferedBsplinePolynomialEquidistant3D{
            Float64
        }(
            4, 10, 1
        )
        bufferedcopy = MultilevelKernelInterpolation.BufferedBsplinePolynomialEquidistant3D{
            Float64
        }(
            4, 10, 1
        )
        @test !(buffered == bufferedcopy)          # documents WHY `==` is not usable here
        @test MultilevelKernelInterpolation.toeplitzcompatible(buffered, bufferedcopy)

        # ...and the predicate must not have become a bare same-type/same-size test in the process:
        # a different grid of the SAME type stays incompatible, because differing node spacing gives
        # a different difference set and so no single Toeplitz kernel.
        @test !MultilevelKernelInterpolation.toeplitzcompatible(
            buffered,
            MultilevelKernelInterpolation.BufferedBsplinePolynomialEquidistant3D{Float64}(
                4, 12, 1
            ),
        )
    end

    @testset "assembletreepolynomials rejects a misaligned protrusion vector" begin
        # The requirement-level reconciliation passes a protrusion vector into `assembletreepolynomials`
        # that MUST stay aligned with `levels(tree)` -- it is consumed positionally and its `maximum`
        # seeds the whole hierarchy, so a stray entry would silently build the wrong grids. Pin the
        # guard so this positional invariant is not eroded later.
        space = lagrangecxd0(m)
        tree = H2Trees.buildtree(space; builder=H2Trees.TwoNTreeBuilder(; minvalues=120))
        ps = MultilevelKernelInterpolation.PolynomialStrategy(
            op;
            polynomial=MultilevelKernelInterpolation.BarycentricLagrangePolynomialEquidistant3D(
                6
            ),
        )
        n = length(H2Trees.levels(tree))
        # Correct length constructs; a wrong length errors rather than proceeding.
        @test MultilevelKernelInterpolation.assembletreepolynomials(
            tree, space, ps; protrusion=zeros(n)
        ) isa Tuple
        @test_throws ErrorException MultilevelKernelInterpolation.assembletreepolynomials(
            tree, space, ps; protrusion=zeros(n + 1)
        )
    end
end

end # module TestPetrovToeplitz
