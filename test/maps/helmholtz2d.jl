module TestHelmholtz2D

using Test
using LinearAlgebra
using Random
using CompScienceMeshes, BEAST
using H2Trees, H2Factory
using H2FactoryTestUtils
using MultilevelKernelInterpolation

# 2-D SCALAR HELMHOLTZ, END TO END.
#
# WHY THIS TEST IS SHAPED AS A CONVERGENCE SWEEP AND NOT A SINGLE TOLERANCE. Everything genuinely new
# for 2-D is a CONVENTION: which Hankel (`hankelh2`, outgoing), which amplitude (`-i/4`), which side
# the double layer's normal derivative sits on, and whether `α` picks up a sign. Each of those is a
# factor of `-1`, `i`, or a swap, none of which a single "is the error small" assertion can
# separate from ordinary interpolation error, because a fixture that is loose enough to pass at a
# usable polynomial order is loose enough to hide a wrong sign on a subset of the interactions.
#
# A CONVERGENCE SWEEP CAN separate them. The interpolation-based factorization is spectrally accurate
# in the polynomial order, so a correct set of conventions falls by orders of magnitude per order
# step, while ANY wrong convention leaves a fixed O(1)-relative residue that the sweep flattens out
# against. So the sweep asserts the DROP, not just the floor.
#
# The reference is `H2Factory.farinteractions(Adense, A)`: the TRUE far block, recovered by indexing
# BEAST's own dense assembly with `A`'s own near/far pattern. So the comparison is against BEAST,
# not against another MultilevelKernelInterpolation code path, and the near-field quadrature is skipped (`Dummy...`) without
# weakening anything, the factorization is only claimed on far pairs in the first place.

const κ = 2.0

# `meshcircle` gives segments in R², which is what makes `lagrangecxd0`/`lagrangec0d1` 2-D spaces and
# `buildtree` produce a quadtree. This is the same 2-D geometry stack MLFMA's `Operators/helmholtz2d.jl`
# validates against.
const MESH = CompScienceMeshes.meshcircle(1.0, 0.05)
const X = lagrangecxd0(MESH)
const TREE = H2Trees.buildtree(X; builder=H2Trees.TwoNTreeBuilder(; minvalues=30))
# Needed only by the stored-matrix count, which builds the two couplings directly rather than
# through a whole map, counting what a map stores would mean reaching through the map's internals.
const PLAN = H2Trees.buildplans(TREE).testdisaggregationplan

dummynear(op, testspace, trialspace) = H2Factory.DummyNearInteractionsStrategy(
    BEAST.defaultquadstrat(op, testspace, trialspace)
)

function galerkinmap(op, space, tree, polynomial; symmetryreduced=false)
    return MultilevelKernelInterpolation.GalerkinKernelInterpolation(
        op,
        space,
        tree;
        strategy=MultilevelKernelInterpolation.InterpolationStrategy(
            op,
            space;
            polynomialstrategy=MultilevelKernelInterpolation.PolynomialStrategy(
                op; polynomial=polynomial
            ),
            translatorstrategy=MultilevelKernelInterpolation.TranslatorStrategy(;
                i2ostrategy=MultilevelKernelInterpolation.DenseI2OStrategy(;
                    symmetryreduced=symmetryreduced
                ),
            ),
            nearinteractionsstrategy=dummynear(op, space, space),
        ),
    )
end

relativeerror(A, reference, x) = norm(A * x - reference * x) / norm(reference * x)

# The Toeplitz I2O route, which is the DEFAULT translator, so this is what a 2-D user gets by
# leaving the translator alone, and it is the configuration that has to work for 2-D to be usable
# without an opt-out. `i2ostrategy` is named explicitly rather than left to the default so the test
# states which route it exercises; `toeplitzmap(op, 6)` and the bare default are the same map.
#
# EQUIDISTANT, not Chebyshev: the Toeplitz form exists because a node-index DIFFERENCE alone
# determines the kernel entry, which needs equal node spacing. `toeplitzi2ocoupling` asserts
# `isequidistantpolynomial`.
function toeplitzmap(op, order; symmetryreduced=false)
    return MultilevelKernelInterpolation.GalerkinKernelInterpolation(
        op,
        X,
        TREE;
        strategy=MultilevelKernelInterpolation.InterpolationStrategy(
            op,
            X;
            polynomialstrategy=MultilevelKernelInterpolation.PolynomialStrategy(
                op;
                polynomial=MultilevelKernelInterpolation.BarycentricLagrangePolynomialEquidistant{
                    2,Float64
                }(
                    order
                ),
            ),
            translatorstrategy=MultilevelKernelInterpolation.TranslatorStrategy(;
                i2ostrategy=MultilevelKernelInterpolation.ToeplitzI2OStrategy(;
                    symmetryreduced=symmetryreduced
                ),
            ),
            nearinteractionsstrategy=dummynear(op, X, X),
        ),
    )
end

@testset verbose = true "2-D scalar Helmholtz" begin
    @testset "the Green's function matches BEAST's own closed form" begin
        op = Helmholtz2D.singlelayer(; wavenumber=κ)

        # Independently written out here rather than read back from the package: this is the ONE
        # place the 2-D convention is stated, so restating it through `scalargreen2Dkernel` itself
        # would assert nothing. `MultilevelKernelInterpolation.hankelh2` is the same `SpecialFunctions` binding the package
        # imports, used so the test environment needs no dependency of its own.
        for (x, y) in (
            ([0.3, -0.7], [-2.1, 1.4]), ([0.0, 0.0], [1.0, 0.0]), ([5.0, 5.0], [-5.0, -4.0])
        )
            R = hypot(x[1] - y[1], x[2] - y[2])
            @test MultilevelKernelInterpolation.scalargreen2Dkernel(op, x, y) ≈
                -im / 4 * MultilevelKernelInterpolation.hankelh2(0, κ * R)
        end

        # The wavenumber recovery has two branches (real `k` fast path vs complex). A purely
        # imaginary γ is the branch every call above takes, so exercise the other one too.
        lossy = BEAST.HH2DSingleLayerFDBIO(1.0, 0.3 + 2.0im)
        @test MultilevelKernelInterpolation.operatorgamma(lossy) == 0.3 + 2.0im
        let R = hypot(0.3 + 2.1, -0.7 - 1.4)
            @test MultilevelKernelInterpolation.scalargreen2Dkernel(
                lossy, [0.3, -0.7], [-2.1, 1.4]
            ) ≈ -im / 4 *
                  MultilevelKernelInterpolation.hankelh2(0, -im * (0.3 + 2.0im) * R)
        end
    end

    @testset "the operator table routes the 2-D operators" begin
        single = Helmholtz2D.singlelayer(; wavenumber=κ)
        double = Helmholtz2D.doublelayer(; wavenumber=κ)
        doubletransposed = Helmholtz2D.doublelayer_transposed(; wavenumber=κ)

        for op in (single, double, doubletransposed)
            @test MultilevelKernelInterpolation.i2otranslation(op) ==
                MultilevelKernelInterpolation.ScalarGreen2DTranslation()
            # The whole point of the supertype: every downstream rule is shared with the 3-D scalar
            # trait, so only the closed form differs.
            @test MultilevelKernelInterpolation.i2otranslation(op) isa
                MultilevelKernelInterpolation.ScalarI2OTranslation
            @test MultilevelKernelInterpolation.symmetrycomponentaction(
                MultilevelKernelInterpolation.i2otranslation(op)
            ) === nothing
            @test MultilevelKernelInterpolation.i2otranslationtrait(
                MultilevelKernelInterpolation.i2otranslation(op)
            ) == MultilevelKernelInterpolation.DenseTranslation()
            @test MultilevelKernelInterpolation.α(op, nothing) == op.alpha
            @test MultilevelKernelInterpolation.λ(op) ≈ 2π / κ
        end

        # The moment operators are reused UNCHANGED from 3-D because their parameter is the
        # component count, not the spatial dimension. If that ever stops being true these break.
        @test MultilevelKernelInterpolation.moment(single, nothing) ==
            MultilevelKernelInterpolation.LaplaceMoment{1,Float64}()
        # The normal derivative sits on the TRIAL side for the double layer and on the TEST side for
        # its transpose, matching BEAST's `dot(n_y, -∇ₓG)` and `dot(n_x, ∇ₓG)` respectively.
        @test MultilevelKernelInterpolation.trialmoment(double, nothing) ==
            MultilevelKernelInterpolation.NormalGradLaplaceMoment{1,Float64}()
        @test MultilevelKernelInterpolation.testmoment(double, nothing) ==
            MultilevelKernelInterpolation.LaplaceMoment{1,Float64}()
        @test MultilevelKernelInterpolation.testmoment(doubletransposed, nothing) ==
            MultilevelKernelInterpolation.NormalGradLaplaceMoment{1,Float64}()
        @test MultilevelKernelInterpolation.trialmoment(doubletransposed, nothing) ==
            MultilevelKernelInterpolation.LaplaceMoment{1,Float64}()
    end

    @testset "the far field converges spectrally in the polynomial order" begin
        op = Helmholtz2D.singlelayer(; wavenumber=κ)
        Adense = assemble(op, X, X)

        errors = map((2, 4, 6, 8)) do order
            Random.seed!(20260901)
            A = galerkinmap(
                op,
                X,
                TREE,
                MultilevelKernelInterpolation.BarycentricLagrangePolynomialChebyshev{
                    2,Float64
                }(
                    order
                ),
            )
            reference = H2Factory.farinteractions(Adense, A)
            # `checkrng()` INSIDE the loop on purpose, unlike everywhere else in this suite. A fresh
            # RNG on the same fixed seed gives every order the SAME probe vector, and that is what
            # makes the convergence assertion below a measurement of the ORDER rather than of the
            # draw: the ratio `errors[i]/errors[i-1]` would otherwise carry draw noise from two
            # different vectors. Hoisting an advancing rng out of this `map` would reintroduce it.
            return relativeerror(
                A, reference, randn(H2FactoryTestUtils.checkrng(), ComplexF64, size(A, 2))
            )
        end

        # THE DROP IS THE ASSERTION. Each order step must gain at least an order of magnitude; a
        # wrong sign or amplitude anywhere in the conventions flattens this immediately.
        for i in 2:length(errors)
            @test errors[i] < errors[i - 1] / 10
        end
        # And the sweep must actually reach a useful accuracy, so that "monotone decrease" cannot be
        # satisfied by a sequence that is uniformly bad.
        @test errors[end] < 1e-7
        # Non-vacuity for the ratio test above: the coarsest order must be genuinely inaccurate,
        # otherwise the whole sweep sits on the floor and the ratios mean nothing.
        @test errors[1] > 1e-4
    end

    @testset "transpose and adjoint" begin
        op = Helmholtz2D.singlelayer(; wavenumber=κ)
        Adense = assemble(op, X, X)
        A = galerkinmap(
            op,
            X,
            TREE,
            MultilevelKernelInterpolation.BarycentricLagrangePolynomialChebyshev{2,Float64}(
                6
            ),
        )
        x = randn(Random.Xoshiro(4711), ComplexF64, size(A, 2))

        @test relativeerror(A, H2Factory.farinteractions(Adense, A), x) < 1e-5
        @test relativeerror(
            transpose(A), H2Factory.farinteractions(transpose(Adense), transpose(A)), x
        ) < 1e-5
        @test relativeerror(
            adjoint(A), H2Factory.farinteractions(adjoint(Adense), adjoint(A)), x
        ) < 1e-5
    end

    @testset "the lattice-symmetry reduction carries over to 2-D" begin
        op = Helmholtz2D.singlelayer(; wavenumber=κ)
        polynomial = MultilevelKernelInterpolation.BarycentricLagrangePolynomialEquidistant{
            2,Float64
        }(
            6
        )
        Adense = assemble(op, X, X)

        plain = galerkinmap(op, X, TREE, polynomial; symmetryreduced=false)
        reduced = galerkinmap(op, X, TREE, polynomial; symmetryreduced=true)
        x = randn(Random.Xoshiro(4711), ComplexF64, size(plain, 2))

        # AGREEMENT IS AT ROUNDING, NOT BITWISE, and the distinction is the point. The reduction is
        # an exact identity in exact arithmetic, one stored matrix per orbit, every orbit-mate read
        # through an index permutation, nothing approximated. But it is realized by permuting the
        # VECTORS around a shared `mul!` rather than permuting the matrix, so BLAS accumulates the
        # same terms in a different ORDER and the two results differ in the last few ULP. Measured
        # here at ~1e-15 in all three modes, matching the 3-D route.
        #
        # So the bound is tight enough that a genuinely wrong permutation cannot pass, a wrong
        # permutation reorders whole node contributions, which is an O(1) relative change, fifteen
        # orders of magnitude above this, while not asserting a bitwise equality the
        # implementation never promised.
        for (plainmap, reducedmap) in (
            (plain, reduced),
            (transpose(plain), transpose(reduced)),
            (adjoint(plain), adjoint(reduced)),
        )
            plainresult = plainmap * x
            @test maximum(abs, plainresult - reducedmap * x) / maximum(abs, plainresult) <
                1e-13
        end
        @test relativeerror(reduced, H2Factory.farinteractions(Adense, reduced), x) < 1e-5
        @test relativeerror(
            transpose(reduced),
            H2Factory.farinteractions(transpose(Adense), transpose(reduced)),
            x,
        ) < 1e-5
        @test relativeerror(
            adjoint(reduced),
            H2Factory.farinteractions(adjoint(Adense), adjoint(reduced)),
            x,
        ) < 1e-5

        # AGREEMENT ALONE WOULD BE VACUOUS, which is the trap the 3-D reduction test calls out and
        # this one needs guarding against for the same reason: `symmetryreduceddensei2ocoupling`
        # RETURNS THE UNREDUCED COUPLING when the displacements are not lattice aligned. That
        # fallback agrees with the plain map perfectly. It IS the plain map, so every check above
        # would still pass on a build where the reduction never engaged at all.
        #
        # So count what is actually stored, by object identity on the matrix each entry reads from,
        # so orbit-mates sharing one matrix are counted once.
        schedule = H2Factory.planschedule(PLAN, Base.Threads.nthreads())
        polynomials = MultilevelKernelInterpolation.levelpolynomials(
            polynomial, H2Trees.levels(TREE), (level) -> false
        )
        plaincoupling, reducedcoupling = H2Factory.withnchunks(
            H2Factory.schedulechunkcounts(schedule)
        ) do
            return (
                MultilevelKernelInterpolation.densei2ocoupling(
                    op, TREE, polynomials, PLAN; verbose=false
                ),
                MultilevelKernelInterpolation.symmetryreduceddensei2ocoupling(
                    op, TREE, polynomials, PLAN; verbose=false
                ),
            )
        end

        # The type is the direct statement that the fallback did not fire.
        @test reducedcoupling isa H2Factory.SymmetryReducedI2OTranslationCollection
        @test keys(reducedcoupling) == keys(plaincoupling)

        root(t::H2Factory.PermutedDenseTranslation) = t.canonical
        root(t) = t
        stored(coupling) =
            length(Set(objectid(root(coupling[key])) for key in keys(coupling)))

        # Measured 40 -> 7 here. The ceiling is the group order, `D! * 2^D` = 8 in two dimensions
        # against 48 in three, so 2-D reduces by a genuinely smaller factor, and the bound is set
        # against what 2-D can actually deliver rather than carried over from the 3-D test.
        @test stored(reducedcoupling) < stored(plaincoupling) / 4
        # Non-vacuity for that ratio: it would also hold if the plain side stored almost nothing.
        @test stored(plaincoupling) > 20
    end

    @testset "the Toeplitz I2O route, the default translator, serves 2-D" begin
        op = Helmholtz2D.singlelayer(; wavenumber=κ)
        Adense = assemble(op, X, X)
        x = randn(Random.Xoshiro(4711), ComplexF64, size(Adense, 2))

        # THE SAME SHAPE OF ASSERTION AS THE DENSE SWEEP, for a sharper version of the same reason.
        # The Toeplitz route reaches the kernel through a completely different index path: a signed
        # difference per axis, folded into a circulant embedding, transformed, multiplied in the
        # frequency domain and transformed back. A rank generalization that mismatched an axis,
        # the difference vectors against the node lookup, or either against the tensor's own
        # ordering, still produces a plausible matrix on a cubic grid, because the axes have the
        # same length and a swap is invisible to any size check. It does not produce a CONVERGING
        # one, so the sweep is what separates a right generalization from a lucky-looking one.
        errors = map((2, 4, 6)) do order
            A = toeplitzmap(op, order)
            return relativeerror(A, H2Factory.farinteractions(Adense, A), x)
        end
        for i in 2:length(errors)
            @test errors[i] < errors[i - 1] / 10
        end
        # Equidistant nodes, so the floor is set above the Chebyshev sweep's: at order 8 equidistant
        # interpolation is already conditioning-limited and the per-step gain collapses (measured
        # 15x, against 89x for the step before it), which is a property of the node family and not
        # of this route.
        @test errors[end] < 1e-6
        @test errors[1] > 1e-4

        # AGREEMENT WITH THE DENSE ROUTE, AT ROUNDING. The two share the kernel closed form and the
        # polynomial and differ only in how the coupling is stored and applied, so on one equidistant
        # grid they are the same operator up to summation order, measured 7.1e-16. The sweep above
        # says the Toeplitz route converges; this says it converges to the ALREADY-VERIFIED dense
        # answer rather than to something else that also happens to converge.
        toeplitz = toeplitzmap(op, 6)
        dense = galerkinmap(
            op,
            X,
            TREE,
            MultilevelKernelInterpolation.BarycentricLagrangePolynomialEquidistant{
                2,Float64
            }(
                6
            ),
        )
        @test norm(toeplitz * x - dense * x) / norm(dense * x) < 1e-13

        @test relativeerror(
            transpose(toeplitz),
            H2Factory.farinteractions(transpose(Adense), transpose(toeplitz)),
            x,
        ) < 1e-5
        @test relativeerror(
            adjoint(toeplitz),
            H2Factory.farinteractions(adjoint(Adense), adjoint(toeplitz)),
            x,
        ) < 1e-5

        # and the symmetry reduction on top of it ------------------------------------------
        reduced = toeplitzmap(op, 6; symmetryreduced=true)
        @test norm(reduced * x - toeplitz * x) / norm(toeplitz * x) < 1e-13
        @test relativeerror(reduced, H2Factory.farinteractions(Adense, reduced), x) < 1e-5

        # AGREEMENT ALONE IS VACUOUS HERE TOO. `symmetryreducedtoeplitzi2ocoupling` has the same
        # non-lattice fallback to the unreduced coupling as its dense counterpart, and that fallback
        # would satisfy every check above by being the plain map. So count what is stored, and check
        # the type that says the fallback did not fire.
        polynomials = MultilevelKernelInterpolation.levelpolynomials(
            MultilevelKernelInterpolation.BarycentricLagrangePolynomialEquidistant{
                2,Float64
            }(
                6
            ),
            H2Trees.levels(TREE),
            (level) -> false,
        )
        schedule = H2Factory.planschedule(PLAN, Base.Threads.nthreads())
        plaincoupling, reducedcoupling = H2Factory.withnchunks(
            H2Factory.schedulechunkcounts(schedule)
        ) do
            return (
                MultilevelKernelInterpolation.toeplitzi2ocoupling(
                    op, TREE, polynomials, PLAN; verbose=false
                ),
                MultilevelKernelInterpolation.symmetryreducedtoeplitzi2ocoupling(
                    op, TREE, polynomials, PLAN; verbose=false
                ),
            )
        end
        @test reducedcoupling.i2ocollection isa
            H2Factory.SymmetryReducedI2OTranslationCollection

        tensorroot(t::H2Factory.ToeplitzMatrixRepresentation) = tensorroot(t.toeplitz)
        tensorroot(t::H2Factory.PermutedTranslation) = t.canonical
        tensorroot(t) = t
        storedtensors(coupling) = length(
            Set(
                objectid(tensorroot(coupling.i2ocollection[key])) for
                key in keys(coupling.i2ocollection)
            ),
        )

        # Measured 40 -> 7 over 100 interactions: the SAME orbit structure the dense route finds on
        # this tree, as it has to be, the orbits are a property of the tree's displacements, not of
        # how a coupling chooses to store itself.
        @test storedtensors(reducedcoupling) < storedtensors(plaincoupling) / 4
        @test storedtensors(plaincoupling) > 20

        # The stored difference tensors are RANK 2, stated as a type rather than inferred from the
        # numbers coming out right. Before this the tensor was `Array{T,3}` unconditionally and 2-D
        # was refused at construction; a generalization that reached rank 2 only by accident of a
        # trailing singleton axis would fail here while passing everything above.
        for coupling in (plaincoupling, reducedcoupling)
            entry = coupling.i2ocollection[first(keys(coupling.i2ocollection))]
            @test ndims(entry.toeplitz) == 2
            @test eltype(entry.toeplitz) == ComplexF64
        end
    end

    @testset "B-splines serve 2-D on both I2O routes" begin
        op = Helmholtz2D.singlelayer(; wavenumber=κ)
        Adense = assemble(op, X, X)
        x = randn(Random.Xoshiro(4711), ComplexF64, size(Adense, 2))

        # BUFFERED, on a protrusion-checked tree, not an incidental choice. A B-spline is ZERO
        # outside its knot span, unlike a Lagrange polynomial, which extrapolates. Elements
        # protruding past their box are therefore not merely badly approximated but unrepresented,
        # and the error that leaves is a FIXED residue: measured 0.136 here, flat to five digits
        # across orders 2/3/4, and 0.380 equally flat for the same unbuffered configuration in 3-D.
        # Which is to say a sweep run on an unbuffered B-spline would be measuring the protrusion,
        # not the reduction, at every order, so the buffer is what makes this testset able to see
        # its own subject at all.
        bsplinetree = H2Trees.buildtree(
            X;
            builder=H2Trees.TwoNTreeBuilder(;
                minvalues=30,
                protrusion=H2Trees.ProtrusionCheck(;
                    max=0.25, compute=H2Trees.BEASTProtrusionFunctor(X)
                ),
            ),
        )
        bsplinepolynomial(degree, nbasis) =
            MultilevelKernelInterpolation.BufferedBsplinePolynomialEquidistant{2,Float64}(
                degree, nbasis, MultilevelKernelInterpolation.AutoBufferFromProtrusion(0)
            )
        function bsplinemap(degree, nbasis, i2ostrategy)
            return MultilevelKernelInterpolation.GalerkinKernelInterpolation(
                op,
                X,
                bsplinetree;
                strategy=MultilevelKernelInterpolation.InterpolationStrategy(
                    op,
                    X;
                    polynomialstrategy=MultilevelKernelInterpolation.PolynomialStrategy(
                        op; polynomial=bsplinepolynomial(degree, nbasis)
                    ),
                    translatorstrategy=MultilevelKernelInterpolation.TranslatorStrategy(;
                        i2ostrategy=i2ostrategy
                    ),
                    nearinteractionsstrategy=dummynear(op, X, X),
                ),
            )
        end

        # THE SWEEP CROSSES `BSPLINEREDUCTIONKRONECKERCROSSOVER`, and that is the point of these
        # three sizes rather than any three. Below it the reduction is one precomputed Kronecker
        # matrix; at or above it the SAME reduction is applied separably, one 1-D matrix per axis,
        # through `D-1` staged buffers. Those are two different implementations of one identity, and
        # only the separable one has a buffer count that varies with the dimension, so a sweep
        # confined to either side of the crossover would leave half the rank generalization untested.
        grids = map(
            nbasis -> size(LinearIndices(bsplinepolynomial(4, nbasis)), 1), (8, 12, 16)
        )
        @test grids[1] < MultilevelKernelInterpolation.BSPLINEREDUCTIONKRONECKERCROSSOVER
        @test all(
            grids[2:end] .>=
            MultilevelKernelInterpolation.BSPLINEREDUCTIONKRONECKERCROSSOVER,
        )

        errors = map((8, 12, 16)) do nbasis
            A = bsplinemap(4, nbasis, MultilevelKernelInterpolation.DenseI2OStrategy())
            return relativeerror(A, H2Factory.farinteractions(Adense, A), x)
        end

        # ALGEBRAIC, NOT SPECTRAL, so the per-step bound is looser than the Lagrange sweeps above:
        # a degree-`p` B-spline converges as the KNOT SPACING refines, at order `p+1`, not
        # exponentially in the order. Measured 1.7e-6 -> 3.8e-8 -> 4.4e-9, i.e. 43x then 8.6x.
        for i in 2:length(errors)
            @test errors[i] < errors[i - 1] / 5
        end
        @test errors[end] < 1e-8
        @test errors[1] > 1e-7

        # AGREEMENT ACROSS THE TWO ROUTES, at rounding, on BOTH sides of the crossover. The dense and
        # Toeplitz routes share the polynomial and the 1-D inverse and differ in how the coupling is
        # stored, applied, and modulated, the Toeplitz one fusing the basis change with an FFT
        # round trip. Measured 1.5e-15 below the crossover and 3.1e-15 above it.
        for nbasis in (8, 12)
            dense = bsplinemap(4, nbasis, MultilevelKernelInterpolation.DenseI2OStrategy())
            toeplitz = bsplinemap(
                4, nbasis, MultilevelKernelInterpolation.ToeplitzI2OStrategy()
            )
            @test norm(toeplitz * x - dense * x) / norm(dense * x) < 1e-13
            @test relativeerror(toeplitz, H2Factory.farinteractions(Adense, toeplitz), x) <
                1e-5
        end

        toeplitz = bsplinemap(4, 12, MultilevelKernelInterpolation.ToeplitzI2OStrategy())
        @test relativeerror(
            transpose(toeplitz),
            H2Factory.farinteractions(transpose(Adense), transpose(toeplitz)),
            x,
        ) < 1e-5
        @test relativeerror(
            adjoint(toeplitz),
            H2Factory.farinteractions(adjoint(Adense), adjoint(toeplitz)),
            x,
        ) < 1e-5

        # THE KNOT-REMOVAL STABILIZATION IS ENGAGED, not merely available, and this needs saying
        # because nothing else here would notice if it stopped being. Every assertion above is
        # satisfied by an UNSTABILIZED B-spline too: it would converge, and both routes would still
        # agree with each other. Stabilization is on only because the polynomials are buffered and
        # `knotstoremove` happens to return 1 at these orders; were that ever 0, this testset would
        # keep passing while quietly testing a different basis.
        #
        # It also matters WHICH path that selects. `interpolationinverse` returns EARLY in the
        # stabilized branch, with an `SMatrix`, so a stabilized B-spline never reaches the
        # Kronecker form and always modulates through the separable sweep, with a RECTANGULAR
        # matrix. That is the `from != to` case, where the staged buffers are the asymmetric ones
        # (`raw` on the axes not yet converted, `reduced` on those already done). In three
        # dimensions those were the two buffers the route hardcoded; at rank `D` there are `D-1` of
        # them and `separablescratch` derives the shapes. So this is also the assertion that the
        # generalization is exercised in its harder shape and not only on square grids.
        protrusion = H2Trees.maxprotrusion(
            bsplinetree; computeprotrusion=H2Trees.BEASTProtrusionFunctor(X)
        )
        levelpolys = MultilevelKernelInterpolation.levelpolynomials(
            bsplinepolynomial(4, 12),
            H2Trees.levels(bsplinetree),
            (level) -> false;
            protrusion=protrusion,
        )
        for level in H2Trees.levels(bsplinetree)
            p = levelpolys[level]
            # Raw basis strictly larger than the reduced (stabilized) grid: measured 14 against 12.
            @test MultilevelKernelInterpolation.numpolynomials(p)[1] >
                length(MultilevelKernelInterpolation.range(p, 1))
            mlu = MultilevelKernelInterpolation.interpolationinverse(
                MultilevelKernelInterpolation.interpolationmatrix(p), p
            )
            # Not the `degree == 1` no-op, and not square, the two ways the stabilized branch
            # could be absent without any number above changing.
            @test !(mlu isa UniformScaling)
            @test size(mlu, 1) != size(mlu, 2)
            @test length(
                MultilevelKernelInterpolation.separablescratch(
                    ComplexF64, mlu, MultilevelKernelInterpolation.dims(p)
                ),
            ) == MultilevelKernelInterpolation.dims(p) - 1
        end

        # THE SHAPE THAT WAS ACTUALLY WRONG, stated as a size rather than inferred from the numbers
        # above coming out right. `_kroneckerreduction` took THREE copies of the 1-D inverse
        # unconditionally, so on a 2-D grid it built an `n^3 x n^3` matrix to reduce an `n^2 x n^2`
        # translation. That surfaced as a bare `DimensionMismatch` naming neither B-splines nor the
        # dimension, and it is the reason the dense route, which the previous commit's refusal
        # message recommended to 2-D users, did not in fact serve them here.
        below = MultilevelKernelInterpolation._luT(bsplinepolynomial(4, 8))
        n = size(below, 1)
        @test size(MultilevelKernelInterpolation._kroneckerreduction(below, 2)) ==
            (n^2, n^2)
        @test size(MultilevelKernelInterpolation._kroneckerreduction(below, 3)) ==
            (n^3, n^3)
        # Above the crossover it declines to build one at all; that `nothing` is what routes the
        # translation to the separable sweep, so it is part of the same contract, and it has to be
        # read off a DIFFERENT polynomial, since one grid cannot sit on both sides of the threshold.
        @test isnothing(
            MultilevelKernelInterpolation._kroneckerreduction(
                MultilevelKernelInterpolation._luT(bsplinepolynomial(4, 12)), 2
            ),
        )
    end

    @testset "the double layers, which need the Petrov path" begin
        # `GalerkinKernelInterpolation` asserts `trialmoment == testmoment`, and the double layers put the normal
        # derivative on exactly one side, so they are Petrov by construction, in 2-D for the same
        # reason as in 3-D, not for any dimensional reason.
        testspace = lagrangec0d1(MESH)
        trialspace = lagrangecxd0(MESH)
        tree = H2Trees.buildtree(
            testspace,
            trialspace;
            builder=H2Trees.BlockTreeBuilder(;
                test=H2Trees.TwoNTreeBuilder(; minvalues=30),
                trial=H2Trees.TwoNTreeBuilder(; minvalues=30),
            ),
        )
        polynomial = MultilevelKernelInterpolation.BarycentricLagrangePolynomialChebyshev{
            2,Float64
        }(
            6
        )

        @testset "$label" for (label, op) in (
            ("doublelayer", Helmholtz2D.doublelayer(; wavenumber=κ)),
            ("doublelayer_transposed", Helmholtz2D.doublelayer_transposed(; wavenumber=κ)),
        )
            Adense = assemble(op, testspace, trialspace)
            A = MultilevelKernelInterpolation.PetrovGalerkinKernelInterpolation(
                op,
                testspace,
                trialspace,
                tree;
                strategy=MultilevelKernelInterpolation.InterpolationStrategy(
                    op,
                    testspace,
                    trialspace;
                    polynomialstrategy=MultilevelKernelInterpolation.PolynomialStrategy(
                        op; polynomial=polynomial
                    ),
                    translatorstrategy=MultilevelKernelInterpolation.TranslatorStrategy(;
                        i2ostrategy=MultilevelKernelInterpolation.DenseI2OStrategy()
                    ),
                    nearinteractionsstrategy=dummynear(op, testspace, trialspace),
                ),
            )
            x = randn(Random.Xoshiro(4711), ComplexF64, size(A, 2))
            @test relativeerror(A, H2Factory.farinteractions(Adense, A), x) < 1e-3
        end
    end

    @testset "what 2-D does NOT support is refused, not silently wrong" begin
        op = Helmholtz2D.singlelayer(; wavenumber=κ)

        # The recommended default polynomial is 3-D, so this is the mistake a 2-D user makes by
        # doing nothing. Without the guard it surfaces as a `BoundsError` on a shift vector deep
        # inside polynomial evaluation, naming neither the tree nor the polynomial.
        defaulted = try
            MultilevelKernelInterpolation.GalerkinKernelInterpolation(
                op,
                X,
                TREE;
                strategy=MultilevelKernelInterpolation.InterpolationStrategy(
                    op,
                    X;
                    translatorstrategy=MultilevelKernelInterpolation.TranslatorStrategy(;
                        i2ostrategy=MultilevelKernelInterpolation.DenseI2OStrategy()
                    ),
                    nearinteractionsstrategy=dummynear(op, X, X),
                ),
            )
            nothing
        catch exception
            exception
        end
        @test defaulted isa ErrorException
        @test occursin("Dimension mismatch", defaulted.msg)
        @test occursin("tree is 2-D", defaulted.msg)

        # Hypersingular is deliberately absent from the moment table: its 3-D partner splits a
        # 6-component buffer into a 3-vector normal and a 3-vector curl, and neither half carries
        # over. Absent means a `MethodError`, which is a refusal, not a wrong answer.
        @test_throws MethodError MultilevelKernelInterpolation.moment(
            Helmholtz2D.hypersingular(; wavenumber=κ), nothing
        )
    end
end

end
