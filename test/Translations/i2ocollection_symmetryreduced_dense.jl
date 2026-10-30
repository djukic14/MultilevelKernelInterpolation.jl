module TestI2OCollectionSymmetryReducedDense

using Test
using CompScienceMeshes, BEAST
using H2Trees
using MultilevelKernelInterpolation
using H2Factory
using LinearAlgebra
using H2FactoryTestUtils

# THE SYMMETRY-REDUCED **DENSE** COUPLING, AGAINST THE UNREDUCED ONE IT REPLACES.
#
# The sibling file covers the Toeplitz route. This one is a different mechanism, not a different
# parameter: Toeplitz stores a lazy permuted VIEW and lets the elementwise kernel read through it,
# which a BLAS `mul!` cannot do. Here the matrix is left alone and the VECTORS are permuted around it,
# gather, BLAS, scatter, so the two share only `H2Trees`'s tags.
#
# As there, agreement alone would NOT be evidence: the build falls back to the unreduced coupling
# when the displacements are not lattice aligned, and that fallback agrees perfectly. So every
# agreement check is paired with a count of the DISTINCT STORED MATRICES, counted by object identity
# on both sides.

function _withplanchunks(f, plan)
    schedule = H2Factory.planschedule(plan, Base.Threads.nthreads())
    return H2Factory.withnchunks(H2Factory.schedulechunkcounts(schedule)) do
        return f()
    end
end

# The matrix a reduced entry actually reads from, so orbit-mates sharing one matrix are counted once.
_root(t::H2Factory.PermutedDenseTranslation) = t.canonical
_root(t) = t

# The matrix a reduced entry STANDS FOR, materialized by applying the two permutations. Never used
# in production, the whole point is that this matrix is not formed, so it is the independent
# statement of the claim rather than a call back into the code under test.
function _materialize(t::H2Factory.PermutedDenseTranslation)
    full = similar(t.canonical)
    for j in axes(t.canonical, 2), i in axes(t.canonical, 1)
        full[t.rowpermutation[i], t.columnpermutation[j]] = t.canonical[i, j]
    end
    return full
end
_materialize(m::AbstractMatrix) = m

function _compare(plain, reduced)
    worst, scale = 0.0, 0.0
    plainroots, reducedroots = Set{UInt}(), Set{UInt}()
    for key in keys(plain)
        p, r = plain[key], reduced[key]
        # ENTRYWISE, not just a count. Counting alone says the reduction happened; it says nothing
        # about the permutations being the right ones, and a wrong permutation reduces exactly as
        # well as a right one.
        scale = max(scale, maximum(abs, p))
        worst = max(worst, maximum(abs, p - _materialize(r)))
        push!(plainroots, objectid(_root(p)))
        push!(reducedroots, objectid(_root(r)))
    end
    return worst / scale, length(plainroots), length(reducedroots)
end

λ = 1.0
k = 2π / λ
mesh = readmesh(
    joinpath(pkgdir(MultilevelKernelInterpolation), "test", "assets", "in", "sphere.in")
)
Y = lagrangecxd0(mesh)
scalaroperator = Helmholtz3D.singlelayer(; wavenumber=k)
tree = buildtree(Y; builder=TwoNTreeBuilder(; minvalues=120))
plan = H2Trees.buildplans(tree).testdisaggregationplan

function _polynomials(polynomial)
    return MultilevelKernelInterpolation.levelpolynomials(
        polynomial, H2Trees.levels(tree), (level) -> false
    )
end

@testset verbose = true "Symmetry-reduced dense I2OCollection" begin
    @testset "the reduced coupling stores far fewer matrices ($label)" for (
        label, polynomial
    ) in (
        (
            "equidistant",
            MultilevelKernelInterpolation.BarycentricLagrangePolynomialEquidistant3D(6),
        ),
        # Chebyshev nodes are NOT equally spaced, only symmetric about the box centre, which is
        # the actual hypothesis. Running both says the reduction depends on the symmetry and not on
        # the spacing, which is the thing a single equidistant fixture could not distinguish.
        (
            "chebyshev",
            MultilevelKernelInterpolation.BarycentricLagrangePolynomialChebyshev3D(6),
        ),
    )
        polys = _polynomials(polynomial)
        plain = _withplanchunks(plan) do
            return MultilevelKernelInterpolation.densei2ocoupling(
                scalaroperator, tree, polys, plan; verbose=false
            )
        end
        reduced = _withplanchunks(plan) do
            return MultilevelKernelInterpolation.symmetryreduceddensei2ocoupling(
                scalaroperator, tree, polys, plan; verbose=false
            )
        end

        @test reduced isa H2Factory.SymmetryReducedI2OTranslationCollection
        @test H2Factory.i2otrait(reduced) == H2Factory.SymmetryReducedDenseTranslation()
        @test keys(reduced) == keys(plain)

        relative, nplain, nreduced = _compare(plain, reduced)
        # THE NUMBERS, not only the count. This is what says the permutations are the RIGHT ones,
        # and for the Chebyshev family it is the only place that says it, since the end-to-end map
        # test below runs the equidistant family. Chebyshev is the case that matters here: its nodes
        # are NOT equally spaced, only symmetric about the box centre, which is the actual
        # hypothesis the reduction rests on.
        @test relative < 1e-13
        # Measured 316 -> 16, i.e. 19.8x. 5x is headroom against a tree whose orbits are less
        # favourable; without this the lattice-misalignment fallback would pass every other check.
        @test nreduced < nplain / 5
    end

    # A WHOLE MAP, NOT JUST THE COUPLING.
    #
    # Comparing stored couplings never RUNS one, and that gap is not hypothetical: on the Toeplitz
    # route `i2ochannels` sized its scratch by calling the stored container type, which is fine while
    # every collection stores an `Array` and a `MethodError` the moment one stores something else.
    # The dense route allocates its own gather/scatter scratch the same way, so it gets the same
    # end-to-end check, written first rather than after.
    @testset "a full map with the reduced coupling agrees with the unreduced one" begin
        build(reduced) = MultilevelKernelInterpolation.GalerkinMap(
            scalaroperator,
            Y;
            tree=tree,
            verbose=false,
            strategy=MultilevelKernelInterpolation.InterpolationStrategy(
                scalaroperator,
                Y;
                polynomialstrategy=MultilevelKernelInterpolation.PolynomialStrategy(
                    scalaroperator;
                    polynomial=MultilevelKernelInterpolation.BarycentricLagrangePolynomialEquidistant3D(
                        6
                    ),
                ),
                nearinteractionsstrategy=H2Factory.DummyNearInteractionsStrategy(
                    BEAST.defaultquadstrat(scalaroperator, Y, Y)
                ),
                translatorstrategy=MultilevelKernelInterpolation.TranslatorStrategy(;
                    i2ostrategy=MultilevelKernelInterpolation.DenseI2OStrategy(;
                        symmetryreduced=reduced
                    ),
                ),
            ),
        )

        Aplain, Areduced = build(false), build(true)
        rng = H2FactoryTestUtils.checkrng()
        x = randn(rng, ComplexF64, size(Aplain, 2))
        z = randn(rng, ComplexF64, size(Aplain, 1))

        # All three modes: forward gathers over COLUMNS and scatters over ROWS, transpose and adjoint
        # do the opposite, and each reaches the coupling through a different kernel.
        @test Areduced * x ≈ Aplain * x rtol = 1e-10
        @test transpose(Areduced) * z ≈ transpose(Aplain) * z rtol = 1e-10
        @test Areduced' * z ≈ Aplain' * z rtol = 1e-10

        # And it really is the reduced build, the flag falls back silently on a non-lattice tree,
        # which would pass every check above.
        inner = H2Factory.coupling(H2Factory.i2otranslator(Areduced))
        @test inner isa H2Factory.SymmetryReducedI2OTranslationCollection
        @test H2Factory.i2otrait(inner) == H2Factory.SymmetryReducedDenseTranslation()

        # The saving is the point, so it is asserted on the OBJECT the map actually holds rather
        # than only on a coupling built beside it. Measured 577 MiB -> 39 MiB for the whole map.
        @test Base.summarysize(Areduced) < Base.summarysize(Aplain) / 5
    end

    # THE 3x3 KERNELS, END TO END.
    #
    # ONE CASE PER PARITY RULE, and that is the whole point of running two: `efie` is a TRUE tensor
    # and `MWDoubleLayer3D` (built from a curl) a PSEUDO one, they are stored in the same 3x3 shape,
    # and each is EXACTLY a sign flip away under the other's rule. Half of the 48-element group has
    # `det == -1`, so choosing by shape is right on the rotation half and O(1) wrong on the
    # reflection half, invisible to any test using only the identity or a rotation. Measured
    # directly on the stored couplings: each is 2.0 wrong under the other's rule.
    @testset "3x3 kernels: $name" for (name, operator) in (
        ("efie (TrueTensor)", MultilevelKernelInterpolation.efie(; wavenumber=k)),
        (
            "MWDoubleLayer3D (PseudoTensor)",
            MultilevelKernelInterpolation.MWDoubleLayer3D(; wavenumber=k),
        ),
    )
        W = raviartthomas(mesh)
        tensortree = buildtree(W; builder=TwoNTreeBuilder(; minvalues=120))

        build(reduced) = MultilevelKernelInterpolation.GalerkinMap(
            operator,
            W;
            tree=tensortree,
            verbose=false,
            strategy=MultilevelKernelInterpolation.InterpolationStrategy(
                operator,
                W;
                polynomialstrategy=MultilevelKernelInterpolation.PolynomialStrategy(
                    operator;
                    polynomial=MultilevelKernelInterpolation.BarycentricLagrangePolynomialEquidistant3D(
                        3
                    ),
                ),
                nearinteractionsstrategy=H2Factory.DummyNearInteractionsStrategy(
                    BEAST.defaultquadstrat(operator, W, W)
                ),
                translatorstrategy=MultilevelKernelInterpolation.TranslatorStrategy(;
                    i2ostrategy=MultilevelKernelInterpolation.DenseI2OStrategy(;
                        symmetryreduced=reduced
                    ),
                ),
            ),
        )

        Aplain, Areduced = build(false), build(true)
        rng = H2FactoryTestUtils.checkrng()
        x = randn(rng, ComplexF64, size(Aplain, 2))
        z = randn(rng, ComplexF64, size(Aplain, 1))

        # All three modes. Transpose and adjoint transpose the BLOCK INDEX as well as the matrices
        # (`(Mᵀ)[a,b] == (M[b,a])ᵀ`), which a symmetric coupling could not distinguish.
        @test Areduced * x ≈ Aplain * x rtol = 1e-10
        @test transpose(Areduced) * z ≈ transpose(Aplain) * z rtol = 1e-10
        @test Areduced' * z ≈ Aplain' * z rtol = 1e-10

        inner = H2Factory.coupling(H2Factory.i2otranslator(Areduced))
        @test inner isa H2Factory.SymmetryReducedI2OTranslationCollection
        @test first(values(inner.dict)) isa H2Factory.PermutedDenseBlockTranslation

        # Measured 2844 -> 144 stored blocks; whole map 186 -> 18 MiB.
        @test Base.summarysize(Areduced) < Base.summarysize(Aplain) / 5
    end

    @testset "VectorField operators are refused, because they cannot be Galerkin anyway" begin
        # `HH3DDoubleLayerFDBIOGradientI2O` has `trialmoment != testmoment`, so `GalerkinCore`
        # asserts before this route is ever reached, the refusal exists so the reduced build says
        # so plainly rather than failing further in with an assertion about moments.
        polys = _polynomials(
            MultilevelKernelInterpolation.BarycentricLagrangePolynomialEquidistant3D(6)
        )
        gradient = MultilevelKernelInterpolation.HH3DDoubleLayerFDBIOGradientI2O(;
            wavenumber=k
        )
        @test MultilevelKernelInterpolation.symmetrycomponentaction(
            MultilevelKernelInterpolation.i2otranslation(gradient)
        ) == H2Factory.VectorField()
        @test_throws "cannot form a Galerkin map" _withplanchunks(plan) do
            return MultilevelKernelInterpolation.densei2ocoupling(
                gradient, tree, polys, plan, Y, Y; verbose=false, symmetryreduced=true
            )
        end
    end

    # THE GRID HYPOTHESIS, TESTED ON POINTS.
    #
    # `latticegridpermutation` is a statement about INDICES and is silently wrong for a grid whose
    # POINTS are not closed under the group about the box centre, a plausible permutation of the
    # wrong grid, hence a wrong field rather than an error. `latticeclosuredefect` is the build-time
    # guard for that, and it needs both directions or it could be `return nothing`.
    @testset "latticeclosuredefect accepts a symmetric grid and rejects a skewed one" begin
        group = H2Trees.symmetrygroup(H2Trees.FullLatticeSymmetry(), Val(3))
        b = 4
        gridsize = (b, b, b)
        nodes = range(-1.0, 1.0; length=b)
        symmetric = [(nodes[i], nodes[j], nodes[l]) for i in 1:b, j in 1:b, l in 1:b]
        @test MultilevelKernelInterpolation.latticeclosuredefect(
            vec(symmetric), gridsize, group
        ) === nothing

        # Chebyshev nodes: unequally spaced, still symmetric. Accepted, which is what lets the
        # reduction serve a Chebyshev family.
        chebyshev = [-cospi((i - 1) / (b - 1)) for i in 1:b]
        cheb = [(chebyshev[i], chebyshev[j], chebyshev[l]) for i in 1:b, j in 1:b, l in 1:b]
        @test MultilevelKernelInterpolation.latticeclosuredefect(
            vec(cheb), gridsize, group
        ) === nothing

        # One axis slid off centre. Every point is still a legitimate grid point and the linear
        # order is unchanged, so nothing about the SHAPE can detect this, only the points can.
        skewed = [(p[1] + 0.3, p[2], p[3]) for p in symmetric]
        @test MultilevelKernelInterpolation.latticeclosuredefect(
            vec(skewed), gridsize, group
        ) !== nothing
    end

    # WHAT THE ROUTE REFUSES, AND THAT IT REFUSES IT ON EVERY ENTRY.
    #
    # "The option is present on some routes and silently absent on others" is the failure this
    # routing has produced three times. A refusal that only fires on one of the entry points is the
    # same bug wearing a different hat, so each family is asked for through the SPACE-TAKING entry a
    # real map uses, not only through the four-argument builder.
    @testset "unsupported families are refused rather than silently unreduced" begin
        polys = _polynomials(
            MultilevelKernelInterpolation.BarycentricLagrangePolynomialEquidistant3D(6)
        )

        # NOTE the dyadic kernels are NOT here: `efie` and `MWDoubleLayer3D` are SUPPORTED, and are
        # covered end to end above. Only the two FAMILY restrictions remain, plus `VectorField`,
        # which has its own testset because its reason is different in kind (unreachable, not
        # unwritten).

        # B-splines: the dense B-spline route carries its own raw<->reduced modulation.
        bsplines = _polynomials(
            MultilevelKernelInterpolation.BsplinePolynomialEquidistant3D{Float64}(2, 5)
        )
        @test_throws "raw<->reduced modulation" _withplanchunks(plan) do
            return MultilevelKernelInterpolation.densei2ocoupling(
                scalaroperator,
                tree,
                bsplines,
                plan,
                Y,
                Y;
                verbose=false,
                symmetryreduced=true,
            )
        end

        # Petrov: a node's grid resolves through its own side's tree, which is unexercised here.
        sided = MultilevelKernelInterpolation.SidedPolynomialDicts(
            MultilevelKernelInterpolation.leveltopolynomialdict(
                polys, H2Trees.levels(plan)
            ),
            MultilevelKernelInterpolation.leveltopolynomialdict(
                polys, H2Trees.levels(plan)
            ),
        )
        @test_throws "GALERKIN maps only" _withplanchunks(plan) do
            return MultilevelKernelInterpolation.densei2ocoupling(
                scalaroperator, tree, sided, plan, Y, Y; verbose=false, symmetryreduced=true
            )
        end

        # NON-VACUITY: with the flag off, both families build fine. Otherwise these could be passing
        # because the fixture is broken rather than because the refusal fired.
        for (family, operator) in ((bsplines, scalaroperator), (sided, scalaroperator))
            coupling = _withplanchunks(plan) do
                return MultilevelKernelInterpolation.densei2ocoupling(
                    operator, tree, family, plan, Y, Y; verbose=false
                )
            end
            @test coupling !== nothing
        end
    end
end

end # module TestI2OCollectionSymmetryReducedDense
