module TestI2OCollectionSymmetryReduced

using Test
using CompScienceMeshes, BEAST
using H2Trees
using MultilevelKernelInterpolation
using H2Factory
using LinearAlgebra
using H2FactoryTestUtils

# THE SYMMETRY-REDUCED TOEPLITZ COUPLING, AGAINST THE UNREDUCED ONE IT REPLACES.
#
# Agreement alone would NOT be evidence: `symmetryreducedtoeplitzi2ocoupling` falls back to the
# unreduced build when the displacements are not lattice aligned, and that fallback agrees perfectly.
# So every agreement check here is paired with a count of the DISTINCT STORED TENSORS, which is the
# quantity the reduction is for -- counted BY OBJECT IDENTITY ON BOTH SIDES, see `_compare`.

function _withplanchunks(f, plan)
    schedule = H2Factory.planschedule(plan, Base.Threads.nthreads())
    return H2Factory.withnchunks(H2Factory.schedulechunkcounts(schedule)) do
        return f()
    end
end

_tensors(v::H2Factory.ToeplitzMatrixRepresentation) = [v.toeplitz]
_tensors(v::AbstractArray) = [x.toeplitz for x in vec(v)]

# The array a reduced entry actually reads from, so orbit-mates sharing one tensor are counted once.
_root(a::H2Factory.PermutedTranslation) = a.canonical
_root(a::H2Factory.ScaledPermutedTranslation) = a.canonical
_root(a) = a

# `ToeplitzI2OTranslationCollection` forwards `getindex` but not `keys` -- the traversal drives it
# from the plan -- so the key set comes from the wrapped collection.
_keys(collection) = keys(collection.i2ocollection)

# Which levels read each stored tensor, and which tensor sizes occur at each level.
#
# WHY THIS IS NOT COVERED BY THE LENGTH CHECK IN `PermutedTranslation`. When every level carries the
# same polynomial, every level's difference tensor has the SAME SIZE -- so a canonical shared between
# two levels has the right shape and the wrong box size, and no shape guard can see it. The level
# partition is the only thing that can. When the order varies by level the sizes differ and a
# conflation would throw, but then the live risk is the other one: a permutation built from a single
# size assumed for the whole collection rather than from each key's own tensor.
function _levelaudit(tree, plain, reduced)
    levelsof, sizesbylevel = Dict{UInt,Set{Int}}(), Dict{Int,Set{Tuple}}()
    for key in _keys(plain)
        level = H2Trees.level(tree, key[1])
        for (p, r) in zip(_tensors(plain[key]), _tensors(reduced[key]))
            push!(get!(sizesbylevel, level, Set{Tuple}()), size(p))
            push!(get!(levelsof, objectid(_root(r)), Set{Int}()), level)
        end
    end
    spanning = count(levels -> length(levels) > 1, values(levelsof))
    return spanning, sizesbylevel
end

# BOTH SIDES COUNTED THE SAME WAY. Counting the plain side as `interactions x components` and the
# reduced side by object identity compares a NON-deduplicated number against a deduplicated one:
# `DirectionInvariance` already shares one tensor per DIRECTION, so the plain collection never held
# one tensor per interaction. That mistake inflated the measured reduction from 18.7x to 102x. What
# the symmetry reduction actually does is collapse direction classes into ORBIT classes.
function _compare(plain, reduced)
    worst, scale = 0.0, 0.0
    plainroots, reducedroots = Set{UInt}(), Set{UInt}()
    for key in _keys(plain)
        plaintensors, reducedtensors = _tensors(plain[key]), _tensors(reduced[key])
        @test length(plaintensors) == length(reducedtensors)
        for (p, r) in zip(plaintensors, reducedtensors)
            scale = max(scale, maximum(abs, p))
            worst = max(worst, maximum(abs, vec(p) - vec(collect(r))))
            push!(plainroots, objectid(_root(p)))
            push!(reducedroots, objectid(_root(r)))
        end
    end
    return worst / scale, length(plainroots), length(reducedroots)
end

@testset verbose = true "Symmetry-reduced Toeplitz I2OCollection" begin
    λ = 1.0
    k = 2π / λ
    m = readmesh(
        joinpath(pkgdir(MultilevelKernelInterpolation), "test", "assets", "in", "sphere.in")
    )
    X = lagrangec0d1(m)
    tree = buildtree(X; builder=TwoNTreeBuilder(; minhalfsize=λ / 5, minvalues=0))
    plan = H2Trees.buildplans(
        tree;
        builder=H2Trees.PlanBuilder(; aggregatenode=H2Trees.AggregateAllNodesFunctor()),
    ).testdisaggregationplan
    treepolys = MultilevelKernelInterpolation.levelpolynomials(
        MultilevelKernelInterpolation.BsplinePolynomialEquidistant3D{Float64}(2, 5),
        H2Trees.levels(tree),
        (level) -> false,
    )

    # ONE CASE PER COMPONENT CLASS. The four differ only in `symmetrycomponentaction`, and each is
    # O(1) wrong under any of the others on a reflection -- half of the 48-element group.
    @testset "$name" for (name, operator) in (
        ("scalar", Maxwell3D.singlelayer(; wavenumber=k)),
        ("TrueTensor (efie)", MultilevelKernelInterpolation.efie(; wavenumber=k)),
        (
            "PseudoTensor (MWDoubleLayer3D)",
            MultilevelKernelInterpolation.MWDoubleLayer3D(; wavenumber=k),
        ),
        (
            "VectorField (HH3D DL gradient)",
            MultilevelKernelInterpolation.HH3DDoubleLayerFDBIOGradientI2O(; wavenumber=k),
        ),
    )
        plain = _withplanchunks(plan) do
            return MultilevelKernelInterpolation.toeplitzi2ocoupling(
                operator, tree, treepolys, plan; verbose=false
            )
        end
        reduced = _withplanchunks(plan) do
            return MultilevelKernelInterpolation.symmetryreducedtoeplitzi2ocoupling(
                operator, tree, treepolys, plan; verbose=false
            )
        end

        relative, nplain, nreduced = _compare(plain, reduced)
        @test relative < 1e-13
        # THE REDUCTION ACTUALLY HAPPENED. Without this the lattice-misalignment fallback -- which
        # returns the unreduced collection -- passes every agreement check above. Measured 18.7x, so
        # 5x is headroom against a tree whose orbits happen to be less favourable.
        @test nreduced < nplain / 5
    end

    @testset "the component declaration is per trait, not per shape" begin
        # `∇∇`, `Dyadic` and `∇Cross` are all stored 3x3; only the declaration separates them, and
        # `∇Cross` (a curl) is the one that is a pseudo-tensor.
        @test MultilevelKernelInterpolation.symmetrycomponentaction(
            MultilevelKernelInterpolation.ScalarGreen3DTranslation()
        ) === nothing
        @test MultilevelKernelInterpolation.symmetrycomponentaction(
            MultilevelKernelInterpolation.∇Green3DTranslation()
        ) == H2Factory.VectorField()
        @test MultilevelKernelInterpolation.symmetrycomponentaction(
            MultilevelKernelInterpolation.∇∇Green3DTranslation()
        ) == H2Factory.TrueTensor()
        @test MultilevelKernelInterpolation.symmetrycomponentaction(
            MultilevelKernelInterpolation.DyadicGreen3DTranslation()
        ) == H2Factory.TrueTensor()
        @test MultilevelKernelInterpolation.symmetrycomponentaction(
            MultilevelKernelInterpolation.∇CrossGreen3DTranslation()
        ) == H2Factory.PseudoTensor()
    end

    @testset "the B-spline modulation route reduces too" begin
        # The route a real map takes. It composes the SAME coupling with the interpolation matrices,
        # so the only thing that could go wrong is the coupling not being threaded through -- which
        # the stored-tensor count below is what detects.
        operator = Maxwell3D.singlelayer(; wavenumber=k)
        plainbspline = _withplanchunks(plan) do
            return MultilevelKernelInterpolation._bsplinetoeplitzcoupling(
                operator, tree, treepolys, plan; verbose=false
            )
        end
        reducedbspline = _withplanchunks(plan) do
            return MultilevelKernelInterpolation._bsplinetoeplitzcoupling(
                operator,
                tree,
                treepolys,
                plan;
                verbose=false,
                couplingbuilder=MultilevelKernelInterpolation.symmetryreducedtoeplitzi2ocoupling,
            )
        end
        relative, nplain, nreduced = _compare(
            H2Factory.coupling(plainbspline), H2Factory.coupling(reducedbspline)
        )
        @test relative < 1e-13
        @test nreduced < nplain / 5
    end

    # A WHOLE MAP, NOT JUST THE COUPLING.
    #
    # Every check above compares stored couplings; none of them ever RUNS one. That gap was not
    # hypothetical: `i2ochannels` sized its FFT scratch by calling the stored container type as
    # `T(undef, dims)`, which is fine while every collection stores an `Array` and a `MethodError`
    # the moment one stores a lazy view. Nothing that only inspects couplings can catch that, so
    # this builds two real maps differing ONLY in the flag and requires the same answer in all
    # three modes.
    @testset "a full map with the reduced coupling agrees with the unreduced one" begin
        mapmesh = readmesh(
            joinpath(
                pkgdir(MultilevelKernelInterpolation), "test", "assets", "in", "sphere.in"
            ),
        )
        Y = lagrangecxd0(mapmesh)
        mapop = Helmholtz3D.singlelayer(; wavenumber=k)
        maptree = buildtree(Y; builder=TwoNTreeBuilder(; minvalues=120))

        build(reduced) = MultilevelKernelInterpolation.GalerkinKernelInterpolation(
            mapop,
            Y;
            tree=maptree,
            verbose=false,
            strategy=MultilevelKernelInterpolation.InterpolationStrategy(
                mapop,
                Y;
                polynomialstrategy=MultilevelKernelInterpolation.PolynomialStrategy(
                    mapop;
                    polynomial=MultilevelKernelInterpolation.BsplinePolynomialEquidistant3D{
                        Float64
                    }(
                        2, 5
                    ),
                ),
                nearinteractionsstrategy=H2Factory.DummyNearInteractionsStrategy(
                    BEAST.defaultquadstrat(mapop, Y, Y)
                ),
                translatorstrategy=MultilevelKernelInterpolation.TranslatorStrategy(;
                    i2ostrategy=MultilevelKernelInterpolation.ToeplitzI2OStrategy(;
                        symmetryreduced=reduced
                    ),
                ),
            ),
        )

        Aplain, Areduced = build(false), build(true)
        rng = H2FactoryTestUtils.checkrng()
        x = randn(rng, ComplexF64, size(Aplain, 2))
        z = randn(rng, ComplexF64, size(Aplain, 1))

        # All three modes: the reduction rewrites what the coupling READS, and transpose/adjoint
        # reach it through different code paths than the forward matvec does.
        @test Areduced * x ≈ Aplain * x rtol = 1e-10
        @test transpose(Areduced) * z ≈ transpose(Aplain) * z rtol = 1e-10
        @test Areduced' * z ≈ Aplain' * z rtol = 1e-10

        # And it really is the reduced build -- `ToeplitzI2OStrategy` silently falls back to the
        # unreduced coupling when the displacements are not lattice aligned, which would pass
        # every check above.
        inner = H2Factory.coupling(H2Factory.i2otranslator(Areduced)).i2ocollection
        @test inner isa H2Factory.SymmetryReducedI2OTranslationCollection
    end

    # THE POLYNOMIAL ORDER NEED NOT BE THE SAME ON EVERY LEVEL. `cond` decides whether a level's
    # order is increased, so the two settings produce genuinely different situations and each hides
    # a different mistake -- see `_levelaudit`.
    @testset "orbits are per level ($label)" for (label, cond) in (
        ("uniform order", (level) -> false), ("order increases per level", (level) -> true)
    )
        operator = Maxwell3D.singlelayer(; wavenumber=k)
        polys = MultilevelKernelInterpolation.levelpolynomials(
            MultilevelKernelInterpolation.BsplinePolynomialEquidistant3D{Float64}(2, 5),
            H2Trees.levels(tree),
            cond,
        )
        plain = _withplanchunks(plan) do
            return MultilevelKernelInterpolation.toeplitzi2ocoupling(
                operator, tree, polys, plan; verbose=false
            )
        end
        reduced = _withplanchunks(plan) do
            return MultilevelKernelInterpolation.symmetryreducedtoeplitzi2ocoupling(
                operator, tree, polys, plan; verbose=false
            )
        end

        spanning, sizesbylevel = _levelaudit(tree, plain, reduced)
        @test spanning == 0
        @test length(sizesbylevel) > 1               # more than one level, or this says nothing
        for sizes in values(sizesbylevel)
            @test length(sizes) == 1                 # one tensor size per level
        end
        # NON-VACUITY of the varying case: it must actually vary, or it is the uniform case again.
        allsizes = union(values(sizesbylevel)...)
        cond(0) ? (@test length(allsizes) > 1) : (@test length(allsizes) == 1)

        relative, _, _ = _compare(plain, reduced)
        @test relative < 1e-13
    end
end

end # module TestI2OCollectionSymmetryReduced
