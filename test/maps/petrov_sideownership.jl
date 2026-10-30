module TestPetrovSideOwnership

using Test
using CompScienceMeshes, BEAST
using H2Trees, H2Factory
using MultilevelKernelInterpolation

# Part E Phases 2+3 invariant, asserted where it can actually fail:
#
#     the trial tree owns the trial level transfers; the test tree owns the test level transfers
#
# This lives in MultilevelKernelInterpolation, not in H2Factory's harness, because it CANNOT be tested with the fake maps:
# `fakepetrovmap` uses one `identitytranslator()` for both sides, so test and trial transfers are the
# same object there and every side-selection bug passes. Only a real Petrov map with genuinely
# distinct per-side polynomials (different spaces -> different per-side protrusion) can distinguish
# them.
#
# It asserts the TRAVERSAL-FACING functor, not just the accessor. That distinction is the whole
# point: the accessors were already correct while `I2ITranslatorFunctor` stored the *map* and
# re-resolved `i2itranslator(H2map)` at call time -- and since the stored map is the unwrapped
# forward one, that re-resolution always returned the TRIAL side, silently discarding the flip for
# transpose/adjoint. The full suites, the factorization oracle and the end-to-end smoke test all
# passed with that bug present.
@testset "Petrov side ownership reaches the traversal" begin
    m = readmesh(
        joinpath(pkgdir(MultilevelKernelInterpolation), "test", "assets", "in", "sphere.in")
    )
    op = Helmholtz3D.doublelayer(; wavenumber=1.0)
    testspace = lagrangec0d1(m)
    trialspace = lagrangecxd0(m)

    A = MultilevelKernelInterpolation.PetrovGalerkinKernelInterpolation(
        op,
        testspace,
        trialspace;
        # DENSE explicitly. This test's premise is MAXIMALLY distinct per-side transfers, and dense
        # keeps independent per-side grids -- whereas the Petrov DEFAULT (Toeplitz) reconciles the two
        # sides to one shared grid per shared I2O level, weakening the `!==` precondition below. The
        # side-ownership flip being verified is I2O-strategy-independent (it is about the per-side
        # leveltranslators), so dense fully exercises it; the default (Toeplitz) path is covered by
        # `petrov_endtoend_smoke.jl` and `petrov_toeplitz.jl`.
        strategy=MultilevelKernelInterpolation.InterpolationStrategy(
            op,
            testspace,
            trialspace;
            nearinteractionsstrategy=H2Factory.DummyNearInteractionsStrategy(
                BEAST.defaultquadstrat(op, testspace, trialspace)
            ),
            translatorstrategy=MultilevelKernelInterpolation.TranslatorStrategy(;
                i2ostrategy=MultilevelKernelInterpolation.DenseI2OStrategy()
            ),
        ),
    )

    testi2i = H2Factory.testi2itranslator(A)
    triali2i = H2Factory.triali2itranslator(A)
    testo2o = H2Factory.testo2otranslator(A)
    trialo2o = H2Factory.trialo2otranslator(A)

    # Precondition: without this the rest of the testset is vacuous.
    @test testi2i !== triali2i
    @test testo2o !== trialo2o

    # Since the level-transfer refactor, disaggregation reconstructs the O2O action as
    # transpose/adjoint(I2I): the O2O functor carries the disaggregation SIDE's I2I translator
    # (`disaggregationi2itranslator`, mirror of aggregation's side) plus an explicit direction, NOT the
    # materialized O2O. The materialized-O2O accessors still resolve (a cache) but are off the traversal.
    @testset "$label" for (label, M, wanti2i, wanto2o, wantdisaggi2i, wantdir) in (
        ("forward", A, triali2i, testo2o, testi2i, H2Factory.TransposeTranslation),
        (
            "transpose",
            transpose(A),
            testi2i,
            trialo2o,
            triali2i,
            H2Factory.TransposeTranslation,
        ),
        ("adjoint", adjoint(A), testi2i, trialo2o, triali2i, H2Factory.AdjointTranslation),
    )
        # Accessor level.
        @test H2Factory.i2itranslator(M) === wanti2i
        @test H2Factory.o2otranslator(M) === wanto2o   # materialized O2O accessor (cache; off traversal)
        # Traversal level -- what the upward/downward passes actually apply.
        @test H2Factory.i2itranslatorfunctor(M).translator === wanti2i
        @test H2Factory.o2otranslatorfunctor(M).translator === wantdisaggi2i
        @test H2Factory.o2otranslatorfunctor(M).direction isa wantdir
    end
end

end # module TestPetrovSideOwnership
