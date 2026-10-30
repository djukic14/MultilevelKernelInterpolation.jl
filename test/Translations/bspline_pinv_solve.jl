module TestBsplinePinvSolve

using Test
using StaticArrays
using LinearAlgebra
using BEAST
using MultilevelKernelInterpolation
using H2FactoryTestUtils

# `BsplineKernelTranslator` (src/translations/I2Otranslations/I2Obsplinetranslationcollection.jl)
# reduces raw Green's-function samples at B-spline collocation points into interpolation
# coefficients via `pinv(interpolationmatrix(p))`, applied along each tensor dimension by
# `_six1dinterpolations!`. This pins down that `pinv`-based reduction is mathematically
# equivalent to a direct dense linear solve, before deleting the unused `_hackyldiv!`
# dead-code path (an `lu`-based alternative with zero live callers, see A.3.1 in the plan).
@testset verbose=true "pinv matches direct dense solve" begin
    for poly in [
        MultilevelKernelInterpolation.BsplinePolynomialEquidistant3D{Float64}(2, 5),
        MultilevelKernelInterpolation.BsplinePolynomialEquidistant3D{Float64}(3, 7),
        MultilevelKernelInterpolation.BufferedBsplinePolynomialEquidistant3D{Float64}(
            2, 5, 1
        ),
    ]
        M = collect(MultilevelKernelInterpolation.interpolationmatrix(poly))
        b = size(LinearIndices(poly), 1)

        @testset "$(typeof(poly).name.name), degree $(MultilevelKernelInterpolation.polynomialorders(poly)[1])" begin
            @test size(M) == (b, b)

            # ONE rng for the loop, ADVANCED per iteration. `checkrng()` returns a fresh RNG on the
            # same fixed seed, so calling it inside the loop would hand all five iterations the
            # identical `rhs`, quietly reducing this to one assertion repeated five times. Nothing
            # would fail; only the diversity would be gone.
            rng = H2FactoryTestUtils.checkrng()
            for _ in 1:5
                rhs = randn(rng, ComplexF64, b, 3)
                @test pinv(M) * rhs ≈ M \ rhs atol = 1e-8
            end
        end
    end

    @testset "translator's stored luT matches pinv(interpolationmatrix)" begin
        poly = MultilevelKernelInterpolation.BufferedBsplinePolynomialEquidistant3D{Float64}(
            2, 5, 1
        )
        op = Helmholtz3D.singlelayer(; wavenumber=1.0)

        translator = MultilevelKernelInterpolation.BsplineKernelTranslator(
            op, [poly], nothing, 1:1, nothing
        )

        b = size(LinearIndices(poly), 1)
        expected = pinv(collect(MultilevelKernelInterpolation.interpolationmatrix(poly)))
        # Per-side and keyed by LEVEL since Part E Phase 5b (was one dict keyed by size; size is a
        # dimension, not a grid identity). A single family populates both sides identically, which
        # this pins as well.
        @test translator.testluTs[1] ≈ expected atol = 1e-10
        @test translator.trialluTs[1] ≈ expected atol = 1e-10
    end
end

end
