module TestInterpolation

using Test
using StaticArrays
using BEAST
using H2Trees
using MultilevelKernelInterpolation

# `evaluateallpolynomials` sizes its output columns by `numpolynomials(poly)` (the raw,
# pre-stability-reduction basis count), while for `BufferedBsplinePolynomialEquidistant`
# with active near-boundary knot removal (see `MultilevelKernelInterpolation.knotstoremove`), `calculateI2Otranslation!`
# expresses `green` in the reduced, numerically-stable basis (`size(LinearIndices(poly))`).
# `_promotetorawbasis` bridges the two via the paper's stabilization transfer matrix `S`
# (`i2icheapaveraging`, Eq 42: S maps raw -> reduced, so `S' * green * S` promotes `green`
# back to the raw basis `evaluateallpolynomials` expects).
function _hasactiveknotremoval(poly)
    return MultilevelKernelInterpolation.numpolynomials(poly) != size(LinearIndices(poly))
end

function _promotetorawbasis(green::AbstractMatrix{<:Number}, poly)
    S = MultilevelKernelInterpolation.i2icheapaveraging(
        MultilevelKernelInterpolation.bspline(poly, 1).bspline
    )
    S3 = kron(kron(S, S), S)
    return S3' * green * S3
end

function _promotetorawbasis(green::AbstractArray{<:AbstractMatrix}, poly)
    return map(g -> _promotetorawbasis(g, poly), green)
end

@testset verbose=true "Lagrange" begin
    @testset "Scalar Green's" begin
        λ = 1.0
        k = 2π / λ
        op = Maxwell3D.singlelayer(; wavenumber=k)

        translator = MultilevelKernelInterpolation.KernelTranslator(
            op, nothing, nothing, nothing
        )
        comparisonpoly = MultilevelKernelInterpolation.BarycentricLagrangePolynomialChebyshev3D2(
            9
        )

        h = λ / 4

        testdata = H2Trees.BoxData(0, Int[], SVector(0.0, 0.0, 0.0), h, 1)
        trialdata = H2Trees.BoxData(0, Int[], SVector(0.0, 4 * h, 0.0), h, 1)

        testcomparisonpoly = MultilevelKernelInterpolation.shiftedpolynomial(
            testdata, comparisonpoly
        )
        trialcomparisonpoly = MultilevelKernelInterpolation.shiftedpolynomial(
            trialdata, comparisonpoly
        )

        comparisongreen = MultilevelKernelInterpolation.calculateI2Otranslation!(
            translator,
            MultilevelKernelInterpolation.i2otranslation(op),
            testcomparisonpoly,
            trialcomparisonpoly,
        )

        for poly in [
            MultilevelKernelInterpolation.BarycentricLagrangePolynomialEquidistant3D(7),
            MultilevelKernelInterpolation.BarycentricLagrangePolynomialChebyshev3D(7),
            MultilevelKernelInterpolation.BarycentricLagrangePolynomialChebyshev3D2(7),
        ]
            testpolynomial = MultilevelKernelInterpolation.shiftedpolynomial(testdata, poly)
            trialpolynomial = MultilevelKernelInterpolation.shiftedpolynomial(
                trialdata, poly
            )

            green = MultilevelKernelInterpolation.calculateI2Otranslation!(
                translator,
                MultilevelKernelInterpolation.i2otranslation(op),
                testpolynomial,
                trialpolynomial,
            )

            interpolationmatrix = MultilevelKernelInterpolation.evaluateallpolynomials(
                poly, comparisonpoly.ranges[1]
            )

            interpolatedgreen = interpolationmatrix * green * interpolationmatrix'
            er =
                maximum(abs, comparisongreen - interpolatedgreen) /
                maximum(abs, comparisongreen)
            @test er < 1e-4
        end
    end

    @testset "Dyadic Green's" begin
        λ = 1.0
        k = 2π / λ
        op = MultilevelKernelInterpolation.efie(; wavenumber=k)

        translator = MultilevelKernelInterpolation.KernelTranslator(
            op, nothing, nothing, nothing
        )
        comparisonpoly = MultilevelKernelInterpolation.BarycentricLagrangePolynomialChebyshev3D2(
            9
        )

        h = λ / 4

        testdata = H2Trees.BoxData(0, Int[], SVector(0.0, 0.0, 0.0), h, 1)
        trialdata = H2Trees.BoxData(0, Int[], SVector(0.0, 4 * h, 0.0), h, 1)

        testcomparisonpoly = MultilevelKernelInterpolation.shiftedpolynomial(
            testdata, comparisonpoly
        )
        trialcomparisonpoly = MultilevelKernelInterpolation.shiftedpolynomial(
            trialdata, comparisonpoly
        )

        comparisongreen = MultilevelKernelInterpolation.calculateI2Otranslation!(
            translator,
            MultilevelKernelInterpolation.DyadicGreen3DTranslation(),
            testcomparisonpoly,
            trialcomparisonpoly,
        )

        for poly in [
            MultilevelKernelInterpolation.BarycentricLagrangePolynomialEquidistant3D(7),
            MultilevelKernelInterpolation.BarycentricLagrangePolynomialChebyshev3D(7),
            MultilevelKernelInterpolation.BarycentricLagrangePolynomialChebyshev3D2(7),
        ]
            testpolynomial = MultilevelKernelInterpolation.shiftedpolynomial(testdata, poly)
            trialpolynomial = MultilevelKernelInterpolation.shiftedpolynomial(
                trialdata, poly
            )

            green = MultilevelKernelInterpolation.calculateI2Otranslation!(
                translator,
                MultilevelKernelInterpolation.i2otranslation(op),
                testpolynomial,
                trialpolynomial,
            )

            interpolationmatrix = MultilevelKernelInterpolation.evaluateallpolynomials(
                poly, comparisonpoly.ranges[1]
            )

            for i in eachindex(green)
                interpolatedgreen = interpolationmatrix * green[i] * interpolationmatrix'
                er =
                    maximum(abs, comparisongreen[i] - interpolatedgreen) /
                    maximum(abs, comparisongreen[i])
                @test er < 1e-3
            end
        end
    end

    @testset "∇∇ Green's" begin
        λ = 1.0
        k = 2π / λ
        op = MultilevelKernelInterpolation.MWHyperSingular3D(; wavenumber=k)

        translator = MultilevelKernelInterpolation.KernelTranslator(
            op, nothing, nothing, nothing
        )
        comparisonpoly = MultilevelKernelInterpolation.BarycentricLagrangePolynomialChebyshev3D2(
            9
        )

        h = λ / 4

        testdata = H2Trees.BoxData(0, Int[], SVector(0.0, 0.0, 0.0), h, 1)
        trialdata = H2Trees.BoxData(0, Int[], SVector(0.0, 4 * h, 0.0), h, 1)

        testcomparisonpoly = MultilevelKernelInterpolation.shiftedpolynomial(
            testdata, comparisonpoly
        )
        trialcomparisonpoly = MultilevelKernelInterpolation.shiftedpolynomial(
            trialdata, comparisonpoly
        )

        comparisongreen = MultilevelKernelInterpolation.calculateI2Otranslation!(
            translator,
            MultilevelKernelInterpolation.∇∇Green3DTranslation(),
            testcomparisonpoly,
            trialcomparisonpoly,
        )

        for poly in [
            MultilevelKernelInterpolation.BarycentricLagrangePolynomialEquidistant3D(7),
            MultilevelKernelInterpolation.BarycentricLagrangePolynomialChebyshev3D(7),
            MultilevelKernelInterpolation.BarycentricLagrangePolynomialChebyshev3D2(7),
        ]
            testpolynomial = MultilevelKernelInterpolation.shiftedpolynomial(testdata, poly)
            trialpolynomial = MultilevelKernelInterpolation.shiftedpolynomial(
                trialdata, poly
            )

            green = MultilevelKernelInterpolation.calculateI2Otranslation!(
                translator,
                MultilevelKernelInterpolation.i2otranslation(op),
                testpolynomial,
                trialpolynomial,
            )

            interpolationmatrix = MultilevelKernelInterpolation.evaluateallpolynomials(
                poly, comparisonpoly.ranges[1]
            )

            for i in eachindex(green)
                interpolatedgreen = interpolationmatrix * green[i] * interpolationmatrix'
                er =
                    maximum(abs, comparisongreen[i] - interpolatedgreen) /
                    maximum(abs, comparisongreen[i])
                @test er < 1e-3
            end
        end
    end

    @testset "∇ Green's" begin
        λ = 1.0
        k = 2π / λ
        op = MultilevelKernelInterpolation.HH3DDoubleLayerFDBIOGradientI2O(; wavenumber=k)

        translator = MultilevelKernelInterpolation.KernelTranslator(
            op, nothing, nothing, nothing
        )
        comparisonpoly = MultilevelKernelInterpolation.BarycentricLagrangePolynomialChebyshev3D2(
            9
        )

        h = λ / 4

        testdata = H2Trees.BoxData(0, Int[], SVector(0.0, 0.0, 0.0), h, 1)
        trialdata = H2Trees.BoxData(0, Int[], SVector(0.0, 4 * h, 0.0), h, 1)

        testcomparisonpoly = MultilevelKernelInterpolation.shiftedpolynomial(
            testdata, comparisonpoly
        )
        trialcomparisonpoly = MultilevelKernelInterpolation.shiftedpolynomial(
            trialdata, comparisonpoly
        )

        comparisongreen = MultilevelKernelInterpolation.calculateI2Otranslation!(
            translator,
            MultilevelKernelInterpolation.∇Green3DTranslation(),
            testcomparisonpoly,
            trialcomparisonpoly,
        )

        for poly in [
            MultilevelKernelInterpolation.BarycentricLagrangePolynomialEquidistant3D(7),
            MultilevelKernelInterpolation.BarycentricLagrangePolynomialChebyshev3D(7),
            MultilevelKernelInterpolation.BarycentricLagrangePolynomialChebyshev3D2(7),
        ]
            testpolynomial = MultilevelKernelInterpolation.shiftedpolynomial(testdata, poly)
            trialpolynomial = MultilevelKernelInterpolation.shiftedpolynomial(
                trialdata, poly
            )

            green = MultilevelKernelInterpolation.calculateI2Otranslation!(
                translator,
                MultilevelKernelInterpolation.i2otranslation(op),
                testpolynomial,
                trialpolynomial,
            )

            interpolationmatrix = MultilevelKernelInterpolation.evaluateallpolynomials(
                poly, comparisonpoly.ranges[1]
            )

            for i in eachindex(green)
                interpolatedgreen = interpolationmatrix * green[i] * interpolationmatrix'
                er =
                    maximum(abs, comparisongreen[i] - interpolatedgreen) /
                    maximum(abs, comparisongreen[i])
                @test er < 1e-3
            end
        end
    end

    @testset "∇ × Green's" begin
        λ = 1.0
        k = 2π / λ
        op = MultilevelKernelInterpolation.MWDoubleLayer3D(; wavenumber=k)

        translator = MultilevelKernelInterpolation.KernelTranslator(
            op, nothing, nothing, nothing
        )
        comparisonpoly = MultilevelKernelInterpolation.BarycentricLagrangePolynomialChebyshev3D2(
            9
        )

        h = λ / 4

        testdata = H2Trees.BoxData(0, Int[], SVector(0.0, 0.0, 0.0), h, 1)
        trialdata = H2Trees.BoxData(0, Int[], SVector(0.0, 4 * h, 0.0), h, 1)

        testcomparisonpoly = MultilevelKernelInterpolation.shiftedpolynomial(
            testdata, comparisonpoly
        )
        trialcomparisonpoly = MultilevelKernelInterpolation.shiftedpolynomial(
            trialdata, comparisonpoly
        )

        comparisongreen = MultilevelKernelInterpolation.calculateI2Otranslation!(
            translator,
            MultilevelKernelInterpolation.∇CrossGreen3DTranslation(),
            testcomparisonpoly,
            trialcomparisonpoly,
        )

        for poly in [
            MultilevelKernelInterpolation.BarycentricLagrangePolynomialEquidistant3D(7),
            MultilevelKernelInterpolation.BarycentricLagrangePolynomialChebyshev3D(7),
            MultilevelKernelInterpolation.BarycentricLagrangePolynomialChebyshev3D2(7),
        ]
            testpolynomial = MultilevelKernelInterpolation.shiftedpolynomial(testdata, poly)
            trialpolynomial = MultilevelKernelInterpolation.shiftedpolynomial(
                trialdata, poly
            )

            green = MultilevelKernelInterpolation.calculateI2Otranslation!(
                translator,
                MultilevelKernelInterpolation.i2otranslation(op),
                testpolynomial,
                trialpolynomial,
            )

            interpolationmatrix = MultilevelKernelInterpolation.evaluateallpolynomials(
                poly, comparisonpoly.ranges[1]
            )

            for i in eachindex(green)
                all(iszero, green[i]) && continue
                interpolatedgreen = interpolationmatrix * green[i] * interpolationmatrix'
                er =
                    maximum(abs, comparisongreen[i] - interpolatedgreen) /
                    maximum(abs, comparisongreen[i])
                @test er < 1e-3
            end
        end
    end
end

@testset verbose=true "B-spline" begin
    @testset "Scalar Green's" begin
        λ = 1.0
        k = 2π / λ
        op = Maxwell3D.singlelayer(; wavenumber=k)

        translator = MultilevelKernelInterpolation.KernelTranslator(
            op, nothing, nothing, nothing
        )
        comparisonpoly = MultilevelKernelInterpolation.BarycentricLagrangePolynomialEquidistant3D(
            9
        )

        h = λ / 4

        testdata = H2Trees.BoxData(0, Int[], SVector(0.0, 0.0, 0.0), h, 1)
        trialdata = H2Trees.BoxData(0, Int[], SVector(0.0, 4 * h, 0.0), h, 1)

        testcomparisonpoly = MultilevelKernelInterpolation.shiftedpolynomial(
            testdata, comparisonpoly
        )
        trialcomparisonpoly = MultilevelKernelInterpolation.shiftedpolynomial(
            trialdata, comparisonpoly
        )

        comparisongreen = MultilevelKernelInterpolation.calculateI2Otranslation!(
            translator,
            MultilevelKernelInterpolation.i2otranslation(op),
            testcomparisonpoly,
            trialcomparisonpoly,
        )

        for poly in [
            MultilevelKernelInterpolation.BsplinePolynomialEquidistant3D{Float64}(4, 10),
            MultilevelKernelInterpolation.BufferedBsplinePolynomialEquidistant3D{Float64}(
                4, 10, 1
            ),
        ]
            testpolynomial = MultilevelKernelInterpolation.shiftedpolynomial(testdata, poly)
            trialpolynomial = MultilevelKernelInterpolation.shiftedpolynomial(
                trialdata, poly
            )
            translator = MultilevelKernelInterpolation.BsplineKernelTranslator(
                op, [poly], nothing, 1:1, nothing
            )

            @test comparisongreen ≈ MultilevelKernelInterpolation.calculateI2Otranslation!(
                translator,
                MultilevelKernelInterpolation.i2otranslation(op),
                MultilevelKernelInterpolation.shiftedpolynomial(
                    testdata,
                    MultilevelKernelInterpolation.BsplinePolynomialEquidistant3D{Float64}(
                        1, 10
                    ),
                ),
                MultilevelKernelInterpolation.shiftedpolynomial(
                    trialdata,
                    MultilevelKernelInterpolation.BsplinePolynomialEquidistant3D{Float64}(
                        1, 10
                    ),
                ),
            )

            green = MultilevelKernelInterpolation.calculateI2Otranslation!(
                translator,
                MultilevelKernelInterpolation.i2otranslation(op),
                testpolynomial,
                trialpolynomial,
            )

            _hasactiveknotremoval(poly) && (green = _promotetorawbasis(green, poly))

            interpolationmatrix = MultilevelKernelInterpolation.evaluateallpolynomials(
                poly, comparisonpoly.ranges[1]
            )

            interpolatedgreen = interpolationmatrix * green * interpolationmatrix'
            er =
                maximum(abs, comparisongreen - interpolatedgreen) /
                maximum(abs, comparisongreen)
            @test er < 1e-4
        end
    end

    @testset "Dyadic Green's" begin
        λ = 1.0
        k = 2π / λ
        op = MultilevelKernelInterpolation.efie(; wavenumber=k)

        translator = MultilevelKernelInterpolation.KernelTranslator(
            op, nothing, nothing, nothing
        )
        comparisonpoly = MultilevelKernelInterpolation.BarycentricLagrangePolynomialEquidistant3D(
            9
        )

        h = λ / 4

        testdata = H2Trees.BoxData(0, Int[], SVector(0.0, 0.0, 0.0), h, 1)
        trialdata = H2Trees.BoxData(0, Int[], SVector(0.0, 4 * h, 0.0), h, 1)

        testcomparisonpoly = MultilevelKernelInterpolation.shiftedpolynomial(
            testdata, comparisonpoly
        )
        trialcomparisonpoly = MultilevelKernelInterpolation.shiftedpolynomial(
            trialdata, comparisonpoly
        )

        comparisongreen = MultilevelKernelInterpolation.calculateI2Otranslation!(
            translator,
            MultilevelKernelInterpolation.DyadicGreen3DTranslation(),
            testcomparisonpoly,
            trialcomparisonpoly,
        )

        for poly in [
            MultilevelKernelInterpolation.BsplinePolynomialEquidistant3D{Float64}(4, 10),
            MultilevelKernelInterpolation.BufferedBsplinePolynomialEquidistant3D{Float64}(
                4, 10, 1
            ),
        ]
            testpolynomial = MultilevelKernelInterpolation.shiftedpolynomial(testdata, poly)
            trialpolynomial = MultilevelKernelInterpolation.shiftedpolynomial(
                trialdata, poly
            )
            translator = MultilevelKernelInterpolation.BsplineKernelTranslator(
                op, [poly], nothing, 1:1, nothing
            )

            @test comparisongreen ≈ MultilevelKernelInterpolation.calculateI2Otranslation!(
                translator,
                MultilevelKernelInterpolation.i2otranslation(op),
                MultilevelKernelInterpolation.shiftedpolynomial(
                    testdata,
                    MultilevelKernelInterpolation.BsplinePolynomialEquidistant3D{Float64}(
                        1, 10
                    ),
                ),
                MultilevelKernelInterpolation.shiftedpolynomial(
                    trialdata,
                    MultilevelKernelInterpolation.BsplinePolynomialEquidistant3D{Float64}(
                        1, 10
                    ),
                ),
            )

            green = MultilevelKernelInterpolation.calculateI2Otranslation!(
                translator,
                MultilevelKernelInterpolation.i2otranslation(op),
                testpolynomial,
                trialpolynomial,
            )

            _hasactiveknotremoval(poly) && (green = _promotetorawbasis(green, poly))

            interpolationmatrix = MultilevelKernelInterpolation.evaluateallpolynomials(
                poly, comparisonpoly.ranges[1]
            )

            for i in eachindex(green)
                interpolatedgreen = interpolationmatrix * green[i] * interpolationmatrix'
                er =
                    maximum(abs, comparisongreen[i] - interpolatedgreen) /
                    maximum(abs, comparisongreen[i])
                @test er < 1e-3
            end
        end
    end

    @testset "∇∇ Green's" begin
        λ = 1.0
        k = 2π / λ
        op = MultilevelKernelInterpolation.MWHyperSingular3D(; wavenumber=k)

        translator = MultilevelKernelInterpolation.KernelTranslator(
            op, nothing, nothing, nothing
        )
        comparisonpoly = MultilevelKernelInterpolation.BarycentricLagrangePolynomialChebyshev3D2(
            9
        )

        h = λ / 4

        testdata = H2Trees.BoxData(0, Int[], SVector(0.0, 0.0, 0.0), h, 1)
        trialdata = H2Trees.BoxData(0, Int[], SVector(0.0, 4 * h, 0.0), h, 1)

        testcomparisonpoly = MultilevelKernelInterpolation.shiftedpolynomial(
            testdata, comparisonpoly
        )
        trialcomparisonpoly = MultilevelKernelInterpolation.shiftedpolynomial(
            trialdata, comparisonpoly
        )

        comparisongreen = MultilevelKernelInterpolation.calculateI2Otranslation!(
            translator,
            MultilevelKernelInterpolation.∇∇Green3DTranslation(),
            testcomparisonpoly,
            trialcomparisonpoly,
        )

        for poly in [
            MultilevelKernelInterpolation.BarycentricLagrangePolynomialEquidistant3D(7),
            MultilevelKernelInterpolation.BarycentricLagrangePolynomialChebyshev3D(7),
            MultilevelKernelInterpolation.BarycentricLagrangePolynomialChebyshev3D2(7),
        ]
            testpolynomial = MultilevelKernelInterpolation.shiftedpolynomial(testdata, poly)
            trialpolynomial = MultilevelKernelInterpolation.shiftedpolynomial(
                trialdata, poly
            )

            green = MultilevelKernelInterpolation.calculateI2Otranslation!(
                translator,
                MultilevelKernelInterpolation.i2otranslation(op),
                testpolynomial,
                trialpolynomial,
            )

            interpolationmatrix = MultilevelKernelInterpolation.evaluateallpolynomials(
                poly, comparisonpoly.ranges[1]
            )

            for i in eachindex(green)
                interpolatedgreen = interpolationmatrix * green[i] * interpolationmatrix'
                er =
                    maximum(abs, comparisongreen[i] - interpolatedgreen) /
                    maximum(abs, comparisongreen[i])
                @test er < 1e-3
            end
        end
    end

    @testset "∇ Green's" begin
        λ = 1.0
        k = 2π / λ
        op = MultilevelKernelInterpolation.HH3DDoubleLayerFDBIOGradientI2O(; wavenumber=k)

        translator = MultilevelKernelInterpolation.KernelTranslator(
            op, nothing, nothing, nothing
        )
        comparisonpoly = MultilevelKernelInterpolation.BarycentricLagrangePolynomialEquidistant3D(
            9
        )

        h = λ / 4

        testdata = H2Trees.BoxData(0, Int[], SVector(0.0, 0.0, 0.0), h, 1)
        trialdata = H2Trees.BoxData(0, Int[], SVector(0.0, 4 * h, 0.0), h, 1)

        testcomparisonpoly = MultilevelKernelInterpolation.shiftedpolynomial(
            testdata, comparisonpoly
        )
        trialcomparisonpoly = MultilevelKernelInterpolation.shiftedpolynomial(
            trialdata, comparisonpoly
        )

        comparisongreen = MultilevelKernelInterpolation.calculateI2Otranslation!(
            translator,
            MultilevelKernelInterpolation.∇Green3DTranslation(),
            testcomparisonpoly,
            trialcomparisonpoly,
        )

        for poly in [
            MultilevelKernelInterpolation.BsplinePolynomialEquidistant3D{Float64}(4, 10),
            MultilevelKernelInterpolation.BufferedBsplinePolynomialEquidistant3D{Float64}(
                4, 10, 1
            ),
        ]
            testpolynomial = MultilevelKernelInterpolation.shiftedpolynomial(testdata, poly)
            trialpolynomial = MultilevelKernelInterpolation.shiftedpolynomial(
                trialdata, poly
            )
            translator = MultilevelKernelInterpolation.BsplineKernelTranslator(
                op, [poly], nothing, 1:1, nothing
            )

            @test comparisongreen ≈ MultilevelKernelInterpolation.calculateI2Otranslation!(
                translator,
                MultilevelKernelInterpolation.i2otranslation(op),
                MultilevelKernelInterpolation.shiftedpolynomial(
                    testdata,
                    MultilevelKernelInterpolation.BsplinePolynomialEquidistant3D{Float64}(
                        1, 10
                    ),
                ),
                MultilevelKernelInterpolation.shiftedpolynomial(
                    trialdata,
                    MultilevelKernelInterpolation.BsplinePolynomialEquidistant3D{Float64}(
                        1, 10
                    ),
                ),
            )

            green = MultilevelKernelInterpolation.calculateI2Otranslation!(
                translator,
                MultilevelKernelInterpolation.i2otranslation(op),
                testpolynomial,
                trialpolynomial,
            )

            _hasactiveknotremoval(poly) && (green = _promotetorawbasis(green, poly))

            interpolationmatrix = MultilevelKernelInterpolation.evaluateallpolynomials(
                poly, comparisonpoly.ranges[1]
            )

            for i in eachindex(green)
                interpolatedgreen = interpolationmatrix * green[i] * interpolationmatrix'
                er =
                    maximum(abs, comparisongreen[i] - interpolatedgreen) /
                    maximum(abs, comparisongreen[i])
                @test er < 1e-3
            end
        end
    end

    @testset "∇ × Green's" begin
        λ = 1.0
        k = 2π / λ
        op = MultilevelKernelInterpolation.MWDoubleLayer3D(; wavenumber=k)

        translator = MultilevelKernelInterpolation.KernelTranslator(
            op, nothing, nothing, nothing
        )
        comparisonpoly = MultilevelKernelInterpolation.BarycentricLagrangePolynomialChebyshev3D2(
            9
        )

        h = λ / 4

        testdata = H2Trees.BoxData(0, Int[], SVector(0.0, 0.0, 0.0), h, 1)
        trialdata = H2Trees.BoxData(0, Int[], SVector(0.0, 4 * h, 0.0), h, 1)

        testcomparisonpoly = MultilevelKernelInterpolation.shiftedpolynomial(
            testdata, comparisonpoly
        )
        trialcomparisonpoly = MultilevelKernelInterpolation.shiftedpolynomial(
            trialdata, comparisonpoly
        )

        comparisongreen = MultilevelKernelInterpolation.calculateI2Otranslation!(
            translator,
            MultilevelKernelInterpolation.∇CrossGreen3DTranslation(),
            testcomparisonpoly,
            trialcomparisonpoly,
        )

        for poly in [
            MultilevelKernelInterpolation.BarycentricLagrangePolynomialEquidistant3D(7),
            MultilevelKernelInterpolation.BarycentricLagrangePolynomialChebyshev3D(7),
            MultilevelKernelInterpolation.BarycentricLagrangePolynomialChebyshev3D2(7),
        ]
            testpolynomial = MultilevelKernelInterpolation.shiftedpolynomial(testdata, poly)
            trialpolynomial = MultilevelKernelInterpolation.shiftedpolynomial(
                trialdata, poly
            )

            green = MultilevelKernelInterpolation.calculateI2Otranslation!(
                translator,
                MultilevelKernelInterpolation.i2otranslation(op),
                testpolynomial,
                trialpolynomial,
            )

            interpolationmatrix = MultilevelKernelInterpolation.evaluateallpolynomials(
                poly, comparisonpoly.ranges[1]
            )

            for i in eachindex(green)
                all(iszero, green[i]) && continue
                interpolatedgreen = interpolationmatrix * green[i] * interpolationmatrix'
                er =
                    maximum(abs, comparisongreen[i] - interpolatedgreen) /
                    maximum(abs, comparisongreen[i])
                @test er < 1e-3
            end
        end
    end
end

end
