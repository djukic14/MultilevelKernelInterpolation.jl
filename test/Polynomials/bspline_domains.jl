module TestBsplineDomains

using Test
using MultilevelKernelInterpolation
using MultilevelKernelInterpolation.NURBS

@testset "B-spline domains and registered NURBS API" begin
    for T in (Float32, Float64), buffer in (0, 2)
        poly = if buffer == 0
            MultilevelKernelInterpolation.BsplinePolynomialEquidistant{1,T}(3, 8)
        else
            MultilevelKernelInterpolation.BufferedBsplinePolynomialEquidistant{1,T}(
                3, 8, buffer
            )
        end
        spline = MultilevelKernelInterpolation.bspline(poly, 1)
        extension = buffer / (8 - 3)
        lo, hi = -extension, 1 + extension
        values = zeros(T, MultilevelKernelInterpolation.numpolynomials(poly)[1])
        derivatives = similar(values)
        for extrapolate in (false, true)
            for x in (lo, 0.0, 0.37, 1.0, hi)
                MultilevelKernelInterpolation.evaluateall1dpolynomial!(
                    values, poly, x, 1; extrapolate
                )
                @test sum(values) ≈ 1
                expected = [NURBS.evalNaive(spline, i, x) for i in eachindex(values)]
                @test values ≈ expected
                MultilevelKernelInterpolation.evaluateall1dpolynomialderivative!(
                    (values, derivatives), poly, x, 1; extrapolate
                )
                @test sum(derivatives) ≈ 0 atol=100eps(T)
            end
            for x in (lo - 0.01, hi + 0.01, NaN)
                @test_throws DomainError MultilevelKernelInterpolation.evaluateall1dpolynomial!(
                    values, poly, x, 1; extrapolate
                )
                @test_throws DomainError MultilevelKernelInterpolation.evaluateall1dpolynomialderivative!(
                    (values, derivatives), poly, x, 1; extrapolate
                )
            end
        end
        if buffer > 0
            x, h = 0.37, 1e-4
            MultilevelKernelInterpolation.evaluateall1dpolynomialderivative!(
                (values, derivatives), poly, x, 1
            )
            finite_difference = [
                (NURBS.evalNaive(spline, i, x + h) - NURBS.evalNaive(spline, i, x - h)) /
                (2h) for i in eachindex(values)
            ]
            @test derivatives ≈ finite_difference rtol=1e-3 atol=1e-4
            scaled = MultilevelKernelInterpolation.ScaledShiftedPolynomial(poly, 2.0, -1.0)
            scaled_values, scaled_derivatives = similar(values), similar(values)
            MultilevelKernelInterpolation.evaluateall1dpolynomialderivative!(
                (scaled_values, scaled_derivatives), scaled, 2x - 1, 1
            )
            @test scaled_values ≈ values
            @test scaled_derivatives ≈ derivatives / 2
        end
    end

    lagrange = MultilevelKernelInterpolation.BarycentricLagrangePolynomialEquidistant(
        dim=1, order=3
    )
    values = zeros(4)
    MultilevelKernelInterpolation.evaluateall1dpolynomial!(
        values, lagrange, 1.1, 1; extrapolate=true
    )
    @test sum(values) ≈ 1
    MultilevelKernelInterpolation.evaluateall1dpolynomial!(
        values, lagrange, 1.1, 1; extrapolate=false
    )
    @test all(iszero, values)

    @testset "Buffered transfers with saturated knots" begin
        for degree in 1:3,
            mode in (
                MultilevelKernelInterpolation.noincrease,
                MultilevelKernelInterpolation.increase,
            )

            child = MultilevelKernelInterpolation.BufferedBsplinePolynomialEquidistant{
                1,Float64
            }(
                degree, degree + 3, 1
            )
            parent = mode(child)
            left, right = MultilevelKernelInterpolation.onedimi2itranslations(child, parent)
            @test size(left) == (
                MultilevelKernelInterpolation.numpolynomials(parent)[1],
                MultilevelKernelInterpolation.numpolynomials(child)[1],
            )
            @test size(right) == size(left)
            @test all(isfinite, left)
            @test all(isfinite, right)
        end
    end
end

end
