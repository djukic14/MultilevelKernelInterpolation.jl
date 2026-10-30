module TestBarycentricLagrange

using Test
using LinearAlgebra
using MultilevelKernelInterpolation

include(
    joinpath(
        pkgdir(MultilevelKernelInterpolation), "test", "support", "lagrangereference.jl"
    ),
)

const BARYCENTRIC_LAGRANGE_TYPES = (
    MultilevelKernelInterpolation.BarycentricLagrangePolynomialEquidistant,
    MultilevelKernelInterpolation.BarycentricLagrangePolynomialChebyshev,
    MultilevelKernelInterpolation.BarycentricLagrangePolynomialChebyshev2,
)

const BARYCENTRIC_LAGRANGE_REFERENCE_PAIRS = (
    (
        MultilevelKernelInterpolation.BarycentricLagrangePolynomialEquidistant,
        LagrangeReference.LagrangePolynomialEquidistant,
    ),
    (
        MultilevelKernelInterpolation.BarycentricLagrangePolynomialChebyshev,
        LagrangeReference.LagrangePolynomialChebyshev,
    ),
    (
        MultilevelKernelInterpolation.BarycentricLagrangePolynomialChebyshev2,
        LagrangeReference.LagrangePolynomialChebyshev2,
    ),
)

@testset verbose=true "Barycentric Lagrange" begin
    @testset verbose=true "convenience constructors and traits" begin
        @test_nowarn MultilevelKernelInterpolation.BarycentricLagrangePolynomialEquidistant3D(
            3
        )
        @test_nowarn MultilevelKernelInterpolation.BarycentricLagrangePolynomialChebyshev3D(
            3
        )
        @test_nowarn MultilevelKernelInterpolation.BarycentricLagrangePolynomialChebyshev3D2(
            3
        )

        keyword_poly = MultilevelKernelInterpolation.BarycentricLagrangePolynomialEquidistant(;
            dim=2, order=3, T=Float32
        )
        @test keyword_poly isa
            MultilevelKernelInterpolation.AbstractBarycentricLagrange{2,Float32}
        @test eltype(keyword_poly) == Float32

        chebyshev_keyword_poly = MultilevelKernelInterpolation.BarycentricLagrangePolynomialChebyshev(;
            dim=2, order=3, T=Float32
        )
        @test chebyshev_keyword_poly isa
            MultilevelKernelInterpolation.AbstractBarycentricLagrange{2,Float32}
        @test eltype(chebyshev_keyword_poly) == Float32
        @test MultilevelKernelInterpolation.dims(chebyshev_keyword_poly) == 2
        @test MultilevelKernelInterpolation.polynomialorders(chebyshev_keyword_poly) ==
            (3, 3)

        chebyshev2_keyword_poly = MultilevelKernelInterpolation.BarycentricLagrangePolynomialChebyshev2(;
            dim=2, order=3, T=Float32
        )
        @test chebyshev2_keyword_poly isa
            MultilevelKernelInterpolation.AbstractBarycentricLagrange{2,Float32}
        @test eltype(chebyshev2_keyword_poly) == Float32
        @test MultilevelKernelInterpolation.dims(chebyshev2_keyword_poly) == 2
        @test MultilevelKernelInterpolation.polynomialorders(chebyshev2_keyword_poly) ==
            (3, 3)

        # the three variants must produce distinct interpolation nodes for the same
        # order (that's the whole point of offering equidistant vs Chebyshev nodes)
        @test MultilevelKernelInterpolation.range(keyword_poly, 1) !=
            MultilevelKernelInterpolation.range(chebyshev_keyword_poly, 1)
        @test MultilevelKernelInterpolation.range(chebyshev_keyword_poly, 1) !=
            MultilevelKernelInterpolation.range(chebyshev2_keyword_poly, 1)

        @test MultilevelKernelInterpolation.isequidistantpolynomial(
            MultilevelKernelInterpolation.BarycentricLagrangePolynomialEquidistant3D(3)
        )
        @test !MultilevelKernelInterpolation.isequidistantpolynomial(
            MultilevelKernelInterpolation.BarycentricLagrangePolynomialChebyshev3D(3)
        )
        @test !MultilevelKernelInterpolation.isequidistantpolynomial(
            MultilevelKernelInterpolation.BarycentricLagrangePolynomialChebyshev3D2(3)
        )
    end

    @testset verbose=true "constructing with an AbstractVector type parameter" begin
        # Poly{D,T}(order) and Poly{D,T,R,W,C,L}(order) both accept T being an
        # AbstractVector (not just a Number) as a convenience: they extract
        # eltype(T) and forward to the Number-based constructor. R,W,C,L are part
        # of the dispatch signature only -- ignored in the body -- so dummy
        # placeholders must work too.
        for Poly in BARYCENTRIC_LAGRANGE_TYPES
            for D in (1, 2, 3)
                for VectorType in (Vector{Float32}, Vector{Float64}, Vector{BigFloat})
                    T = eltype(VectorType)
                    reference = Poly{D,T}(4)

                    # 2-parameter form: Poly{D,T}(order) where T<:AbstractVector
                    via_two_param = Poly{D,VectorType}(4)
                    @test via_two_param isa Poly{D,T}
                    @test eltype(via_two_param) == T
                    @test MultilevelKernelInterpolation.dims(via_two_param) == D
                    @test MultilevelKernelInterpolation.polynomialorders(via_two_param) ==
                        MultilevelKernelInterpolation.polynomialorders(reference)
                    @test MultilevelKernelInterpolation.range(via_two_param, 1) ≈
                        MultilevelKernelInterpolation.range(reference, 1)

                    # 6-parameter form: Poly{D,T,R,W,C,L}(order) where T<:AbstractVector,
                    # dummy R,W,C,L to confirm they're genuinely unused in the body
                    via_six_param = Poly{D,VectorType,Nothing,Nothing,Nothing,Nothing}(4)
                    @test via_six_param isa Poly{D,T}
                    @test eltype(via_six_param) == T
                    @test MultilevelKernelInterpolation.dims(via_six_param) == D
                    @test MultilevelKernelInterpolation.polynomialorders(via_six_param) ==
                        MultilevelKernelInterpolation.polynomialorders(reference)
                    @test MultilevelKernelInterpolation.range(via_six_param, 1) ≈
                        MultilevelKernelInterpolation.range(reference, 1)
                end
            end
        end
    end

    @testset verbose=true "generic numinterpolationnodes and numbasisfunctions" begin
        # AbstractLagrange has no override for these (unlike AbstractBspline /
        # BufferedBsplinePolynomialEquidistant), so this exercises the fully generic
        # AbstractTensorProductPolynomial fallback methods in abstractpolynomial.jl.
        for Poly in BARYCENTRIC_LAGRANGE_TYPES
            for D in (1, 2, 3)
                poly = Poly{D,Float64}(4)

                @test MultilevelKernelInterpolation.numinterpolationnodes(poly) ==
                    MultilevelKernelInterpolation.numpolynomials(poly)
                @test MultilevelKernelInterpolation.numinterpolationnodes(poly) ==
                    Tuple(length.(MultilevelKernelInterpolation.ranges(poly)))

                @test MultilevelKernelInterpolation.numbasisfunctions(poly) ==
                    MultilevelKernelInterpolation.numpolynomials(poly)
            end
        end
    end

    @testset verbose=true "basis contract" begin
        for Poly in BARYCENTRIC_LAGRANGE_TYPES
            for D in (1, 2, 3)
                for T in (Float32, Float64, BigFloat)
                    poly = Poly{D,T}(4)

                    @test MultilevelKernelInterpolation.dims(poly) == D
                    @test eltype(poly) == T
                    @test MultilevelKernelInterpolation.polynomialorders(poly) ==
                        ntuple(_ -> 4, D)
                    @test MultilevelKernelInterpolation.numpolynomials(poly) ==
                        ntuple(_ -> 5, D)
                    @test length(MultilevelKernelInterpolation.ranges(poly)) == D
                    @test CartesianIndices(poly) == CartesianIndices(
                        MultilevelKernelInterpolation.numpolynomials(poly)
                    )
                    @test LinearIndices(poly) ==
                        LinearIndices(MultilevelKernelInterpolation.numpolynomials(poly))

                    buffer = MultilevelKernelInterpolation.kernelbuffer(poly, T)
                    @test length(buffer) == D
                    @test all(
                        length(buffer[dim]) ==
                        MultilevelKernelInterpolation.numpolynomials(poly)[dim] for
                        dim in 1:D
                    )

                    derivative_buffer = MultilevelKernelInterpolation.kernelbufferderivative(
                        poly, T
                    )
                    @test length(derivative_buffer) == D
                    @test all(
                        length(derivative_buffer[dim][1]) ==
                        MultilevelKernelInterpolation.numpolynomials(poly)[dim] for
                        dim in 1:D
                    )
                    @test all(
                        length(derivative_buffer[dim][2]) ==
                        MultilevelKernelInterpolation.numpolynomials(poly)[dim] for
                        dim in 1:D
                    )
                end
            end
        end
    end

    @testset verbose=true "nodal interpolation" begin
        for Poly in BARYCENTRIC_LAGRANGE_TYPES
            poly = Poly{2,Float64}(5)
            values = zeros(Float64, MultilevelKernelInterpolation.numpolynomials(poly)[1])

            for dim in 1:MultilevelKernelInterpolation.dims(poly)
                for (node_index, node) in
                    enumerate(MultilevelKernelInterpolation.range(poly, dim))
                    MultilevelKernelInterpolation.evaluateall1dpolynomial!(
                        values, poly, node, dim
                    )

                    @test values[node_index] ≈ 1.0 atol = 1e-14
                    @test sum(values) ≈ 1.0 atol = 1e-14
                    @test count(!iszero, values) == 1
                end
            end
        end
    end

    @testset verbose=true "partition of unity and derivative consistency" begin
        for Poly in BARYCENTRIC_LAGRANGE_TYPES
            poly = Poly{1,Float64}(6)
            values = zeros(Float64, MultilevelKernelInterpolation.numpolynomials(poly)[1])
            derivative_values = zeros(
                Float64, MultilevelKernelInterpolation.numpolynomials(poly)[1]
            )

            for x in range(0.0, 1.0; length=21)
                MultilevelKernelInterpolation.evaluateall1dpolynomial!(values, poly, x, 1)
                @test sum(values) ≈ 1.0 atol = 1e-12

                MultilevelKernelInterpolation.evaluateall1dpolynomialderivative!(
                    (values, derivative_values), poly, x, 1
                )
                @test sum(values) ≈ 1.0 atol = 1e-12
                @test sum(derivative_values) ≈ 0.0 atol = 1e-10
            end
        end
    end

    @testset verbose=true "derivative extrapolate=false clamps outside [0, 1]" begin
        for Poly in BARYCENTRIC_LAGRANGE_TYPES
            poly = Poly{1,Float64}(6)
            values = zeros(Float64, MultilevelKernelInterpolation.numpolynomials(poly)[1])
            derivative_values = zeros(
                Float64, MultilevelKernelInterpolation.numpolynomials(poly)[1]
            )

            for x in (-0.5, 1.5)
                MultilevelKernelInterpolation.evaluateall1dpolynomialderivative!(
                    (values, derivative_values), poly, x, 1; extrapolate=false
                )
                @test all(iszero, values)
                @test all(iszero, derivative_values)
            end
        end
    end

    @testset verbose=true "3D tensor product evaluation" begin
        for Poly in BARYCENTRIC_LAGRANGE_TYPES
            poly = Poly{3,Float64}(3)
            interpolation = MultilevelKernelInterpolation.evaluateallpolynomials(
                poly, MultilevelKernelInterpolation.range(poly, 1)
            )

            @test size(interpolation) == (
                prod(MultilevelKernelInterpolation.numpolynomials(poly)),
                prod(MultilevelKernelInterpolation.numpolynomials(poly)),
            )
            @test interpolation ≈ I atol = 1e-12
        end
    end

    @testset verbose=true "increase" begin
        for Poly in BARYCENTRIC_LAGRANGE_TYPES
            poly = Poly{3,Float64}(4)
            increased = MultilevelKernelInterpolation.increase(poly)

            @test increased isa typeof(poly)
            @test MultilevelKernelInterpolation.numpolynomials(increased) ==
                2 .* MultilevelKernelInterpolation.numpolynomials(poly) .- 1
        end
    end

    @testset verbose=true "sampling point lookup" begin
        poly = MultilevelKernelInterpolation.BarycentricLagrangePolynomialChebyshev2{
            3,Float64
        }(
            3
        )
        point = zeros(Float64, MultilevelKernelInterpolation.dims(poly))

        for cartesian_index in CartesianIndices(poly)
            linear_index = LinearIndices(poly)[cartesian_index]
            MultilevelKernelInterpolation.getsamplingpoint!(point, poly, linear_index)

            @test point == [
                MultilevelKernelInterpolation.range(poly, dim, cartesian_index[dim]) for
                dim in 1:MultilevelKernelInterpolation.dims(poly)
            ]
        end
    end

    @testset verbose=true "naive reference implementation" begin
        # Kept close to [0, 1]: wild extrapolation of high-order Lagrange
        # polynomials is inherently ill-conditioned (Runge's phenomenon), which
        # amplifies harmless formula-level floating-point differences between
        # the two implementations well beyond any reasonable tolerance.
        evalpoints = LinRange(-0.3, 1.3, 200)

        for p in 1:8
            for T in (Float32, Float64, BigFloat)
                for (Poly, ReferencePoly) in BARYCENTRIC_LAGRANGE_REFERENCE_PAIRS
                    poly = Poly{1,T}(p)
                    reference = ReferencePoly{1,T}(p)

                    values = Vector{T}(undef, p + 1)
                    referencevalues = Vector{T}(undef, p + 1)
                    derivatives = Vector{T}(undef, p + 1)
                    referencederivatives = Vector{T}(undef, p + 1)

                    # evalpoints are Float64 regardless of T, so accuracy for T=BigFloat
                    # is capped by Float64's input precision, not BigFloat's own eps.
                    valuetol = 1e4 * max(eps(T), eps(Float64))
                    derivativetol = 1e6 * max(eps(T), eps(Float64))

                    for extrapolate in (true, false)
                        for x in evalpoints
                            MultilevelKernelInterpolation.evaluateall1dpolynomial!(
                                values, poly, x, 1; extrapolate=extrapolate
                            )
                            MultilevelKernelInterpolation.evaluateall1dpolynomial!(
                                referencevalues, reference, x, 1; extrapolate=extrapolate
                            )
                            @test values ≈ referencevalues rtol = valuetol

                            MultilevelKernelInterpolation.evaluateall1dpolynomialderivative!(
                                (values, derivatives), poly, x, 1
                            )
                            MultilevelKernelInterpolation.evaluateall1dpolynomialderivative!(
                                (referencevalues, referencederivatives), reference, x, 1
                            )
                            @test derivatives ≈ referencederivatives rtol = derivativetol
                        end
                    end
                end
            end
        end
    end
end

end
