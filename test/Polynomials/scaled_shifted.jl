module TestScaledShifted

using Test
using CompScienceMeshes
using H2Trees
using MultilevelKernelInterpolation

@testset verbose=true "ScaledShiftedPolynomial" begin
    base = MultilevelKernelInterpolation.BarycentricLagrangePolynomialEquidistant{3,Float64}(
        4
    )
    scaled = MultilevelKernelInterpolation.ScaledShiftedPolynomial(
        base, 2.0, [-0.5, 1.0, 3.0]
    )

    @test MultilevelKernelInterpolation.polynomial(scaled) === base
    @test MultilevelKernelInterpolation.dims(scaled) ==
        MultilevelKernelInterpolation.dims(base)
    @test MultilevelKernelInterpolation.numpolynomials(scaled) ==
        MultilevelKernelInterpolation.numpolynomials(base)
    @test eltype(scaled) == eltype(base)
    @test MultilevelKernelInterpolation.scaling(scaled) == 2.0
    @test MultilevelKernelInterpolation.shift(scaled) == [-0.5, 1.0, 3.0]

    for dim in 1:MultilevelKernelInterpolation.dims(scaled)
        @test MultilevelKernelInterpolation.range(scaled, dim) ≈
            MultilevelKernelInterpolation.range(base, dim) .* 2.0 .+
              MultilevelKernelInterpolation.shift(scaled)[dim]
        @test MultilevelKernelInterpolation.range(scaled, dim, 2) ≈
            MultilevelKernelInterpolation.range(base, dim, 2) * 2.0 +
              MultilevelKernelInterpolation.shift(scaled)[dim]

        range_buffer = similar(MultilevelKernelInterpolation.range(scaled, dim))
        @test MultilevelKernelInterpolation.range!(range_buffer, scaled, dim) ===
            range_buffer
        @test range_buffer ≈ MultilevelKernelInterpolation.range(scaled, dim)
    end

    values_base = zeros(Float64, MultilevelKernelInterpolation.numpolynomials(base)[1])
    values_scaled = similar(values_base)
    derivatives_base = similar(values_base)
    derivatives_scaled = similar(values_base)

    for dim in 1:MultilevelKernelInterpolation.dims(scaled)
        for x_parent in range(0.0, 1.0; length=7)
            x_scaled =
                x_parent * MultilevelKernelInterpolation.scaling(scaled) +
                MultilevelKernelInterpolation.shift(scaled)[dim]

            MultilevelKernelInterpolation.evaluateall1dpolynomial!(
                values_base, base, x_parent, dim
            )
            MultilevelKernelInterpolation.evaluateall1dpolynomial!(
                values_scaled, scaled, x_scaled, dim
            )
            @test values_scaled ≈ values_base atol = 1e-14

            MultilevelKernelInterpolation.evaluateall1dpolynomialderivative!(
                (values_base, derivatives_base), base, x_parent, dim
            )
            MultilevelKernelInterpolation.evaluateall1dpolynomialderivative!(
                (values_scaled, derivatives_scaled), scaled, x_scaled, dim
            )
            @test values_scaled ≈ values_base atol = 1e-14
            @test derivatives_scaled ≈
                derivatives_base ./ MultilevelKernelInterpolation.scaling(scaled) atol =
                1e-12
        end
    end

    @testset verbose=true "scalar shift broadcasts across dimensions" begin
        scalar_shifted = MultilevelKernelInterpolation.ScaledShiftedPolynomial(
            base, 1.5, 0.25
        )
        @test MultilevelKernelInterpolation.shift(scalar_shifted) ==
            fill(0.25, MultilevelKernelInterpolation.dims(base))
    end

    @testset verbose=true "forwarded indices and traits" begin
        @test CartesianIndices(scaled) == CartesianIndices(base)
        @test LinearIndices(scaled) == LinearIndices(base)
        @test MultilevelKernelInterpolation.isequidistantpolynomial(scaled)

        chebyshev_base = MultilevelKernelInterpolation.BarycentricLagrangePolynomialChebyshev{
            3,Float64
        }(
            4
        )
        chebyshev_scaled = MultilevelKernelInterpolation.ScaledShiftedPolynomial(
            chebyshev_base, 2.0, [0.0, 0.0, 0.0]
        )
        @test !MultilevelKernelInterpolation.isequidistantpolynomial(chebyshev_scaled)
    end

    @testset verbose=true "increase and noincrease" begin
        # both grow/reset the wrapped polynomial (via increase/noincrease on the
        # inner polynomial) and recompute scaling/shift from a "protrusion" amount:
        # pro = max(protrusion, maximum(abs, shift(p)) / 2), scaling = 1 + 2*pro,
        # shift = fill(-pro, dims(p)).
        shift_magnitude = maximum(abs, MultilevelKernelInterpolation.shift(scaled))
        expected_default_pro = shift_magnitude / 2

        increased = MultilevelKernelInterpolation.increase(scaled)
        @test MultilevelKernelInterpolation.polynomial(increased) isa typeof(base)
        @test MultilevelKernelInterpolation.polynomialorders(
            MultilevelKernelInterpolation.polynomial(increased)
        ) == 2 .* MultilevelKernelInterpolation.polynomialorders(base)
        @test MultilevelKernelInterpolation.scaling(increased) ≈
            1 + 2 * expected_default_pro
        @test MultilevelKernelInterpolation.shift(increased) ≈
            fill(-expected_default_pro, MultilevelKernelInterpolation.dims(scaled))

        noincreased = MultilevelKernelInterpolation.noincrease(scaled)
        @test MultilevelKernelInterpolation.polynomial(noincreased) isa typeof(base)
        @test MultilevelKernelInterpolation.polynomialorders(
            MultilevelKernelInterpolation.polynomial(noincreased)
        ) == MultilevelKernelInterpolation.polynomialorders(base)
        @test MultilevelKernelInterpolation.scaling(noincreased) ≈
            1 + 2 * expected_default_pro
        @test MultilevelKernelInterpolation.shift(noincreased) ≈
            fill(-expected_default_pro, MultilevelKernelInterpolation.dims(scaled))

        # a protrusion keyword larger than the shift-derived value takes precedence
        large_protrusion = shift_magnitude * 10
        increased_large_pro = MultilevelKernelInterpolation.increase(
            scaled; protrusion=large_protrusion
        )
        @test MultilevelKernelInterpolation.scaling(increased_large_pro) ≈
            1 + 2 * large_protrusion
        @test MultilevelKernelInterpolation.shift(increased_large_pro) ≈
            fill(-large_protrusion, MultilevelKernelInterpolation.dims(scaled))

        # a protrusion keyword smaller than the shift-derived value has no effect
        increased_small_pro = MultilevelKernelInterpolation.increase(
            scaled; protrusion=shift_magnitude / 100
        )
        @test MultilevelKernelInterpolation.scaling(increased_small_pro) ≈
            1 + 2 * expected_default_pro
    end

    @testset verbose=true "realizedprotrusion" begin
        # A plain (unwrapped) Lagrange polynomial has no domain-extension mechanism -- always
        # zero, matching the generic `realizedprotrusion(::Any) = 0` fallback.
        @test MultilevelKernelInterpolation.realizedprotrusion(base) == 0

        # Unlike the buffered-B-spline case, there is no integer/rounding discretization here
        # -- `shift(p)` already stores exactly the realized `pro` value `increase`/`noincrease`
        # computed (including their own `max(...)`-based floor against the previous level), so
        # this is a direct read.
        @test MultilevelKernelInterpolation.realizedprotrusion(scaled) ==
            maximum(abs, MultilevelKernelInterpolation.shift(scaled))

        increased = MultilevelKernelInterpolation.increase(scaled)
        @test MultilevelKernelInterpolation.realizedprotrusion(increased) ≈
            maximum(abs, MultilevelKernelInterpolation.shift(increased))

        large_protrusion = maximum(abs, MultilevelKernelInterpolation.shift(scaled)) * 10
        increased_large_pro = MultilevelKernelInterpolation.increase(
            scaled; protrusion=large_protrusion
        )
        @test MultilevelKernelInterpolation.realizedprotrusion(increased_large_pro) ≈
            large_protrusion
    end

    @testset verbose=true "leaf polynomials" begin
        mesh = readmesh(
            joinpath(
                pkgdir(MultilevelKernelInterpolation), "test", "assets", "in", "sphere.in"
            ),
        )
        tree = buildtree(
            vertices(mesh); builder=TwoNTreeBuilder(; minhalfsize=0.1, minvalues=0)
        )
        poly = MultilevelKernelInterpolation.BarycentricLagrangePolynomialEquidistant{
            3,Float64
        }(
            4
        )
        tree_polys = MultilevelKernelInterpolation.levelpolynomials(
            poly, H2Trees.levels(tree), _ -> false
        )
        leaf_polys = MultilevelKernelInterpolation.leafpolynomials(tree, tree_polys)

        @test length(leaf_polys) == length(H2Trees.leaves(tree))

        range_buffer = Vector{Float64}(
            undef, MultilevelKernelInterpolation.numpolynomials(poly)[1]
        )
        values = similar(range_buffer)
        derivative_values = similar(range_buffer)

        for (leaf_poly, leaf) in zip(leaf_polys, H2Trees.leaves(tree))
            @test H2Trees.center(leaf_poly) ≈ H2Trees.center(tree, leaf)
            @test MultilevelKernelInterpolation.polynomial(leaf_poly) === poly
            @test MultilevelKernelInterpolation.dims(leaf_poly) ==
                MultilevelKernelInterpolation.dims(poly)
            @test MultilevelKernelInterpolation.numpolynomials(leaf_poly) ==
                MultilevelKernelInterpolation.numpolynomials(poly)
            @test typeof(MultilevelKernelInterpolation.kernelbuffer(leaf_poly, Float32)) ==
                typeof(MultilevelKernelInterpolation.kernelbuffer(poly, Float32))
            @test typeof(
                MultilevelKernelInterpolation.kernelbufferderivative(leaf_poly, Float32)
            ) == typeof(
                MultilevelKernelInterpolation.kernelbufferderivative(poly, Float32)
            )
            @test eltype(leaf_poly) == eltype(poly)

            for dim in 1:MultilevelKernelInterpolation.dims(leaf_poly)
                @test first(MultilevelKernelInterpolation.range(leaf_poly, dim)) ≈
                    H2Trees.center(tree, leaf)[dim] - H2Trees.halfsize(tree, leaf)
                @test last(MultilevelKernelInterpolation.range(leaf_poly, dim)) ≈
                    H2Trees.center(tree, leaf)[dim] + H2Trees.halfsize(tree, leaf)
                @test H2Trees.center(leaf_poly, dim) ≈ H2Trees.center(tree, leaf)[dim]
                @test MultilevelKernelInterpolation.range!(range_buffer, leaf_poly, dim) ≈
                    MultilevelKernelInterpolation.range(leaf_poly, dim)

                for (point_index, point) in
                    enumerate(MultilevelKernelInterpolation.range(leaf_poly, dim))
                    MultilevelKernelInterpolation.evaluateall1dpolynomial!(
                        values, leaf_poly, point, dim
                    )
                    @test sum(values) ≈ 1
                    @test values[point_index] ≈ 1

                    MultilevelKernelInterpolation.evaluateall1dpolynomialderivative!(
                        (values, derivative_values), leaf_poly, point, dim
                    )
                    @test sum(values) ≈ 1
                    @test values[point_index] ≈ 1
                end
            end
        end
    end
end

end
