module TestBsplines

using Test
using LinearAlgebra
using StaticArrays
using BandedMatrices
using MultilevelKernelInterpolation

@testset verbose=true "B-splines" begin
    @testset verbose=true "traits" begin
        @test MultilevelKernelInterpolation.isequidistantpolynomial(
            MultilevelKernelInterpolation.BsplinePolynomialEquidistant{3,Float64}(3, 8)
        )
        @test MultilevelKernelInterpolation.isequidistantpolynomial(
            MultilevelKernelInterpolation.BufferedBsplinePolynomialEquidistant{3,Float64}(
                3, 8, 2
            ),
        )

        keyword_poly = MultilevelKernelInterpolation.BsplinePolynomialEquidistant(;
            dim=2, degree=3, nbasis=8, T=Float32
        )
        buffered_keyword_poly = MultilevelKernelInterpolation.BufferedBsplinePolynomialEquidistant(;
            dim=2, degree=3, nbasis=8, buffer=2, T=Float32
        )

        @test keyword_poly isa MultilevelKernelInterpolation.AbstractBspline{2,Float32}
        @test buffered_keyword_poly isa
            MultilevelKernelInterpolation.AbstractBspline{2,Float32}
        @test MultilevelKernelInterpolation.numbasisfunctions(keyword_poly) ==
            MultilevelKernelInterpolation.numpolynomials(keyword_poly)
        @test MultilevelKernelInterpolation.numinterpolationnodes(buffered_keyword_poly) ==
            Tuple(length.(MultilevelKernelInterpolation.ranges(buffered_keyword_poly)))
    end

    @testset verbose=true "increase accepts protrusion keyword" begin
        # increase(::AbstractBspline; kwargs...) must accept (and ignore) protrusion,
        # matching the AbstractLagrange/ScaledShiftedPolynomial increase signature,
        # callers like levelpolynomials always pass protrusion as a keyword.
        poly = MultilevelKernelInterpolation.BsplinePolynomialEquidistant{1,Float64}(3, 8)
        increased = MultilevelKernelInterpolation.increase(poly; protrusion=0.5)
        @test increased isa typeof(poly)

        increased_no_protrusion = MultilevelKernelInterpolation.increase(poly)
        @test MultilevelKernelInterpolation.numpolynomials(increased) ==
            MultilevelKernelInterpolation.numpolynomials(increased_no_protrusion)
        @test MultilevelKernelInterpolation.range(increased, 1) ≈
            MultilevelKernelInterpolation.range(increased_no_protrusion, 1)
    end

    @testset verbose=true "buffered noincrease/increase resolve buffer from bufferpolicy" begin
        # `noincrease`/`increase` re-resolve `bufferpolicy(poly)` against the *current*
        # `protrusion` rather than reinterpreting `numbufferpoints(poly)` the way the
        # pre-refactor inline formula did (see `bspline_buffer.md`). The old integer
        # constructor now maps to `ExplicitBuffer`, which `resolvebuffer` always resolves to
        # the same fixed count, regardless of level or protrusion.
        poly = MultilevelKernelInterpolation.BufferedBsplinePolynomialEquidistant{1,Float64}(
            3, 8, 4
        )
        @test MultilevelKernelInterpolation.numpolynomials(poly) == (16,)  # 8 + 2*4 buffer points
        @test MultilevelKernelInterpolation.bufferpolicy(poly) ==
            MultilevelKernelInterpolation.ExplicitBuffer(4)

        default = MultilevelKernelInterpolation.noincrease(poly)
        @test default isa typeof(poly)
        @test MultilevelKernelInterpolation.polynomialorders(default) ==
            MultilevelKernelInterpolation.polynomialorders(poly)
        @test MultilevelKernelInterpolation.numbufferpoints(default) == 4  # ExplicitBuffer: unaffected by protrusion=0
        @test MultilevelKernelInterpolation.numpolynomials(default) == (16,)  # 8 + 2*4

        # ExplicitBuffer never grows from protrusion either, unlike the pre-refactor formula.
        custom = MultilevelKernelInterpolation.noincrease(poly; protrusion=0.9)
        @test MultilevelKernelInterpolation.numbufferpoints(custom) == 4
        @test MultilevelKernelInterpolation.bufferpolicy(custom) ==
            MultilevelKernelInterpolation.ExplicitBuffer(4)

        increased = MultilevelKernelInterpolation.increase(poly; protrusion=0.9)
        @test MultilevelKernelInterpolation.numbufferpoints(increased) == 4
        @test MultilevelKernelInterpolation.bufferpolicy(increased) ==
            MultilevelKernelInterpolation.ExplicitBuffer(4)
    end

    @testset verbose=true "buffer policies" begin
        @testset verbose=true "resolvebuffer" begin
            # NoBuffer always resolves to zero, even with nonzero protrusion.
            @test MultilevelKernelInterpolation.resolvebuffer(
                MultilevelKernelInterpolation.NoBuffer(), 3, 8, 0.0
            ) == 0
            @test MultilevelKernelInterpolation.resolvebuffer(
                MultilevelKernelInterpolation.NoBuffer(), 3, 8, 0.9
            ) == 0

            # ExplicitBuffer(n) always resolves to n, regardless of protrusion.
            @test MultilevelKernelInterpolation.resolvebuffer(
                MultilevelKernelInterpolation.ExplicitBuffer(0), 3, 8, 0.9
            ) == 0
            @test MultilevelKernelInterpolation.resolvebuffer(
                MultilevelKernelInterpolation.ExplicitBuffer(4), 3, 8, 0.0
            ) == 4
            @test MultilevelKernelInterpolation.resolvebuffer(
                MultilevelKernelInterpolation.ExplicitBuffer(4), 3, 8, 0.9
            ) == 4

            p, b, protrusion = 3, 8, 0.9
            @test MultilevelKernelInterpolation.resolvebuffer(
                MultilevelKernelInterpolation.AutoBufferFromProtrusion(0), p, b, protrusion
            ) == ceil(Int, protrusion * (b - p))
            @test MultilevelKernelInterpolation.resolvebuffer(
                MultilevelKernelInterpolation.AutoBufferFromProtrusion(), p, b, protrusion
            ) == ceil(Int, protrusion * (b - p))

            # AutoBufferFromProtrusion(n) never resolves below n, even at zero protrusion...
            @test MultilevelKernelInterpolation.resolvebuffer(
                MultilevelKernelInterpolation.AutoBufferFromProtrusion(3), p, b, 0.0
            ) == 3
            # ...but still grows past n once protrusion demands more.
            @test MultilevelKernelInterpolation.resolvebuffer(
                MultilevelKernelInterpolation.AutoBufferFromProtrusion(1), p, b, protrusion
            ) == ceil(Int, protrusion * (b - p))
        end

        @testset verbose=true "negative buffer counts are rejected at construction" begin
            # A negative buffer would flow straight into `BufferedBspline`'s knot-vector length
            # (`b + 2*numbufferpoints`), silently shrinking the basis instead of failing at the
            # point the caller made the mistake, so both policies validate at construction.
            @test_throws ArgumentError MultilevelKernelInterpolation.ExplicitBuffer(-1)
            @test_throws ArgumentError MultilevelKernelInterpolation.AutoBufferFromProtrusion(
                -1
            )

            # Zero is still fine (the boundary, not the violation).
            @test MultilevelKernelInterpolation.ExplicitBuffer(0).points == 0
            @test MultilevelKernelInterpolation.AutoBufferFromProtrusion(0).minimum == 0
        end

        @testset verbose=true "old integer constructor maps to ExplicitBuffer" begin
            poly = MultilevelKernelInterpolation.BufferedBsplinePolynomialEquidistant{
                1,Float64
            }(
                3, 8, 2
            )
            @test MultilevelKernelInterpolation.bufferpolicy(poly) ==
                MultilevelKernelInterpolation.ExplicitBuffer(2)
            @test MultilevelKernelInterpolation.numbufferpoints(poly) == 2

            zero_poly = MultilevelKernelInterpolation.BufferedBsplinePolynomialEquidistant{
                1,Float64
            }(
                3, 8, 0
            )
            @test MultilevelKernelInterpolation.bufferpolicy(zero_poly) ==
                MultilevelKernelInterpolation.ExplicitBuffer(0)
            @test MultilevelKernelInterpolation.numbufferpoints(zero_poly) == 0
        end

        @testset verbose=true "policy-based constructor resolves at construction time" begin
            # No protrusion is known yet at construction time, so this resolves against
            # protrusion=0.0 by default, exactly like the seed polynomial `levelpolynomials`
            # starts from before it calls `noincrease`/`increase` with the real per-level
            # protrusion.
            auto_poly = MultilevelKernelInterpolation.BufferedBsplinePolynomialEquidistant{
                1,Float64
            }(
                3, 8, MultilevelKernelInterpolation.AutoBufferFromProtrusion(0)
            )
            @test MultilevelKernelInterpolation.bufferpolicy(auto_poly) ==
                MultilevelKernelInterpolation.AutoBufferFromProtrusion(0)
            @test MultilevelKernelInterpolation.numbufferpoints(auto_poly) == 0

            auto_poly_min = MultilevelKernelInterpolation.BufferedBsplinePolynomialEquidistant{
                1,Float64
            }(
                3, 8, MultilevelKernelInterpolation.AutoBufferFromProtrusion(2)
            )
            @test MultilevelKernelInterpolation.numbufferpoints(auto_poly_min) == 2

            auto_poly_protrusion = MultilevelKernelInterpolation.BufferedBsplinePolynomialEquidistant{
                1,Float64
            }(
                3,
                8,
                MultilevelKernelInterpolation.AutoBufferFromProtrusion(0);
                protrusion=0.9,
            )
            @test MultilevelKernelInterpolation.numbufferpoints(auto_poly_protrusion) ==
                ceil(Int, 0.9 * (8 - 3))

            no_buffer_poly = MultilevelKernelInterpolation.BufferedBsplinePolynomialEquidistant{
                1,Float64
            }(
                3, 8, MultilevelKernelInterpolation.NoBuffer(); protrusion=0.9
            )
            @test MultilevelKernelInterpolation.numbufferpoints(no_buffer_poly) == 0
        end

        @testset verbose=true "keyword constructor accepts a policy" begin
            poly = MultilevelKernelInterpolation.BufferedBsplinePolynomialEquidistant(;
                dim=2,
                degree=3,
                nbasis=8,
                buffer=MultilevelKernelInterpolation.AutoBufferFromProtrusion(1),
                T=Float32,
            )
            @test poly isa MultilevelKernelInterpolation.AbstractBspline{2,Float32}
            @test MultilevelKernelInterpolation.bufferpolicy(poly) ==
                MultilevelKernelInterpolation.AutoBufferFromProtrusion(1)
            @test MultilevelKernelInterpolation.numbufferpoints(poly) == 1
        end

        @testset verbose=true "levelpolynomials realizes buffer per policy" begin
            levels = 1:3
            protrusion = [0.0, 0.5, 0.9]
            p, b = 3, 8

            explicit_poly = MultilevelKernelInterpolation.BufferedBsplinePolynomialEquidistant{
                1,Float64
            }(
                p, b, 4
            )
            explicit_levels = MultilevelKernelInterpolation.levelpolynomials(
                explicit_poly, levels, level -> false; protrusion=protrusion
            )
            @test all(
                poly -> MultilevelKernelInterpolation.numbufferpoints(poly) == 4,
                explicit_levels,
            )

            auto_poly = MultilevelKernelInterpolation.BufferedBsplinePolynomialEquidistant{
                1,Float64
            }(
                p, b, MultilevelKernelInterpolation.AutoBufferFromProtrusion(0)
            )
            auto_levels = MultilevelKernelInterpolation.levelpolynomials(
                auto_poly, levels, level -> false; protrusion=protrusion
            )
            @test [
                MultilevelKernelInterpolation.numbufferpoints(poly) for poly in auto_levels
            ] == [ceil(Int, pro * (b - p)) for pro in protrusion]
        end

        @testset verbose=true "bsplinebuffersummary" begin
            levels = 1:3
            protrusion = [0.0, 0.5, 0.9]
            p, b = 3, 8
            poly = MultilevelKernelInterpolation.BufferedBsplinePolynomialEquidistant{
                1,Float64
            }(
                p, b, MultilevelKernelInterpolation.AutoBufferFromProtrusion(0)
            )
            treepolys = MultilevelKernelInterpolation.levelpolynomials(
                poly, levels, level -> false; protrusion=protrusion
            )

            summary = MultilevelKernelInterpolation.bsplinebuffersummary(
                treepolys, levels, protrusion
            )
            @test length(summary) == length(levels)
            @test [row.level for row in summary] == collect(levels)
            @test [row.realizedbuffer for row in summary] == [
                MultilevelKernelInterpolation.numbufferpoints(poly) for poly in treepolys
            ]
            @test all(
                row ->
                    row.policy == MultilevelKernelInterpolation.AutoBufferFromProtrusion(0),
                summary,
            )

            # A length mismatch must error loudly rather than silently truncate via `zip`,
            # exactly the kind of misalignment this diagnostic exists to catch.
            @test_throws ArgumentError MultilevelKernelInterpolation.bsplinebuffersummary(
                treepolys, 1:2, protrusion
            )
            @test_throws ArgumentError MultilevelKernelInterpolation.bsplinebuffersummary(
                treepolys, levels, [0.0, 0.5]
            )
        end
    end

    @testset verbose=true "realizedprotrusion" begin
        # Plain (non-buffered) B-spline has no domain-extension mechanism, always zero,
        # matching the generic `realizedprotrusion(::Any) = 0` fallback.
        plain = MultilevelKernelInterpolation.BsplinePolynomialEquidistant{1,Float64}(3, 8)
        @test MultilevelKernelInterpolation.realizedprotrusion(plain) == 0

        # Buffered B-spline: realized protrusion is `buffer / (b - p)`, read back from the
        # already-constructed polynomial's own stored buffer count, i.e. the *realized*
        # value (whatever integer `buffer` this concrete polynomial was actually built with),
        # not a re-derivation via `increase`/`noincrease`'s own `ceil`-based formula.
        for p in 1:4
            for b in (p + 1):(p + 6)
                for buffer in 1:3
                    poly = MultilevelKernelInterpolation.BufferedBsplinePolynomialEquidistant{
                        1,Float64
                    }(
                        p, b, buffer
                    )
                    @test MultilevelKernelInterpolation.realizedprotrusion(poly) ≈
                        buffer / (b - p)
                end
            end
        end
    end

    @testset verbose=true "equidistant basis contract" begin
        for D in (1, 2, 3)
            poly = MultilevelKernelInterpolation.BsplinePolynomialEquidistant{D,Float64}(
                3, 8
            )

            @test MultilevelKernelInterpolation.dims(poly) == D
            @test eltype(poly) == Float64
            @test MultilevelKernelInterpolation.polynomialorders(poly) == ntuple(_ -> 3, D)
            @test MultilevelKernelInterpolation.numpolynomials(poly) == ntuple(_ -> 8, D)
            @test length(MultilevelKernelInterpolation.ranges(poly)) == D
            @test CartesianIndices(poly) ==
                CartesianIndices(MultilevelKernelInterpolation.numpolynomials(poly))
            @test LinearIndices(poly) ==
                LinearIndices(MultilevelKernelInterpolation.numpolynomials(poly))
        end
    end

    @testset verbose=true "basis evaluation" begin
        poly = MultilevelKernelInterpolation.BsplinePolynomialEquidistant{1,Float64}(3, 8)
        values = zeros(Float64, MultilevelKernelInterpolation.numpolynomials(poly)[1])
        derivative_values = similar(values)

        for x in range(0.0, 1.0; length=17)
            MultilevelKernelInterpolation.evaluateall1dpolynomial!(values, poly, x, 1)
            @test sum(values) ≈ 1.0 atol = 1e-12

            MultilevelKernelInterpolation.evaluateall1dpolynomialderivative!(
                (values, derivative_values), poly, x, 1
            )
            @test sum(values) ≈ 1.0 atol = 1e-12
            @test sum(derivative_values) ≈ 0.0 atol = 1e-10
        end
    end

    @testset verbose=true "interpolation matrix" begin
        for p in 1:4
            for b in (p + 1):(p + 6)
                poly = MultilevelKernelInterpolation.BsplinePolynomialEquidistant{1,Float64}(
                    p, b
                )
                expected = MultilevelKernelInterpolation.evalbsplines(
                    MultilevelKernelInterpolation.bspline(poly, 1),
                    MultilevelKernelInterpolation.range(poly, 1),
                )

                @test Matrix(MultilevelKernelInterpolation.interpolationmatrix(poly)) ≈
                    expected atol = 1e-12
                @test Matrix(
                    MultilevelKernelInterpolation.transposeinterpolationmatrix(poly)
                ) ≈ transpose(expected) atol = 1e-12
            end
        end
    end

    @testset verbose=true "buffered interpolation matrix" begin
        for p in 1:4
            for b in (p + 1):(p + 6)
                for bufferpoints in 1:2
                    poly = MultilevelKernelInterpolation.BufferedBsplinePolynomialEquidistant{
                        1,Float64
                    }(
                        p, b, bufferpoints
                    )
                    interpolation = MultilevelKernelInterpolation.interpolationmatrix(poly)
                    transposed = MultilevelKernelInterpolation.transposeinterpolationmatrix(
                        poly
                    )

                    @test size(interpolation) == (
                        length(MultilevelKernelInterpolation.range(poly, 1)),
                        length(MultilevelKernelInterpolation.range(poly, 1)),
                    )
                    @test Matrix(transposed) ≈ transpose(Matrix(interpolation)) atol = 1e-12
                    @test all(isfinite, Matrix(interpolation))
                end
            end
        end
    end

    @testset verbose=true "buffered basis contract" begin
        poly = MultilevelKernelInterpolation.BufferedBsplinePolynomialEquidistant{2,Float64}(
            3, 8, 2
        )

        @test MultilevelKernelInterpolation.dims(poly) == 2
        @test MultilevelKernelInterpolation.numbufferpoints(poly) == 2
        @test MultilevelKernelInterpolation.polynomialorders(poly) == (3, 3)
        @test MultilevelKernelInterpolation.numpolynomials(poly) == (12, 12)
        @test first(MultilevelKernelInterpolation.range(poly, 1)) < 0
        @test last(MultilevelKernelInterpolation.range(poly, 1)) > 1
    end

    # SWEPT OVER THE RANK rather than fixed at three. `separable1dinterpolations!` applies the 1-D
    # matrix along one axis at a time, staging through `D-1` intermediate buffers, so the buffer
    # COUNT, not just the indexing, changes with the dimension, and two ranks are the smallest
    # sweep that can tell a rank-generic implementation from one that happens to be right at three.
    @testset verbose=true "$(D)-D interpolation inversion" for D in (2, 3)
        for Poly in (
            (p, b) ->
                MultilevelKernelInterpolation.BsplinePolynomialEquidistant{D,Float64}(p, b),
            (p, b) -> MultilevelKernelInterpolation.BufferedBsplinePolynomialEquidistant{
                D,Float64
            }(
                p, b, 1
            ),
        )
            for p in 2:4
                for b in (p + 1):(p + 3)
                    poly = Poly(p, b)
                    interpolation = MultilevelKernelInterpolation.interpolationmatrix(poly)
                    interpolationnd = MultilevelKernelInterpolation.computend(
                        ntuple(_ -> Matrix(interpolation), D)...
                    )
                    inverse_interpolation = pinv(Matrix(interpolation))
                    static_inverse_interpolation = SMatrix{size(interpolation)...}(
                        inverse_interpolation
                    )
                    inverse_interpolationnd = MultilevelKernelInterpolation.computend(
                        ntuple(_ -> inverse_interpolation, D)...
                    )
                    interpolation_size = ntuple(_ -> size(interpolation, 1), D)
                    x = reshape(ComplexF64.(1:prod(interpolation_size)), interpolation_size)
                    by_1d_inverse = similar(x)
                    by_nd_inverse = similar(x)

                    # The two halves of the crossover: the 1-D matrix applied separably per axis
                    # (`SMatrix`), and its `D`-fold Kronecker product applied flat (`Matrix`).
                    MultilevelKernelInterpolation.separable1dinterpolations!(
                        by_1d_inverse, static_inverse_interpolation, x
                    )
                    MultilevelKernelInterpolation.separable1dinterpolations!(
                        by_nd_inverse, inverse_interpolationnd, x
                    )

                    @test interpolationnd * vec(by_1d_inverse) ≈ vec(x) rtol = 1e-10
                    @test interpolationnd * vec(by_nd_inverse) ≈ vec(x) rtol = 1e-10

                    # THE FORM EVERY PRODUCTION CALLER ACTUALLY USES. The buffer-free call above
                    # allocates its own scratch; the modulation paths hand in a preallocated set
                    # from a channel, so the two must agree bit for bit, a scratch sized or
                    # ordered wrongly is otherwise only visible inside a running matvec.
                    scratch = MultilevelKernelInterpolation.separablescratch(
                        ComplexF64, static_inverse_interpolation, D
                    )
                    @test length(scratch) == D - 1
                    withscratch = similar(x)
                    MultilevelKernelInterpolation.separable1dinterpolations!(
                        withscratch, static_inverse_interpolation, x, scratch
                    )
                    @test withscratch == by_1d_inverse
                end
            end
        end
    end
end

end
