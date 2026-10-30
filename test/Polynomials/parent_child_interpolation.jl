module TestParentChildInterpolation

using Test
using MultilevelKernelInterpolation

const LAGRANGE_TYPES = (
    MultilevelKernelInterpolation.BarycentricLagrangePolynomialEquidistant,
    MultilevelKernelInterpolation.BarycentricLagrangePolynomialChebyshev,
    MultilevelKernelInterpolation.BarycentricLagrangePolynomialChebyshev2,
)

function assert_child_interpolates_parent(
    child, parent, evalpoints_left, evalpoints_right; atol=1e-10
)
    i2i_left, i2i_right = MultilevelKernelInterpolation.onedimi2itranslations(child, parent)

    parent_values = zeros(
        Float64,
        MultilevelKernelInterpolation.numpolynomials(parent)[1],
        length(evalpoints_left),
    )
    child_values = zeros(
        Float64,
        MultilevelKernelInterpolation.numpolynomials(child)[1],
        length(evalpoints_left),
    )

    for (i, x) in enumerate(evalpoints_left)
        MultilevelKernelInterpolation.evaluateall1dpolynomial!(
            view(parent_values, :, i), parent, x, 1; extrapolate=false
        )
        MultilevelKernelInterpolation.evaluateall1dpolynomial!(
            view(child_values, :, i), child, 2x, 1; extrapolate=false
        )
    end
    @test parent_values ≈ i2i_left * child_values atol = atol

    parent_values = zeros(
        Float64,
        MultilevelKernelInterpolation.numpolynomials(parent)[1],
        length(evalpoints_right),
    )
    child_values = zeros(
        Float64,
        MultilevelKernelInterpolation.numpolynomials(child)[1],
        length(evalpoints_right),
    )

    for (i, x) in enumerate(evalpoints_right)
        MultilevelKernelInterpolation.evaluateall1dpolynomial!(
            view(parent_values, :, i), parent, x, 1; extrapolate=false
        )
        MultilevelKernelInterpolation.evaluateall1dpolynomial!(
            view(child_values, :, i), child, 2x - 1, 1; extrapolate=false
        )
    end
    @test parent_values ≈ i2i_right * child_values atol = atol
end

function interior(points)
    return points[(begin + 1):(end - 1)]
end

@testset verbose=true "Parent-child interpolation" begin
    @testset verbose=true "Lagrange constant order" begin
        for Poly in LAGRANGE_TYPES
            for p in 1:8
                poly = Poly{1,Float64}(p)
                left = range(
                    first(MultilevelKernelInterpolation.range(poly, 1)) / 2,
                    last(MultilevelKernelInterpolation.range(poly, 1)) / 2;
                    length=33,
                )
                right = range(
                    (first(MultilevelKernelInterpolation.range(poly, 1)) + 1) / 2,
                    (last(MultilevelKernelInterpolation.range(poly, 1)) + 1) / 2;
                    length=33,
                )

                assert_child_interpolates_parent(poly, poly, left, right)
            end
        end
    end

    @testset verbose=true "Lagrange increasing order" begin
        for Poly in LAGRANGE_TYPES
            for p in 1:8
                child = Poly{1,Float64}(p)
                parent = MultilevelKernelInterpolation.increase(child)
                left = MultilevelKernelInterpolation.range(child, 1) ./ 2
                right = (MultilevelKernelInterpolation.range(child, 1) .+ 1) ./ 2

                assert_child_interpolates_parent(child, parent, left, right)
            end
        end
    end

    @testset verbose=true "scaled Lagrange mixed parent" begin
        protrusion = 0.25
        child = MultilevelKernelInterpolation.ScaledShiftedPolynomial(
            MultilevelKernelInterpolation.BarycentricLagrangePolynomialEquidistant{
                3,Float64
            }(
                4
            ),
            1 + 2protrusion,
            fill(-protrusion, 3),
        )
        parent = MultilevelKernelInterpolation.ScaledShiftedPolynomial(
            MultilevelKernelInterpolation.BarycentricLagrangePolynomialChebyshev{3,Float64}(
                4
            ),
            1 + protrusion,
            fill(-protrusion / 2, 3),
        )

        left = range(-protrusion / 2, 0.5 - protrusion / 2; length=33)
        right = range(0.5 - protrusion / 2, 1 + protrusion / 2; length=33)

        assert_child_interpolates_parent(child, parent, left, right)
    end

    @testset verbose=true "B-spline constant order" begin
        for p in 1:4
            for b in (p + 1):(p + 8)
                poly = MultilevelKernelInterpolation.BsplinePolynomialEquidistant{1,Float64}(
                    p, b
                )
                left = range(
                    first(MultilevelKernelInterpolation.range(poly, 1)) / 2,
                    last(MultilevelKernelInterpolation.range(poly, 1)) / 2;
                    length=33,
                )
                right = range(
                    (first(MultilevelKernelInterpolation.range(poly, 1)) + 1) / 2,
                    (last(MultilevelKernelInterpolation.range(poly, 1)) + 1) / 2;
                    length=33,
                )

                assert_child_interpolates_parent(poly, poly, left, right)
            end
        end
    end

    @testset verbose=true "B-spline increasing order" begin
        for p in 1:4
            for b in (p + 1):(p + 8)
                child = MultilevelKernelInterpolation.BsplinePolynomialEquidistant{
                    1,Float64
                }(
                    p, b
                )
                parent = MultilevelKernelInterpolation.increase(child)
                left = range(
                    first(MultilevelKernelInterpolation.range(child, 1)) / 2,
                    last(MultilevelKernelInterpolation.range(child, 1)) / 2;
                    length=33,
                )
                right = range(
                    (first(MultilevelKernelInterpolation.range(child, 1)) + 1) / 2,
                    (last(MultilevelKernelInterpolation.range(child, 1)) + 1) / 2;
                    length=33,
                )

                assert_child_interpolates_parent(child, parent, left, right)
            end
        end
    end

    @testset verbose=true "buffered B-spline constant order" begin
        for p in 2:2
            for b in (p + 1):(p + 8)
                for bufferpoints in 1:2
                    poly = MultilevelKernelInterpolation.BufferedBsplinePolynomialEquidistant{
                        1,Float64
                    }(
                        p, b, bufferpoints
                    )
                    left = interior(
                        range(
                            first(MultilevelKernelInterpolation.range(poly, 1)) / 2,
                            last(MultilevelKernelInterpolation.range(poly, 1)) / 2;
                            length=35,
                        ),
                    )
                    right = interior(
                        range(
                            (first(MultilevelKernelInterpolation.range(poly, 1)) + 1) / 2,
                            (last(MultilevelKernelInterpolation.range(poly, 1)) + 1) / 2;
                            length=35,
                        ),
                    )

                    assert_child_interpolates_parent(poly, poly, left, right)
                end
            end
        end
    end

    @testset verbose=true "buffered B-spline increasing order" begin
        for p in 1:4
            for b in (p + 1):(p + 8)
                for bufferpoints in 1:2
                    child = MultilevelKernelInterpolation.BufferedBsplinePolynomialEquidistant{
                        1,Float64
                    }(
                        p, b, bufferpoints
                    )
                    parent = MultilevelKernelInterpolation.increase(child)
                    left = interior(
                        range(
                            first(MultilevelKernelInterpolation.range(child, 1)) / 2,
                            last(MultilevelKernelInterpolation.range(child, 1)) / 2;
                            length=35,
                        ),
                    )
                    right = interior(
                        range(
                            (first(MultilevelKernelInterpolation.range(child, 1)) + 1) / 2,
                            (last(MultilevelKernelInterpolation.range(child, 1)) + 1) / 2;
                            length=35,
                        ),
                    )

                    assert_child_interpolates_parent(child, parent, left, right)
                end
            end
        end
    end

    @testset verbose=true "i2icheapaveraging exactly reproduces the reduced-knot spline" begin
        # i2icheapaveraging removes knotstoremove(p, b) knot pairs near each boundary
        # to build a coarser `newspline`, then re-inserts them to express newspline's
        # basis functions as linear combinations of the original (fine) basis. Knot
        # insertion is exact for B-splines, so the reconstructed values must match
        # newspline's own basis evaluation to floating-point precision.
        #
        # Must use the b-aware MultilevelKernelInterpolation.knotstoremove(p, b), matching what
        # i2icheapaveraging itself now uses internally -- not the raw
        # knotstoremoveforstability[p] table lookup, which ignores b and, for small
        # b relative to p, requests removing more knot pairs than actually exist as
        # interior knots. That previously corrupted the knot vector's clamped
        # boundary and crashed NURBS.jl's insertKnot (fixed in refinement.jl by
        # switching i2icheapaveraging to the b-aware helper).
        points = collect(range(0.05, 0.95; length=21))

        for p in 3:6
            for b in (p + 3):(p + 12)
                knotstoremove = MultilevelKernelInterpolation.knotstoremove(p, b)

                poly = MultilevelKernelInterpolation.Bspline(
                    p, MultilevelKernelInterpolation.generateKnotVec(b, p)
                )
                res = MultilevelKernelInterpolation.i2icheapaveraging(poly)

                U2 = copy(poly.knotVec)
                for _ in 1:knotstoremove
                    popat!(U2, p + 2)
                    popat!(U2, length(U2) - p - 1)
                end
                newspline = MultilevelKernelInterpolation.Bspline(p, U2)

                @test size(res) ==
                    (MultilevelKernelInterpolation.NURBS.numBasisFunctions(newspline), b)

                newspline_values = [
                    MultilevelKernelInterpolation.NURBS.evalNaive(newspline, ib, x) for
                    x in points,
                    ib in 1:MultilevelKernelInterpolation.NURBS.numBasisFunctions(newspline)
                ]
                poly_values = [
                    MultilevelKernelInterpolation.NURBS.evalNaive(poly, j, x) for
                    x in points, j in 1:b
                ]
                reconstructed = poly_values * transpose(Matrix(res))

                @test newspline_values ≈ reconstructed atol = 1e-10
            end
        end

        @testset verbose=true "some (p, b) combinations legitimately remove nothing" begin
            # sanity check that the b-aware guard actually triggers within the swept
            # range above, not just in principle
            triggered = any(
                MultilevelKernelInterpolation.knotstoremove(p, b) == 0 for p in 3:6 for
                b in (p + 3):(p + 12)
            )
            @test triggered
        end
    end

    @testset verbose=true "suboptimali2iweights approximates i2iknotinsertion" begin
        # suboptimali2iweights computes I2I translation weights via least-squares
        # evaluation-matrix inversion instead of exact knot insertion -- it should
        # agree closely (not necessarily bit-exactly) with the exact i2iknotinsertion
        # result for both the constant-order and increasing-order cases.
        @testset verbose=true "constant order" begin
            for p in 1:3
                for b in (p + 1):(p + 6)
                    for bufferpoints in 1:2
                        poly = MultilevelKernelInterpolation.BufferedBsplinePolynomialEquidistant{
                            1,Float64
                        }(
                            p, b, bufferpoints
                        )
                        exact0, exact1 = MultilevelKernelInterpolation.i2iknotinsertion(
                            poly, poly
                        )
                        sub0, sub1 = MultilevelKernelInterpolation.suboptimali2iweights(
                            poly, poly
                        )

                        @test size(sub0) == size(exact0)
                        @test size(sub1) == size(exact1)
                        @test Matrix(sub0) ≈ Matrix(exact0) atol = 1e-8
                        @test Matrix(sub1) ≈ Matrix(exact1) atol = 1e-8
                    end
                end
            end
        end

        @testset verbose=true "increasing order" begin
            for p in 1:3
                for b in (p + 1):(p + 6)
                    for bufferpoints in 1:2
                        child = MultilevelKernelInterpolation.BufferedBsplinePolynomialEquidistant{
                            1,Float64
                        }(
                            p, b, bufferpoints
                        )
                        parent = MultilevelKernelInterpolation.increase(child)
                        exact0, exact1 = MultilevelKernelInterpolation.i2iknotinsertion(
                            child, parent
                        )
                        sub0, sub1 = MultilevelKernelInterpolation.suboptimali2iweights(
                            child, parent
                        )

                        @test size(sub0) == size(exact0)
                        @test size(sub1) == size(exact1)
                        @test Matrix(sub0) ≈ Matrix(exact0) atol = 1e-8
                        @test Matrix(sub1) ≈ Matrix(exact1) atol = 1e-8
                    end
                end
            end
        end
    end
end

end
