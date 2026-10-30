module TestKernelComputation

using Test
using CompScienceMeshes, BEAST
using H2Trees
using MultilevelKernelInterpolation
using LinearAlgebra

function _kernel(k, x, y)
    diff = norm(x - y)
    return cis(-k * diff) / (4 * pi * diff)
end

function _Greensdivdiv(k, x, y, dipoledir)
    R = x - y
    d = norm(R)
    kd = (k * d)
    kd² = kd^2
    ℓeᵣeᵣ = R * (dot(R, dipoledir)) / (d^2)

    return (
        _kernel(k, x, y) *
        ((3.0 / kd² + 3.0im / kd - 1) * ℓeᵣeᵣ + (-1.0 / kd² - 1.0im / kd + 1) * dipoledir)[:]
    )
end

@testset verbose=true "Kernels" begin
    @testset "Scalar Green's" begin
        λ = 1.0
        k = 2π / λ
        m = readmesh(
            joinpath(
                pkgdir(MultilevelKernelInterpolation), "test", "assets", "in", "cuboid.in"
            ),
        )

        X = raviartthomas(m)

        tree = buildtree(X; builder=TwoNTreeBuilder(; minhalfsize=λ / 10, minvalues=0))
        poly = MultilevelKernelInterpolation.BarycentricLagrangePolynomialEquidistant3D(3)

        op = Maxwell3D.singlelayer(; wavenumber=k)
        tf = Matrix{ComplexF64}(
            undef,
            prod(MultilevelKernelInterpolation.numpolynomials(poly)),
            prod(MultilevelKernelInterpolation.numpolynomials(poly)),
        )

        x = Vector{Float64}(undef, 3)
        y = Vector{Float64}(undef, 3)
        indices = LinearIndices(poly)

        for node in H2Trees.LevelIterator(tree, 3)[1:5]
            nodepoly = MultilevelKernelInterpolation.shiftedpolynomial(tree, poly, node)
            for farnode in H2Trees.WellSeparatedIterator(tree, node)
                farnodepoly = MultilevelKernelInterpolation.shiftedpolynomial(
                    tree, poly, farnode
                )

                MultilevelKernelInterpolation.translation!(
                    tf,
                    op,
                    MultilevelKernelInterpolation.ScalarGreen3DTranslation(),
                    nodepoly,
                    farnodepoly,
                )

                for j in indices
                    MultilevelKernelInterpolation.getsamplingpoint!(x, nodepoly, j)
                    for i in indices
                        MultilevelKernelInterpolation.getsamplingpoint!(y, farnodepoly, i)

                        @test tf[j, i] ≈ _kernel(k, x, y)
                    end
                end
            end
        end
    end

    @testset "Dyadic Green's" begin
        λ = 1.0
        k = 2π / λ
        m = readmesh(
            joinpath(
                pkgdir(MultilevelKernelInterpolation), "test", "assets", "in", "cuboid.in"
            ),
        )

        X = raviartthomas(m)

        tree = buildtree(X; builder=TwoNTreeBuilder(; minhalfsize=λ / 10, minvalues=0))
        poly = MultilevelKernelInterpolation.BarycentricLagrangePolynomialEquidistant3D(3)

        op = MultilevelKernelInterpolation.efie(; wavenumber=k)
        tf = [
            Matrix{ComplexF64}(
                undef,
                prod(MultilevelKernelInterpolation.numpolynomials(poly)),
                prod(MultilevelKernelInterpolation.numpolynomials(poly)),
            ) for _ in 1:3, _ in 1:3
        ]

        x = Vector{Float64}(undef, 3)
        y = Vector{Float64}(undef, 3)
        indices = LinearIndices(poly)

        for node in H2Trees.LevelIterator(tree, 3)[1:5]
            nodepoly = MultilevelKernelInterpolation.shiftedpolynomial(tree, poly, node)
            for farnode in H2Trees.WellSeparatedIterator(tree, node)
                farnodepoly = MultilevelKernelInterpolation.shiftedpolynomial(
                    tree, poly, farnode
                )

                MultilevelKernelInterpolation.translation!(
                    tf,
                    op,
                    MultilevelKernelInterpolation.DyadicGreen3DTranslation(),
                    nodepoly,
                    farnodepoly,
                )

                for j in indices
                    MultilevelKernelInterpolation.getsamplingpoint!(x, nodepoly, j)
                    for i in indices
                        MultilevelKernelInterpolation.getsamplingpoint!(y, farnodepoly, i)

                        ref1 = _Greensdivdiv(k, x, y, [1, 0, 0])

                        @test tf[1, 1][j, i] ≈ ref1[1]
                        @test tf[1, 2][j, i] ≈ ref1[2]
                        @test tf[1, 3][j, i] ≈ ref1[3]

                        ref2 = _Greensdivdiv(k, x, y, [0, 1, 0])

                        @test tf[2, 1][j, i] ≈ ref2[1]
                        @test tf[2, 2][j, i] ≈ ref2[2]
                        @test tf[2, 3][j, i] ≈ ref2[3]

                        ref3 = _Greensdivdiv(k, x, y, [0, 0, 1])

                        @test tf[3, 1][j, i] ≈ ref3[1]
                        @test tf[3, 2][j, i] ≈ ref3[2]
                        @test tf[3, 3][j, i] ≈ ref3[3]
                    end
                end

                MultilevelKernelInterpolation.translation!(
                    tf,
                    op,
                    MultilevelKernelInterpolation.∇∇Green3DTranslation(),
                    nodepoly,
                    farnodepoly,
                )

                for j in indices
                    MultilevelKernelInterpolation.getsamplingpoint!(x, nodepoly, j)
                    for i in indices
                        MultilevelKernelInterpolation.getsamplingpoint!(y, farnodepoly, i)

                        ref1 =
                            (
                                _Greensdivdiv(k, x, y, [1, 0, 0]) -
                                MultilevelKernelInterpolation.scalargreen3Dkernel(
                                    op, x, y
                                ) * [1, 0, 0]
                            ) * -k^2

                        @test tf[1, 1][j, i] ≈ ref1[1]
                        @test tf[1, 2][j, i] ≈ ref1[2]
                        @test tf[1, 3][j, i] ≈ ref1[3]

                        ref2 =
                            (
                                _Greensdivdiv(k, x, y, [0, 1, 0]) -
                                MultilevelKernelInterpolation.scalargreen3Dkernel(
                                    op, x, y
                                ) * [0, 1, 0]
                            ) * -k^2

                        @test tf[2, 1][j, i] ≈ ref2[1]
                        @test tf[2, 2][j, i] ≈ ref2[2]
                        @test tf[2, 3][j, i] ≈ ref2[3]

                        ref3 =
                            (
                                _Greensdivdiv(k, x, y, [0, 0, 1]) -
                                MultilevelKernelInterpolation.scalargreen3Dkernel(
                                    op, x, y
                                ) * [0, 0, 1]
                            ) * -k^2

                        @test tf[3, 1][j, i] ≈ ref3[1]
                        @test tf[3, 2][j, i] ≈ ref3[2]
                        @test tf[3, 3][j, i] ≈ ref3[3]
                    end
                end
            end
        end
    end
end

end
