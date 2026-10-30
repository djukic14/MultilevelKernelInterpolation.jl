module TestBeastOperators

using Test
using MultilevelKernelInterpolation
using BEAST

@testset verbose=true "BEAST operators Helmholtz3D" begin
    poly = MultilevelKernelInterpolation.BarycentricLagrangePolynomialEquidistant3D(4)

    ## singlelayer
    @test MultilevelKernelInterpolation.testmoment(
        Helmholtz3D.singlelayer(; wavenumber=1.0), poly
    ) == MultilevelKernelInterpolation.LaplaceMoment{1,Float64}()
    @test MultilevelKernelInterpolation.trialmoment(
        Helmholtz3D.singlelayer(; wavenumber=1.0), poly
    ) == MultilevelKernelInterpolation.LaplaceMoment{1,Float64}()
    @test MultilevelKernelInterpolation.moment(
            Helmholtz3D.singlelayer(; wavenumber=1.0), poly
        ) ==
        MultilevelKernelInterpolation.testmoment(
            Helmholtz3D.singlelayer(; wavenumber=1.0), poly
        ) ==
        MultilevelKernelInterpolation.trialmoment(
            Helmholtz3D.singlelayer(; wavenumber=1.0), poly
        )
    @test MultilevelKernelInterpolation.α(
        Helmholtz3D.singlelayer(; wavenumber=1.0, alpha=2.0), poly
    ) == 2.0

    ## doublelayer
    @test MultilevelKernelInterpolation.testmoment(
        Helmholtz3D.doublelayer(; wavenumber=1.0), poly
    ) == MultilevelKernelInterpolation.LaplaceMoment{1,Float64}()
    @test MultilevelKernelInterpolation.trialmoment(
        Helmholtz3D.doublelayer(; wavenumber=1.0), poly
    ) == MultilevelKernelInterpolation.NormalGradLaplaceMoment{1,Float64}()
    @test_throws MethodError MultilevelKernelInterpolation.moment(
        Helmholtz3D.doublelayer(; wavenumber=1.0), poly
    )
    @test MultilevelKernelInterpolation.α(
        Helmholtz3D.doublelayer(; wavenumber=1.0, alpha=2.0), poly
    ) == 2.0

    ## doublelayer transposed
    @test MultilevelKernelInterpolation.testmoment(
        Helmholtz3D.doublelayer_transposed(; wavenumber=1.0), poly
    ) == MultilevelKernelInterpolation.NormalGradLaplaceMoment{1,Float64}()
    @test MultilevelKernelInterpolation.trialmoment(
        Helmholtz3D.doublelayer_transposed(; wavenumber=1.0), poly
    ) == MultilevelKernelInterpolation.LaplaceMoment{1,Float64}()
    @test_throws MethodError MultilevelKernelInterpolation.moment(
        Helmholtz3D.doublelayer_transposed(; wavenumber=1.0), poly
    )
    @test MultilevelKernelInterpolation.α(
        Helmholtz3D.doublelayer_transposed(; wavenumber=1.0, alpha=2.0), poly
    ) == 2.0

    ## MultilevelKernelInterpolation doublelayer
    @test MultilevelKernelInterpolation.testmoment(
        MultilevelKernelInterpolation.HH3DDoubleLayerFDBIOGradientI2O(; wavenumber=1.0),
        poly,
    ) == MultilevelKernelInterpolation.LaplaceMoment{1,Float64}()
    @test MultilevelKernelInterpolation.trialmoment(
        MultilevelKernelInterpolation.HH3DDoubleLayerFDBIOGradientI2O(; wavenumber=1.0),
        poly,
    ) == MultilevelKernelInterpolation.NormalLaplaceMoment{3,Float64}()
    @test_throws MethodError MultilevelKernelInterpolation.moment(
        MultilevelKernelInterpolation.HH3DDoubleLayerFDBIOGradientI2O(; wavenumber=1.0),
        poly,
    )
    @test MultilevelKernelInterpolation.α(
        MultilevelKernelInterpolation.HH3DDoubleLayerFDBIOGradientI2O(;
            wavenumber=1.0, alpha=2.0
        ),
        poly,
    ) == -2.0

    ## MultilevelKernelInterpolation doublelayer transposed
    @test MultilevelKernelInterpolation.testmoment(
        MultilevelKernelInterpolation.HH3DDoubleLayerTransposedFDBIOGradientI2O(;
            wavenumber=1.0
        ),
        poly,
    ) == MultilevelKernelInterpolation.NormalLaplaceMoment{3,Float64}()
    @test MultilevelKernelInterpolation.trialmoment(
        MultilevelKernelInterpolation.HH3DDoubleLayerTransposedFDBIOGradientI2O(;
            wavenumber=1.0
        ),
        poly,
    ) == MultilevelKernelInterpolation.LaplaceMoment{1,Float64}()
    @test_throws MethodError MultilevelKernelInterpolation.moment(
        MultilevelKernelInterpolation.HH3DDoubleLayerTransposedFDBIOGradientI2O(;
            wavenumber=1.0
        ),
        poly,
    )
    @test MultilevelKernelInterpolation.α(
        MultilevelKernelInterpolation.HH3DDoubleLayerTransposedFDBIOGradientI2O(;
            wavenumber=1.0, alpha=2.0
        ),
        poly,
    ) == 2.0

    ## hypersingular
    @test MultilevelKernelInterpolation.testmoment(
        Helmholtz3D.hypersingular(; wavenumber=1.0), poly
    ) == MultilevelKernelInterpolation.HyperSingularMoment{6,Float64}()
    @test MultilevelKernelInterpolation.trialmoment(
        Helmholtz3D.hypersingular(; wavenumber=1.0), poly
    ) == MultilevelKernelInterpolation.HyperSingularMoment{6,Float64}()
    @test MultilevelKernelInterpolation.moment(
            Helmholtz3D.hypersingular(; wavenumber=1.0), poly
        ) ==
        MultilevelKernelInterpolation.testmoment(
            Helmholtz3D.hypersingular(; wavenumber=1.0), poly
        ) ==
        MultilevelKernelInterpolation.trialmoment(
            Helmholtz3D.hypersingular(; wavenumber=1.0), poly
        )
    @test MultilevelKernelInterpolation.α(
        Helmholtz3D.hypersingular(; wavenumber=1.0), poly
    ) == [
        Helmholtz3D.hypersingular(; wavenumber=1.0).alpha,
        Helmholtz3D.hypersingular(; wavenumber=1.0).alpha,
        Helmholtz3D.hypersingular(; wavenumber=1.0).alpha,
        Helmholtz3D.hypersingular(; wavenumber=1.0).beta,
        Helmholtz3D.hypersingular(; wavenumber=1.0).beta,
        Helmholtz3D.hypersingular(; wavenumber=1.0).beta,
    ]
end

@testset verbose=true "BEAST operators Maxwell3D" begin
    poly = MultilevelKernelInterpolation.BarycentricLagrangePolynomialEquidistant3D(4)

    ## singlelayer
    @test MultilevelKernelInterpolation.testmoment(
        Maxwell3D.singlelayer(; wavenumber=1.0), poly
    ) == MultilevelKernelInterpolation.LaplaceMomentDiv{4,Float64}()
    @test MultilevelKernelInterpolation.trialmoment(
        Maxwell3D.singlelayer(; wavenumber=1.0), poly
    ) == MultilevelKernelInterpolation.LaplaceMomentDiv{4,Float64}()
    @test MultilevelKernelInterpolation.moment(
            Maxwell3D.singlelayer(; wavenumber=1.0), poly
        ) ==
        MultilevelKernelInterpolation.testmoment(
            Maxwell3D.singlelayer(; wavenumber=1.0), poly
        ) ==
        MultilevelKernelInterpolation.trialmoment(
            Maxwell3D.singlelayer(; wavenumber=1.0), poly
        )
    @test MultilevelKernelInterpolation.α(
        Maxwell3D.singlelayer(; wavenumber=1.0, alpha=Complex(2.0)), poly
    ) == [
        Maxwell3D.singlelayer(; wavenumber=1.0, alpha=Complex(2.0)).α,
        Maxwell3D.singlelayer(; wavenumber=1.0, alpha=Complex(2.0)).α,
        Maxwell3D.singlelayer(; wavenumber=1.0, alpha=Complex(2.0)).α,
        Maxwell3D.singlelayer(; wavenumber=1.0, alpha=Complex(2.0)).β,
    ]

    ## efie
    @test MultilevelKernelInterpolation.testmoment(
        MultilevelKernelInterpolation.efie(; wavenumber=1.0), poly
    ) == MultilevelKernelInterpolation.LaplaceMoment{3,Float64}()
    @test MultilevelKernelInterpolation.trialmoment(
        MultilevelKernelInterpolation.efie(; wavenumber=1.0), poly
    ) == MultilevelKernelInterpolation.LaplaceMoment{3,Float64}()
    @test MultilevelKernelInterpolation.moment(
            MultilevelKernelInterpolation.efie(; wavenumber=1.0), poly
        ) ==
        MultilevelKernelInterpolation.testmoment(
            MultilevelKernelInterpolation.efie(; wavenumber=1.0), poly
        ) ==
        MultilevelKernelInterpolation.trialmoment(
            MultilevelKernelInterpolation.efie(; wavenumber=1.0), poly
        )
    @test MultilevelKernelInterpolation.α(
        MultilevelKernelInterpolation.efie(; wavenumber=1.0, alpha=2.0), poly
    ) == 2.0

    ## hypersingular
    @test MultilevelKernelInterpolation.testmoment(
        MultilevelKernelInterpolation.MWHyperSingular3D(; wavenumber=1.0), poly
    ) == MultilevelKernelInterpolation.LaplaceMoment{3,Float64}()
    @test MultilevelKernelInterpolation.trialmoment(
        MultilevelKernelInterpolation.MWHyperSingular3D(; wavenumber=1.0), poly
    ) == MultilevelKernelInterpolation.LaplaceMoment{3,Float64}()
    @test MultilevelKernelInterpolation.moment(
            MultilevelKernelInterpolation.MWHyperSingular3D(; wavenumber=1.0), poly
        ) ==
        MultilevelKernelInterpolation.testmoment(
            MultilevelKernelInterpolation.MWHyperSingular3D(; wavenumber=1.0), poly
        ) ==
        MultilevelKernelInterpolation.trialmoment(
            MultilevelKernelInterpolation.MWHyperSingular3D(; wavenumber=1.0), poly
        )
    @test MultilevelKernelInterpolation.α(
        MultilevelKernelInterpolation.MWHyperSingular3D(; wavenumber=1.0), poly
    ) == im * 1.0

    ## weakly singular
    @test MultilevelKernelInterpolation.testmoment(
        MultilevelKernelInterpolation.MWWeaklySingular3D(; wavenumber=1.0), poly
    ) == MultilevelKernelInterpolation.LaplaceMoment{3,Float64}()
    @test MultilevelKernelInterpolation.trialmoment(
        MultilevelKernelInterpolation.MWWeaklySingular3D(; wavenumber=1.0), poly
    ) == MultilevelKernelInterpolation.LaplaceMoment{3,Float64}()
    @test MultilevelKernelInterpolation.moment(
            MultilevelKernelInterpolation.MWWeaklySingular3D(; wavenumber=1.0), poly
        ) ==
        MultilevelKernelInterpolation.testmoment(
            MultilevelKernelInterpolation.MWWeaklySingular3D(; wavenumber=1.0), poly
        ) ==
        MultilevelKernelInterpolation.trialmoment(
            MultilevelKernelInterpolation.MWWeaklySingular3D(; wavenumber=1.0), poly
        )
    @test MultilevelKernelInterpolation.α(
        MultilevelKernelInterpolation.MWWeaklySingular3D(; wavenumber=1.0), poly
    ) == -im * 1.0

    ## doublelayer
    @test MultilevelKernelInterpolation.testmoment(
        Maxwell3D.doublelayer(; wavenumber=1.0), poly
    ) == MultilevelKernelInterpolation.GradLaplaceMoment{3,Float64}()
    @test MultilevelKernelInterpolation.trialmoment(
        Maxwell3D.doublelayer(; wavenumber=1.0), poly
    ) == MultilevelKernelInterpolation.LaplaceMoment{3,Float64}()
    @test_throws MethodError MultilevelKernelInterpolation.moment(
        Maxwell3D.doublelayer(; wavenumber=1.0), poly
    )
    @test MultilevelKernelInterpolation.α(
        Maxwell3D.doublelayer(; wavenumber=1.0, alpha=2.0), poly
    ) == 2.0

    ## MultilevelKernelInterpolation doublelayer
    @test MultilevelKernelInterpolation.testmoment(
        MultilevelKernelInterpolation.MWDoubleLayer3D(; wavenumber=1.0), poly
    ) == MultilevelKernelInterpolation.LaplaceMoment{3,Float64}()
    @test MultilevelKernelInterpolation.trialmoment(
        MultilevelKernelInterpolation.MWDoubleLayer3D(; wavenumber=1.0), poly
    ) == MultilevelKernelInterpolation.LaplaceMoment{3,Float64}()
    @test MultilevelKernelInterpolation.moment(
            MultilevelKernelInterpolation.MWDoubleLayer3D(; wavenumber=1.0), poly
        ) ==
        MultilevelKernelInterpolation.testmoment(
            MultilevelKernelInterpolation.MWDoubleLayer3D(; wavenumber=1.0), poly
        ) ==
        MultilevelKernelInterpolation.trialmoment(
            MultilevelKernelInterpolation.MWDoubleLayer3D(; wavenumber=1.0), poly
        )
    @test MultilevelKernelInterpolation.α(
        MultilevelKernelInterpolation.MWDoubleLayer3D(; wavenumber=1.0, alpha=2.0), poly
    ) == 2.0

    ## doublelayer rotated
    @test MultilevelKernelInterpolation.testmoment(
        BEAST.DoubleLayerRotatedMW3D(1.0, im * 1.0), poly
    ) == MultilevelKernelInterpolation.NormalGradLaplaceMoment{3,Float64}()
    @test MultilevelKernelInterpolation.trialmoment(
        BEAST.DoubleLayerRotatedMW3D(1.0, im * 1.0), poly
    ) == MultilevelKernelInterpolation.LaplaceMoment{3,Float64}()
    @test_throws MethodError MultilevelKernelInterpolation.moment(
        BEAST.DoubleLayerRotatedMW3D(1.0, im * 1.0), poly
    )
    @test MultilevelKernelInterpolation.α(
        BEAST.DoubleLayerRotatedMW3D(2.0, im * 1.0), poly
    ) == 2.0
end

@testset verbose=true "I2Otranslation types" begin
    @testset "Helmholtz3D" begin
        @test MultilevelKernelInterpolation.i2otranslation(
            Helmholtz3D.singlelayer(; wavenumber=1.0)
        ) == MultilevelKernelInterpolation.ScalarGreen3DTranslation()
        @test MultilevelKernelInterpolation.i2otranslation(
            Helmholtz3D.doublelayer(; wavenumber=1.0)
        ) == MultilevelKernelInterpolation.ScalarGreen3DTranslation()
        @test MultilevelKernelInterpolation.i2otranslation(
            Helmholtz3D.doublelayer_transposed(; wavenumber=1.0)
        ) == MultilevelKernelInterpolation.ScalarGreen3DTranslation()
        @test MultilevelKernelInterpolation.i2otranslation(
            Helmholtz3D.hypersingular(; wavenumber=1.0)
        ) == MultilevelKernelInterpolation.ScalarGreen3DTranslation()

        @test MultilevelKernelInterpolation.i2otranslation(
            MultilevelKernelInterpolation.HH3DDoubleLayerFDBIOGradientI2O(; wavenumber=1.0)
        ) == MultilevelKernelInterpolation.∇Green3DTranslation()
        @test MultilevelKernelInterpolation.i2otranslation(
            MultilevelKernelInterpolation.HH3DDoubleLayerTransposedFDBIOGradientI2O(;
                wavenumber=1.0
            ),
        ) == MultilevelKernelInterpolation.∇Green3DTranslation()
    end

    @testset "Maxwell3D" begin
        @test MultilevelKernelInterpolation.i2otranslation(
            Maxwell3D.singlelayer(; wavenumber=1.0)
        ) == MultilevelKernelInterpolation.ScalarGreen3DTranslation()
        @test MultilevelKernelInterpolation.i2otranslation(
            Maxwell3D.doublelayer(; wavenumber=1.0)
        ) == MultilevelKernelInterpolation.ScalarGreen3DTranslation()
        @test MultilevelKernelInterpolation.i2otranslation(
            Maxwell3D.hypersingular(; wavenumber=1.0)
        ) == MultilevelKernelInterpolation.ScalarGreen3DTranslation()
        @test MultilevelKernelInterpolation.i2otranslation(
            MultilevelKernelInterpolation.MWWeaklySingular3D(; wavenumber=1.0)
        ) == MultilevelKernelInterpolation.ScalarGreen3DTranslation()
        @test MultilevelKernelInterpolation.i2otranslation(
            BEAST.DoubleLayerRotatedMW3D(1.0, im * 1.0)
        ) == MultilevelKernelInterpolation.ScalarGreen3DTranslation()

        @test MultilevelKernelInterpolation.i2otranslation(
            MultilevelKernelInterpolation.efie(; wavenumber=1.0)
        ) == MultilevelKernelInterpolation.DyadicGreen3DTranslation()
        @test MultilevelKernelInterpolation.i2otranslation(
            MultilevelKernelInterpolation.MWHyperSingular3D(; wavenumber=1.0)
        ) == MultilevelKernelInterpolation.∇∇Green3DTranslation()
        @test MultilevelKernelInterpolation.i2otranslation(
            MultilevelKernelInterpolation.MWDoubleLayer3D(; wavenumber=1.0)
        ) == MultilevelKernelInterpolation.∇CrossGreen3DTranslation()
    end

    @testset "I2Otranslationtrait" begin
        @test MultilevelKernelInterpolation.i2otranslationtrait(
            MultilevelKernelInterpolation.ScalarGreen3DTranslation()
        ) == MultilevelKernelInterpolation.DenseTranslation()

        @test MultilevelKernelInterpolation.i2otranslationtrait(
            MultilevelKernelInterpolation.∇Green3DTranslation()
        ) == MultilevelKernelInterpolation.DenseTranslation()

        @test MultilevelKernelInterpolation.i2otranslationtrait(
            MultilevelKernelInterpolation.∇∇Green3DTranslation()
        ) == MultilevelKernelInterpolation.DenseTranslation()

        @test MultilevelKernelInterpolation.i2otranslationtrait(
            MultilevelKernelInterpolation.DyadicGreen3DTranslation()
        ) == MultilevelKernelInterpolation.DenseTranslation()

        @test MultilevelKernelInterpolation.i2otranslationtrait(
            MultilevelKernelInterpolation.∇CrossGreen3DTranslation()
        ) == MultilevelKernelInterpolation.DenseTranslation()

        struct FakeTranslator{T}
            operator::T
        end

        @test MultilevelKernelInterpolation.eltypei2otranslation(
            MultilevelKernelInterpolation.ScalarGreen3DTranslation(),
            FakeTranslator(Maxwell3D.singlelayer(; wavenumber=1.0)),
        ) == Matrix{ComplexF64}

        @test MultilevelKernelInterpolation.eltypei2otranslation(
            MultilevelKernelInterpolation.∇Green3DTranslation(),
            FakeTranslator(Maxwell3D.singlelayer(; wavenumber=1.0)),
        ) == Vector{Matrix{ComplexF64}}

        @test MultilevelKernelInterpolation.eltypei2otranslation(
            MultilevelKernelInterpolation.∇∇Green3DTranslation(),
            FakeTranslator(Maxwell3D.singlelayer(; wavenumber=1.0)),
        ) == Matrix{Matrix{ComplexF64}}

        @test MultilevelKernelInterpolation.eltypei2otranslation(
            MultilevelKernelInterpolation.DyadicGreen3DTranslation(),
            FakeTranslator(Maxwell3D.singlelayer(; wavenumber=1.0)),
        ) == Matrix{Matrix{ComplexF64}}

        @test MultilevelKernelInterpolation.eltypei2otranslation(
            MultilevelKernelInterpolation.∇CrossGreen3DTranslation(),
            FakeTranslator(Maxwell3D.singlelayer(; wavenumber=1.0)),
        ) == Matrix{Matrix{ComplexF64}}
    end
end

end
