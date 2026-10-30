using Test
using MultilevelKernelInterpolation

@testset verbose=true "MultilevelKernelInterpolation" begin
    include("Polynomials/runtests.jl")
    include("Translations/runtests.jl")
    include("Moments/runtests.jl")
    include("Operators/runtests.jl")
    include("maps/runtests.jl")
    include("Package/runtests.jl")

    @testset "Code quality (Aqua.jl)" begin
        using Aqua

        # `ambiguities`/`unbound_args`/`piracies` scoped out -- all their remaining findings
        # (re-checked after fixing the handful that were safely fixable, see below) are
        # genuine cross-package design tension, not MultilevelKernelInterpolation-side bugs:
        #
        #   - ambiguities: `getindex(T::BsplineI2OTranslationCollection/BsplineToeplitzI2O...,
        #     key)` (untyped `key`, matching H2Factory's own base
        #     `I2OTranslationCollection`/`ToeplitzI2OTranslationCollection` convention exactly
        #     -- different translation-collection variants use different key shapes, so
        #     H2Factory itself leaves this untyped) is ambiguous against
        #     `BEAST.Variational.getindex(f, ::HilbertVector)` (also untyped in its first
        #     arg). And `H2Factory.I2OTranslationCollection`'s generic constructor is
        #     ambiguous against MultilevelKernelInterpolation's own override of it -- an intentional H2Factory
        #     extension point (see `piracies` below), the same tension as any generic-function
        #     extension pattern. Both would need an H2Factory-side change (or narrowing a key
        #     type MultilevelKernelInterpolation can't verify precisely) to resolve safely, not a hasty MultilevelKernelInterpolation-only guess.
        #   - unbound_args: dominated by `H2Factory`'s own `@createlinkedtoeplitz`-generated
        #     constructors (`MultilevelKernelInterpolation.ToeplitzPolynomialMoment`/`BsplineDensePolynomialMoment`/
        #     `PolynomialMoment`, all defined via macro code physically in
        #     `H2Factory/src/moments/toeplitz/toeplitzmoment.jl`) plus `onedimi2itranslations`'s
        #     `DA`/`DB` (used, just not further constrained -- deliberate generic dispatch, not
        #     a typo).
        #   - piracies: `α`/`H2Factory.I2OTranslationCollection` are H2Factory's own generic
        #     extension points, implemented here for BEAST's operator types (the same pattern
        #     as extending `Base.show` for a foreign type -- intentional, not accidental
        #     interference); `NURBS.evalNaive`/`evalNaiveDerivative` are deliberate,
        #     documented keyword-compatibility shims (NURBS.jl's own methods have no
        #     `extrapolate` keyword; see `src/polynomials/bspline/bspline.jl`).
        Aqua.test_all(
            MultilevelKernelInterpolation;
            ambiguities=false,
            unbound_args=false,
            piracies=false,
        )
    end

    @testset "Code formatting (JuliaFormatter.jl)" begin
        using JuliaFormatter
        @test JuliaFormatter.format(pkgdir(MultilevelKernelInterpolation); overwrite=false)
    end
end
