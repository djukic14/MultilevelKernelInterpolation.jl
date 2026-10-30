# H2Factory.InteractionStrategy selects near/far pairs and traversal plans.

"""
    AbstractI2IStrategy

Strategy for choosing an I2I (aggregation) translator.
Define a subtype and a [`buildi2itranslator`](@ref) method to add an implementation.
"""
abstract type AbstractI2IStrategy end

"""
    DefaultI2IStrategy()

Store each I2I transfer as one tensor-product matrix with the `DenseI2I` trait.
For `N` nodes per axis in `D` dimensions, a dense matrix needs `O(N^(2D))` storage.
B-spline matrices can stay sparse because their basis functions have compact support.
"""
struct DefaultI2IStrategy <: AbstractI2IStrategy end

"""
    OneDimI2IStrategy()

Store one matrix per axis with the `SeparableI2I` trait and apply them successively.
Storage is `O(D*N^2)` for `N` nodes per axis, rather than `O(N^(2D))` for a combined
dense matrix. This saves memory at high Lagrange orders, but uses more, smaller operations.
[`DefaultO2OStrategy`](@ref) builds the matching O2O cache automatically.
"""
struct OneDimI2IStrategy <: AbstractI2IStrategy end

"""
    AbstractO2OStrategy

Strategy for choosing an O2O (disaggregation) translator through [`buildo2otranslator`](@ref).
"""
abstract type AbstractO2OStrategy end

"""
    DefaultO2OStrategy()

Selects [`O2OTranslator`](@ref) as the O2O (disaggregation) translator implementation.
"""
struct DefaultO2OStrategy <: AbstractO2OStrategy end

"""
    AbstractI2OStrategy

Strategy for choosing an I2O (far-field kernel) coupling through [`buildi2ocoupling`](@ref).
"""
abstract type AbstractI2OStrategy end

"""
    DenseI2OStrategy(; symmetryreduced=false)

Use dense [`I2OTranslationCollection`](@ref) couplings.

With `symmetryreduced=true`, store one matrix for each group of lattice directions related
by symmetry. Other directions reuse it through grid permutations, without an extra approximation.
In 3-D, these permutations have `O(b^3)` entries versus `O(b^6)` for a dense coupling.

Reduction supports Galerkin maps with non-B-spline polynomials and scalar or supported
3-by-3 kernels (`efie`, `MWHyperSingular3D`, `MWDoubleLayer3D`). Unsupported combinations
error; non-lattice trees use unreduced couplings. Reduction is off by default.
"""
struct DenseI2OStrategy <: AbstractI2OStrategy
    symmetryreduced::Bool
end

DenseI2OStrategy(; symmetryreduced=false) = DenseI2OStrategy(symmetryreduced)

"""
    ToeplitzI2OStrategy(; symmetryreduced=false)

Use FFT-accelerated [`ToeplitzI2OTranslationCollection`](@ref) couplings.

With `symmetryreduced=true`, directions related by lattice symmetry share one difference
tensor through index permutations. This adds no approximation. Reduction is off by default;
non-lattice displacements, such as offset block-tree roots, use unreduced couplings.
"""
struct ToeplitzI2OStrategy <: AbstractI2OStrategy
    symmetryreduced::Bool
end

ToeplitzI2OStrategy(; symmetryreduced=false) = ToeplitzI2OStrategy(symmetryreduced)

# Toeplitz needs a common grid on both Petrov sides; dense couplings allow separate grids.
# New strategies keep separate grids unless they opt in.
_sharedi2ogrids(::AbstractI2OStrategy) = false
_sharedi2ogrids(::ToeplitzI2OStrategy) = true

"""
    AbstractTranslatorStrategy

Configuration of I2I, O2O, and I2O translator strategies.
"""
abstract type AbstractTranslatorStrategy end

"""
    TranslatorStrategy(; i2istrategy, i2ostrategy, o2ostrategy)

Choose an implementation for each translation kind using strategy instances.
Defaults are `DefaultI2IStrategy()`, `DenseI2OStrategy()`, and `DefaultO2OStrategy()`.
"""
struct TranslatorStrategy{I2I,I2O,O2O} <: AbstractTranslatorStrategy
    i2istrategy::I2I
    i2ostrategy::I2O
    o2ostrategy::O2O
end

function TranslatorStrategy(;
    i2istrategy=DefaultI2IStrategy(),
    i2ostrategy=DenseI2OStrategy(),
    o2ostrategy=DefaultO2OStrategy(),
)
    return TranslatorStrategy(i2istrategy, i2ostrategy, o2ostrategy)
end

"""
    i2istrategy(strategy::AbstractTranslatorStrategy)

Return the I2I (aggregation) strategy.
"""
i2istrategy(strategy::AbstractTranslatorStrategy) = strategy.i2istrategy

"""
    i2ostrategy(strategy::AbstractTranslatorStrategy)

Return the I2O (far-field kernel) strategy.
"""
i2ostrategy(strategy::AbstractTranslatorStrategy) = strategy.i2ostrategy

"""
    o2ostrategy(strategy::AbstractTranslatorStrategy)

Return the O2O (disaggregation) strategy.
"""
o2ostrategy(strategy::AbstractTranslatorStrategy) = strategy.o2ostrategy

"""
    AbstractMomentAssemblyStrategy

Strategy for assembling leaf moments.
"""
abstract type AbstractMomentAssemblyStrategy end

"""
    MomentAssemblyStrategy(; MomentAssembler, momentquadstrategy)

Choose the moment assembler and its quadrature strategy.
Near interactions use a separate `H2Factory.AbstractNearInteractionsStrategy`.
"""
struct MomentAssemblyStrategy{MA,MQ} <: AbstractMomentAssemblyStrategy
    MomentAssembler::MA
    momentquadstrategy::MQ
end

function MomentAssemblyStrategy(;
    MomentAssembler=BEASTMomentAssembler, momentquadstrategy=MOMENTSQUADSTRATEGYRECOM
)
    return MomentAssemblyStrategy(MomentAssembler, momentquadstrategy)
end

"""
    MomentAssembler(strategy::AbstractMomentAssemblyStrategy)

Return the moment-assembler constructor.
"""
MomentAssembler(strategy::AbstractMomentAssemblyStrategy) = strategy.MomentAssembler

"""
    momentquadstrategy(strategy::AbstractMomentAssemblyStrategy)

Return the quadrature strategy for moment assembly.
"""
momentquadstrategy(strategy::AbstractMomentAssemblyStrategy) = strategy.momentquadstrategy

"""
    AbstractPolynomialStrategy

Strategy for choosing and growing the polynomial basis.
"""
abstract type AbstractPolynomialStrategy end

"""
    PolynomialStrategy(operator; polynomial, increase, hfhalfsize)

Choose the polynomial basis, its growth function, and the high-frequency box half-size threshold.
"""
struct PolynomialStrategy{P,I,H} <: AbstractPolynomialStrategy
    polynomial::P
    increase::I
    hfhalfsize::H
end

function PolynomialStrategy(
    operator;
    polynomial=POLYNOMIALRECOM,
    increase=MultilevelKernelInterpolation.increase,
    hfhalfsize=λ(operator) * HFHALFSIZERECOM,
)
    return PolynomialStrategy(polynomial, increase, hfhalfsize)
end

"""
    polynomial(strategy::AbstractPolynomialStrategy)

Return the polynomial basis.
"""
polynomial(strategy::AbstractPolynomialStrategy) = strategy.polynomial

"""
    increase(strategy::AbstractPolynomialStrategy)

Return the function that grows the basis between levels.
"""
increase(strategy::AbstractPolynomialStrategy) = strategy.increase

"""
    hfhalfsize(strategy::AbstractPolynomialStrategy)

Return the high-frequency box half-size threshold.
"""
hfhalfsize(strategy::AbstractPolynomialStrategy) = strategy.hfhalfsize

"""
    AbstractInterpolationStrategy

Configuration for building an interpolation map; see [`InterpolationStrategy`](@ref).
"""
abstract type AbstractInterpolationStrategy end

"""
    InterpolationStrategy(operator, testspace, trialspace=testspace; polynomialstrategy, momentassemblystrategy, nearinteractionsstrategy, nearfieldstrategy, translatorstrategy, scheduler)

Configure construction and scheduling for `GalerkinKernelInterpolation` or `PetrovGalerkinKernelInterpolation`.
Pass one space for Galerkin, or both test and trial spaces for Petrov-Galerkin.
The spaces determine the default near-interaction quadrature through `BEAST.defaultquadstrat`.

Defaults to order-6 equidistant Lagrange polynomials and FFT-accelerated Toeplitz I2O.
These differ from the standalone `PolynomialStrategy` and `TranslatorStrategy` defaults
(Chebyshev and dense I2O). Toeplitz requires an equidistant grid, so keep the polynomial
and I2O choices compatible when overriding either.
"""
struct InterpolationStrategy{P,M,N,F,T,S} <: AbstractInterpolationStrategy
    polynomialstrategy::P
    momentassemblystrategy::M
    nearinteractionsstrategy::N
    nearfieldstrategy::F
    translatorstrategy::T
    scheduler::S
end

function InterpolationStrategy(
    operator,
    testspace,
    trialspace=testspace;
    polynomialstrategy=PolynomialStrategy(
        operator; polynomial=BarycentricLagrangePolynomialEquidistant3D(6)
    ),
    momentassemblystrategy=MomentAssemblyStrategy(),
    nearinteractionsstrategy=BlockBEASTNearInteractionsStrategy(
        BEAST.defaultquadstrat(operator, testspace, trialspace)
    ),
    nearfieldstrategy=InteractionStrategy(),
    translatorstrategy=TranslatorStrategy(; i2ostrategy=ToeplitzI2OStrategy()),
    scheduler=DynamicScheduler(),
)
    return InterpolationStrategy(
        polynomialstrategy,
        momentassemblystrategy,
        nearinteractionsstrategy,
        nearfieldstrategy,
        translatorstrategy,
        scheduler,
    )
end

"""
    polynomialstrategy(strategy::AbstractInterpolationStrategy)

Return the polynomial strategy.
"""
polynomialstrategy(strategy::AbstractInterpolationStrategy) = strategy.polynomialstrategy

"""
    momentassemblystrategy(strategy::AbstractInterpolationStrategy)

Return the moment-assembly strategy.
"""
momentassemblystrategy(strategy::AbstractInterpolationStrategy) =
    strategy.momentassemblystrategy

"""
    nearinteractionsstrategy(strategy::AbstractInterpolationStrategy)

Return the near-interaction assembly strategy (`H2Factory.AbstractNearInteractionsStrategy`).
"""
nearinteractionsstrategy(strategy::AbstractInterpolationStrategy) =
    strategy.nearinteractionsstrategy

"""
    nearfieldstrategy(strategy::AbstractInterpolationStrategy)

Return the interaction strategy that selects near/far pairs and traversal plans.
"""
nearfieldstrategy(strategy::AbstractInterpolationStrategy) = strategy.nearfieldstrategy

"""
    translatorstrategy(strategy::AbstractInterpolationStrategy)

Return the translator strategy.
"""
translatorstrategy(strategy::AbstractInterpolationStrategy) = strategy.translatorstrategy

"""
    scheduler(strategy::AbstractInterpolationStrategy)

Return the execution scheduler. This is separate from `H2Factory.scheduler`,
which selects a scheduler for a constructed map and operation.
"""
scheduler(strategy::AbstractInterpolationStrategy) = strategy.scheduler
