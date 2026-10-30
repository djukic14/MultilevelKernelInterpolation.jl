
"""
    testmoment(operator, polynomial)

Return the test-side moment operator for this factorization.
The fallback calls `moment(operator, polynomial)`, using the same moment on both sides.
"""
function testmoment(operator, polynomial)
    return moment(operator, polynomial)
end

"""
    trialmoment(operator, polynomial)

Return the trial-side moment operator for this factorization.
The fallback calls `moment(operator, polynomial)`, using the same moment on both sides.
"""
function trialmoment(operator, polynomial)
    return moment(operator, polynomial)
end

# Helmholtz moments ########################################################################

function moment(op::BEAST.HH3DSingleLayerFDBIO, ::Any)
    return LaplaceMoment{1,real(BEAST.scalartype(op))}()
end

function testmoment(op::BEAST.HH3DDoubleLayerFDBIO, ::Any)
    return LaplaceMoment{1,real(BEAST.scalartype(op))}()
end

function trialmoment(op::BEAST.HH3DDoubleLayerFDBIO, ::Any)
    return NormalGradLaplaceMoment{1,real(BEAST.scalartype(op))}()
end

function trialmoment(op::BEAST.HH3DDoubleLayerTransposedFDBIO, ::Any)
    return LaplaceMoment{1,real(BEAST.scalartype(op))}()
end

function testmoment(op::BEAST.HH3DDoubleLayerTransposedFDBIO, ::Any)
    return NormalGradLaplaceMoment{1,real(BEAST.scalartype(op))}()
end

function trialmoment(
    op::MultilevelKernelInterpolation.HH3DDoubleLayerFDBIOGradientI2O, ::Any
)
    return NormalLaplaceMoment{3,real(BEAST.scalartype(op))}()
end

function testmoment(
    op::MultilevelKernelInterpolation.HH3DDoubleLayerFDBIOGradientI2O, ::Any
)
    return LaplaceMoment{1,real(BEAST.scalartype(op))}()
end

function trialmoment(
    op::MultilevelKernelInterpolation.HH3DDoubleLayerTransposedFDBIOGradientI2O, ::Any
)
    return LaplaceMoment{1,real(BEAST.scalartype(op))}()
end

function testmoment(
    op::MultilevelKernelInterpolation.HH3DDoubleLayerTransposedFDBIOGradientI2O, ::Any
)
    return NormalLaplaceMoment{3,real(BEAST.scalartype(op))}()
end

function moment(op::BEAST.HH3DHyperSingularFDBIO, ::Any)
    return HyperSingularMoment{6,real(BEAST.scalartype(op))}()
end

"""
    operatorgamma(operator)

Return the operator's `γ` parameter.
Delegate to `BEAST.gamma` except for 2-D Helmholtz operators, which expose only a field.
Keeping this accessor in MultilevelKernelInterpolation avoids adding a method to BEAST's function for its own type.
"""
operatorgamma(operator) = BEAST.gamma(operator)
operatorgamma(operator::BEAST.HelmholtzOperator2D) = operator.gamma

# 2-D scalar moments reuse the same integrands: the type parameter counts components,
# not spatial dimensions. HH2DHyperSingularFDBIO is unsupported because the 3-D
# HyperSingularMoment integrand uses a 3+3 normal/curl split.
function moment(op::BEAST.HH2DSingleLayerFDBIO, ::Any)
    return LaplaceMoment{1,real(BEAST.scalartype(op))}()
end

function testmoment(op::BEAST.HH2DDoubleLayerFDBIO, ::Any)
    return LaplaceMoment{1,real(BEAST.scalartype(op))}()
end

function trialmoment(op::BEAST.HH2DDoubleLayerFDBIO, ::Any)
    return NormalGradLaplaceMoment{1,real(BEAST.scalartype(op))}()
end

function trialmoment(op::BEAST.HH2DDoubleLayerTransposedFDBIO, ::Any)
    return LaplaceMoment{1,real(BEAST.scalartype(op))}()
end

function testmoment(op::BEAST.HH2DDoubleLayerTransposedFDBIO, ::Any)
    return NormalGradLaplaceMoment{1,real(BEAST.scalartype(op))}()
end

# Maxwell moments ##########################################################################

function moment(op::BEAST.MWSingleLayer3D, ::Any)
    # [x, y, z, div]
    return LaplaceMomentDiv{4,real(BEAST.scalartype(op))}()
end

function moment(
    op::Union{
        MWEFIE3D,
        MWHyperSingular3D,
        MWWeaklySingular3D,
        MultilevelKernelInterpolation.MWDoubleLayer3D,
    },
    ::Any,
)
    return LaplaceMoment{3,real(BEAST.scalartype(op))}()
end

function testmoment(op::BEAST.MWDoubleLayer3D, ::Any)
    return GradLaplaceMoment{3,real(BEAST.scalartype(op))}()
end

function trialmoment(op::BEAST.MWDoubleLayer3D, ::Any)
    return LaplaceMoment{3,real(BEAST.scalartype(op))}()
end

function trialmoment(op::BEAST.DoubleLayerRotatedMW3D, ::Any)
    return LaplaceMoment{3,real(BEAST.scalartype(op))}()
end

function testmoment(op::BEAST.DoubleLayerRotatedMW3D, ::Any)
    # [x, y, z]
    return NormalGradLaplaceMoment{3,real(BEAST.scalartype(op))}()
end

# α values #################################################################################

"""
    α(operator, polynomial)

Returns the factor in front of the factorization used for the given operator and polynomial.
"""
function α(
    operator::Union{
        BEAST.HH3DSingleLayerFDBIO,
        BEAST.HH3DDoubleLayerFDBIO,
        BEAST.HH3DDoubleLayerTransposedFDBIO,
        BEAST.MWDoubleLayer3D,
        BEAST.DoubleLayerRotatedMW3D,
        # BEAST uses the same double-layer sign convention in 2-D and 3-D.
        BEAST.HH2DSingleLayerFDBIO,
        BEAST.HH2DDoubleLayerFDBIO,
        BEAST.HH2DDoubleLayerTransposedFDBIO,
    },
    polynomial,
)
    return operator.alpha
end

function α(operator::BEAST.HH3DHyperSingularFDBIO, polynomial)
    return [
        operator.alpha,
        operator.alpha,
        operator.alpha,
        operator.beta,
        operator.beta,
        operator.beta,
    ]
end

function α(operator::BEAST.MWSingleLayer3D, polynomial)
    return [operator.α, operator.α, operator.α, operator.β]
end

function α(op::Union{MWEFIE3D,MWWeaklySingular3D}, polynomial)
    return operator(op).α
end

function α(op::MWHyperSingular3D, polynomial)
    return operator(op).β
end

function α(op::MultilevelKernelInterpolation.HH3DDoubleLayerFDBIOGradientI2O, polynomial)
    return -operator(op).alpha
end

function α(
    op::Union{
        MultilevelKernelInterpolation.HH3DDoubleLayerTransposedFDBIOGradientI2O,
        MultilevelKernelInterpolation.MWDoubleLayer3D,
    },
    polynomial,
)
    return operator(op).alpha
end

"""
    αtranspose(operator, polynomial)

Return the transpose coefficient, defaulting to `α(operator, polynomial)`.
`MultilevelKernelInterpolation.MWDoubleLayer3D` reverses its sign. Resolve once at construction;
the adjoint uses the conjugate of this coefficient.
"""
function αtranspose(operator, polynomial)
    return α(operator, polynomial)
end

function αtranspose(operator::MultilevelKernelInterpolation.MWDoubleLayer3D, polynomial)
    return -α(operator, polynomial)
end

# I2O translations #########################################################################

"""
    abstract type I2OTranslation

Base type for operator-specific I2O kernel traits.
"""
abstract type I2OTranslation end

"""
    abstract type ScalarI2OTranslation <: I2OTranslation

Scalar kernel trait: apply the same kernel independently to each moment component.
"""
abstract type ScalarI2OTranslation <: I2OTranslation end

"""
    struct ScalarGreen3DTranslation <: ScalarI2OTranslation

Translate each moment component independently with the scalar 3-D Green's function.
"""
struct ScalarGreen3DTranslation <: ScalarI2OTranslation end

"""
    struct ScalarGreen2DTranslation <: ScalarI2OTranslation

Translate each component with the 2-D Green's function `-i/4 H₀⁽²⁾(k|x-y|)`.
Shares scalar storage and application with [`ScalarGreen3DTranslation`](@ref);
only the kernel formula differs.
"""
struct ScalarGreen2DTranslation <: ScalarI2OTranslation end

"""
    struct ∇∇Green3DTranslation <: I2OTranslation

Use the Hessian of the 3-D Green's function as a dyadic coupling.
Store a 3-by-3 matrix of operators mapping input to output field components.
"""
struct ∇∇Green3DTranslation <: I2OTranslation end

"""
    struct ∇Green3DTranslation <: I2OTranslation

Use the gradient of the 3-D Green's function, stored as a vector of three component matrices.
"""
struct ∇Green3DTranslation <: I2OTranslation end

"""
    struct DyadicGreen3DTranslation <: I2OTranslation

Use the dyadic 3-D Green's function, stored as a 3-by-3 matrix of component operators.
"""
struct DyadicGreen3DTranslation <: I2OTranslation end

"""
    struct ∇CrossGreen3DTranslation <: I2OTranslation

Use the cross-product matrix of the 3-D Green's function gradient.
Store a 3-by-3 matrix of component operators; it transforms as a pseudo-tensor.
"""
struct ∇CrossGreen3DTranslation <: I2OTranslation end

"""
    i2otranslation(operator)

Returns the I2O translation trait of the (BEAST) operator.
"""
function i2otranslation(
    ::Union{
        BEAST.MWSingleLayer3D,
        BEAST.MWDoubleLayer3D,
        BEAST.DoubleLayerRotatedMW3D,
        BEAST.HH3DSingleLayerFDBIO,
        BEAST.HH3DHyperSingularFDBIO,
        BEAST.HH3DDoubleLayerFDBIO,
        BEAST.HH3DDoubleLayerTransposedFDBIO,
        MultilevelKernelInterpolation.MWWeaklySingular3D,
    },
)
    return ScalarGreen3DTranslation()
end

function i2otranslation(
    ::Union{
        BEAST.HH2DSingleLayerFDBIO,
        BEAST.HH2DDoubleLayerFDBIO,
        BEAST.HH2DDoubleLayerTransposedFDBIO,
    },
)
    return ScalarGreen2DTranslation()
end

function i2otranslation(::MWEFIE3D)
    return DyadicGreen3DTranslation()
end

function i2otranslation(::MWHyperSingular3D)
    return ∇∇Green3DTranslation()
end

function i2otranslation(::MultilevelKernelInterpolation.MWDoubleLayer3D)
    return ∇CrossGreen3DTranslation()
end

function i2otranslation(
    ::Union{HH3DDoubleLayerFDBIOGradientI2O,HH3DDoubleLayerTransposedFDBIOGradientI2O}
)
    return ∇Green3DTranslation()
end

"""
    symmetrycomponentaction(i2otranslation)

Return the lattice-symmetry action on field components: `nothing`, `VectorField()`,
`TrueTensor()`, or `PseudoTensor()` from H2Factory.

Container shape cannot choose this action: Hessian, dyadic, and curl kernels all use
3-by-3 blocks, but curl is a pseudo-tensor. For symmetry `S`:

    true tensor:   G(Sx) = S G(x) Sᵀ
    pseudo-tensor: K(Sx) = det(S) S K(x) Sᵀ

Scalar kernels need only the sample permutation, so return `nothing`.
"""
symmetrycomponentaction(::ScalarGreen3DTranslation) = nothing
# A scalar has no field components to transform, in either spatial dimension.
symmetrycomponentaction(::ScalarGreen2DTranslation) = nothing
symmetrycomponentaction(::∇Green3DTranslation) = H2Factory.VectorField()
symmetrycomponentaction(::∇∇Green3DTranslation) = H2Factory.TrueTensor()
symmetrycomponentaction(::DyadicGreen3DTranslation) = H2Factory.TrueTensor()
# Curl changes sign under reflections: it is a pseudo-tensor.
symmetrycomponentaction(::∇CrossGreen3DTranslation) = H2Factory.PseudoTensor()

function i2otranslationtrait(
    ::Union{
        ScalarI2OTranslation,
        ∇Green3DTranslation,
        ∇∇Green3DTranslation,
        DyadicGreen3DTranslation,
        ∇CrossGreen3DTranslation,
    },
)
    return DenseTranslation()
end

"""
    eltypei2otranslation(i2otranslationtrait, translator)

Return the stored coupling type for this kernel trait and translator.
"""
function eltypei2otranslation(::ScalarI2OTranslation, translator)
    return Matrix{scalartype(translator.operator)}
end

function eltypei2otranslation(::∇Green3DTranslation, translator)
    return Vector{Matrix{scalartype(translator.operator)}}
end

function eltypei2otranslation(
    ::Union{∇∇Green3DTranslation,DyadicGreen3DTranslation,∇CrossGreen3DTranslation},
    translator,
)
    return Matrix{Matrix{scalartype(translator.operator)}}
end
