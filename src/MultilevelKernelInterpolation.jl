module MultilevelKernelInterpolation

using NURBS
using StaticArrays, SparseArrays
using BlockSparseMatrices
using CompScienceMeshes, BEAST
using LinearAlgebra
using Dictionaries
using LinearMaps
using AbstractFFTs
using BandedMatrices
using OhMyThreads
using H2Factory, H2Trees

# The 2-D Green's function uses only hankelh2 from SpecialFunctions.
import SpecialFunctions: hankelh2

import H2Trees: numberoflevels
import H2Trees: leafclusters, halfsize, halfsizes, sector, treetrait
import H2Trees: data, level, values
import H2Trees: AbstractPlan, tree, DirectionInvariance, isTwoNTree, isBlockTree, BoxData
import H2Trees: translatingplan, levels, leaves, computevectorbuffers, testtree, trialtree
import H2Factory: BlockBEASTNearInteractionsAssembler, assemblemoments!, storetrait
import H2Factory:
    storagemomentsplan, α, isTranspose, isAdjoint, isForward, buildstorer, operator
import H2Factory: AbstractInteractionStrategy, InteractionStrategy, assemblenearinteractions
import H2Factory: AbstractNearInteractionsStrategy
import H2Factory: BlockBEASTNearInteractionsStrategy, DummyNearInteractionsStrategy
import H2Factory: AbstractI2OTranslationCollection, _getminmomenttype, applycouplingkernel!
import H2Factory: I2OTranslationCollection
import H2Factory: ToeplitzI2OTranslationTrait, AbstractToeplitzMoment, takefftplan!
import H2Factory: toeplitzmoment, toeplitzvector, _copytobuffer!
import H2Factory: _copytomoment!, i2otrait, takeifftplan!, createchannel
import H2Factory: ToeplitzMatrixRepresentation, StoreAsIs, toeplitzeltype
import H2Factory: EmptyChannels

include("polynomials/abstractpolynomial.jl")
include("polynomials/levelpolynomials.jl")
include("polynomials/scaledshiftedpolynomial.jl")
include("polynomials/lagrange/barycentriclagrange.jl")
include("polynomials/lagrange/evaluatebarycentriclagrange.jl")
include("polynomials/lagrange/refinement.jl")
include("polynomials/bspline/bspline.jl")
include("polynomials/bspline/bsplineequidistant.jl")
include("polynomials/bspline/bsplinebuffered.jl")
include("polynomials/bspline/interpolationmatrix.jl")
include("polynomials/bspline/refinement.jl")

"""
    isequidistantpolynomial(polynomial)

Return whether the polynomial uses equidistant interpolation nodes.
"""
function isequidistantpolynomial(polynomial)
    return isequidistantpolynomial(typeof(polynomial))
end

function isequidistantpolynomial(::Type{T}) where {T}
    return false
end

for type in [BarycentricLagrangePolynomialEquidistant, AbstractBspline]
    eval(
        quote
            function MultilevelKernelInterpolation.isequidistantpolynomial(::Type{<:$type})
                return true
            end
        end
    )
end

include("operators/wrappers.jl")
include("operators/momentoperators.jl")
include("operators/beastoperators.jl")

include("moments/moment.jl")
include("moments/momentcollection.jl")
include("moments/beastmomentassembler.jl")

include("translations/I2Itranslations.jl")
include("translations/I2ItranslationsOneDim.jl")
include("translations/I2Otranslations/abstracttranslator.jl")
include("translations/I2Otranslations/I2Otranslations.jl")
include("translations/I2Otranslations/I2Otranslationcollection.jl")
include("translations/I2Otranslations/I2Obsplinetranslationcollection.jl")
include("translations/I2Otranslations/ToeplitzI2OTranslationCollection.jl")
include("translations/I2Otranslations/BsplineToeplitzModulation.jl")
include("translations/O2Otranslations.jl")
include("translations/O2OtranslationsOneDim.jl")

include("tree/leafpolynomials.jl")

include("maps/construction/strategies.jl")
include("maps/construction/bundles.jl")
include("maps/construction/polynomials.jl")
include("maps/construction/translators.jl")

include("maps/GalerkinMap.jl")
include("maps/PetrovGalerkinMap.jl")
include("maps/mapoperations.jl")

function assemble(operator, testspace, trialspace; kwargs...)
    if testspace === trialspace
        return assemble(operator, testspace; kwargs...)
    else
        return PetrovGalerkinMap(operator, testspace, trialspace; kwargs...)
    end
end

function assemble(operator, space; kwargs...)
    return GalerkinMap(operator, space; kwargs...)
end

export LaplaceMoment, LaplaceMomentDiv, GradLaplaceMoment
export PolynomialMoment
export GalerkinMap, PetrovGalerkinMap

end # module MultilevelKernelInterpolation
