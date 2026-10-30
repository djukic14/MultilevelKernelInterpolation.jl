# create PolynomialMoment, ToeplitzPolynomialMoment, and moment collections
@createmoment PolynomialMoment
@createtoeplitzmoment ToeplitzPolynomialMoment
@createmomentcollection TrialMomentCollection
@createmomentcollection TestMomentCollection

# link all moments to all collections
@linkmomenttocollection PolynomialMoment TestMomentCollection
@linkmomenttocollection PolynomialMoment TrialMomentCollection
@linkmomenttocollection ToeplitzPolynomialMoment TestMomentCollection
@linkmomenttocollection ToeplitzPolynomialMoment TrialMomentCollection

"""
    abstract type AbstractBsplineDenseMoment{D,T} <: H2Factory.AbstractToeplitzMoment{D,T}

Moment with raw B-spline arrays and reduced-basis scratch arrays.
I2I/O2O and moment assembly use the raw basis; dense I2O uses the reduced basis.
`BsplineDenseModulation` converts between them.

Inherits `AbstractToeplitzMoment` to reuse its two-array pool allocator, not its FFTs.
The modulation policy, rather than this supertype, selects the conversion.
"""
abstract type AbstractBsplineDenseMoment{D,T} <: H2Factory.AbstractToeplitzMoment{D,T} end

@createtoeplitzmoment BsplineDensePolynomialMoment AbstractBsplineDenseMoment
@linkmomenttocollection BsplineDensePolynomialMoment TestMomentCollection
@linkmomenttocollection BsplineDensePolynomialMoment TrialMomentCollection

"""
    reducedmoment(moment::AbstractBsplineDenseMoment, i::Int)

Return the `i`-th reduced-basis array used by the dense B-spline I2O conversion.
"""
function reducedmoment(moment::AbstractBsplineDenseMoment, i::Int)
    return H2Factory.toeplitzmoment(moment, i)
end

"""
    reducedvector(moment::AbstractBsplineDenseMoment, i::Int)

Return the cached flat view of `reducedmoment(moment, i)`.
Reuse this view in `mul!` to avoid allocating a new array header per translation.
"""
function reducedvector(moment::AbstractBsplineDenseMoment, i::Int)
    return H2Factory.toeplitzvector(moment, i)
end

"""
    defaultmomenttype(i2ostrategy)

Return the moment type for an I2O strategy. The fallback is `PolynomialMoment`.
"""
function defaultmomenttype(::Any)
    return PolynomialMoment
end
