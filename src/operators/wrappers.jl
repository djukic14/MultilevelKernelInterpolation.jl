@efie MWEFIE3D

"""
    MWEFIE3D

Wrap `Maxwell3D.singlelayer` with coefficients chosen for EFIE.
Interpolate the dyadic kernel directly.
"""
MWEFIE3D

function efie(; gamma=nothing, wavenumber=nothing, alpha=nothing)
    return MWEFIE3D(; gamma=gamma, wavenumber=wavenumber, alpha=alpha)
end

@mw3Dhypersingular MWHyperSingular3D
"""
    MWHyperSingular3D

Wrap `Maxwell3D.singlelayer` with `α=0` and interpolate its dyadic kernel.
"""
MWHyperSingular3D

@mw3Dweaklysingular MWWeaklySingular3D
"""
    MWWeaklySingular3D

Wrap `Maxwell3D.singlelayer` with `β=0` and interpolate its dyadic kernel.
"""
MWWeaklySingular3D

@mw3Ddoublelayer MWDoubleLayer3D
"""
    MWDoubleLayer3D

Wrap `Maxwell3D.doublelayer` and interpolate the kernel gradient directly.
"""
MWDoubleLayer3D

@hh3Ddoublelayer HH3DDoubleLayerFDBIOGradientI2O

"""
    HH3DDoubleLayerFDBIOGradientI2O

Wrap `Helmholtz3D.doublelayer` and interpolate the kernel gradient directly.
"""
HH3DDoubleLayerFDBIOGradientI2O

@hh3Ddoublelayer_transposed HH3DDoubleLayerTransposedFDBIOGradientI2O

"""
    HH3DDoubleLayerTransposedFDBIOGradientI2O

Wrap `Helmholtz3D.doublelayer_transposed` and interpolate the kernel gradient directly.
"""
HH3DDoubleLayerTransposedFDBIOGradientI2O
