# Custom translations

Strategies select builders. The built translations then participate in
H2Factory's execution protocol. Construction data belongs in the core;
mutable scratch belongs in each workspace.

## Level transfers: I2I and O2O

For a custom subtype of `MultilevelKernelInterpolation.AbstractI2IStrategy`, define:

```julia
MultilevelKernelInterpolation.buildi2itranslator(polynomials, levels, strategy)
```

`polynomials` and `levels` are matching sequences. The standard builders
return an `H2Factory.I2ITranslationCollection`, with either combined matrices
or one factor per axis. Returning that representation lets you reuse its
level-transfer implementation.

The result also feeds:

```julia
MultilevelKernelInterpolation.buildo2otranslator(i2itranslator, strategy)
```

The standard O2O builder makes a reference cache of transposed transfers.
H2Factory's traversal resolves its downward transfers from the I2I collection;
that cache is not a second execution path. A different transfer representation
must support H2Factory's level-transfer and scratch protocols as well.

Select the builders with `MultilevelKernelInterpolation.TranslatorStrategy(i2istrategy=...,
o2ostrategy=...)`. New strategy types derive from `MultilevelKernelInterpolation.AbstractI2IStrategy`
and `MultilevelKernelInterpolation.AbstractO2OStrategy`, respectively.

## Far-field coupling: I2O

For a subtype of `MultilevelKernelInterpolation.AbstractI2OStrategy`, the builder signature is:

```julia
MultilevelKernelInterpolation.buildi2ocoupling(op, tree, polynomials, translatingplan,
    testspace, trialspace, strategy; verbose=false)
```

Return the read-only coupling, without workspace scratch. The spaces are
significant: the standard builders use them to select B-spline conversion.
Skipping that route can produce a coupling that samples the right kernel
but applies it in the wrong basis.

## Pair coupling with modulation

`MultilevelKernelInterpolation.assemblei2otranslator` composes the result as an
`H2Factory.I2OTranslation(coupling, modulation)`. Standard dense and Toeplitz
couplings have default policies. A new representation needing its own
conversion must supply the composition deliberately; it must not silently
fall back to `NoModulation()`.

The MultilevelKernelInterpolation construction hooks are `MultilevelKernelInterpolation._composei2otranslation` and
`MultilevelKernelInterpolation._i2oselfmodulates`. They are internal extension points, not interchangeable
with the public H2Factory coupling type. In particular, `MultilevelKernelInterpolation.I2OTranslation`
is an operator-kernel trait, not `H2Factory.I2OTranslation`.

## Allocate scratch separately

`MultilevelKernelInterpolation.i2ochannelfor(translation, polynomials, translatingplan, T)` builds fresh
scratch for a workspace. Its lower-level dispatch separates the coupling
and modulation policy. FFT plans and basis-conversion buffers must follow
the workspace's per-level chunk counts.

For a new coupling application, extend
`H2Factory.Interfaces.applycouplingkernel!`, not the guarded orchestration
function `H2Factory.applycoupling!`. Input and output conversions extend
`H2Factory.modulate_in!` and `H2Factory.modulate_out!`.

The exact direction, accumulation, and context arguments are described in
the [H2Factory translation guide](https://djukic14.github.io/H2Factory.jl/dev/).
The [MultilevelKernelInterpolation translation reference](../api/translations.md) contains the existing
implementations to compare against.
