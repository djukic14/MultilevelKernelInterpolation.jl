# B-spline I2O

B-spline moments are integrals against basis functions, not kernel samples.
The coupling therefore needs the [sampling matrix](../polynomials/bsplines.md).

With sample-row matrices `S_test` and `S_trial`, the coefficient matrix is

```math
W=S_{\mathrm{test}}^{-1}\,G\,S_{\mathrm{trial}}^{-\mathsf T}.
```

This is a description of the square, unreduced case. The buffered route also
maps between raw and reduced bases after boundary-knot removal.

## Dense route

`MultilevelKernelInterpolation.BsplineKernelTranslator` converts kernel samples to interpolation
coefficients. `MultilevelKernelInterpolation.BsplineI2OTranslationCollection` pairs the stored coupling
with `MultilevelKernelInterpolation.BsplineDenseModulation`.

The policy converts raw incoming moments to reduced storage before coupling,
then expands the outgoing result. Test and trial conversion data are stored
separately, so dense Petrov coupling can use different grids.

## Toeplitz route

The raw coefficient matrix `W` is not itself Toeplitz. Move the sampling
conversions onto the moments instead:

```math
m_t^{\mathsf T}Wm_s
=\left(S_t^{-\mathsf T}m_t\right)^{\mathsf T}
 G\left(S_s^{-\mathsf T}m_s\right).
```

Now the middle operation uses only kernel samples on a uniform grid.
`MultilevelKernelInterpolation.BsplineToeplitzModulation` combines the basis conversions with FFTs.
`MultilevelKernelInterpolation.BsplineToeplitzI2OChannel` holds the workspace's conversion scratch
alongside the Toeplitz channel.

```text
source raw moments
        ↓ basis conversion and FFT
frequency-domain coupling
        ↓ inverse FFT and output conversion
receiving raw moments
```

Level transfers always use raw moments. Reduction belongs to I2O only; it
does not replace the bases used by I2I and O2O.

Choose either `MultilevelKernelInterpolation.DenseI2OStrategy()` or `MultilevelKernelInterpolation.ToeplitzI2OStrategy()` with
the same buffered polynomial strategy. The constructors select the matching
modulation policy; normal usage does not assemble these conversions by hand.
