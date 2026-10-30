```@meta
CurrentModule = MultilevelKernelInterpolation
```

# Moments

Moment types, the collections that map DOF amplitudes to and from them, and the assemblers that
populate them from a BEAST space.

The generated `mul!` and `conjmul!` methods over these types are on their own pages, since their
rendered signatures are long enough to dominate any page they share.

```@autodocs
Modules = [MultilevelKernelInterpolation]
Filter = inarea(:moments)
```
