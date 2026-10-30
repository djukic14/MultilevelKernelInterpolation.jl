module TestBsplineI2OShapes

using Test
using Random
using LinearAlgebra
using H2Factory
using MultilevelKernelInterpolation

# The dense B-spline I2O apply, tested directly on synthetic moments and dense kernels -- no mesh,
# no BEAST, no translator construction. `_bsplinei2oapply!` is pure linear algebra over
# `reducedmoment(m, i)`, so it can be pinned exactly, and the three `green` shapes it dispatches on
# are the three `_tfcontainershape` cases:
#
#   AbstractMatrix{<:Number}          scalar kernel, applied to every component independently
#   AbstractVector{<:AbstractMatrix}  3 independent, uncoupled kernels (`∇Green3DTranslation`)
#   AbstractMatrix{<:AbstractMatrix}  3x3 cross-component-coupled (`∇∇`/`Dyadic`/`∇Cross`)
#
# Each has an overwrite form and an α/β accumulate form. `green` is never pre-transposed/adjointed;
# instead `_bsplinei2oapply!` takes the RAW `green` plus a `H2Factory.isForward/isTranspose/isAdjoint`
# mode and selects the per-block transform inside its own loop (`_bsplineblock`/`_bsplineindex`), so
# the O2I (transpose) and adjoint directions reuse the same dispatch as forward.

const RAW = (3, 3, 3)      # raw storage; irrelevant to the apply, but the moment type carries it
const RED = (2, 2, 2)      # reduced storage -- what the apply actually multiplies
const N = prod(RED)        # 8

# A moment whose reduced arrays are filled with reproducible values.
function randmoment(D, rng)
    m = MultilevelKernelInterpolation.BsplineDensePolynomialMoment{D,ComplexF64}(RAW, RED)
    for i in 1:D
        MultilevelKernelInterpolation.reducedmoment(m, i) .= reshape(
            randn(rng, ComplexF64, N), RED
        )
    end
    return m
end

reduced(m, i) = copy(vec(MultilevelKernelInterpolation.reducedmoment(m, i)))
randmat(rng) = randn(rng, ComplexF64, N, N)

@testset "reducedvector aliases reducedmoment (cached view, not a copy)" begin
    # `reducedvector` replaces `vec(reducedmoment(m, i))` in `_bsplinei2oapply!`'s hot path -- it
    # must be a live view of the SAME storage `reducedmoment` exposes shaped, not an independent copy,
    # or writes through one would silently stop being visible through the other.
    D = 3
    rng = MersenneTwister(11)
    m = randmoment(D, rng)

    for i in 1:D
        v = MultilevelKernelInterpolation.reducedvector(m, i)
        r = MultilevelKernelInterpolation.reducedmoment(m, i)

        @test length(v) == length(r)

        v[1] = 3 + 4im
        @test r[1] == 3 + 4im

        r[2] = 5 - 2im
        @test v[2] == 5 - 2im
    end
end

@testset "_bsplinei2oapply!: scalar kernel applies to every component independently" begin
    # `dims(in)` need NOT be 1 here: e.g. Maxwell3D.singlelayer's 4-component `[x,y,z,div]` moment
    # still translates through a single scalar kernel matrix. So exercise D=4, not D=1.
    rng = MersenneTwister(1)
    D = 4
    green = randmat(rng)
    in = randmoment(D, rng)
    tin = [reduced(in, i) for i in 1:D]

    out = randmoment(D, rng)
    MultilevelKernelInterpolation._bsplinei2oapply!(out, green, in, H2Factory.isForward())
    for i in 1:D
        @test vec(MultilevelKernelInterpolation.reducedmoment(out, i)) ≈ green * tin[i]
    end

    for α in (true, false), β in (true, false)
        out = randmoment(D, rng)
        tout0 = [reduced(out, i) for i in 1:D]
        MultilevelKernelInterpolation._bsplinei2oapply!(
            out, green, in, H2Factory.isForward(), α, β
        )
        for i in 1:D
            expected =
                (β ? tout0[i] : zero(tout0[i])) + (α ? green * tin[i] : zero(tout0[i]))
            @test vec(MultilevelKernelInterpolation.reducedmoment(out, i)) ≈ expected
        end
    end
end

@testset "_bsplinei2oapply!: vector kernel = 3 independent, uncoupled operators" begin
    # `∇Green3DTranslation`: component i is translated by green[i] ONLY -- no cross-coupling. A bug
    # that coupled them (or reused green[1] for all) breaks here.
    rng = MersenneTwister(2)
    D = 3
    green = [randmat(rng) for _ in 1:D]
    in = randmoment(D, rng)
    tin = [reduced(in, i) for i in 1:D]

    out = randmoment(D, rng)
    MultilevelKernelInterpolation._bsplinei2oapply!(out, green, in, H2Factory.isForward())
    for i in 1:D
        @test vec(MultilevelKernelInterpolation.reducedmoment(out, i)) ≈ green[i] * tin[i]
    end

    for α in (true, false), β in (true, false)
        out = randmoment(D, rng)
        tout0 = [reduced(out, i) for i in 1:D]
        MultilevelKernelInterpolation._bsplinei2oapply!(
            out, green, in, H2Factory.isForward(), α, β
        )
        for i in 1:D
            expected =
                (β ? tout0[i] : zero(tout0[i])) + (α ? green[i] * tin[i] : zero(tout0[i]))
            @test vec(MultilevelKernelInterpolation.reducedmoment(out, i)) ≈ expected
        end
    end
end

@testset "_bsplinei2oapply!: matrix kernel couples all components (out_i = Σ_j green[i,j]*in_j)" begin
    # `∇∇`/`Dyadic`/`∇Cross`: every output component sums contributions from every input component.
    # The implementation overwrites on j==1 then accumulates, so a bug there drops all but one term.
    rng = MersenneTwister(3)
    D = 3
    green = [randmat(rng) for _ in 1:D, _ in 1:D]
    in = randmoment(D, rng)
    tin = [reduced(in, i) for i in 1:D]

    out = randmoment(D, rng)
    MultilevelKernelInterpolation._bsplinei2oapply!(out, green, in, H2Factory.isForward())
    for i in 1:D
        @test vec(MultilevelKernelInterpolation.reducedmoment(out, i)) ≈
            sum(green[i, j] * tin[j] for j in 1:D)
    end

    for α in (true, false), β in (true, false)
        out = randmoment(D, rng)
        tout0 = [reduced(out, i) for i in 1:D]
        MultilevelKernelInterpolation._bsplinei2oapply!(
            out, green, in, H2Factory.isForward(), α, β
        )
        for i in 1:D
            # Same α/β contract as the scalar and vector shapes: β scales what `out` held, α scales
            # the WHOLE sum. (`β` is applied once, on the first term; every later term accumulates.)
            expected = (β ? tout0[i] : zero(tout0[i]))
            if α
                expected = expected + sum(green[i, j] * tin[j] for j in 1:D)
            end
            @test vec(MultilevelKernelInterpolation.reducedmoment(out, i)) ≈ expected
        end
    end
end

@testset "_bsplinei2oapply!: transpose/adjoint mode selects the correct per-block transform" begin
    # `green` is passed RAW (never pre-transposed/pre-adjointed); `mode` alone must select the right
    # per-block transform inside `_bsplinei2oapply!`'s own loop. Ground truth is `transformfn` applied
    # directly to the relevant block(s), mirroring what the removed `_bsplinetranspose`/`_bsplineadjoint`
    # helpers used to compute up front.
    rng = MersenneTwister(4)
    D = 3

    for (modelabel, mode, transformfn) in (
        ("transpose", H2Factory.isTranspose(), transpose),
        ("adjoint", H2Factory.isAdjoint(), adjoint),
    )
        # Scalar: mode applies to the whole matrix (same block used for every component).
        g = randmat(rng)
        insc = randmoment(4, rng)
        tinsc = [reduced(insc, i) for i in 1:4]
        outsc = randmoment(4, rng)
        MultilevelKernelInterpolation._bsplinei2oapply!(outsc, g, insc, mode)
        for i in 1:4
            @test vec(MultilevelKernelInterpolation.reducedmoment(outsc, i)) ≈
                transformfn(g) * tinsc[i]
        end
        for α in (true, false), β in (true, false)
            outαβ = randmoment(4, rng)
            tout0 = [reduced(outαβ, i) for i in 1:4]
            MultilevelKernelInterpolation._bsplinei2oapply!(outαβ, g, insc, mode, α, β)
            for i in 1:4
                expected =
                    (β ? tout0[i] : zero(tout0[i])) +
                    (α ? transformfn(g) * tinsc[i] : zero(tout0[i]))
                @test vec(MultilevelKernelInterpolation.reducedmoment(outαβ, i)) ≈ expected
            end
        end

        # Vector of independent operators: mode applies PER BLOCK, no index swap -- there is no
        # cross-component coupling to swap for this shape (unlike the matrix-of-matrices case below).
        gv = [randmat(rng) for _ in 1:D]
        inv_ = randmoment(D, rng)
        tinv = [reduced(inv_, i) for i in 1:D]
        outv = randmoment(D, rng)
        MultilevelKernelInterpolation._bsplinei2oapply!(outv, gv, inv_, mode)
        for i in 1:D
            @test vec(MultilevelKernelInterpolation.reducedmoment(outv, i)) ≈
                transformfn(gv[i]) * tinv[i]
        end

        # Matrix of operators: mode must swap the OUTER (i,j) index AND transform the block, matching
        # `transformfn(G)[i,j] == transformfn(G[j,i])` -- the load-bearing identity this file used to
        # pin directly on `_bsplinetranspose`'s reliance on Julia's recursive `Transpose`/`Adjoint`
        # wrapper; now pinned end-to-end through `_bsplinei2oapply!`'s explicit `_bsplineindex` swap.
        gm = [randmat(rng) for _ in 1:D, _ in 1:D]
        inm = randmoment(D, rng)
        tinm = [reduced(inm, i) for i in 1:D]
        outm = randmoment(D, rng)
        MultilevelKernelInterpolation._bsplinei2oapply!(outm, gm, inm, mode)
        for i in 1:D
            @test vec(MultilevelKernelInterpolation.reducedmoment(outm, i)) ≈
                sum(transformfn(gm[j, i]) * tinm[j] for j in 1:D)
        end
    end
end

@testset "_bsplinei2oapply!: zero allocation, every shape x mode (reducedvector, not vec(reducedmoment(...)))" begin
    # Two allocation sources are fixed by this point: the vector shape (`∇Green3DTranslation`) used
    # to build `[transpose(g) for g in green]` (a new `Vector` sized D) on every translation via the
    # now-removed `_bsplinetranspose`/`_bsplineadjoint`; and EVERY shape used `vec(reducedmoment(m,
    # i))`, which reallocates a new `Array` header each call since `reducedmoment` is the raw shaped
    # array, not a cached vector view. `reducedvector` (backed by `H2Factory.toeplitzvector`'s cached
    # view, computed once at moment construction) fixes the second source too, so this can now assert
    # literal zero -- not just "no more than forward" -- for every shape and every mode.
    D = 3
    rng = MersenneTwister(6)
    scalargreen = randmat(rng)
    vectorgreen = [randmat(rng) for _ in 1:D]
    matrixgreen = [randmat(rng) for _ in 1:D, _ in 1:D]

    for (label, green, Dgreen) in
        (("scalar", scalargreen, 4), ("vector", vectorgreen, D), ("matrix", matrixgreen, D))
        for mode in (H2Factory.isForward(), H2Factory.isTranspose(), H2Factory.isAdjoint())
            in_ = randmoment(Dgreen, rng)
            out = randmoment(Dgreen, rng)
            MultilevelKernelInterpolation._bsplinei2oapply!(out, green, in_, mode)   # warm up (JIT)
            a = @allocated MultilevelKernelInterpolation._bsplinei2oapply!(
                out, green, in_, mode
            )
            @test a == 0

            # α/β accumulate form.
            outαβ = randmoment(Dgreen, rng)
            MultilevelKernelInterpolation._bsplinei2oapply!(
                outαβ, green, in_, mode, true, true
            )
            aαβ = @allocated MultilevelKernelInterpolation._bsplinei2oapply!(
                outαβ, green, in_, mode, true, true
            )
            @test aαβ == 0
        end
    end
end

@testset "O2I/adjoint == I2O with the transposed/adjointed kernel (the duality the traversal relies on)" begin
    # O2I is `_bsplinei2oapply!(out, green, in, isTranspose())` with the node pair SWAPPED at the
    # collection lookup; adjoint is the same with `isAdjoint()`. Pin both dualities directly: transpose
    # gives the bilinear form <y,Gx> == <G^T y,x> (no conjugation); adjoint gives the true sesquilinear
    # inner-product duality <y,Gx> == <G'y,x> (with conjugation) -- for every shape.
    rng = MersenneTwister(5)
    D = 3

    for (label, green) in (
        ("scalar", randmat(rng)),
        ("vector", [randmat(rng) for _ in 1:D]),
        ("matrix", [randmat(rng) for _ in 1:D, _ in 1:D]),
    )
        x = randmoment(D, rng)
        y = randmoment(D, rng)
        tx = [reduced(x, i) for i in 1:D]
        ty = [reduced(y, i) for i in 1:D]

        Gx = randmoment(D, rng)
        MultilevelKernelInterpolation._bsplinei2oapply!(Gx, green, x, H2Factory.isForward())

        # Transpose duality: bilinear form, no conjugation (`dot(conj(a),b)` cancels `dot`'s own
        # conjugation of its first argument, leaving the plain bilinear sum).
        GTy = randmoment(D, rng)
        MultilevelKernelInterpolation._bsplinei2oapply!(
            GTy, green, y, H2Factory.isTranspose()
        )
        lhsT = sum(
            dot(conj(ty[i]), vec(MultilevelKernelInterpolation.reducedmoment(Gx, i))) for
            i in 1:D
        )
        rhsT = sum(
            dot(conj(vec(MultilevelKernelInterpolation.reducedmoment(GTy, i))), tx[i]) for
            i in 1:D
        )
        @test lhsT ≈ rhsT rtol = 1e-9
        @test abs(lhsT) > 1e-8   # guard against a vacuous pass

        # Adjoint duality: true sesquilinear inner product, <a,b> = dot(a,b) = sum(conj(a).*b).
        GHy = randmoment(D, rng)
        MultilevelKernelInterpolation._bsplinei2oapply!(
            GHy, green, y, H2Factory.isAdjoint()
        )
        lhsA = sum(
            dot(ty[i], vec(MultilevelKernelInterpolation.reducedmoment(Gx, i))) for i in 1:D
        )
        rhsA = sum(
            dot(vec(MultilevelKernelInterpolation.reducedmoment(GHy, i)), tx[i]) for
            i in 1:D
        )
        @test lhsA ≈ rhsA rtol = 1e-9
        @test abs(lhsA) > 1e-8   # guard against a vacuous pass
    end
end

end # module TestBsplineI2OShapes
