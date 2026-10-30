# Run with: julia --project=docs docs/check-explorers.jl (requires Node.js).
using MultilevelKernelInterpolation
using Test

@testset "Level growth and automatic buffers" begin
    lagrange = MultilevelKernelInterpolation.BarycentricLagrangePolynomialEquidistant(
        dim=1, order=2
    )
    splines = MultilevelKernelInterpolation.BsplinePolynomialEquidistant(
        dim=1, degree=3, nbasis=9
    )
    @test first.(
        MultilevelKernelInterpolation.polynomialorders.(
            MultilevelKernelInterpolation.levelpolynomials(lagrange, 1:4, l -> l <= 2)
        ),
    ) == [8, 4, 2, 2]
    @test first.(
        MultilevelKernelInterpolation.numpolynomials.(
            MultilevelKernelInterpolation.levelpolynomials(splines, 1:4, l -> l <= 2)
        ),
    ) == [27, 15, 9, 9]
    buffered = MultilevelKernelInterpolation.BufferedBsplinePolynomialEquidistant(
        dim=1,
        degree=3,
        nbasis=9,
        buffer=MultilevelKernelInterpolation.AutoBufferFromProtrusion(),
    )
    requirements = [0.10, 0.15, 0.20, 0.30]
    polys = MultilevelKernelInterpolation.levelpolynomials(
        buffered, 1:4, l -> false; protrusion=requirements
    )
    @test MultilevelKernelInterpolation.numbufferpoints.(polys) == [1, 1, 2, 2]
    @test all(MultilevelKernelInterpolation.realizedprotrusion.(polys) .>= requirements)
end

function samples(poly, xs)
    return [
        begin
            values = zeros(first(MultilevelKernelInterpolation.numpolynomials(poly)))
            MultilevelKernelInterpolation.evaluateall1dpolynomial!(values, poly, x, 1)
            values
        end for x in xs
    ]
end

# These numeric Julia vector literals are also JavaScript array literals.
jsrows(rows) = "[" * join(repr.(rows), ",") * "]"

mktemp() do path, io
    explorer = joinpath(@__DIR__, "src", "assets", "basis-explorer.js")
    println(io, "const assert = require('node:assert/strict');")
    println(io, "const {basis, transfer, parentBasis} = require(", repr(explorer), ");")
    println(
        io,
        "let checks = 0; const near = (a,b) => { assert.equal(a.length,b.length); a.forEach((x,i) => assert.ok(Math.abs(x-b[i]) < 2e-10, x+' != '+b[i])); checks++; };",
    )
    for p in (1, 3, 5), family in ("equidistant", "chebyshev1", "chebyshev2")
        constructor = if family == "equidistant"
            MultilevelKernelInterpolation.BarycentricLagrangePolynomialEquidistant
        elseif family == "chebyshev1"
            MultilevelKernelInterpolation.BarycentricLagrangePolynomialChebyshev
        else
            MultilevelKernelInterpolation.BarycentricLagrangePolynomialChebyshev2
        end
        child = constructor(; dim=1, order=p)
        xs = sort(
            unique([
                0.0, 0.13, 0.5, 0.83, 1.0, MultilevelKernelInterpolation.range(child, 1)...
            ]),
        )
        println(io, "{const c=basis('lagrange',", p, ",", p+1, ",0,", repr(family), ");")
        println(
            io,
            repr(xs),
            ".forEach((x,i)=>near(c.eval(x),",
            jsrows(samples(child, xs)),
            "[i]));",
        )
        for pp in (p, 2p)
            parent = constructor(; dim=1, order=pp)
            left, right = MultilevelKernelInterpolation.onedimi2itranslations(child, parent)
            println(
                io, "{const p=basis('lagrange',", pp, ",", pp+1, ",0,", repr(family), ");"
            )
            for (side, mat) in enumerate((left, right))
                println(
                    io,
                    "transfer(c,p,",
                    side-1,
                    ").forEach((r,i)=>near(r,",
                    jsrows([collect(mat[i, :]) for i in axes(mat, 1)]),
                    "[i]));",
                )
            end
            println(io, "}")
        end
        println(io, "}")
    end
    for p in (1, 3, 5), count in (p+1, p+3), buffer in (0, 1, 2)
        # Construct the raw buffered basis; stabilization changes sampling, not these functions.
        poly = MultilevelKernelInterpolation.BufferedBsplinePolynomialEquidistant(;
            dim=1, degree=p, nbasis=count, buffer=buffer
        )
        lo = -buffer / (count-p)
        xs = [lo, lo/2, 0.0, 0.13, 0.5, 0.83, 1.0, 1-lo/2, 1-lo]
        println(io, "{const c=basis('bspline',", p, ",", count, ",", buffer, ");")
        println(
            io,
            repr(xs),
            ".forEach((x,i)=>near(c.eval(x),",
            jsrows(samples(poly, xs)),
            "[i]));}",
        )
    end
    for count in (6, 9), parentcount in (count, 2count-3)
        child = MultilevelKernelInterpolation.BsplinePolynomialEquidistant(;
            dim=1, degree=3, nbasis=count
        )
        parent = MultilevelKernelInterpolation.BsplinePolynomialEquidistant(;
            dim=1, degree=3, nbasis=parentcount
        )
        left, right = MultilevelKernelInterpolation.onedimi2itranslations(child, parent)
        println(
            io,
            "{const c=basis('bspline',3,",
            count,
            ");const p=basis('bspline',3,",
            parentcount,
            ");",
        )
        for (side, mat) in enumerate((left, right))
            println(
                io,
                "transfer(c,p,",
                side-1,
                ").forEach((r,i)=>near(r,",
                jsrows([collect(mat[i, :]) for i in axes(mat, 1)]),
                "[i]));",
            )
        end
        println(io, "}")
    end
    for p in (1, 3, 5)
        count = p + 3
        child = MultilevelKernelInterpolation.BufferedBsplinePolynomialEquidistant(;
            dim=1, degree=p, nbasis=count, buffer=2
        )
        parent = MultilevelKernelInterpolation.BufferedBsplinePolynomialEquidistant(;
            dim=1, degree=p, nbasis=count, buffer=1
        )
        n = first(MultilevelKernelInterpolation.numpolynomials(child))
        lo, hi = -2 / (count-p), 1 + 2 / (count-p)
        knots = [fill(lo, p); collect(range(lo, hi; length=n-p+1)); fill(hi, p)]
        points = [sum(knots[(i + 1):(i + p)]) / p for i in 1:n]
        points[1], points[end] = lo, hi
        m = transpose(reduce(hcat, samples(child, points)))
        println(
            io,
            "{const c=basis('bspline',",
            p,
            ",",
            count,
            ",2);const p=basis('bspline',",
            p,
            ",",
            count,
            ",1);",
        )
        for side in (0, 1)
            rhs = transpose(reduce(hcat, samples(parent, (points .+ side) ./ 2)))
            mat = transpose(m \ rhs)
            println(
                io,
                "transfer(c,p,",
                side,
                ").forEach((r,i)=>near(r,",
                jsrows([collect(mat[i, :]) for i in axes(mat, 1)]),
                "[i]));",
            )
        end
        println(io, "}")
    end
    for p in (1, 3, 5),
        nodes in ("equidistant", "chebyshev1", "chebyshev2"),
        mode in ("increase", "noincrease")

        constructor = if nodes == "equidistant"
            MultilevelKernelInterpolation.BarycentricLagrangePolynomialEquidistant
        elseif nodes == "chebyshev1"
            MultilevelKernelInterpolation.BarycentricLagrangePolynomialChebyshev
        else
            MultilevelKernelInterpolation.BarycentricLagrangePolynomialChebyshev2
        end
        child = constructor(; dim=1, order=p)
        parent = if mode == "increase"
            MultilevelKernelInterpolation.increase(child)
        else
            MultilevelKernelInterpolation.noincrease(child)
        end
        xs = [0.0, 0.13, 0.5, 0.83, 1.0]
        println(
            io,
            "{const p=parentBasis(basis('lagrange',",
            p,
            ",",
            p+1,
            ",0,",
            repr(nodes),
            "),",
            repr(mode),
            ");",
        )
        println(
            io,
            "assert.equal(p.p,",
            first(MultilevelKernelInterpolation.polynomialorders(parent)),
            ");assert.equal(p.n,",
            first(MultilevelKernelInterpolation.numpolynomials(parent)),
            ");",
        )
        println(
            io,
            repr(xs),
            ".forEach((x,i)=>near(p.eval(x),",
            jsrows(samples(parent, xs)),
            "[i]));}",
        )
    end
    for p in (1, 3, 5), buffer in (0, 1, 2), mode in ("increase", "noincrease")
        count = p + 3
        child = MultilevelKernelInterpolation.BufferedBsplinePolynomialEquidistant(;
            dim=1,
            degree=p,
            nbasis=count,
            buffer=MultilevelKernelInterpolation.ExplicitBuffer(buffer),
        )
        parent = if mode == "increase"
            MultilevelKernelInterpolation.increase(child)
        else
            MultilevelKernelInterpolation.noincrease(child)
        end
        xs = [0.0, 0.13, 0.5, 0.83, 1.0]
        println(
            io,
            "{const p=parentBasis(basis('bspline',",
            p,
            ",",
            count,
            ",",
            buffer,
            "),",
            repr(mode),
            ");",
        )
        println(
            io,
            "assert.equal(p.p,",
            first(MultilevelKernelInterpolation.polynomialorders(parent)),
            ");assert.equal(p.n,",
            first(MultilevelKernelInterpolation.numpolynomials(parent)),
            ");assert.equal(p.buffer,",
            MultilevelKernelInterpolation.numbufferpoints(parent),
            ");",
        )
        println(
            io,
            repr(xs),
            ".forEach((x,i)=>near(p.eval(x),",
            jsrows(samples(parent, xs)),
            "[i]));}",
        )
    end
    println(io, "assert.throws(()=>parentBasis(basis('bspline',3,6),'invalid'));")
    println(
        io,
        "assert.throws(()=>basis('bspline',3,3)); assert.throws(()=>basis('bspline',3,9,-1));",
    )
    println(
        io,
        "function residual(c,p){let error=0; for(const side of [0,1]) {const mat=transfer(c,p,side); for(let k=0;k<=200;k++){const u=k/200, v=c.eval(u), ref=p.eval((u+side)/2); mat.forEach((r,i)=>{error=Math.max(error,Math.abs(r.reduce((s,a,j)=>s+a*v[j],0)-ref[i]));});}}return error;}",
    )
    println(io, "assert.ok(residual(basis('lagrange',4,5),basis('lagrange',4,5))<1e-10);")
    println(io, "assert.ok(residual(basis('lagrange',4,5),basis('lagrange',8,9))>1e-3);")
    println(io, "assert.ok(residual(basis('bspline',3,6),basis('bspline',3,9))<1e-10);")
    println(io, "assert.ok(residual(basis('bspline',3,6),basis('bspline',3,8))>1e-3);")
    println(
        io,
        "console.log(checks+' Julia/browser comparisons passed, plus exact/non-exact and invalid-input controls.');",
    )
    flush(io)
    @test success(pipeline(`node $path`; stdout=stdout, stderr=stderr))
end
