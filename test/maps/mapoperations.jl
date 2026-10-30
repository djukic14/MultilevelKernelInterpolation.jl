module TestMapOperations

using Test
using CompScienceMeshes, BEAST
using H2Trees
using MultilevelKernelInterpolation

@testset verbose=true "MultilevelKernelInterpolation moments" begin
    λ = 1.0
    k = 2π / λ
    op = Maxwell3D.singlelayer(; wavenumber=k)

    m = readmesh(
        joinpath(pkgdir(MultilevelKernelInterpolation), "test", "assets", "in", "cuboid.in")
    )

    X = lagrangec0d1(m)
    tree = buildtree(X; builder=TwoNTreeBuilder(; minhalfsize=λ / 10, minvalues=0))

    plans = H2Trees.buildplans(
        tree; builder=H2Trees.PlanBuilder(; aggregatenode=(tree) -> (node) -> true)
    )

    poly = MultilevelKernelInterpolation.BarycentricLagrangePolynomialEquidistant3D(7)

    treepolynomials = MultilevelKernelInterpolation.levelpolynomials(
        poly,
        H2Trees.levels(tree),
        MultilevelKernelInterpolation.HFConditionFunctor(tree, Inf);
        increase=MultilevelKernelInterpolation.increase,
    )

    matvecbuffers, testmoments, trialmoments = MultilevelKernelInterpolation.moments(
        im * k,
        MultilevelKernelInterpolation.testmoment(op, poly),
        MultilevelKernelInterpolation.trialmoment(op, poly),
        tree,
        treepolynomials,
        plans.testaggregationplan,
        plans.trialaggregationplan,
        MultilevelKernelInterpolation.PolynomialMoment,
    )

    for i in eachindex(testmoments)
        @test size(testmoments[i][1]) == MultilevelKernelInterpolation.numpolynomials(poly)
        @test size(trialmoments[i][1]) == MultilevelKernelInterpolation.numpolynomials(poly)
        @test typeof(testmoments[i]) == typeof(trialmoments[i])
    end

    matvecbuffers, testmoments, trialmoments = MultilevelKernelInterpolation.moments(
        im * k,
        MultilevelKernelInterpolation.testmoment(op, poly),
        MultilevelKernelInterpolation.trialmoment(op, poly),
        tree,
        treepolynomials,
        plans.testaggregationplan,
        plans.trialaggregationplan,
        MultilevelKernelInterpolation.ToeplitzPolynomialMoment,
    )

    for i in eachindex(testmoments)
        @test size(testmoments[i][1]) == MultilevelKernelInterpolation.numpolynomials(poly)
        @test size(trialmoments[i][1]) == MultilevelKernelInterpolation.numpolynomials(poly)

        @test size(testmoments[i].toeplitzmoments[1]) ==
            (2 .* MultilevelKernelInterpolation.numpolynomials(poly) .- 1)
        @test size(trialmoments[i].toeplitzmoments[1]) ==
            (2 .* MultilevelKernelInterpolation.numpolynomials(poly) .- 1)

        @test typeof(testmoments[i]) == typeof(trialmoments[i])
    end

    @test MultilevelKernelInterpolation.λ(op) ≈ λ
end

end
