function postprocessmoment!(F, ::Any)
    return F
end

function postprocessmoment!(F, ::A) where {A<:AbstractBspline}
    return SVector{length(F)}(sparse.(F))
end

function postprocessmoment!(
    F, ::A
) where {A<:ScaledShiftedPolynomial{D,<:AbstractBspline}} where {D}
    return SVector{length(F)}(sparse.(F))
end

function momentcollectionconstructors(collection, collections, momentoperator)
    @eval function $collections(
        op,
        tree,
        polynomials,
        space;
        quadstrat=BEAST.SingleNumQStrat(7),
        MomentAssembler=MultilevelKernelInterpolation.BEASTMomentAssembler,
        verbose::Bool=false,
    )
        assembler = MomentAssembler(
            $momentoperator(op, first(polynomials)),
            space,
            quadstrat,
            dims($momentoperator(op, first(polynomials))),
        )

        return $collections(assembler, tree, polynomials; verbose=verbose)
    end

    @eval function $collections(assembler, tree, polynomials; verbose::Bool=false)
        Fs = assembler(tree, polynomials; verbose=verbose)

        firstcollection = $collection(
            MultilevelKernelInterpolation.postprocessmoment!(Fs[begin], polynomials[1])
        )
        collections = Vector{typeof(firstcollection)}(undef, length(Fs))
        collections[1] = firstcollection

        @tasks for i in eachindex(collections)
            collections[i] = $collection(
                MultilevelKernelInterpolation.postprocessmoment!(Fs[i], polynomials[i])
            )
        end

        return collections
    end

    return nothing
end

momentcollectionconstructors(:TrialMomentCollection, :TrialMomentCollections, :trialmoment)
momentcollectionconstructors(:TestMomentCollection, :TestMomentCollections, :testmoment)

"""
    TestMomentCollections(
        operator,
        tree,
        polynomials,
        space;
        quadstrat=BEAST.SingleNumQStrat(7),
        MomentAssembler= MultilevelKernelInterpolation.BEASTMomentAssembler,
        verbose::Bool=false,
    )

Assemble one `TestMomentCollection` per tree leaf using `space` and its leaf polynomial.
`testmoment(operator, first(polynomials))` selects the moment operator.

`MomentAssembler` builds the moments with quadrature strategy `quadstrat`.
B-spline moments are stored as sparse matrices. Set `verbose=true` to show progress.
"""
TestMomentCollections

"""
    TrialMomentCollections(
        operator,
        tree,
        polynomials,
        space;
        quadstrat=BEAST.SingleNumQStrat(7),
        MomentAssembler= MultilevelKernelInterpolation.BEASTMomentAssembler,
        verbose::Bool=false,
    )

Assemble one `TrialMomentCollection` per tree leaf using `space` and its leaf polynomial.
`trialmoment(operator, first(polynomials))` selects the moment operator.

`MomentAssembler` builds the moments with quadrature strategy `quadstrat`.
B-spline moments are stored as sparse matrices. Set `verbose=true` to show progress.
"""
TrialMomentCollections
