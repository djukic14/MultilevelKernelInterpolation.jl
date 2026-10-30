using MultilevelKernelInterpolation
using Documenter
using Pkg

DocMeta.setdocmeta!(
    MultilevelKernelInterpolation,
    :DocTestSetup,
    :(using MultilevelKernelInterpolation);
    recursive=true,
)

# THE API REFERENCE IS SPLIT BY BINDING, NOT BY SOURCE PATH.
#
# A single `@autodocs` page carrying all 238 docstring entries generated 424 KiB of HTML, over
# Documenter's 200 KiB `size_threshold`. The obvious fix, `Pages` filters per source directory, does
# NOT work here: `Pages` matches the file a docstring came from, and 39 of these entries are written
# by H2Factory's moment macros, so their recorded path is inside H2Factory. No `Pages` list rooted in
# MultilevelKernelInterpolation's `src/` can name them, and `checkdocs=:all` rejected that attempt with "39 docstrings not
# included in the manual".
#
# `Filter` receives the EVALUATED OBJECT a binding names, so it routes those by identity instead. It
# also runs per BINDING, before the per-method loop, so a binding's docstrings are never separated.
#
# THIS PARTITION FAILS CLOSED. Every branch below either classifies deliberately or errors: a new
# binding recorded outside `src/`, a new source directory, a binding that will not evaluate, a
# binding spanning several areas, two bindings resolving to one object with different areas, or a
# lookup miss at filter time. That matters because `checkdocs=:all` only proves every docstring
# landed SOMEWHERE; it cannot tell that a docstring landed on the RIGHT page. Silent defaults would
# keep the build green while quietly misfiling things.

# Source directories under `src/`. A directory not listed here is a decision someone has to make, so
# it errors rather than defaulting.
const AREA_OF_SOURCE_DIR = Dict(
    "maps" => :maps,
    "polynomials" => :polynomials,
    "translations" => :translations,
    "moments" => :moments,
    "operators" => :operators,
    "tree" => :tree,
)

# The ONLY bindings whose docstrings Documenter records outside MultilevelKernelInterpolation's `src/`, because H2Factory's
# moment macros generate them. An allowlist, not a rule: a new one means the macros emit something
# new, and that must be classified on purpose rather than swept into a default page.
const EXTERNALLY_RECORDED = Dict(
    ("MultilevelKernelInterpolation", "PolynomialMoment") => :moments,
    ("MultilevelKernelInterpolation", "TestMomentCollection") => :moments,
    ("MultilevelKernelInterpolation", "TrialMomentCollection") => :moments,
    ("H2Factory.Interfaces", "conjmul!") => :conjmul,
    ("LinearAlgebra", "mul!") => :mul,
)

# Generic functions with documented methods in more than one source directory. `Filter` is applied
# per BINDING, so all of a binding's docstrings share one page and the home has to be chosen.
#
# Listed explicitly rather than resolved by a majority rule: a rule would silently repick the page
# when method counts shift and nobody would see it move. No method counts are written down here
# either, for the same reason; instead `build_area_map` CHECKS that each configured home is one of
# the areas the binding actually spans, so a wrong or outdated choice fails the build.
const MULTI_AREA_HOME = Dict(
    ("MultilevelKernelInterpolation", "kernelbuffer") => :translations,
    ("MultilevelKernelInterpolation", "defaultmomenttype") => :maps,
    ("MultilevelKernelInterpolation", "increase") => :polynomials,
    ("MultilevelKernelInterpolation", "dims") => :polynomials,
)

function build_area_map()
    areas = IdDict{Any,Symbol}()
    problems = String[]
    # The allowlists are checked in BOTH directions. Unknown keys already fail inline; these record
    # what was actually encountered so STALE keys fail too. Without this a binding can stop being
    # external or multi-area while its table entry, and the comment explaining it, stay confidently
    # wrong.
    seenexternal = Set{Tuple{String,String}}()
    seenmulti = Set{Tuple{String,String}}()
    root = pkgdir(MultilevelKernelInterpolation)
    # `Base.Docs.meta` rather than `Documenter.DocSystem.getmeta`: the latter is a Documenter
    # INTERNAL with nothing pinning it. Both return the same 189 bindings, and `Base.Docs.meta` is
    # the more stable of the two, so this file no longer depends on Documenter's private surface.
    for (binding, multidoc) in Base.Docs.meta(MultilevelKernelInterpolation)
        key = (string(binding.mod), string(binding.var))
        obj = try
            Core.eval(binding.mod, binding.var)
        catch err
            push!(problems, "binding $(key[1]).$(key[2]) does not evaluate: $(err)")
            continue
        end
        found = Symbol[]
        for (_, docstr) in multidoc.docs
            parts = splitpath(relpath(normpath(docstr.data[:path]), root))
            if isempty(parts) || parts[1] != "src"
                if haskey(EXTERNALLY_RECORDED, key)
                    push!(seenexternal, key)
                    push!(found, EXTERNALLY_RECORDED[key])
                else
                    push!(
                        problems,
                        "binding $(key[1]).$(key[2]) has a docstring recorded outside `src/` and " *
                        "is not in EXTERNALLY_RECORDED; classify it there deliberately",
                    )
                end
            elseif length(parts) == 2
                # `src/MultilevelKernelInterpolation.jl`, the module file itself.
                push!(found, :tree)
            elseif haskey(AREA_OF_SOURCE_DIR, parts[2])
                push!(found, AREA_OF_SOURCE_DIR[parts[2]])
            else
                push!(
                    problems,
                    "binding $(key[1]).$(key[2]) lives in unlisted source directory " *
                    "`src/$(parts[2])`; add it to AREA_OF_SOURCE_DIR",
                )
            end
        end
        isempty(found) && continue
        area = if length(unique(found)) == 1
            first(found)
        elseif haskey(MULTI_AREA_HOME, key)
            push!(seenmulti, key)
            home = MULTI_AREA_HOME[key]
            if !(home in found)
                push!(
                    problems,
                    "MULTI_AREA_HOME sends $(key[1]).$(key[2]) to $(home), but its documented " *
                    "methods are in $(sort(unique(found))); the configured home is not one of them",
                )
            end
            home
        else
            push!(
                problems,
                "binding $(key[1]).$(key[2]) has documented methods in several areas " *
                "$(sort(unique(found))) and no entry in MULTI_AREA_HOME; pick its home there",
            )
            continue
        end
        if haskey(areas, obj) && areas[obj] != area
            push!(
                problems,
                "the object behind $(key[1]).$(key[2]) is already mapped to $(areas[obj]) but " *
                "this binding wants $(area); two bindings resolving to one object disagree",
            )
            continue
        end
        areas[obj] = area
    end
    for stale in sort(collect(setdiff(keys(EXTERNALLY_RECORDED), seenexternal)))
        push!(
            problems,
            "EXTERNALLY_RECORDED lists $(stale[1]).$(stale[2]), but no docstring for it was " *
            "recorded outside `src/`; the entry is stale and must be removed",
        )
    end
    for stale in sort(collect(setdiff(keys(MULTI_AREA_HOME), seenmulti)))
        push!(
            problems,
            "MULTI_AREA_HOME lists $(stale[1]).$(stale[2]), but it no longer spans several areas; " *
            "the entry is stale and must be removed",
        )
    end
    # Deduplicated: a binding with many docstrings would otherwise report the same problem once per
    # docstring, burying the distinct failures.
    isempty(problems) || error(
        "the documentation page partition is incomplete or ambiguous:\n  " *
        join(sort(unique(problems)), "\n  "),
    )
    return areas
end

const AREAS = build_area_map()

# STRICT lookup. A miss is a partition bug and must surface, not silently file a docstring under a
# default page where nobody would notice it is in the wrong place.
function inarea(area)
    return function (obj)
        haskey(AREAS, obj) || error(
            "no documentation area for $(obj); build_area_map did not classify it, so the " *
            "partition and the docsystem have diverged",
        )
        return AREAS[obj] === area
    end
end

include(joinpath(@__DIR__, "ensureplots.jl"))

# Generated moment docstrings point into H2Factory, including when it is a cached Git dependency.
const DEPENDENCY_REMOTES = Dict{String,Any}()
let factory = MultilevelKernelInterpolation.H2Factory,
    info = Pkg.dependencies()[Base.PkgId(factory).uuid]

    realpath(info.source) == realpath(pkgdir(factory)) || error(
        "Loaded H2Factory does not match the documentation environment; start a fresh Julia session",
    )
    if info.is_tracking_repo &&
        info.git_source == "https://github.com/djukic14/H2Factory.jl.git"
        DEPENDENCY_REMOTES[pkgdir(factory)] = (
            Documenter.Remotes.GitHub("djukic14", "H2Factory.jl"), info.git_revision
        )
    end
end

makedocs(;
    modules=[MultilevelKernelInterpolation],
    remotes=DEPENDENCY_REMOTES,
    authors="djukic14 <danijel.jukic14@gmail.com> and contributors",
    sitename="MultilevelKernelInterpolation.jl",
    format=Documenter.HTML(;
        prettyurls=true,
        canonical="https://djukic14.github.io/MultilevelKernelInterpolation.jl",
        edit_link="main",
        assets=["assets/guide.css", "assets/basis-explorer.js"],
    ),
    # `checkdocs=:all` holds every docstring to a page. It caught an earlier, incomplete split, so it
    # stays. Note what it does NOT do: it proves coverage, not correct placement. The fail-closed
    # partition above is what covers placement.
    checkdocs=:all,
    pages=[
        "Home" => "index.md",
        "Using the package" => [
            "Constructing a map" => "getting-started.md",
            "Choosing parameters" => "guide/approximation.md",
            "Supported operators" => "guide/operators.md",
        ],
        "Polynomials" => [
            "Choosing a basis" => "polynomials/choosing.md",
            "Lagrange interpolation" => "polynomials/lagrange.md",
            "B-splines" => "polynomials/bsplines.md",
        ],
        "Level translations: I2I and O2O" => "transfers/i2i.md",
        "Far-field coupling" => [
            "Dense I2O" => "coupling/dense.md",
            "Toeplitz I2O" => "coupling/toeplitz.md",
            "B-spline I2O" => "coupling/bspline.md",
        ],
        "Moments" => [
            "Moment integrals" => "moments/integrals.md",
            "Assembling collections" => "moments/assembly.md",
            "Moment traits and storage" => "moments/storage.md",
        ],
        "Extending the method" => [
            "Adding an operator" => "extending/operators.md",
            "Custom moment assembly" => "extending/moments.md",
            "Custom translations" => "extending/translations.md",
        ],
        "API reference" => [
            "Maps and construction" => "api/maps.md",
            "Polynomials" => "api/polynomials.md",
            "Translations" => "api/translations.md",
            "Moments" => "api/moments.md",
            "Moment multiplication" => "api/mul.md",
            "Conjugated moment multiplication" => "api/conjmul.md",
            "Operators" => "api/operators.md",
            "Tree and module surface" => "api/tree.md",
        ],
    ],
)
