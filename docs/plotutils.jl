using Markdown
using Pkg
using SHA

const PLOT_NAMES = ("lagrange_basis", "bspline_basis", "node_choices", "spline_solve")

# Show the same source that generates the figure, without display-only setup.
function displayedcode(path::AbstractString)
    lines = filter(line -> !endswith(rstrip(line), "# hide"), readlines(path))
    return Markdown.parse("```julia\n" * strip(join(lines, "\n")) * "\n```")
end

function sourcehash(root::AbstractString)
    entries = (
        "src",
        "docs/plots",
        "docs/src/assets/code/figures.jl",
        "docs/plotutils.jl",
        "docs/genplots.jl",
        "docs/ensureplots.jl",
        "docs/Project.toml",
        "docs/Manifest.toml",
    )
    io = IOBuffer()
    for entry in entries
        path = joinpath(root, entry)
        files = if isdir(path)
            sort([joinpath(dir, file) for (dir, _, names) in walkdir(path) for file in names])
        else
            [path]
        end
        for file in files
            print(io, relpath(file, root), '\0', bytes2hex(sha256(read(file))), '\0')
        end
    end
    for info in sort(collect(values(Pkg.dependencies())); by=info -> info.name)
        print(io, info.name, '@', info.version, '\0')
    end
    return bytes2hex(sha256(take!(io)))
end
