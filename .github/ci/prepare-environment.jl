using Pkg
using TOML
using UUIDs

function ci_project(kind; root=dirname(dirname(@__DIR__)))
    kind in ("test", "docs") || throw(ArgumentError("Unknown CI environment: $kind"))
    project = TOML.parsefile(joinpath(root, kind, "Project.toml"))
    pop!(project, "sources", nothing)
    runtime = TOML.parsefile(joinpath(root, "Project.toml"))
    # Direct dependencies keep Pkg from borrowing paths from the package's own manifest.
    merge!(project["deps"], runtime["deps"])
    compat = get!(project, "compat", Dict{String,Any}())
    for (name, bound) in runtime["compat"]
        if name == "julia" || haskey(project["deps"], name)
            get!(compat, name, bound)
        end
    end
    return project
end

function prepare_environment(kind; root=dirname(dirname(@__DIR__)))
    root = realpath(root)
    project = ci_project(kind; root)
    # Neither the development manifest nor sibling-checkout overrides enter CI.
    environment = mktempdir(; cleanup=false)
    open(joinpath(environment, "Project.toml"), "w") do io
        return TOML.print(io, project; sorted=true)
    end
    Pkg.Registry.add("General")
    Pkg.Registry.update()
    Pkg.activate(environment)
    Pkg.develop(Pkg.PackageSpec(; path=root))
    Pkg.instantiate()

    dependencies = Pkg.dependencies()
    for info in values(dependencies)
        info.is_tracking_path &&
            info.name != "MultilevelKernelInterpolation" &&
            error("Unexpected path dependency: $(info.name) at $(info.source)")
    end
    info = dependencies[UUID(project["deps"]["MultilevelKernelInterpolation"])]
    info.is_tracking_path && realpath(info.source) == root ||
        error("MultilevelKernelInterpolation did not resolve to the intended checkout")
    println("MultilevelKernelInterpolation: ", info.source)
    registry_names = ["BEAST", "CompScienceMeshes", "H2Trees", "NURBS", "H2Factory"]
    kind == "test" && push!(registry_names, "H2FactoryTestUtils")
    for name in registry_names
        matches = [info for info in values(dependencies) if info.name == name]
        length(matches) == 1 || error("Expected exactly one $name dependency")
        info = only(matches)
        info.is_tracking_registry || error("$name must resolve from a registry")
        println("$name: registry version ", info.version, ", tree ", info.tree_hash)
    end
    return environment
end
