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

function checkout_dependency(name, spec, directory)
    revision = spec["revision"]
    occursin(r"^[0-9a-f]{40}$", revision) ||
        error("$name must be pinned to a full commit hash")
    destination = joinpath(directory, name)
    run(`git clone --quiet --no-checkout $(spec["url"]) $destination`)
    run(`git -C $destination checkout --quiet --detach $revision`)
    readchomp(`git -C $destination rev-parse HEAD`) == revision ||
        error("$name checkout does not match its pin")
    isempty(readchomp(`git -C $destination status --porcelain`)) ||
        error("$name checkout is dirty")
    project = TOML.parsefile(joinpath(destination, "Project.toml"))
    project["name"] == name && project["uuid"] == spec["uuid"] ||
        error("$name checkout has an unexpected package identity")
    return destination
end

function prepare_environment(kind; root=dirname(dirname(@__DIR__)))
    root = realpath(root)
    project = ci_project(kind; root)
    # Neither the development manifest nor sibling-checkout overrides enter CI.
    environment = mktempdir(; cleanup=false)
    open(joinpath(environment, "Project.toml"), "w") do io
        return TOML.print(io, project; sorted=true)
    end
    pins = TOML.parsefile(joinpath(root, ".github", "ci", "dependencies.toml"))
    checkouts = Dict("MultilevelKernelInterpolation" => root)
    ids = Dict(
        "MultilevelKernelInterpolation" => UUID("5e691252-11fb-440e-a586-271db86c0410")
    )
    for name in sort!(collect(keys(pins)))
        spec = pins[name]
        kind in spec["environments"] || continue
        checkouts[name] = checkout_dependency(name, spec, environment)
        ids[name] = UUID(spec["uuid"])
    end

    Pkg.Registry.add("General")
    Pkg.activate(environment)
    Pkg.develop([Pkg.PackageSpec(; path=path) for path in values(checkouts)])
    Pkg.instantiate()

    dependencies = Pkg.dependencies()
    for info in values(dependencies)
        info.is_tracking_path &&
            !haskey(checkouts, info.name) &&
            error("Unexpected path dependency: $(info.name) at $(info.source)")
    end
    for name in sort!(collect(keys(checkouts)))
        info = dependencies[ids[name]]
        info.is_tracking_path && realpath(info.source) == realpath(checkouts[name]) ||
            error("$name did not resolve to the intended checkout")
        println("$name: ", info.source)
    end
    for name in ("BEAST", "CompScienceMeshes", "H2Trees", "NURBS")
        matches = [info for info in values(dependencies) if info.name == name]
        length(matches) == 1 || error("Expected exactly one $name dependency")
        info = only(matches)
        info.is_tracking_registry || error("$name must resolve from a registry")
        println("$name: registry version ", info.version, ", tree ", info.tree_hash)
    end
    return environment
end
