using Test
include("prepare-environment.jl")

@testset "Clean CI project" begin
    root = dirname(dirname(@__DIR__))
    for kind in ("test", "docs")
        projectfile = joinpath(root, kind, "Project.toml")
        original = read(projectfile, String)
        project = ci_project(kind; root)
        @test !haskey(project, "sources")
        runtime = TOML.parsefile(joinpath(root, "Project.toml"))
        @test project["deps"] == merge(TOML.parse(original)["deps"], runtime["deps"])
        @test project["compat"]["julia"] == runtime["compat"]["julia"]
        @test all(
            name -> name == "julia" || haskey(project["deps"], name),
            keys(project["compat"]),
        )
        @test read(projectfile, String) == original
    end
    @test_throws ArgumentError ci_project("unknown"; root)
    mktempdir() do directory
        @test_throws ErrorException checkout_dependency(
            "Invalid", Dict("revision" => "main"), directory
        )
        @test isempty(readdir(directory))
    end
end
