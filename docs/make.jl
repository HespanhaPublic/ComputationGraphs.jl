# For development use:
#=
```julia
using Pkg
Pkg.activate("docs")
using LiveServer
servedocs()
```
=#
# ```sh
# http://localhost:8000/
# ````
# This will 
# 1) track any changes to the .md files 
# 2) call make.jl when changes are detected
# 3) refresh the browser as needed

using Documenter
using ComputationGraphs

makedocs(
    modules=Module[ComputationGraphs],
    sitename="ComputationGraphs",
    format=Documenter.HTML(
        # Use clean URLs, unless built as a "local" build
        #prettyurls=!("local" in ARGS),
        prettyurls=false,
        canonical="https://documenter.juliadocs.org/stable/",
        #assets=["assets/favicon.ico"],
        mathengine=Documenter.HTMLWriter.MathJax3(),
        highlights=["yaml"], size_threshold=500000,
        size_threshold_warn=200000,
        #size_threshold_ignore=[
        #    "examples.md",
        #    "lib_public.md",
        #],
    ),
    pages=[
        "Home" => "index.md",
        "User manual" => Any[
            "man_guide.md",
            "man_differentiation.md",
            "man_recipes.md",
            "man_code_generation.md",
            "examples.md",
        ],
        "Reference" => Any[
            "lib_representation.md",
            "lib_public.md",
        ],
    ],
    pagesonly=true, # process only pages in `pages`` keyword
    doctest=false,  # false only for debug 
    warnonly=true,  # simply prints a warning on error
)

#https://documenter.juliadocs.org/stable/lib/public/#Documenter.deploydocs
deploydocs(
    repo="github.com/HespanhaPublic/ComputationGraphs.jl.git",
    branch="gh-pages", # branch where the generated documentation, using default
    target="build",    # directory to be deployed, using default
    deploy_config=Documenter.GitHubActions(),
    versions=["stable" => "v^", "v#.#.#", "dev" => "dev"], # default
    #push_preview=true,
    push_preview=false,
    # see https://documenter.juliadocs.org/stable/man/hosting/#Documentation-Versions
    tag_prefix="docs-", # only version tags with that prefix will trigger deployment
)
