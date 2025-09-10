# Hosting

## Using GitHub actions

1) Get `docs` to have it's own `Project.toml` as suggested in the Documenter's
   [guide](https://documenter.juliadocs.org/stable/man/guide/). This project must have all the
   packages needed to run `docs/make.jl`

    ```julia
    pkg> activate docs
    pkg> add Documenter
    pkg> add BenchmarkTools
    pkg> add Plots
    pkg> dev ../ComputationGraphs
    ```

2) Create the file `.github/workflows/documentation.yml` as in
   [https://documenter.juliadocs.org/stable/man/hosting/#GitHub-Actions]

    !!! Attention:
        + Get the right branch for the trigger: `main`
        + Use `GITHUB_TOKEN`, which means commenting the `DOCUMENTER_KEY` line

3) Add to `docs/make.jl`

    ```julia
    deploydocs(
        repo="github.com/hespanha/ComputationGraphs.git",
        branch="gh-pages", # branch where the generated documentation, using default
        target="build",    # directory to be deployed, using default
        deploy_config=Documenter.GitHubActions(),
        #push_preview=true,
        push_preview=false
    )
    ```

    This pushes documentation to the `gh-pages` branch. If the branch does not exist it will be
    created automatically by `deploydocs`

4) To trigger the action, just do a pull to `main`

    !!! warning
        The first time seems to take much longer to run

    !! note
        Changed workflow to trigger on tags matching 'v*docs', for example `docs-v0.0.0+take1`

        To trigger the workflow from VS-Code:

        1) create tag (source control/.../Tags/Create Tag)
        2) push tags (Ctrl+Shift+P / Git: Push Tags)

        To trigger the workflow from the command line:

        ```bash
        git tag docs-v0.0.0+take1
        git push origin --tags
        ```

    !!! attention

    When make.jl is triggered by a tag, GIT_REF is constructed from the tag and must be a valid version name

    See:

    + "GITHUB_REF: must match the devbranch keyword to deploydocs, alternatively correspond to a git tag."
    
        (https://documenter.juliadocs.org/stable/man/hosting/#Documenter.GitHubActions)

    + "devbranch is the branch that "tracks" the in-development version of the generated
      documentation. By default Documenter tries to figure this out using git. Can be set
      explicitly as a string (typically "master" or "main")."

        (https://documenter.juliadocs.org/stable/lib/public/#Documenter.deploydocs)

    + https://github.com/JuliaDocs/Documenter.jl/issues/1230#issuecomment-579450827

5) The documentation will appear in the branch `gh-pages`

    (https://github.com/hespanha/ComputationGraphs/tree/gh-pages)

    https://hespanha.github.io/ComputationGraphs/

    !!! error
        github pages are not available for private repositories

## Hosting in readthedocs.org

1) Login to [readthedocs](https://app.readthedocs.org/dashboard/)
2) Add project
    + Create from repository [ComputationGraphs](https://github.com/hespanha/ComputationGraphs)
  
!!! error
    Will only find public repositories
