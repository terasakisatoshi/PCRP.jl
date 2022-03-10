using PCRP
using Documenter

DocMeta.setdocmeta!(PCRP, :DocTestSetup, :(using PCRP); recursive=true)

makedocs(;
    modules=[PCRP],
    authors="Satoshi Terasaki <terasakisatoshi.math@gmail.com> and contributors",
    repo="https://github.com/terasakisatoshi/PCRP.jl/blob/{commit}{path}#{line}",
    sitename="PCRP.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://terasakisatoshi.github.io/PCRP.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/terasakisatoshi/PCRP.jl",
    devbranch="main",
)
