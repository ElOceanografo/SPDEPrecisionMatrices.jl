using SPDEPrecisionMatrices
using Documenter

makedocs(;
    modules=[SPDEPrecisionMatrices],
    authors="Sam Urmy <oceanographerschoice@gmail.com>",
    repo="https://github.com/eloceanografo/SPDEPrecisionMatrices.jl/blob/{commit}{path}#L{line}",
    sitename="SPDEPrecisionMatrices.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://eloceanografo.github.io/SPDEPrecisionMatrices.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/eloceanografo/SPDEPrecisionMatrices.jl",
)
