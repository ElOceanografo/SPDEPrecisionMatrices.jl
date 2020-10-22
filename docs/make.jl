using SPDEPrecisionMatrices
using Documenter

makedocs(;
    modules=[SPDEPrecisionMatrices],
    authors="Sam Urmy <oceanographerschoice@gmail.com>, John K Best <isposdef@gmail.com>",
    repo="https://github.com/jkbest2/SPDEPrecisionMatrices.jl/blob/{commit}{path}#L{line}",
    sitename="SPDEPrecisionMatrices.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://jkbest2.github.io/SPDEPrecisionMatrices.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/jkbest2/SPDEPrecisionMatrices.jl",
)
