using SPDEPrecisionMatrices2
using Documenter

makedocs(;
    modules=[SPDEPrecisionMatrices2],
    authors="Sam Urmy <oceanographerschoice@gmail.com>, John K Best <isposdef@gmail.com>",
    repo="https://github.com/jkbest2/SPDEPrecisionMatrices2.jl/blob/{commit}{path}#L{line}",
    sitename="SPDEPrecisionMatrices2.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://jkbest2.github.io/SPDEPrecisionMatrices2.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/jkbest2/SPDEPrecisionMatrices2.jl",
)
