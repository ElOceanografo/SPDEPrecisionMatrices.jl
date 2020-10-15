using SparseSpatialPrecisionMatrices
using Random
using SparseArrays
using TriangleMesh
using LinearAlgebra
using SpecialFunctions
using Test

@testset "Meshes" begin
    Random.seed!(0)
    n = 200
    points = rand(2, n) .* [100, 50]
    mesh = refine(create_mesh(collect(points')))

    A = observation_matrix(mesh, points)
    @test size(A) == (n, mesh.n_point)
    @test nnz(A) == n * 3
    @test all(sum(A, dims=2) .≈ 1)
end

@testset "Precision matrices" begin
    Random.seed!(0)
    n = 200
    points = rand(2, n) .* [100, 50]
    mesh = refine(create_mesh(collect(points')))

    r = 15 # decorrelation range
    σ = 5 # marginal variance
    ν = 2  # smoothness parameter
    d = size(mesh.point, 1)

    for ν in 1:3
        println("ν = $ν")
        κ = calculate_κ(ν, r)
        τ = calculate_τ(ν, d, κ, σ)
        @test κ ≈ sqrt(8ν) / r
        @test τ ≈ sqrt(gamma(ν) / (gamma(ν + d/2) * (4π)^(d/2))) / (σ * κ^ν)

        Q0 = unscaled_precision_matrix(mesh, κ, ν)
        Q = precision_matrix(mesh, r, σ, ν)
        @test all((Q .- τ^2 * Q0) .< 1e-6)
        @test isposdef(Q0 + 1e-3*I)
        @test isposdef(Q + 1e-3*I)
    end
end
