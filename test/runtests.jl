using SparseSpatialPrecisionMatrices
using Random
using SparseArrays
using LinearAlgebra
using SpecialFunctions
using Test

@testset "Meshes" begin
    Random.seed!(0)
    n = 2000
    nnodes = 400
    points = rand(2, n) .* [100, 50]
    mesh = setup_model_mesh(points, nnodes)
    @test n > mesh.n_point > nnodes

    xbounds, ybounds = extrema(mesh.point, dims=2)
    @test all(xbounds[1] .< points[1, :] .< xbounds[2])
    @test all(ybounds[1] .< points[2, :] .< ybounds[2])

    A = observation_matrix(mesh, points)
    @test size(A) == (n, mesh.n_point)
    @test nnz(A) == n * 3
    @test all(sum(A, dims=2) .≈ 1)
end

@testset "Precision matrices" begin
    Random.seed!(0)
    n = 2000
    nnodes = 400
    points = rand(2, n) .* [100, 50]
    mesh = setup_model_mesh(points, nnodes)

    r = 15 # decorrelation range
    σ = 5 # marginal variance
    ν = 2  # smoothness parameter
    d = size(mesh.point, 1)

    for ν in 1:5
        κ = calculate_κ(ν, r)
        τ = calculate_τ(ν, d, κ, σ)
        @test κ ≈ sqrt(8ν) / r
        @test τ ≈ sqrt(gamma(ν) / (gamma(ν + d/2) * (4π)^(d/2))) / (σ * κ^ν)
        Q0 = unscaled_precision_matrix(mesh, κ, ν)
        Q = precision_matrix(mesh, r, σ, ν)
        @test all(Q .≈ τ^2 * Q0)
        # @test isposdef(Q0)
        # @test isposdef(Q)
    end
end
