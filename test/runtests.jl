using SparseSpatialPrecisionMatrices
using Random
using SparseArrays
using Test


Random.seed!(1)

@testset "Meshes" begin
    Random.seed!(0)
    n = 2_000
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
    @test 1 == 1
end
