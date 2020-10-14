using SparseSpatialPrecisionMatrices
using Plots
using TriangleMesh
using Clustering
using LinearAlgebra, SparseArrays
using Distributions
using PDMats
using Random

n = 2_000
Random.seed!(0)
corners = 100 * [0 0; 0 1; 1 1; 1 0.0]
pts = 100 * rand(n-4,2)
nodes = [kmeans(pts', 400).centers'; corners]
scatter(pts[:,1], pts[:, 2])
scatter!(nodes[:,1], nodes[:, 2])
mesh = create_mesh(nodes, voronoi=true)
mesh = refine(mesh, divide_cell_into=2, voronoi=true)


r = 15 # decorrelation range
σ = 5 # marginal variance
ν = 2  # smoothness parameter

Q = precision_matrix(mesh, r, σ, ν)
isposdef(Q)
length(nnz(Q.data)) / length(Q)

U = cholesky(Q).PtL'

x = U \ randn(mesh.n_point)
surface(mesh.point[1, :], mesh.point[2, :], x, camera=(45, 80))

xg = U \ rand(Gamma(0.02, 1000), mesh.n_point)
surface(mesh.point[1, :], mesh.point[2, :], xg, camera=(45, 80))

D = MvNormalCanon(PDSparseMat(sparse(Q)))
s = MersenneTwister(1234)
z = randn(s, mesh.n_point)
x1 = rand(D)
# x1 = D.J.chol.PtL' \ z

x2 = U \ randn(s, mesh.n_point)
surface(mesh.point[1, :], mesh.point[2, :], x, camera=(45, 80))
surface(mesh.point[1, :], mesh.point[2, :], x1, camera=(45, 80))
surface(mesh.point[1, :], mesh.point[2, :], x2, camera=(45, 80))
logpdf(D, x)

mesh2 = add_border(mesh, 50, 1.1)
plotmesh(mesh2)

Q2 = precision_matrix(mesh2, r, σ, ν)
D2 = MvNormalCanon(PDSparseMat(sparse(Q2)))
x2 = rand(D2)
surface(mesh2.point[1, :], mesh2.point[2, :], x2, camera=(45, 80))


A = observation_matrix(mesh2, pts')

xobs = A * x2

scatter(pts[:,1], pts[:,2], zcolor=xobs)
