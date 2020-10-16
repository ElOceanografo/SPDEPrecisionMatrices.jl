using SparseSpatialPrecisionMatrices
using Plots
using TriangleMesh
using Clustering
using LinearAlgebra, SparseArrays
using Distributions
using PDMats
using Random

Random.seed!(1)
n = 2_000
pts = 100 * rand(n-4, 2)

nodes = collect(kmeans(pts', 500).centers')
# corners = [-20 -20; 120 -20; 120 120; -20 120]
corners = [0 0; 100 0; 100 100; 0 100]

mesh = create_mesh([nodes; corners])
mesh = refine(mesh, divide_cell_into=4, voronoi=true)
scatter(mesh.point[1, :], mesh.point[2, :])

r = 15 # decorrelation range
σ = 5 # marginal variance
ν = 2  # smoothness parameter

Q = precision_matrix(mesh, r, σ, ν)
isposdef(Q)
length(nnz(Q.data)) / length(Q)

U = cholesky(Q).PtL'

x = U \ randn(mesh.n_point)
surface(mesh.point[1, :], mesh.point[2, :], x, zlim=4*[-σ, σ], camera=(45, 80))

xg = U \ rand(Gamma(0.1, 100), mesh.n_point)
surface(mesh.point[1, :], mesh.point[2, :], xg, camera=(45, 80))

D = MvNormalCanon(PDSparseMat(sparse(Q)))
logpdf(D, x)
