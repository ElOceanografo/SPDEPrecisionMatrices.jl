using SparseSpatialPrecisionMatrices
using Plots
using TriangleMesh

n = 10_000
Random.seed!(0)
corners = 100 * [0 0; 0 1; 1 1; 1 0.0]
pts = [100 * rand(n-4,2); corners]

nodes = [kmeans(pts', 1000).centers'; corners]
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
