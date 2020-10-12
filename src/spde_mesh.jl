using TriangleMesh
using Interpolations
using Clustering
using SparseArrays
using LinearAlgebra
using Distributions, PDMats
using Random
using Plots

function plotmesh(m :: TriMesh;
                        linewidth :: Real = 1,
                        marker :: String = "None",
                        markersize :: Real = 10,
                        color = :black)
    p = plot()
    for edge in eachcol(m.edge)
        xy = m.point[:, edge]
        plot!(p, xy[1, :], xy[2, :], linewidth=linewidth,
            markersize=markersize, color=color, legend=false)
    end
    return p
end

function component_matrices(mesh, κ)
    d = size(mesh.point, 1)
    n_edge = size(mesh.edge, 2)
    ii = zeros(Int, n_edge)
    jj = zeros(Int, n_edge)

    for (ei, edge) in enumerate(eachcol(mesh.edge))
        ii[ei], jj[ei] = edge
    end

    ii = [ii; jj]
    jj = [jj; ii[1:n_edge]]

    G = sparse(ii, jj, zeros(2*n_edge))
    C = sparse(ii, jj, zeros(2*n_edge))
    C_tilde = spdiagm(0 => zeros(mesh.n_point))

    for triangle in eachcol(mesh.cell)
        i, j, k = triangle
        vi, vj, vk = [mesh.point[:, ind] for ind in [i, j, k]]
        ei = vk - vj
        ej = vi - vk
        ek = vj - vi
        edges = Dict(i => ei, j => ej, k => ek)
        θk = acos(dot(ei, ej) / (norm(ei)*norm(ej)))
        area = 0.5 * norm(ei) * norm(ej) * sin(θk)

        for m in [i, j, k]
            C_tilde[m, m] += area / 3
        end

        for m in [i, j, k], n in [i, j, k]
            if m == n
                C[m, n] += area / 6
            else
                C[m, n] += area / 12
            end
            G[m, n] += 1 / (4 * area) * dot(edges[m], edges[n])
        end
    end

    return C, C_tilde, G
end

function unscaled_precision_matrix(mesh, κ, ν)
    d = size(mesh.point, 1)
    α = ν + div(d, 2)
    α::Integer

    C, C_tilde, G = component_matrices(mesh, κ)
    C_inv = spdiagm(0 => 1 ./ diag(C_tilde))
    K = Symmetric(κ^2 * C + G)

    if α == 1
        return K
    elseif α == 2
        return Symmetric(K * C_inv * K)
    else
        Qminus2 = K
        Qminus1 = Symmetric(K * C_inv * K)
        Q = Symmetric(K * C_inv * Qminus2 * C_inv * K)
        i = 3
        while i < α
            Qminus2 = Qminus1
            Qminus1 = Q
            Q = Symmetric(K * C_inv * Qminus2 * C_inv * K)
            i += 1
        end
        return Symmetric(Q)
    end
end

function precision_matrix(mesh, r, σ, ν::Integer)
    d = size(mesh.point, 1)
    κ = sqrt(8ν) / r
    τ = sqrt(gamma(ν) / (gamma(ν + d/2) * (4π)^(d/2))) / (σ * κ^ν)
    Q = unscaled_precision_matrix(mesh, κ, ν)
    return Q * τ^2
end

function add_border(mesh, npoints, expansion=1.2)
    ii = vec(mesh.point_marker .== 1)
    edge_points = mesh.point[:, ii]
    cm = mean(mesh.point, dims=2)

    θ = [atan(y, x) for (x, y) in eachcol(edge_points .- cm)]
    ds = vec(sqrt.(sum(diff(edge_points[:, sortperm(θ)], dims=2).^2, dims=1)))
    s = cumsum([0; ds])
    xinterp = LinearInterpolation(s, edge_points[1, sortperm(θ)])
    yinterp = LinearInterpolation(s, edge_points[2, sortperm(θ)])

    Δs = (s[end] - s[1]) / npoints
    ss = s[1]:Δs:(s[end] - Δs)
    border_points = [xinterp(ss) yinterp(ss)]'
    border_points = (border_points .- cm) * 1.2 .+ cm

    nodes = collect([mesh.point border_points]')
    return create_mesh(nodes)
end


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

using Optim
using ForwardDiff

function obj(θ)
    r, σ = exp.(θ)
    Q = precision_matrix(mesh, r, σ, 2)
    d = MvNormalCanon(PDSparseMat(sparse(Q)))
    return -logpdf(d, x)
end

obj(log.([1, 2]))

opt = optimize(obj, randn(2))

exp.(opt.minimizer)
