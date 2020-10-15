# function plotmesh(m :: TriMesh;
#                         linewidth :: Real = 1,
#                         marker :: String = "None",
#                         markersize :: Real = 10,
#                         color = :black)
#     p = plot()
#     for edge in eachcol(m.edge)
#         xy = m.point[:, edge]
#         plot!(p, xy[1, :], xy[2, :], linewidth=linewidth,
#             markersize=markersize, color=color, legend=false)
#     end
#     return p
# end
#


function make_border_points(points, n, expansion, k=size(points, 2))
    hull = concave_hull(collect(eachcol(points)), k)
    cm = mean(points, dims=2)
    points2 = cm .+ 1.01 * (points .- cm)
    edge_points = reduce(hcat, [p for p in eachcol(points2) if ! in_hull(p, hull)])

    θ = [atan(y, x) for (x, y) in eachcol(edge_points .- cm)]
    x = edge_points[1, sortperm(θ)]
    y = edge_points[2, sortperm(θ)]
    θ = [θ; θ[1]]
    x = [x; x[1]]
    y = [y; y[1]]

    ds = vec(sqrt.(sum(diff([x y]', dims=2).^2, dims=1)))
    s = cumsum([0; ds])
    xinterp = LinearInterpolation(s, x)
    yinterp = LinearInterpolation(s, y)

    Δs = (s[end] - s[1]) / n
    ss = s[1]:Δs:s[end]
    border_points = [xinterp(ss) yinterp(ss)]'
    border_points = (border_points .- cm) * expansion .+ cm
end

function inverse_distance_weights(dists::AbstractVector)
    w = [d > 0 ? 1/d : 1/(d+eps()) for d in dists]
    return w / sum(w)
end

function observation_matrix(mesh::TriMesh, points::AbstractMatrix)
    npoint = size(points, 2)
    nmesh = size(mesh.point, 2)
    tree = KDTree(mesh.point)
    idx, dists = knn(tree, points, 3)

    ii = repeat(1:npoint, inner=3)
    jj = reduce(vcat, idx)
    ww = mapreduce(inverse_distance_weights, vcat, dists)
    return sparse(ii, jj, ww, npoint, nmesh)
end
