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


function add_border(mesh::TriMesh, npoints::Integer, expansion=1.2)
    ii = vec(mesh.point_marker .== 1)
    edge_points = mesh.point[:, ii]
    cm = mean(mesh.point, dims=2)

    θ = [atan(y, x) for (x, y) in eachcol(edge_points .- cm)]
    ds = vec(sqrt.(sum(diff(edge_points[:, sortperm(θ)], dims=2).^2, dims=1)))
    s = cumsum([0; ds])
    xinterp = LinearInterpolation(s, edge_points[1, sortperm(θ)])
    yinterp = LinearInterpolation(s, edge_points[2, sortperm(θ)])

    Δs = (s[end] - s[1]) / (npoints-1)
    ss = s[1]:Δs:(s[end] - Δs)
    border_points = [xinterp(ss) yinterp(ss)]'
    border_points = (border_points .- cm) * 1.2 .+ cm

    nodes = collect([mesh.point border_points]')
    return create_mesh(nodes)
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

function setup_model_mesh(points::AbstractMatrix, nnodes::Integer;
        refine=0, nborder=round(Int, sqrt(nnodes)), border_expansion=1.1)
    nodes = collect(kmeans(points, nnodes).centers')
    mesh = create_mesh(nodes)
    mesh = add_border(mesh, nborder, border_expansion)
    if refine > 0
        mesh = refine(mesh, divide_cell_into=refine)
    end
    return mesh
end
