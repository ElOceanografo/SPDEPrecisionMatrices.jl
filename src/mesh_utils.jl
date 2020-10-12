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