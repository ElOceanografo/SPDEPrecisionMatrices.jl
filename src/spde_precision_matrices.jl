
function component_matrices(mesh::TriMesh, κ::Real)
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

        # the following is a hack to make sure the argument of acos doesn't
        # end up outside [-1, 1] due to floating point error
        x = dot(ei, ej) / (norm(ei)*norm(ej))
        θk = acos(sign(x) * min(abs(x), 1))
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

function unscaled_precision_matrix(mesh::TriMesh, κ::Real, ν::Real)
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

calculate_κ(ν, r) = sqrt(8ν) / r
calculate_τ(ν, d, κ, σ) = sqrt(gamma(ν) / (gamma(ν + d/2) * (4π)^(d/2))) / (σ * κ^ν)

function precision_matrix(mesh::TriMesh, r::Real, σ::Real, ν::Integer)
    d = size(mesh.point, 1)
    κ = calculate_κ(ν, r)
    τ = calculate_τ(ν, d, κ, σ)
    Q = unscaled_precision_matrix(mesh, κ, ν)
    return Q * τ^2
end
