"""
    component_matrices(mesh::TriMesh, κ::real) -> (C::Matrix, C̃::Matrix, G::Matrix)

From a triangulated mesh of a 2-dimensional domain, generate the finite element
method matrices C, C̃, and G as described in Lindgren et al. 2011.

# Arguments
- `mesh::TriMesh`: a mesh from TriangleMesh.jl
- `κ::Real`: Matérn correlation decay rate parameter

# Returns
- `(Matrix, Matrix, Matrix)`: the C, C̃, and G matrices, respectively
"""
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

"""
    unscaled_precision_matrix(mesh::TriMesh, κ::real, ν::Real) -> Q::Symmetric{Matrix}

Construct a sparse precision matrix representing a Matérn random field on a
triangulated mesh using the finite element method. The returned matrix is not
scaled to either a correlation matrix or have a particular marginal variance.

# Arguments
- `mesh::TriMesh`: a mesh from TriangleMesh.jl
- `κ::Real`: Matérn correlation decay rate parameter
- `ν::Real`: Matérn smoothness parameter, ν + d / 2 must be an integer

# Throws
- Error if ν + d / 2 is not an integer (where d is the dimension of the domain)

# Returns
- `Symmetric`: the precision matrix of a random field on `mesh` with Matérn
  covariance.
"""
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

"""
    calculate_κ(ν, r) -> Real

Calculate the value of the correlation decay rate parameter given a smoothness `ν`
and range `r` where correlation drops to roughly 0.1, calculate the value of κ

```math
κ = \frac{\sqrt{8 ν}}{r}\\
```

# Arguments
- `ν`: Matérn smoothness
- `r`: correlation range

# Returns
- `κ`: correlation decay rate parameter value
"""
calculate_κ(ν, r) = sqrt(8ν) / r

"""
    calculate_τ(ν, d, κ, σ) -> Real

Calculate the scaling factor for a Matérn precision matrix given a smoothness,
domain dimension, correlation distance, and marginal standard deviation.

```math
τ = \frac{1}{\sigma \kappa^\nu}{\Gamma(\nu)}{\Gamma(\nu + d / 2) (4 \pi)^{d / 2}}
```

# Arguments
- `ν`: Matérn smoothness
- `d`: Dimension of the domain (i.e. 2 for spatial domains)
- `κ`: correlation decay rate
- `σ`: marginal standard deviation

# Returns
- `τ`: Precision matrix scaling parameter
"""
calculate_τ(ν, d, κ, σ) = sqrt(gamma(ν) / (gamma(ν + d/2) * (4π)^(d/2))) / (σ * κ^ν)

"""
    precision_matrix(mesh::TriMesh, r::Real, σ::Real, ν::Integer)

Generate the precision matrix of a Markov random field with Matérn covariance.

# Arguments
- `mesh::TriMesh`: a mesh from TriangleMesh.jl
- `r::Real`: range where correlation is reduced to 0.1
- `σ::Real`: marginal standard deviation
- `ν::Integer`: Matérn smoothness parameter; for spatial (d = 2) must be an integer

# Returns
- `Symmetric`: precision matrix scaled to given marginal standard deviation
"""
function precision_matrix(mesh::TriMesh, r::Real, σ::Real, ν::Integer)
    d = size(mesh.point, 1)
    κ = calculate_κ(ν, r)
    τ = calculate_τ(ν, d, κ, σ)
    Q = unscaled_precision_matrix(mesh, κ, ν)
    return Q * τ^2
end
