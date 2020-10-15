module SparseSpatialPrecisionMatrices

using TriangleMesh
using Interpolations
using SparseArrays
using LinearAlgebra
using SpecialFunctions
using Statistics
using Clustering
using NearestNeighbors
using ConcaveHull


export component_matrices,
    unscaled_precision_matrix,
    calculate_κ,
    calculate_τ,
    precision_matrix,
    make_border_points,
    observation_matrix

include("mesh_utils.jl")
include("spde_precision_matrices.jl")


end # module
