module SparseSpatialPrecisionMatrices

using TriangleMesh
using Interpolations
using SparseArrays
using LinearAlgebra
using SpecialFunctions
using Statistics
using Clustering
using NearestNeighbors


export component_matrices,
    unscaled_precision_matrix,
    precision_matrix,
    setup_model_mesh,
    add_border,
    observation_matrix

include("mesh_utils.jl")
include("spde_precision_matrices.jl")


end # module
