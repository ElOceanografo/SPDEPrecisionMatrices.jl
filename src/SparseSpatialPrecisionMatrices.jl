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
    add_border

include("mesh_utils.jl")
include("spde_precision_matrices.jl")


end # module
