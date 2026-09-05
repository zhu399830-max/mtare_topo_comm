"""Deterministic world-generation primitives for structural-topology research."""

from .tng import (
    GenerationError,
    SeedBundle,
    SplitLeakageError,
    TNGEdge,
    TNGNode,
    TNGParameters,
    TunnelNetworkGraph,
    Vec3,
    generate_tng,
    minimum_nonincident_node_edge_sample_distance,
    minimum_nonincident_edge_sample_distance,
    validate_parent_split,
    tng_from_dict,
)
from .tunnel_mesh import (
    MeshGenerationError,
    TriangleMesh,
    TunnelMeshParameters,
    generate_tunnel_mesh,
    mesh_stats,
    write_obj,
    write_usda,
)

__all__ = [
    "GenerationError",
    "MeshGenerationError",
    "SeedBundle",
    "SplitLeakageError",
    "TNGEdge",
    "TNGNode",
    "TNGParameters",
    "TriangleMesh",
    "TunnelNetworkGraph",
    "TunnelMeshParameters",
    "Vec3",
    "generate_tng",
    "generate_tunnel_mesh",
    "mesh_stats",
    "minimum_nonincident_node_edge_sample_distance",
    "minimum_nonincident_edge_sample_distance",
    "validate_parent_split",
    "tng_from_dict",
    "write_obj",
    "write_usda",
]
