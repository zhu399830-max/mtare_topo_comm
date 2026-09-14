"""Reconstruct source-axis arc bounds for the original swept mesh triangles.

This is a surface geometry index, not point labels. An actual return needs
independent, source-bound surface matching before these intervals are usable.
No nearest axis/GT-center assignment or ray-crossing substitution is made.
"""
from dataclasses import dataclass
import numpy as np
from .csg_mesh_provenance import ClosedPrimitiveMesh,mesh_swept_superellipse
from .swept_superellipse_field import _sample_operand


@dataclass(frozen=True)
class SurfaceArcIndex:
    primitive_id: str
    vertex_arc_m: np.ndarray
    triangle_arc_bounds_m: np.ndarray
    point_membership_qualified: bool = False


def index_surface_arcs(primitive, mesh, *, axial_spacing_m, angular_segments):
    """Explicit original tessellation settings; exact mesh equality required."""
    if not isinstance(mesh,ClosedPrimitiveMesh) or mesh.primitive_id!=primitive.primitive_id:
        raise ValueError('same source primitive mesh required')
    expected=mesh_swept_superellipse(primitive,axial_spacing_m=axial_spacing_m,
                                    angular_segments=angular_segments)
    if any(not np.array_equal(getattr(mesh,k),getattr(expected,k)) for k in (
        'vertices_xyz_m','triangle_vertex_indices','triangle_normals')):
        raise ValueError('original mesh geometry/order or tessellation settings drift')
    sampled=_sample_operand(primitive,axial_spacing_m)
    vertex_arc=np.concatenate((np.repeat(sampled.arc_m,angular_segments),sampled.arc_m[[0,-1]]))
    if len(vertex_arc)!=len(mesh.vertices_xyz_m):raise ValueError('vertex provenance mismatch')
    face_arcs=vertex_arc[mesh.triangle_vertex_indices]
    bounds=np.stack((face_arcs.min(axis=1),face_arcs.max(axis=1)),axis=1)
    vertex_arc.setflags(write=False);bounds.setflags(write=False)
    return SurfaceArcIndex(primitive.primitive_id,vertex_arc,bounds)
