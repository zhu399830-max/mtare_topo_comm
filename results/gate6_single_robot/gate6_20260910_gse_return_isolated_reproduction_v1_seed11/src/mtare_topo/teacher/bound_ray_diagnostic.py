"""Same-snapshot origin/crossing/reference comparison; NOT an export caster."""
from dataclasses import dataclass
import numpy as np
from .csg_mesh_provenance import ClosedPrimitiveMesh,CSGMeshProvenanceRaycaster
from .mesh_origin_check import PreparedOriginCheck
from .ordered_exit_candidate import ExitCandidate,ordered_exit_candidate
from .csg_interval_raycaster_v2 import verified_interval_hit


def frozen_array(value,dtype):
    value=np.asarray(value,dtype=dtype)
    return np.frombuffer(value.tobytes(),dtype=dtype).reshape(value.shape)


@dataclass(frozen=True)
class BoundRayComparison:
    geometry_sha256: str
    origin_status: str
    candidate: ExitCandidate
    reference: object
    candidate_matches_reference: bool | None


class BoundRayDiagnostic:
    def __init__(self,meshes):
        self._meshes=tuple(ClosedPrimitiveMesh(m.primitive_id,
            frozen_array(m.vertices_xyz_m,np.float64),frozen_array(m.triangle_vertex_indices,np.int64),
            frozen_array(m.triangle_normals,np.float64)) for m in meshes)
        self._origin=PreparedOriginCheck(self._meshes)
        self._caster=CSGMeshProvenanceRaycaster(self._meshes)

    def compare(self,origin,direction,initial_inside,*,maximum_m=50.):
        origin=np.asarray(origin,float);direction=np.asarray(direction,float);initial=np.asarray(initial_inside)
        if direction.shape!=(3,) or not np.isfinite(direction).all() or np.linalg.norm(direction)==0:
            raise ValueError('finite nonzero direction required')
        if initial.shape!=(len(self._meshes),) or not np.isin(initial,[0,1]).all():
            raise ValueError('binary initial occupancy required')
        direction=direction/np.linalg.norm(direction);checked=self._origin.check(origin)
        import open3d as o3d
        raw={k:v.numpy() for k,v in self._caster.scene.list_intersections(
            o3d.core.Tensor(np.r_[origin,direction][None].astype(np.float32))).items()}
        ids=[self._caster.geometry_to_operand[int(g)] for g in raw['geometry_ids']]
        if checked.status!='origin_checked':
            candidate=ExitCandidate('needs_reference',reason='origin:'+checked.reason)
        elif checked.inside!=tuple(bool(x) for x in initial):
            candidate=ExitCandidate('needs_reference',reason='initial_occupancy_mismatch')
        else:
            dots=[float(self._meshes[i].triangle_normals[int(t)]@direction) for i,t in zip(ids,raw['primitive_ids'])]
            candidate=ordered_exit_candidate(raw['t_hit'],ids,dots,checked.inside,maximum_m=maximum_m)
        reference=verified_interval_hit(self._meshes,origin,direction,initial,raw['t_hit'],ids,maximum_m=maximum_m)
        match=None
        if candidate.status=='candidate':
            identities=tuple(self._meshes[i].primitive_id for i in candidate.operands)
            match=reference is not None and candidate.distance_m==reference.distance_m and identities==reference.source_primitive_ids
        return BoundRayComparison(checked.geometry_sha256,checked.status,candidate,reference,match)
