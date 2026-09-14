"""Versioned mesh-only interval authority for ALL rays, including valid far hits.

Independent of legacy 1cm event grouping. Slow reference implementation, not
yet qualified for dataset export. Ambiguous winding produces no return; it
never falls back to the known gap-bridging legacy result.
"""
import numpy as np
from .csg_mesh_provenance import CSGMeshProvenanceRaycaster
from .mesh_interval_exit import interval_winding_exit
from .primitive_provenance_field import PrimitiveRayHit


def verified_interval_hit(meshes,origin,direction,initial_inside,distances,sources,*,maximum_m=50.):
    direction=np.asarray(direction,float)
    if direction.shape!=(3,) or not np.isfinite(direction).all() or np.linalg.norm(direction)==0:
        raise ValueError('finite nonzero direction required')
    if not np.isfinite(maximum_m) or maximum_m<=0:raise ValueError('positive finite maximum required')
    if len(distances)!=len(sources):raise ValueError('intersection/source length mismatch')
    if any(type(i) not in (int,np.int32,np.int64) or i<0 or i>=len(meshes) for i in sources):
        raise ValueError('invalid intersection source')
    direction=direction/np.linalg.norm(direction)
    result=interval_winding_exit(meshes,origin,direction,initial_inside,distances,sources,maximum_m=maximum_m)
    if result is None:return None
    distance,operands=result
    identities=tuple(meshes[i].primitive_id for i in operands)
    point=np.asarray(origin,float)+distance*direction
    return PrimitiveRayHit(distance,tuple(float(v) for v in point),identities,len(identities)==1)


class CSGIntervalRaycasterV2(CSGMeshProvenanceRaycaster):
    def __init__(self,meshes):
        # Reuse native scene construction, but not the legacy event reducer.
        super().__init__(meshes)

    def ray_exit_hits(self,origins_xyz_m,directions_xyz,initial_inside,*,maximum_m=50.):
        origins=np.asarray(origins_xyz_m,float);directions=np.asarray(directions_xyz,float)
        inside=np.asarray(initial_inside)
        if origins.ndim!=2 or origins.shape[1]!=3 or origins.shape!=directions.shape:
            raise ValueError('matching Nx3 rays required')
        if not np.isfinite(origins).all() or not np.isfinite(directions).all() or (np.linalg.norm(directions,axis=1)==0).any():
            raise ValueError('finite nonzero rays required')
        if inside.shape!=(len(origins),len(self.meshes)) or not np.isin(inside,[0,1]).all():
            raise ValueError('binary ray/operand initial occupancy required')
        directions=directions/np.linalg.norm(directions,axis=1,keepdims=True)
        import open3d as o3d
        raw={k:v.numpy() for k,v in self.scene.list_intersections(o3d.core.Tensor(
            np.concatenate((origins,directions),axis=1).astype(np.float32))).items()}
        output=[]
        for i in range(len(origins)):
            a,b=map(int,raw['ray_splits'][i:i+2])
            output.append(verified_interval_hit(self.meshes,origins[i],directions[i],inside[i].astype(bool),
                raw['t_hit'][a:b],[self.geometry_to_operand[int(g)] for g in raw['geometry_ids'][a:b]],maximum_m=maximum_m))
        return tuple(output)
