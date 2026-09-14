"""Independent float64 linear solve of archived float32 ray/triangle hits.

Reports residuals; intentionally no tuned pass threshold or label production.
"""
import numpy as np


def ray_triangle_audit(vertices, faces, rays, ray_indices, triangle_indices, stored_t):
    vertices=np.asarray(vertices);faces=np.asarray(faces);rays=np.asarray(rays)
    ri=np.asarray(ray_indices,dtype=np.int64);ti=np.asarray(triangle_indices,dtype=np.int64)
    t=np.asarray(stored_t,dtype=np.float32)
    if vertices.dtype!=np.float32 or rays.dtype!=np.float32 or rays.ndim!=2 or rays.shape[1]!=6:
        raise ValueError('original float32 vertices and packed rays required')
    if (ri.ndim!=1 or ri.shape!=ti.shape or ri.shape!=t.shape or not len(ri)
            or (ri<0).any() or (ri>=len(rays)).any() or (ti<0).any() or (ti>=len(faces)).any()):
        raise ValueError('nonempty bounded archived hit identities required')
    tri=vertices[faces[ti]].astype(np.float64)
    origin=rays[ri,:3].astype(np.float64);direction=rays[ri,3:].astype(np.float64)
    e1,e2=tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]
    matrix=np.stack((direction,-e1,-e2),axis=-1)
    solution=np.linalg.solve(matrix,(tri[:,0]-origin)[...,None])[...,0]
    if not np.isfinite(solution).all():raise ValueError('nonfinite triangle solve')
    hit_t,u,v=solution.T
    bary=np.stack((1-u-v,u,v),axis=-1)
    delta=hit_t-t.astype(np.float64)
    normals=np.cross(e1,e2);normals/=np.linalg.norm(normals,axis=1)[:,None]
    cosines=np.sum(normals*direction,axis=1)
    return dict(hits=len(ri),max_abs_t_error_m=float(np.max(np.abs(delta))),
        max_abs_t_error_float32_ulps=float(np.max(np.abs(delta)/np.abs(np.spacing(t)).astype(np.float64))),
        exact_float32_t_matches=int(np.sum(hit_t.astype(np.float32)==t)),
        minimum_barycentric=float(bary.min()),negative_barycentric_hits=int(np.sum((bary<0).any(axis=1))),
        minimum_t=float(hit_t.min()),entering_hits=int(np.sum(cosines<0)),leaving_hits=int(np.sum(cosines>0)),
        min_normal_dot_ray=float(cosines.min()),max_normal_dot_ray=float(cosines.max()),
        all_hits_in_triangle_at_float64=bool((bary>=0).all()),label_qualification=False)
