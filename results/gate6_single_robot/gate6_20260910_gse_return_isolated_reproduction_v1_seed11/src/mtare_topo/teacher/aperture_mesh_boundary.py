"""Original-mesh geometry proxy with bounded work and whole-cell guards.

Unsigned distance is a float32 Open3D numerical query, not a formally verified
distance interval. Results are engineering diagnostics, not safety proofs or
qualified labels. Near-surface cells and inconsistent winding stay unknown.
"""
from dataclasses import dataclass
import time
import numpy as np
from mtare_topo.evaluation.mesh_winding_diagnostic import oriented_winding


@dataclass(frozen=True)
class MeshBoundaryResult:
    state: np.ndarray
    winding_point_queries: int
    triangle_point_evaluations: int
    elapsed_s: float
    physical_safety_certified: bool=False
    training_labels_qualified: bool=False
    distance_query_is_numerical_proxy: bool=True


def _closed_oriented(indices):
    edges=np.concatenate([indices[:,[0,1]],indices[:,[1,2]],indices[:,[2,0]]])
    canonical=np.sort(edges,axis=1)
    _,inverse,counts=np.unique(canonical,axis=0,return_inverse=True,return_counts=True)
    balance=np.bincount(inverse,weights=np.where(edges[:,0]<edges[:,1],1,-1))
    return bool(np.all(counts==2) and np.all(balance==0))


class OriginalMeshBoundary:
    def __init__(self,meshes):
        import open3d as o3d
        self.o3d=o3d;self.data=[]
        for mesh in meshes:
            if not _closed_oriented(mesh.triangle_vertex_indices):raise ValueError('closed consistently oriented mesh required')
            vertices=mesh.vertices_xyz_m;packed=vertices.astype(np.float32)
            scene=o3d.t.geometry.RaycastingScene()
            scene.add_triangles(o3d.core.Tensor(packed),o3d.core.Tensor(mesh.triangle_vertex_indices.astype(np.uint32)))
            self.data.append((vertices,packed.astype(float),mesh.triangle_vertex_indices,scene))
        if not self.data:raise ValueError('nonempty original mesh union required')

    def classify(self,vertices,faces,*,center_m=(0.,0.,0.),radius_m=10.,max_triangle_point_evaluations=200000000):
        started=time.monotonic();unit=np.asarray(vertices,dtype=float);tri=unit[np.asarray(faces,dtype=int)]
        center=np.asarray(center_m,dtype=float)
        if center.shape!=(3,) or not np.isfinite(center).all() or not np.isfinite(radius_m) or radius_m<=0:raise ValueError('finite sphere required')
        if not np.allclose(np.linalg.norm(unit,axis=1),1.,rtol=0,atol=1e-12):raise ValueError('unit boundary vertices required')
        directions=tri.sum(axis=1);directions/=np.linalg.norm(directions,axis=1)[:,None]
        if np.any(np.einsum('ijk,ik->ij',tri,directions)<=0):raise ValueError('convex covering cap required')
        chord=np.linalg.norm(tri-directions[:,None],axis=2).max(axis=1)
        points=center+radius_m*directions;radii=radius_m*(chord+256*np.finfo(float).eps)
        free=np.zeros(len(tri),bool);blocked=np.ones(len(tri),bool);work=0;queries=0
        for original,packed,indices,scene in self.data:
            states=np.full(len(tri),-1,dtype=np.int8)
            scale=max(1.,radius_m,float(np.max(np.abs(original))),float(np.max(np.abs(points))))
            # Explicit proxy guard, not a claim of a certified Open3D error bound.
            guard=512*np.finfo(np.float32).eps*scale+float(np.linalg.norm(original-packed,axis=1).max())
            outside=np.any((points+radii[:,None]<original.min(axis=0)-guard)|
                           (points-radii[:,None]>original.max(axis=0)+guard),axis=1)
            states[outside]=0;ids=np.flatnonzero(~outside)
            if len(ids):
                distances=scene.compute_distance(self.o3d.core.Tensor(points[ids].astype(np.float32)),nthreads=1).numpy().astype(float)
                if not np.isfinite(distances).all():raise ValueError('nonfinite mesh distance')
                query_error=np.linalg.norm(points[ids]-points[ids].astype(np.float32).astype(float),axis=1)
                safe=ids[distances-query_error-guard>radii[ids]]
                work+=2*len(safe)*len(indices);queries+=2*len(safe)
                if work>max_triangle_point_evaluations:raise RuntimeError('mesh winding work cap exceeded; no truncation')
                if len(safe):
                    try:
                        w0=oriented_winding(original[indices],points[safe])
                        w1=oriented_winding(packed[indices],points[safe])
                    except ValueError:
                        # Leave the batch unknown, do not choose a favorable side.
                        pass
                    else:
                        r0=np.rint(w0);r1=np.rint(w1)
                        good=(np.abs(w0-r0)<1e-6)&(np.abs(w1-r1)<1e-6)&(r0>=0)&(r0==r1)
                        states[safe[good]]=(r0[good]>0).astype(np.int8)
            free|=states==1;blocked&=states==0
        out=np.full(len(tri),-1,dtype=np.int8);out[free]=1;out[blocked]=0
        return MeshBoundaryResult(out,queries,work,time.monotonic()-started)
