"""Reusable numerical origin precondition, not global mesh/ray qualification."""
from collections import OrderedDict
from dataclasses import dataclass
import hashlib
import numpy as np
from mtare_topo.evaluation.mesh_winding_diagnostic import oriented_winding


@dataclass(frozen=True)
class OriginCheck:
    geometry_sha256: str
    xyz_m: tuple[float, float, float]
    status: str
    inside: tuple[bool, ...] = ()
    reason: str = ''


class PreparedOriginCheck:
    def __init__(self,meshes,*,cache_capacity=128):
        if type(cache_capacity) is not int or cache_capacity<1:raise ValueError('positive cache capacity required')
        if not len(meshes):raise ValueError('nonempty geometry required')
        digest=hashlib.sha256();triangles=[]
        for mesh in meshes:
            values=np.asarray(mesh.vertices_xyz_m[mesh.triangle_vertex_indices],dtype='<f8')
            raw=values.tobytes()
            identity=mesh.primitive_id.encode()
            digest.update(len(identity).to_bytes(8,'little'));digest.update(identity)
            digest.update(len(raw).to_bytes(8,'little'));digest.update(raw)
            # Immutable bytes-backed copies: external mesh mutation cannot poison cache.
            original=np.frombuffer(raw,dtype='<f8').reshape(values.shape)
            packed=np.frombuffer(values.astype(np.float32).astype('<f8').tobytes(),dtype='<f8').reshape(values.shape)
            triangles.append((original,packed))
        self.geometry_sha256=digest.hexdigest()
        self._triangles=tuple(triangles);self._capacity=cache_capacity;self._cache=OrderedDict()

    def check(self,origin):
        origin=np.asarray(origin,dtype=float)
        if origin.shape!=(3,) or not np.isfinite(origin).all():raise ValueError('finite xyz required')
        key=tuple(float(v) for v in origin)
        if key in self._cache:
            self._cache.move_to_end(key);return self._cache[key]
        def fail(reason):return OriginCheck(self.geometry_sha256,key,'needs_reference',reason=reason)
        occupancies=[];result=None
        for original,packed in self._triangles:
            states=[]
            for triangles,point in ((original,origin),(packed,origin.astype(np.float32).astype(float))):
                try:w=float(oriented_winding(triangles,point[None])[0])
                except ValueError:result=fail('surface_or_invalid_winding');break
                rounded=np.rint(w)
                if not np.isfinite(w) or abs(w-rounded)>1e-6:
                    result=fail('noninteger_winding');break
                if rounded not in (0,1):result=fail('nonbinary_origin_multiplicity');break
                states.append(bool(rounded))
            if result is not None:break
            if states[0]!=states[1]:result=fail('precision_disagreement');break
            occupancies.append(states[0])
        if result is None:
            result=OriginCheck(self.geometry_sha256,key,'origin_checked',tuple(occupancies)) if any(occupancies) else fail('outside_union')
        self._cache[key]=result
        if len(self._cache)>self._capacity:self._cache.popitem(last=False)
        return result
