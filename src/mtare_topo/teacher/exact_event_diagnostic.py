"""Unified exact-root stream diagnostic; not a certified dataset renderer."""
from fractions import Fraction
import numpy as np
from .bound_ray_diagnostic import BoundRayDiagnostic
from .exact_hit_attribution import attribute_hit
from mtare_topo.evaluation.exact_triangle_ray import exact_triangle_ray,vector,sub,cross,dot


def reduce_exact_events(events,inside,maximum=Fraction(50)):
    # events: (Fraction root, operand, canonical plane, outward sign).
    active={i for i,x in enumerate(inside) if x};first=None
    if not active or not events:return {'status':'needs_reference','reason':'missing_origin_or_events'}
    if any(not isinstance(e[0],Fraction) or e[0]<=0 or e[1]<0 or e[1]>=len(inside) or e[3] not in (-1,1) for e in events):
        return {'status':'needs_reference','reason':'invalid_event'}
    unique=set(events)
    for root in sorted({e[0] for e in unique}):
        group=[e for e in unique if e[0]==root];ids=[e[1] for e in group]
        if len(set(ids))!=len(ids):return {'status':'needs_reference','reason':'multiple_planes_same_operand_root'}
        signs={e[3] for e in group}
        if len(signs)!=1:return {'status':'needs_reference','reason':'coincident_entry_exit'}
        exits=next(iter(signs))==1
        if any((i in active)!=exits for i in ids):return {'status':'needs_reference','reason':'nonalternating_stream'}
        was=bool(active)
        if exits:active.difference_update(ids)
        else:active.update(ids)
        if was and not active and first is None:first=(root,tuple(sorted(ids)))
    if active or first is None:return {'status':'needs_reference','reason':'unclosed_stream'}
    if first[0]>maximum:return {'status':'out_of_range'}
    return {'status':'candidate','exact_t':first[0],'operands':first[1],'unique_events':len(unique)}


class ExactEventDiagnostic(BoundRayDiagnostic):
    def query(self,origin,direction,*,maximum_m=50.):
        origin=np.asarray(origin,float);direction=np.asarray(direction,float)
        if direction.shape!=(3,) or not np.isfinite(direction).all() or np.linalg.norm(direction)==0:
            raise ValueError('finite nonzero direction required')
        if not np.isfinite(maximum_m) or maximum_m<=0:raise ValueError('positive maximum required')
        direction=direction/np.linalg.norm(direction);checked=self._origin.check(origin)
        if checked.status!='origin_checked':return {'status':'needs_reference','reason':'origin:'+checked.reason}
        import open3d as o3d
        raw={k:v.numpy() for k,v in self._caster.scene.list_intersections(o3d.core.Tensor(np.r_[origin,direction][None].astype(np.float32))).items()}
        events=[];records=[]
        for g,t,native in zip(raw['geometry_ids'],raw['primitive_ids'],raw['t_hit']):
            operand=self._caster.geometry_to_operand[int(g)];mesh=self._meshes[operand]
            attr=attribute_hit(mesh,int(t),origin,direction)
            if attr is None:return {'status':'needs_reference','reason':'no_exact_attribution','records':records}
            tri=mesh.vertices_xyz_m[mesh.triangle_vertex_indices[attr.valid_triangles[0]]]
            hit=exact_triangle_ray(tri,origin,direction);a,b,c=map(vector,tri)
            outward=dot(cross(sub(b,a),sub(c,a)),vector(direction))
            if not outward:return {'status':'needs_reference','reason':'tangent'}
            events.append((attr.exact_t,operand,hit['plane'],1 if outward>0 else -1))
            records.append({'operand':operand,'native_triangle':int(t),'native_t':float(native),
                            'valid_triangles':attr.valid_triangles,'exact_t':str(attr.exact_t),'replaced':attr.replaced})
        result=reduce_exact_events(events,checked.inside,Fraction(float(maximum_m)))
        result['records']=records;result['geometry_sha256']=checked.geometry_sha256
        if result['status']=='candidate':
            result['distance_m']=float(result['exact_t']);result['exact_t']=str(result['exact_t'])
            result['sources']=[self._meshes[i].primitive_id for i in result['operands']]
        return result
