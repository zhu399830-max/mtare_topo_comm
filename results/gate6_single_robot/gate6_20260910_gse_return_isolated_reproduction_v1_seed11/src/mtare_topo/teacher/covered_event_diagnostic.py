"""Versioned native covered-event diagnostic; no export authority."""
from fractions import Fraction
import numpy as np
from .bound_ray_diagnostic import BoundRayDiagnostic
from .exact_hit_attribution import attribute_hit
from mtare_topo.evaluation.exact_triangle_ray import exact_triangle_ray,vector,sub,cross,dot


from .covered_event_reducer import reduce_covered_events


class CoveredEventDiagnostic(BoundRayDiagnostic):
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
        result=reduce_covered_events(events,checked.inside,Fraction(float(maximum_m)))
        result['records']=records;result['geometry_sha256']=checked.geometry_sha256
        if result['status']=='candidate':
            result['covered_mixed_events']=[{'exact_t':str(t),'persistent_operands':list(ids)} for t,ids in result['covered_mixed_events']]
            result['distance_m']=float(result['exact_t']);result['exact_t']=str(result['exact_t'])
            result['sources']=[self._meshes[i].primitive_id for i in result['operands']]
        return result
