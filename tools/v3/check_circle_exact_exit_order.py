"""Pinned all-outgoing stream: preserve exact roots until the final output.

This checks recorded events, not completeness of arbitrary native intersections.
"""
import json
from pathlib import Path
from fractions import Fraction
from mtare_topo.data.gse_synthetic_matrix import matrix,construction_document
from mtare_topo.data.primitive_relation_materialization import load_p1a_realized_construction
from mtare_topo.teacher.csg_mesh_provenance import mesh_swept_superellipse
from mtare_topo.teacher.mesh_origin_check import PreparedOriginCheck
from mtare_topo.evaluation.exact_triangle_ray import exact_triangle_ray,vector,sub,cross,dot


def main():
    rows=[json.loads(s) for s in Path('docs/evidence/unknown_hit_planes_20260908.jsonl').read_text().splitlines()]
    row=next(r for r in rows if r['case']=='double_junction__circle__view0')
    assert row['frame']==4 and row['axis']==4 and len(row['hits'])==5
    case=next(c for c in matrix() if c['case_id']==row['case'])
    _,primitives=load_p1a_realized_construction(construction_document(case))
    meshes=[mesh_swept_superellipse(p,axial_spacing_m=.05,angular_segments=64) for p in primitives]
    origin=case['poses_world_m'][4];direction=[0,0,1]
    checked=PreparedOriginCheck(meshes).check(origin);assert checked.status=='origin_checked'
    initial={i for i,x in enumerate(checked.inside) if x};events={}
    for index,h in enumerate(row['hits']):
        m=meshes[h['operand']];tri=m.vertices_xyz_m[m.triangle_vertex_indices[h['triangle']]]
        exact=exact_triangle_ray(tri,origin,direction);assert exact['status']=='hit'
        a,b,c=map(vector,tri)
        assert dot(cross(sub(b,a),sub(c,a)),vector(direction))>0,'not an outgoing event'
        event=(exact['t'],h['operand'],exact['plane'])
        events.setdefault(event,[]).append(index)
    # The diagnostic only handles this observed all-outgoing population.
    assert {e[1] for e in events}==initial
    assert len(events)==len(initial),'more than one distinct outgoing face per occupied operand'
    roots=sorted({e[0] for e in events});assert len(roots)==2 and float(roots[0])==float(roots[1])
    active=set(initial);trace=[];first_exit=None
    for root in roots:
        leaving={e[1] for e in events if e[0]==root}
        assert leaving<=active
        active-=leaving
        trace.append({'exact_t':str(root),'leaving':[meshes[i].primitive_id for i in sorted(leaving)],
                      'remaining':[meshes[i].primitive_id for i in sorted(active)]})
        if not active and first_exit is None:first_exit=root
    assert first_exit==roots[-1]
    # Numeric conversion occurs only after union order/source has been determined.
    print(json.dumps({'case':row['case'],'frame':4,'axis':4,'raw_hits':5,'unique_plane_operand_events':len(events),
        'distinct_exact_roots':len(roots),'trace':trace,'exact_root_separation_m':str(roots[1]-roots[0]),
        'root_separation_float_m':float(roots[1]-roots[0]),'final_exact_t':str(first_exit),'final_distance_m':float(first_exit),
        'final_sources':trace[-1]['leaving'],'scope':'recorded_stream_order_not_native_completeness'},indent=2))


if __name__=='__main__':main()
