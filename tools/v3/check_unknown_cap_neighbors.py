"""Exact cap/edge-neighbor diagnostics for the 18 recorded outside hits."""
import json
from pathlib import Path
import numpy as np
from mtare_topo.data.gse_synthetic_matrix import matrix,construction_document
from mtare_topo.data.primitive_relation_materialization import load_p1a_realized_construction
from mtare_topo.teacher.csg_mesh_provenance import mesh_swept_superellipse
from mtare_topo.evaluation.exact_triangle_ray import exact_triangle_ray


def main():
    records=[json.loads(s) for s in Path('docs/evidence/unknown_hit_planes_20260908.jsonl').read_text().splitlines()]
    directions=json.loads(Path('configs/v3/gate3/double_axis_software_contract_v1.json').read_text())['directions']
    cases={c['case_id']:c for c in matrix()};key=None;count=0
    for row in records:
        invalid=[h for h in row['hits'] if h['original_status']=='outside_triangle']
        if not invalid:continue
        case=cases[row['case']]
        if key!=row['case']:
            _,primitives=load_p1a_realized_construction(construction_document(case))
            meshes=[mesh_swept_superellipse(p,axial_spacing_m=.05,angular_segments=64) for p in primitives];key=row['case']
        origin=case['poses_world_m'][row['frame']];direction=directions[row['axis']]
        for h in invalid:
            m=meshes[h['operand']];ti=h['triangle'];indices=m.triangle_vertex_indices
            assert ti>=len(indices)-128,'recorded target is not in generated 64-sector end caps'
            triangle=m.vertices_xyz_m[indices[ti]];exact=exact_triangle_ray(triangle,origin,direction)
            weights=[1-exact['u']-exact['v'],exact['u'],exact['v']]
            shared=np.isin(indices,indices[ti]).sum(axis=1)
            neighbors=set(int(i) for i in np.flatnonzero(shared>=2) if i!=ti)
            valid=[]
            # Full generated cap, not only neighbors, proves where this cap is crossed.
            for other in range(len(indices)-128,len(indices)):
                candidate=exact_triangle_ray(m.vertices_xyz_m[indices[other]],origin,direction)
                if candidate['status']=='hit' and candidate['plane']==exact['plane']:
                    valid.append({'triangle':other,'shared_edge':other in neighbors,'same_exact_t':candidate['t']==exact['t'],
                                  't_m':float(candidate['t']),'boundary':candidate['boundary']})
            count+=1
            print(json.dumps({'case':key,'frame':row['frame'],'axis':row['axis'],'operand':h['operand'],'triangle':ti,
                'area_m2':float(np.linalg.norm(np.cross(triangle[1]-triangle[0],triangle[2]-triangle[0]))/2),
                'weights_exact':[str(w) for w in weights],'minimum_weight':float(min(weights)),
                'native_t_m':h['native_m'],'exact_plane_t_m':float(exact['t']),
                'same_plane_valid_cap_hits':valid}),flush=True)
    assert count==18


if __name__=='__main__':main()
