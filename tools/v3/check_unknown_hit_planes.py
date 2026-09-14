"""Exact arithmetic on the hit triangles of the same seven unknown rays."""
import json
from pathlib import Path
import numpy as np
import open3d as o3d
from mtare_topo.data.gse_synthetic_matrix import matrix,construction_document
from mtare_topo.data.primitive_relation_materialization import load_p1a_realized_construction
from mtare_topo.teacher.csg_mesh_provenance import mesh_swept_superellipse
from mtare_topo.teacher.bound_ray_diagnostic import BoundRayDiagnostic
from mtare_topo.evaluation.exact_triangle_ray import exact_triangle_ray


def main():
    rows=[json.loads(s) for s in Path('docs/evidence/double_axis_reference_20260908.jsonl').read_text().splitlines()]
    rows=[r for r in rows if r.get('status')=='unknown'];assert len(rows)==7
    directions=json.loads(Path('configs/v3/gate3/double_axis_software_contract_v1.json').read_text())['directions']
    cases={c['case_id']:c for c in matrix()};key=None
    for row in rows:
        case=cases[row['case']]
        if key!=row['case']:
            _,primitives=load_p1a_realized_construction(construction_document(case))
            bound=BoundRayDiagnostic([mesh_swept_superellipse(p,axial_spacing_m=.05,angular_segments=64) for p in primitives]);key=row['case']
        origin=np.asarray(case['poses_world_m'][row['frame']]);direction=np.asarray(directions[row['axis']])
        raw={k:v.numpy() for k,v in bound._caster.scene.list_intersections(o3d.core.Tensor(np.r_[origin,direction][None].astype(np.float32))).items()}
        hits=[];groups={}
        for j,(g,tri) in enumerate(zip(raw['geometry_ids'],raw['primitive_ids'])):
            operand=bound._caster.geometry_to_operand[int(g)];mesh=bound._meshes[operand]
            triangle=mesh.vertices_xyz_m[mesh.triangle_vertex_indices[int(tri)]]
            original=exact_triangle_ray(triangle,origin,direction)
            packed=exact_triangle_ray(triangle.astype(np.float32),origin.astype(np.float32),direction.astype(np.float32))
            hits.append(dict(operand=operand,triangle=int(tri),native_m=float(raw['t_hit'][j]),
                original_status=original['status'],original_t=None if 't' not in original else str(original['t']),
                packed_status=packed['status'],packed_t=None if 't' not in packed else str(packed['t']),
                original_boundary=original.get('boundary'),packed_boundary=packed.get('boundary')))
            if original['status']=='hit':groups.setdefault((operand,original['plane'],original['t']),[]).append(j)
        duplicates=[indices for indices in groups.values() if len(indices)>1]
        print(json.dumps(dict(case=key,frame=row['frame'],axis=row['axis'],hits=hits,
                             exact_same_operand_plane_hit_groups=duplicates)),flush=True)


if __name__=='__main__':main()
