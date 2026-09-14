"""Fixed 12 declarations x five origins x six axes; no scan payload reads."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import open3d as o3d
from mtare_topo.data.gse_synthetic_matrix import matrix,construction_document
from mtare_topo.data.primitive_relation_materialization import load_p1a_realized_construction
from mtare_topo.teacher.csg_mesh_provenance import mesh_swept_superellipse
from mtare_topo.teacher.bound_ray_diagnostic import BoundRayDiagnostic
from mtare_topo.teacher.ordered_exit_candidate import ExitCandidate,ordered_exit_candidate
from mtare_topo.evaluation.gse_convex_ray_union import declared_union_exit


def main():
    contract_path=Path('configs/v3/gate3/double_axis_software_contract_v1.json')
    contract=json.loads(contract_path.read_text());directions=np.asarray(contract['directions'],float)
    ids={f"double_junction__{s}__view{v}" for s in contract['sections'] for v in contract['views']}
    cases=[c for c in matrix() if c['case_id'] in ids]
    assert len(cases)==contract['cases']==12
    paths=[contract_path,Path(__file__),Path('src/mtare_topo/data/gse_synthetic_matrix.py'),
           Path('src/mtare_topo/teacher/ordered_exit_candidate.py'),Path('src/mtare_topo/teacher/mesh_origin_check.py'),
           Path('src/mtare_topo/evaluation/gse_convex_ray_union.py')]
    hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    print(json.dumps({'source_hashes':hashes,'contract':contract}),flush=True)
    total=Counter();errors=[];failures=[];start=time.monotonic()
    for case in cases:
        _,primitives=load_p1a_realized_construction(construction_document(case))
        meshes=[mesh_swept_superellipse(p,axial_spacing_m=contract['axial_spacing_m'],
                 angular_segments=contract['angular_segments']) for p in primitives]
        bound=BoundRayDiagnostic(meshes)
        # Diagnostic accesses its bound snapshot; never mix external mutable meshes.
        meshes=bound._meshes;caster=bound._caster;local=Counter();local_errors=[]
        assert len(case['poses_world_m'])==contract['origins_per_case']
        for frame,origin in enumerate(case['poses_world_m']):
            origins=np.tile(origin,(6,1));expected,supported,intervals=declared_union_exit(case,origins,directions)
            assert supported.all()
            checked=bound._origin.check(origin)
            if checked.status=='origin_checked':
                np.testing.assert_array_equal(checked.inside,(intervals[0,:,0]<0)&(intervals[0,:,1]>0))
            raw={k:v.numpy() for k,v in caster.scene.list_intersections(
                o3d.core.Tensor(np.concatenate((origins,directions),axis=1).astype(np.float32))).items()}
            for ray in range(6):
                a,b=map(int,raw['ray_splits'][ray:ray+2])
                operands=[caster.geometry_to_operand[int(g)] for g in raw['geometry_ids'][a:b]]
                dots=[float(meshes[i].triangle_normals[int(t)]@directions[ray])
                      for i,t in zip(operands,raw['primitive_ids'][a:b])]
                result=ordered_exit_candidate(raw['t_hit'][a:b],operands,dots,checked.inside,maximum_m=contract['maximum_m']) if checked.status=='origin_checked' else ExitCandidate('needs_reference',reason='origin:'+checked.reason)
                key=result.status+(':'+result.reason if result.reason else '')
                local[key]+=1;total[key]+=1
                if result.status=='candidate':
                    error=abs(result.distance_m-float(expected[ray]));errors.append(error);local_errors.append(error)
                    if error>contract['native_distance_comparison_atol_m']:
                        failures.append(dict(case=case['case_id'],frame=frame,axis=ray,error_m=error))
        print(json.dumps({'case':case['case_id'],'counts':local,'max_candidate_error_m':max(local_errors,default=None)}),flush=True)
    assert sum(total.values())==contract['rays']
    assert hashes=={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    print(json.dumps({'counts':total,'max_candidate_error_m':max(errors,default=None),'failures':failures,
                      'elapsed_s':time.monotonic()-start,'rays':sum(total.values())}),flush=True)
    assert not failures,'candidate/independent geometry disagreement'


if __name__=='__main__':main()
