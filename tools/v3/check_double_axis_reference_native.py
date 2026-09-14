"""Resolve only the frozen 360-axis diagnostic's needs-reference population."""
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
from mtare_topo.teacher.ordered_exit_candidate import ordered_exit_candidate
from mtare_topo.teacher.csg_interval_raycaster_v2 import verified_interval_hit
from mtare_topo.evaluation.gse_convex_ray_union import declared_union_exit


def main():
    path=Path('configs/v3/gate3/double_axis_software_contract_v1.json')
    assert hashlib.sha256(path.read_bytes()).hexdigest()=='ff778aa7c549c778147647b4ed219ee4456ae0f46d6089376342e627767a97ef'
    old=Path('tools/v3/check_double_axis_candidates_native.py')
    assert hashlib.sha256(old.read_bytes()).hexdigest()=='2533b20d2433b911ed5249c5a3dddcc4e69804932b1f0623eca624dab3bd60ab'
    config=json.loads(path.read_text());directions=np.asarray(config['directions'],float)
    wanted={f'double_junction__{s}__view{v}' for s in config['sections'] for v in config['views']}
    cases=[c for c in matrix() if c['case_id'] in wanted];assert len(cases)==12
    counters=Counter();reasons=Counter();rows=[];started=time.monotonic()
    for case in cases:
        _,primitives=load_p1a_realized_construction(construction_document(case))
        bound=BoundRayDiagnostic([mesh_swept_superellipse(p,axial_spacing_m=config['axial_spacing_m'],
            angular_segments=config['angular_segments']) for p in primitives])
        meshes=bound._meshes;caster=bound._caster
        for frame,origin in enumerate(case['poses_world_m']):
            origins=np.tile(origin,(6,1));expected,supported,intervals=declared_union_exit(case,origins,directions)
            checked=bound._origin.check(origin)
            assert supported.all() and checked.status=='origin_checked'
            np.testing.assert_array_equal(checked.inside,(intervals[0,:,0]<0)&(intervals[0,:,1]>0))
            raw={k:v.numpy() for k,v in caster.scene.list_intersections(o3d.core.Tensor(
                np.concatenate((origins,directions),axis=1).astype(np.float32))).items()}
            for ray in range(6):
                a,b=map(int,raw['ray_splits'][ray:ray+2]);ts=raw['t_hit'][a:b]
                ids=[caster.geometry_to_operand[int(g)] for g in raw['geometry_ids'][a:b]]
                dots=[float(meshes[i].triangle_normals[int(t)]@directions[ray]) for i,t in zip(ids,raw['primitive_ids'][a:b])]
                fast=ordered_exit_candidate(ts,ids,dots,checked.inside,maximum_m=config['maximum_m'])
                counters[fast.status]+=1
                if fast.status=='candidate':continue
                assert fast.status=='needs_reference'
                reasons[fast.reason]+=1
                before=time.monotonic()
                reference=verified_interval_hit(meshes,origin,directions[ray],checked.inside,ts,ids,maximum_m=config['maximum_m'])
                error=None if reference is None else abs(reference.distance_m-float(expected[ray]))
                status='unknown' if reference is None else ('match' if error<=config['native_distance_comparison_atol_m'] else 'mismatch')
                counters['reference_'+status]+=1
                row=dict(case=case['case_id'],frame=frame,axis=ray,fast_reason=fast.reason,
                         expected_m=float(expected[ray]),reference_m=None if reference is None else reference.distance_m,
                         error_m=error,status=status,reference_s=time.monotonic()-before)
                rows.append(row);print(json.dumps(row),flush=True)
        print(json.dumps({'completed_case':case['case_id'],'counts':counters}),flush=True)
    assert counters['candidate']==269 and counters['needs_reference']==91
    assert dict(reasons)=={'duplicate_or_multishell_crossing':61,'nonalternating_or_missing_crossing':27,'unclosed_crossing_stream':3}
    print(json.dumps({'summary':dict(counters),'reasons':dict(reasons),'elapsed_s':time.monotonic()-started,
                     'max_reference_error_m':max((r['error_m'] for r in rows if r['error_m'] is not None),default=None)}),flush=True)
    assert counters['reference_mismatch']==0,'reference disagrees with independent geometry'


if __name__=='__main__':main()
