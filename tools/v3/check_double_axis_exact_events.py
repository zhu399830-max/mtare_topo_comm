"""Unified exact event diagnostic on the unchanged 360-axis contract."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from mtare_topo.data.gse_synthetic_matrix import matrix,construction_document
from mtare_topo.data.primitive_relation_materialization import load_p1a_realized_construction
from mtare_topo.teacher.csg_mesh_provenance import mesh_swept_superellipse
from mtare_topo.teacher.exact_event_diagnostic import ExactEventDiagnostic
from mtare_topo.evaluation.gse_convex_ray_union import declared_union_exit


def main():
    path=Path('configs/v3/gate3/double_axis_software_contract_v1.json')
    assert hashlib.sha256(path.read_bytes()).hexdigest()=='ff778aa7c549c778147647b4ed219ee4456ae0f46d6089376342e627767a97ef'
    config=json.loads(path.read_text());directions=np.asarray(config['directions'],float)
    wanted={f'double_junction__{s}__view{v}' for s in config['sections'] for v in config['views']}
    cases=[c for c in matrix() if c['case_id'] in wanted];assert len(cases)==12
    counts=Counter();errors=[];start=time.monotonic()
    for case in cases:
        _,primitives=load_p1a_realized_construction(construction_document(case))
        diagnostic=ExactEventDiagnostic([mesh_swept_superellipse(p,axial_spacing_m=config['axial_spacing_m'],angular_segments=config['angular_segments']) for p in primitives])
        for frame,origin in enumerate(case['poses_world_m']):
            expected,supported,_=declared_union_exit(case,np.tile(origin,(6,1)),directions);assert supported.all()
            for axis,direction in enumerate(directions):
                result=diagnostic.query(origin,direction,maximum_m=config['maximum_m'])
                error=abs(result['distance_m']-float(expected[axis])) if result['status']=='candidate' else None
                key=result['status']+(':'+result['reason'] if 'reason' in result else '')
                counts[key]+=1
                if error is not None:errors.append(error)
                print(json.dumps({'case':case['case_id'],'frame':frame,'axis':axis,'expected_m':float(expected[axis]),
                    'error_m':error,'result':result}),flush=True)
        print(json.dumps({'completed_case':case['case_id'],'counts':counts}),flush=True)
    print(json.dumps({'summary':dict(counts),'rays':sum(counts.values()),'max_error_m':max(errors,default=None),
                     'elapsed_s':time.monotonic()-start}),flush=True)
    assert sum(counts.values())==config['rays']
    assert all(e<=config['native_distance_comparison_atol_m'] for e in errors)


if __name__=='__main__':main()
