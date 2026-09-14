"""Same contract and versioned fallback, six-case check or complete regression."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from mtare_topo.data.gse_synthetic_matrix import matrix,construction_document
from mtare_topo.data.primitive_relation_materialization import load_p1a_realized_construction
from mtare_topo.teacher.csg_mesh_provenance import mesh_swept_superellipse
from mtare_topo.teacher.exact_reference_diagnostic import ExactReferenceDiagnostic
from mtare_topo.evaluation.gse_convex_ray_union import declared_union_exit


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--remaining-only',action='store_true');args=parser.parse_args()
    path=Path('configs/v3/gate3/double_axis_software_contract_v1.json')
    assert hashlib.sha256(path.read_bytes()).hexdigest()=='ff778aa7c549c778147647b4ed219ee4456ae0f46d6089376342e627767a97ef'
    config=json.loads(path.read_text());directions=np.asarray(config['directions'],float)
    old=[json.loads(s) for s in Path('docs/evidence/double_axis_exact_events_20260908.jsonl').read_text().splitlines()]
    selected={(r['case'],r['frame'],r['axis']) for r in old if r.get('result',{}).get('status')=='needs_reference'};assert len(selected)==6
    wanted={f'double_junction__{s}__view{v}' for s in config['sections'] for v in config['views']}
    counts=Counter();errors=[];start=time.monotonic()
    for case in [c for c in matrix() if c['case_id'] in wanted]:
        if args.remaining_only and not any(s[0]==case['case_id'] for s in selected):continue
        _,primitives=load_p1a_realized_construction(construction_document(case))
        diagnostic=ExactReferenceDiagnostic([mesh_swept_superellipse(p,axial_spacing_m=config['axial_spacing_m'],angular_segments=config['angular_segments']) for p in primitives])
        for frame,origin in enumerate(case['poses_world_m']):
            expected,supported,_=declared_union_exit(case,np.tile(origin,(6,1)),directions);assert supported.all()
            for axis,direction in enumerate(directions):
                if args.remaining_only and (case['case_id'],frame,axis) not in selected:continue
                result=diagnostic.query(origin,direction,maximum_m=config['maximum_m']);counts[result['status']]+=1
                error=None if 'distance_m' not in result else abs(result['distance_m']-float(expected[axis]))
                if error is not None:errors.append(error)
                print(json.dumps({'case':case['case_id'],'frame':frame,'axis':axis,'expected_m':float(expected[axis]),'error_m':error,'result':result}),flush=True)
    print(json.dumps({'summary':dict(counts),'rays':sum(counts.values()),'max_error_m':max(errors,default=None),'elapsed_s':time.monotonic()-start,'remaining_only':args.remaining_only}),flush=True)
    assert sum(counts.values())==(6 if args.remaining_only else 360)
    assert all(e<=config['native_distance_comparison_atol_m'] for e in errors)
    assert len(errors)==sum(counts.values()),'unknown returns remain'


if __name__=='__main__':main()
