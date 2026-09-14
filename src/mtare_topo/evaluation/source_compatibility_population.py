"""Read-only source comparison; discrepancies never rewrite data or labels."""
from collections import Counter
import json
import numpy as np
from mtare_topo.data.covered_sensor_conversion import file_sha
from mtare_topo.data.gse_synthetic_matrix import matrix
from mtare_topo.data.cano_sensor_smoke import lidar_local_directions, world_directions
from mtare_topo.governance_covered_sensor_export import digest
from .closed_prism_surface import declared_surface_candidates


def audit(root,run,scope,progress):
    def verify():
        for p,h in scope['input_sha256'].items():
            if file_sha(root/p)!=h:raise ValueError('input drift: '+p)
    verify()
    manifest=json.loads((root/scope['source_run']/'config/source_config').read_text())['frame_inputs']
    cases={c['case_id']:c for c in matrix()}; totals=Counter(); frame_rows=[]
    local=lidar_local_directions().reshape(-1,3).astype(np.float64)
    for name in scope['cases']:
        case=cases[name];p=root/scope['export_run']/'artifacts'/name
        meta=json.loads(p.with_name(p.name+'.json').read_text())
        book=meta['reported_source_codebook'];ids=[e['id'] for e in case['program']['edges']]
        if book['primitive_ids']!=ids or meta['qualification']['training_eligible'] is not False:
            raise ValueError('source/qualification drift')
        with np.load(str(p)+'.student.npz',allow_pickle=False) as s,np.load(str(p)+'.diagnostic.npz',allow_pickle=False) as d:
            for frame in range(5):
                m=manifest[len(frame_rows)]
                if (m['case_id'],m['frame_index'],m['case_sha256'])!=(name,frame,digest(case)):
                    raise ValueError('declaration drift')
                origin=d['sensor_xyz_m'][frame];yaw=float(d['yaw_deg'][frame])
                if origin.tolist()!=m['origin_m'] or yaw!=m['yaw_deg']:raise ValueError('pose drift')
                directions=world_directions(local,yaw);directions/=np.linalg.norm(directions,axis=1,keepdims=True)
                ranges=s['ranges_m'][frame].reshape(-1);valid=s['valid_mask'][frame].reshape(-1).astype(bool)
                codes=d['primitive_membership_code'][frame].reshape(-1)
                reported=np.zeros((11520,len(ids)),bool)
                for code,indices in enumerate(book['source_sets']):
                    for index in indices:reported[codes==code,index]=True
                if not np.array_equal(reported.any(1),valid):raise ValueError('valid/code mismatch')
                ref=declared_surface_candidates(case,np.tile(origin,(11520,1)),directions,ranges)
                possible=ref['possible_surface'];exits=ref['possible_exit']
                flags=dict(reported_outside_possible=(reported & ~possible).any(1),
                    reported_not_exit=(reported & ~exits).any(1),
                    extra_surface=(possible & ~reported).any(1),
                    exact_candidate_set_match=(reported==possible).all(1),
                    singleton_candidate=ref['singleton_candidate'],
                    reported_singleton=reported.sum(1)==1,
                    coplanar_uncertain=ref['uncertain_operands'].any(1),
                    no_surface_candidate=~possible.any(1))
                row=dict(case_id=name,frame=frame,rays=11520,valid=int(valid.sum()),
                         **{k:int((v & valid).sum()) for k,v in flags.items()})
                totals.update({k:v for k,v in row.items() if isinstance(v,int) and k!='frame'})
                with (run/'artifacts'/f'{name}_frame{frame}.npz').open('xb') as out:
                    np.savez_compressed(out,reported=reported,valid=valid,**dict(ref,**flags))
                frame_rows.append(row);progress(row)
    verify()
    if len(frame_rows)!=60 or totals['rays']!=691200:raise ValueError('population incomplete')
    return dict(totals=dict(totals),frames=frame_rows,numerical_geometry_certified=False,
                training_eligible=False,teacher_labels_generated=0,optimizer_steps=0)
