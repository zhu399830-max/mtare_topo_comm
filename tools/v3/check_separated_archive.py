"""Exact fixed24 archive reads; range residual diagnostic, no tolerance tuning."""
import _bootstrap
import gzip,hashlib,io,json
from pathlib import Path
import numpy as np
from mtare_topo.data.gse_synthetic_matrix import matrix
from mtare_topo.data.gse_synthetic_fit_scope import SOURCE,SEAL_SHA
from mtare_topo.data.cano_sensor_smoke import lidar_local_directions,world_directions
from mtare_topo.evaluation.gse_separated_tube_oracle import separated_tube_expectation,convex_main_exit
from mtare_topo.evaluation.gse_synthetic_field_scoring import match_positions


def binary_validity(value):
    value=np.asarray(value)
    if not np.isin(value,[0,1]).all():raise ValueError('nonbinary validity')
    return value.astype(bool)


def main():
    root=Path(__file__).resolve().parents[2]
    raw=(root/SOURCE/'artifacts/evidence_sha256.txt').read_bytes()
    assert hashlib.sha256(raw).hexdigest()==SEAL_SHA
    pins={p:h for h,p in (line.split('  ',1) for line in raw.decode().splitlines())}
    cases=[c for c in matrix() if c['program']['type'] in ('parallel','stacked')];assert len(cases)==24
    def read(path):
        raw=(root/path).read_bytes();assert hashlib.sha256(raw).hexdigest()==pins[path],path
        return raw
    rows=[];local=lidar_local_directions().reshape(-1,3).astype(float)
    for case in cases:
        stem=SOURCE+'/artifacts/'+case['case_id']
        record=json.loads(gzip.decompress(read(stem+'.json.gz')));assert record['case']==case
        with np.load(io.BytesIO(read(stem+'.npz')),allow_pickle=False) as payload:
            ranges=payload['ranges_m'];valid=binary_validity(payload['valid_mask'])
            assert ranges.shape==valid.shape==(5,16,720)
            assert np.array_equal(payload['sensor_xyz_m'],np.asarray(case['poses_world_m']))
            errors=[]
            for f,(pose,yaw) in enumerate(zip(case['poses_world_m'],case['yaw_deg'])):
                d=world_directions(local,float(yaw));d=d/np.linalg.norm(d,axis=1,keepdims=True)
                expected=convex_main_exit(case,np.broadcast_to(pose,d.shape),d)
                actual=ranges[f].reshape(-1);v=valid[f].reshape(-1)
                if v.any():errors.extend(np.abs(actual[v]-expected[v]).tolist())
            truth=separated_tube_expectation(case);target=record['produced_targets']['record']
            rows.append(dict(case_id=case['case_id'],rays=int(valid.size),invalid=int((~valid).sum()),
                max_range_error_m=max(errors) if errors else None,
                anchors=match_positions(truth['anchors'],[x['position_m'] for x in target['anchors']],1.),
                openings=match_positions([x['position_m'] for x in truth['openings']],[x['position_m'] for x in target['openings']],1.)))
    print(json.dumps(dict(status='SAME24_ARCHIVE_RESIDUAL_DIAGNOSTIC',verified_files=48,
        target_count_match_cases=sum(r['anchors']['exact_count_and_matching_pass'] and r['openings']['exact_count_and_matching_pass'] for r in rows),
        rays=sum(r['rays'] for r in rows),invalid=sum(r['invalid'] for r in rows),
        max_range_error_m=max(r['max_range_error_m'] for r in rows),per_case=rows,
        threshold_selected=False,training_eligible=False,optimizer_steps=0,renders=0)))

if __name__=='__main__':main()
