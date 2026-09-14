"""Fixed12 read-only first-exit and branch-ray diagnostic; not new labels."""
import _bootstrap
import gzip,hashlib,io,json
from pathlib import Path
import numpy as np
from mtare_topo.data.gse_synthetic_matrix import matrix
from mtare_topo.data.gse_synthetic_fit_scope import SOURCE,SEAL_SHA
from mtare_topo.data.cano_sensor_smoke import lidar_local_directions,world_directions
from mtare_topo.evaluation.gse_convex_ray_union import declared_union_exit,origin_component_exit
from check_separated_archive import binary_validity


def main():
    root=Path(__file__).resolve().parents[2]
    seal=(root/SOURCE/'artifacts/evidence_sha256.txt').read_bytes()
    assert hashlib.sha256(seal).hexdigest()==SEAL_SHA
    pins={p:h for h,p in (line.split('  ',1) for line in seal.decode().splitlines())}
    def read(path):
        raw=(root/path).read_bytes();assert hashlib.sha256(raw).hexdigest()==pins[path],path
        return raw
    cases=[c for c in matrix() if c['program']['type']=='double_junction'];assert len(cases)==12
    rows=[];local=lidar_local_directions().reshape(-1,3).astype(float)
    for case in cases:
        stem=SOURCE+'/artifacts/'+case['case_id']
        record=json.loads(gzip.decompress(read(stem+'.json.gz')));assert record['case']==case
        ids=[e['id'] for e in case['program']['edges']]
        assert ids==['left','middle','right','branch0','branch1']
        with np.load(io.BytesIO(read(stem+'.npz')),allow_pickle=False) as payload:
            ranges=payload['ranges_m'];valid=binary_validity(payload['valid_mask'])
            assert ranges.shape==valid.shape==(5,16,720)
            assert np.array_equal(payload['sensor_xyz_m'],np.asarray(case['poses_world_m']))
            errors=[];per_frame=[];unsupported=0;worst=None
            for f,(pose,yaw) in enumerate(zip(case['poses_world_m'],case['yaw_deg'])):
                d=world_directions(local,float(yaw));d/=np.linalg.norm(d,axis=1,keepdims=True)
                origins=np.broadcast_to(pose,d.shape)
                end,support,intervals=declared_union_exit(case,origins,d)
                trunk,trunk_support=origin_component_exit(intervals[:,:3])
                unsupported+=int((~support).sum())
                v=valid[f].reshape(-1);actual=ranges[f].reshape(-1)
                errors.extend(np.abs(actual[v&support]-end[v&support]).tolist())
                indices=np.flatnonzero(v&support)
                if len(indices):
                    k=int(indices[np.argmax(np.abs(actual[indices]-end[indices]))])
                    error=float(abs(actual[k]-end[k]))
                    if worst is None or error>worst['error_m']:
                        worst=dict(frame=f,ray=k,error_m=error,actual_m=float(actual[k]),
                            expected_m=float(end[k]),origin=list(pose),direction=d[k].tolist(),
                            intervals=[[float(a) if np.isfinite(a) else None,float(b) if np.isfinite(b) else None] for a,b in intervals[k]],
                            source_code=int(payload['primitive_membership_code'][f].reshape(-1)[k]))
                crossing=origins+trunk[:,None]*d
                in_window=np.linalg.norm(crossing-np.asarray(case['poses_world_m'][-1]),axis=1)<10
                branch_counts=[]
                for j in (3,4):
                    # Geometric branch witness: actual ray extends past trunk,
                    # and that crossing lies strictly inside the branch prism.
                    mask=v&support&trunk_support&in_window&(actual>trunk)
                    mask&=(intervals[:,j,0]<trunk)&(intervals[:,j,1]>trunk)
                    branch_counts.append(int(mask.sum()))
                per_frame.append(branch_counts)
            target=record['produced_targets']['record']
            rows.append(dict(case_id=case['case_id'],invalid=int((~valid).sum()),
                unsupported_origins=unsupported,max_range_error_m=max(errors) if errors else None,
                branch_witnesses_per_frame=per_frame,branch_total=np.sum(per_frame,axis=0).tolist(),
                original_anchor_positions=[x['position_m'] for x in target['anchors']],
                original_openings=len(target['openings'])))
            rows[-1]['worst_ray']=worst
    print(json.dumps(dict(status='DOUBLE_JUNCTION_ARCHIVE_RAY_DIAGNOSTIC',observations=12,
        verified_files=24,rays=12*5*11520,invalid=sum(r['invalid'] for r in rows),
        unsupported_origins=sum(r['unsupported_origins'] for r in rows),
        max_range_error_m=max(r['max_range_error_m'] for r in rows),per_case=rows,
        diagnostic_only=True,teacher_qualified=False,threshold_selected=False,optimizer_steps=0,renders=0)))

if __name__=='__main__':main()
