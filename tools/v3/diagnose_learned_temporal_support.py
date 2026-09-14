"""Fixed existing-output support and adjacent-window geometry diagnostics."""
from _bootstrap import PROJECT_ROOT as ROOT
import json,hashlib
import numpy as np
from scipy.optimize import linear_sum_assignment
from mtare_topo.evaluation.local_axis_diagnostic import clip_axis_to_ball,sampled_hausdorff


def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(8388608),b''):h.update(block)
    return h.hexdigest()


def main():
    run=ROOT/'results/gate6_single_robot/gate6_20260910_gse_learned_geometry_pair_v1_seed0'
    files=[run/'artifacts/inputs.npz',run/'artifacts/pair/predictions.jsonl']
    seal={l.split('  ',1)[1]:l.split('  ',1)[0] for l in (run/'artifacts/evidence_sha256.txt').read_text().splitlines()}
    for p in files:assert sha(p)==seal[str(p.relative_to(ROOT))]
    poses=np.load(files[0],allow_pickle=False)['poses'];rows=[json.loads(l) for l in files[1].open()]
    thresholds=[.25,.5,1.];records=[]
    for i,row in enumerate(rows):
        center=poses[i+4,:3,3];entry=dict(index=i,timestamp=row['timestamp'],methods={})
        for name,m in row['methods'].items():
            local=[v for p,v in zip(m['targets'],m['current_support']) if np.linalg.norm(np.asarray(p)-center)<=10]
            result=dict(local_targets=len(local),supported=sum(v is True for v in local),
                unsupported=sum(v is False for v in local),unknown=sum(v is None for v in local))
            if i:
                # Both sets clipped to the SAME current ball, not two moving domains.
                def clipped(method):
                    return [s for a in method['axes'] if a is not None for s in [clip_axis_to_ball(a,center)] if s]
                current=clipped(m);previous=clipped(rows[i-1]['methods'][name])
                matches=[]
                if current and previous:
                    costs=np.array([[sampled_hausdorff(a,b) for b in previous] for a in current])
                    rr,cc=linear_sum_assignment(costs)
                    matches=[dict(current_index=int(a),previous_index=int(b),distance_m=float(costs[a,b])) for a,b in zip(rr,cc)]
                result.update(current_axes=len(current),previous_axes=len(previous),assignment=matches,
                    within={str(t):sum(m['distance_m']<=t for m in matches) for t in thresholds})
            entry['methods'][name]=result
        records.append(entry)
    summary={}
    for name in ['learned','nonlearning']:
        values=[r['methods'][name] for r in records];temporal=values[1:]
        summary[name]={k:sum(v[k] for v in values) for k in ['local_targets','supported','unsupported','unknown']}
        summary[name].update(temporal_current_axes=sum(v['current_axes'] for v in temporal),
            temporal_previous_axes=sum(v['previous_axes'] for v in temporal),
            matched_within={str(t):sum(v['within'][str(t)] for v in temporal) for t in thresholds})
    out=ROOT/'docs/figures/gse_learned_geometry_pair_v1/temporal_support.json'
    with out.open('x') as f:json.dump(dict(summary=summary,per_window=records,source_sha256={str(p.relative_to(ROOT)):sha(p) for p in files},
        thresholds_m=thresholds,radius_m=10,sampling_spacing_m=.25,
        limitation='Support uses existing horizontal current sectors, not traversability. Adjacent overlap and odometry shared; temporal consistency is not correctness.'),f,indent=2)
    print(json.dumps(summary))


if __name__=='__main__':main()
