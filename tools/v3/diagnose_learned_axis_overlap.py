"""One read-only local-domain diagnostic of sealed paired predictions."""
from _bootstrap import PROJECT_ROOT as ROOT
import hashlib,json
import numpy as np
from mtare_topo.evaluation.local_axis_diagnostic import clip_axis_to_ball,sampled_hausdorff


def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8388608),b''):h.update(b)
    return h.hexdigest()


def main():
    run=ROOT/'results/gate6_single_robot/gate6_20260910_gse_learned_geometry_pair_v1_seed0'
    paths=[run/'artifacts/inputs.npz',run/'artifacts/pair/predictions.jsonl']
    seal={l.split('  ',1)[1]:l.split('  ',1)[0] for l in (run/'artifacts/evidence_sha256.txt').read_text().splitlines()}
    for p in paths:
        if sha(p)!=seal[str(p.relative_to(ROOT))]:raise ValueError('source drift')
    poses=np.load(paths[0],allow_pickle=False)['poses'];rows=[json.loads(l) for l in paths[1].open()]
    result=[];thresholds=[.25,.5,1.]
    for i,row in enumerate(rows):
        center=poses[i+4,:3,3];entry=dict(index=i,timestamp=row['timestamp'],methods={})
        for name,m in row['methods'].items():
            axes=[]
            for slot,a in enumerate(m['axes']):
                if a is None:continue
                clipped=clip_axis_to_ball(a,center)
                if clipped:axes.append((slot,clipped))
            pairs=[]
            for j,(slot,a) in enumerate(axes):
                for other,b in axes[j+1:]:pairs.append(dict(first=slot,second=other,distance_m=sampled_hausdorff(a,b)))
            entry['methods'][name]=dict(local_axes=len(axes),length_m=sum(float(np.linalg.norm(s[1]-s[0])) for _,a in axes for s in a),
                pair_distances=pairs,close_pair_counts={str(t):sum(p['distance_m']<=t for p in pairs) for t in thresholds})
        result.append(entry)
    summary={name:dict(local_axes=sum(r['methods'][name]['local_axes'] for r in result),
        sampled_close_pairs={str(t):sum(r['methods'][name]['close_pair_counts'][str(t)] for r in result) for t in thresholds},
        frames_with_close_pair={str(t):sum(r['methods'][name]['close_pair_counts'][str(t)]>0 for r in result) for t in thresholds}) for name in ['learned','nonlearning']}
    out=ROOT/'docs/figures/gse_learned_geometry_pair_v1/local_overlap.json'
    with out.open('x') as f:json.dump(dict(summary=summary,per_window=result,source_sha256={str(p.relative_to(ROOT)):sha(p) for p in paths},
        radius_m=10,sampling_spacing_m=.25,thresholds_m=thresholds,
        limitation='Symmetric sampled curve similarity; not false-positive truth, not deduplication, original graph unchanged.'),f,indent=2)
    print(json.dumps(summary))


if __name__=='__main__':main()
