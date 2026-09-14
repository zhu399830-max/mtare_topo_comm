"""Final saved-output support diagnosis; no original payload or estimates."""
from pathlib import Path
import json,hashlib
import numpy as np


def main():
    root=Path(__file__).resolve().parents[2]
    run=root/'results/gate3_semantics/gate3_20260912_gse_observation_axis_comparison_v1_seed0'
    scope=json.loads((root/'docs/figures/gse_conditional_geometry_fit_v1/observation_axis_comparison_scope.json').read_text())
    seal={p:h for h,p in (line.split('  ',1) for line in (run/'artifacts/evidence_sha256.txt').read_text().splitlines())}
    rows=[]
    for entry in scope['entries']:
        case=entry['identity']['case'];p=run/f'artifacts/NORMAL_{case:03d}.npz'
        with p.open('rb') as stream:
            assert hashlib.file_digest(stream,'sha256').hexdigest()==seal[str(p.relative_to(root))]
        with np.load(p,allow_pickle=False) as f:
            ev=f['eigenvalues'];n=f['neighbor_index'];safe=np.maximum(n,0)
            trace=ev.sum(1);rank1=np.divide(ev[:,-1],trace,out=np.zeros(len(ev)),where=trace>0)
            # Ratio is weighted normal concentration, not count of wall types.
            both=np.minimum(rank1[:,None],rank1[safe]);either=np.maximum(rank1[:,None],rank1[safe])
            c=f['centers'];dist=np.linalg.norm(c[:,None]-c[safe],axis=-1)
            radius=np.max(np.where(n>=0,dist,0),axis=1)
            pair_radius=np.maximum(radius[:,None],radius[safe])
            for category,mask in [('all',f['reference_known']),('cross',f['reference_known']&~f['same_reference'])]:
                if not mask.any():continue
                w=f['weights'][mask].astype(float);w/=w.sum()
                rows.append(dict(case=case,parent=entry['identity']['parent_id'],category=category,
                    either_normal_concentration_ge99=float(w@(either[mask]>=.99)),
                    both_normal_concentration_ge99=float(w@(both[mask]>=.99)),
                    mean_larger_neighborhood_radius_m=float(w@pair_radius[mask]),
                    both_neighborhood_radii_below1m=float(w@(pair_radius[mask]<1)),
                    both_neighborhood_radii_below2m=float(w@(pair_radius[mask]<2))))
    summary={}
    for category in ('all','cross'):
        selected=[r for r in rows if r['category']==category]
        parents=sorted({r['parent'] for r in selected})
        keys=[k for k in selected[0] if k not in ('case','parent','category')]
        summary[category]={k:float(np.mean([np.mean([r[k] for r in selected if r['parent']==p]) for p in parents])) for k in keys}
    path=root/'docs/figures/gse_conditional_geometry_fit_v1/observation_axis_comparison/support_diagnostic.json'
    with path.open('x') as f:json.dump(dict(summary=summary,rows=rows,
        interpretation='Normal concentration is not a count of distinct walls. Saved center-neighbour radius is not ray visibility or full point support radius. Reference absolute center/extent is absent; cannot measure reference scale mismatch here.',
        change_to_predictions=False,closes='same saved local-axis support diagnosis; do not repeat'),f,indent=2)
    print(json.dumps(summary))


if __name__=='__main__':main()
