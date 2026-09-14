"""Read-only complete-population attribution of sealed direction outputs.

Fixed descriptive bins, never thresholds for selecting/recomputing predictions.
No original observation or teacher payload is opened; old scores unchanged.
"""
from pathlib import Path
import hashlib,json
import numpy as np


def main():
    root=Path(__file__).resolve().parents[2]
    run=root/'results/gate3_semantics/gate3_20260912_gse_observation_axis_comparison_v1_seed0'
    scope=json.loads((root/'docs/figures/gse_conditional_geometry_fit_v1/observation_axis_comparison_scope.json').read_text())
    seals={p:h for h,p in (line.split('  ',1) for line in (run/'artifacts/evidence_sha256.txt').read_text().splitlines())}
    angle_edges=[0.,5.,15.,30.,90.000001]
    gap_edges=[0.,.01,.1,.5,1.000001]
    records=[]
    for entry in scope['entries']:
        case=entry['identity']['case'];parent=entry['identity']['parent_id']
        for method in ('POINT','NORMAL'):
            path=run/f'artifacts/{method}_{case:03d}.npz'
            with path.open('rb') as stream:
                if hashlib.file_digest(stream,'sha256').hexdigest()!=seals[str(path.relative_to(root))]:
                    raise ValueError('sealed prediction drift')
            with np.load(path,allow_pickle=False) as f:
                known=f['reference_known'];same=f['same_reference'];valid=f['prediction_valid']
                truth=f['reference'];pred=f['predictions'];weight=f['weights'].astype(float)
                ev=f['eigenvalues'];n=f['neighbor_index'];safe=np.maximum(n,0)
                centers=f['centers'];support=f['support_points']
                gap=ev[:,2]-ev[:,1] if method=='POINT' else ev[:,1]-ev[:,0]
                gap=np.clip(gap/np.maximum(abs(ev).sum(1),np.finfo(float).tiny),0,1)
                pair_gap=np.minimum(gap[:,None],gap[safe])
                distance=np.linalg.norm(centers[:,None]-centers[safe],axis=-1)
                error=np.ones(pred.shape);error[valid]=abs(pred[valid]-truth[valid])
                reference_angle=np.degrees(np.arccos(np.clip(truth,0,1)))
                predicted_angle=np.full(pred.shape,np.nan)
                predicted_angle[valid]=np.degrees(np.arccos(np.clip(pred[valid],0,1)))
                for category,mask in [('all',known),('same',known&same),('cross',known&~same)]:
                    if not mask.any():continue
                    w=weight[mask];w=w/w.sum()
                    row=dict(case=case,parent=parent,method=method,category=category,
                        full_error=float(w@error[mask]),constant_error=float(w@abs(1-truth[mask])),
                        mean_reference_angle_deg=float(w@reference_angle[mask]),
                        undefined_mass=float(w@(~valid[mask])),
                        mean_neighbor_distance_m=float(w@distance[mask]),
                        mean_support_points=float(w@np.minimum(support[:,None],support[safe])[mask]),
                        bins={})
                    for kind,values,edges in [('reference_angle',reference_angle,angle_edges),('relative_eigengap',pair_gap,gap_edges)]:
                        bins=[]
                        for lo,hi in zip(edges,edges[1:]):
                            selected=(values[mask]>=lo)&(values[mask]<hi)
                            mass=float(w[selected].sum())
                            good=selected&valid[mask]
                            bins.append(dict(low=lo,high=hi,mass=mass,
                                error_contribution=float(w[selected]@error[mask][selected]),
                                constant_contribution=float(w[selected]@abs(1-truth[mask][selected])),
                                predicted_angle_contribution=float(w[good]@predicted_angle[mask][good]),
                                valid_mass=float(w[good].sum())))
                        row['bins'][kind]=bins
                    records.append(row)
    summary={}
    for method in ('POINT','NORMAL'):
        summary[method]={}
        for category in ('all','same','cross'):
            rows=[r for r in records if r['method']==method and r['category']==category]
            parents=sorted({r['parent'] for r in rows})
            # Equal observations within parent, then equal parent weights.
            row_weights=np.array([1/(len(parents)*sum(t['parent']==r['parent'] for t in rows)) for r in rows])
            out={k:float(row_weights@np.array([r[k] for r in rows])) for k in
                 ('full_error','constant_error','mean_reference_angle_deg','undefined_mass','mean_neighbor_distance_m','mean_support_points')}
            out['bins']={}
            for kind in ('reference_angle','relative_eigengap'):
                bins=[]
                for b in range(4):
                    result={k:float(row_weights@np.array([r['bins'][kind][b][k] for r in rows]))
                            for k in ('mass','error_contribution','constant_contribution','predicted_angle_contribution','valid_mass')}
                    result.update(low=rows[0]['bins'][kind][b]['low'],high=rows[0]['bins'][kind][b]['high'])
                    result['conditional_error']=result['error_contribution']/result['mass'] if result['mass'] else None
                    result['conditional_predicted_angle_deg']=result['predicted_angle_contribution']/result['valid_mass'] if result['valid_mass'] else None
                    bins.append(result)
                out['bins'][kind]=bins
            summary[method][category]=out
    output=root/'docs/figures/gse_conditional_geometry_fit_v1/observation_axis_comparison/error_attribution.json'
    with output.open('x') as stream:json.dump(dict(scope='read_only_saved_outputs_no_new_inference',
        source_seal_sha256=hashlib.sha256((run/'artifacts/evidence_sha256.txt').read_bytes()).hexdigest(),
        fixed_bins=dict(reference_angle_deg=angle_edges,relative_eigengap=gap_edges),
        summary=summary,records=records),stream,indent=2,allow_nan=False)
    for method in summary:
        print(method,json.dumps(summary[method]['cross']))


if __name__=='__main__':main()
