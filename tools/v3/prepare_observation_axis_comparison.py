"""Bind the existing 240-row cache without opening observation/target arrays.

Preparation only: no data-dependent estimator, training or experiment run.
The new operation still needs its frozen run specification and preflight.
"""
import hashlib
import json
from pathlib import Path


def main():
    root=Path(__file__).resolve().parents[2]
    card_path='configs/v3/gate3/data_cards/gse_conditional_development_evaluation_v1.json'
    spec_path='configs/v3/gate3/gse_conditional_development_evaluation_v1.json'
    old=json.loads((root/card_path).read_text())
    spec=json.loads((root/spec_path).read_text())
    entries=old['scope']['entries']
    assert [e['identity']['case'] for e in entries]==list(range(240))
    parents=sorted({e['identity']['parent_id'] for e in entries})
    assert len(parents)==5 and all(p.endswith('_C07') for p in parents)
    rows=[]
    for e in entries:
        rows.append(dict(identity=e['identity'],**{
            k:dict(path=e[k],sha256=spec['input_sha256'][e[k]],bytes=(root/e[k]).stat().st_size)
            for k in ('feature','target')}))
    scope=dict(status='PREPARED_NOT_EXECUTED',observations=240,parents=parents,
        physical_edges=len({e['identity']['physical_edge_id'] for e in entries}),
        variant_frames=len({(e['identity']['task'],f) for e in entries for f in e['identity']['frame_rows']}),
        entries=rows,training_steps=0,gpu_calls=0,
        input_fields=['registered_returns_xyz_m','patch_point_patch_index','patch_centers_m'],
        neighbors='Recompute original observation-only k8; target indices checked only after predictions',
        methods={'POINT':'Pooled point covariance largest-eigenvalue axis',
                 'NORMAL':'Count-weighted reliable patch-normal scatter smallest-eigenvalue axis'},
        shared_support='Each receiver plus same original k8 neighbours; all their observed points; no GT grouping',
        reference_meaning='Construction-conditioned axis abs-dot only; NOT independently certified observable channel identity',
        scoring='Reuse saved component-pair weights; all/same/cross and per parent; report validity coverage and matched-valid MAE, PLUS full-population absolute error assigning undefined estimates worst error 1. Unknown references never become negatives.',
        selection='No parameter calibration, checkpoint selection, sample selection or thresholds from evaluation',
        limits=dict(cpu_workers=1,host_bytes=4*1024**3,wall_seconds=1800,output_bytes=512*1024**2),
        not_claimed=['structure_detection','ground_reachability','graph_advantage','strict_unseen_evaluation'],
        downstream_gap='Patch-reference directions are not channel identities; graph contribution needs observable direction-task interface and suitable sequential evidence separately',
        source_metadata={p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in (card_path,spec_path)})
    out=root/'docs/figures/gse_conditional_geometry_fit_v1/observation_axis_comparison_scope.json'
    with out.open('x') as f:json.dump(scope,f,indent=2)
    print(json.dumps({k:v for k,v in scope.items() if k in ('status','observations','parents','physical_edges','variant_frames','limits')}))


if __name__=='__main__':main()
