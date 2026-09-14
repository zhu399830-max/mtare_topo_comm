"""Post-hoc absolute-axis scoring, not a new prediction experiment.

Original relative scores remain authoritative for the frozen comparison.
References below are original construction-conditioned midpoint directions,
not newly generated observable labels or inference candidates.
"""
from pathlib import Path
import hashlib,json
import numpy as np


def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def main():
    root=Path(__file__).resolve().parents[2]
    run=root/'results/gate3_semantics/gate3_20260912_gse_observation_axis_comparison_v1_seed0'
    refs=root/'results/gate3_semantics/gate3_20260911_gse_conditional_development_geometry_v1_seed0'
    scope=json.loads((root/'docs/figures/gse_conditional_geometry_fit_v1/observation_axis_comparison_scope.json').read_text())
    seals={}
    for r in (run,refs):
        seals.update({p:h for h,p in (l.split('  ',1) for l in (r/'artifacts/evidence_sha256.txt').read_text().splitlines())})
    used={};rows=[]
    def checked(p):
        name=str(p.relative_to(root));h=sha(p)
        if h!=seals[name]:raise ValueError('sealed file drift '+name)
        used[name]=h;return p
    for entry in scope['entries']:
        i=entry['identity']['case']
        ref=json.loads(checked(refs/f'artifacts/case_{i:03d}/references.json').read_text())
        if ref['identity']!=entry['identity'] or ref['observability_certified'] is not False:
            raise ValueError('reference identity/meaning mismatch')
        with np.load(checked(root/entry['target']['path']),allow_pickle=False) as y:
            component=y['patch_reference_component'].copy()
        direction=np.zeros((len(component),3));center=np.zeros_like(direction)
        for p,c in enumerate(component):
            if c<0:continue
            direction[p]=ref['candidates'][c]['normal'];center[p]=ref['candidates'][c]['center_m']
        if not np.allclose(np.linalg.norm(direction[component>=0],axis=1),1,atol=1e-6,rtol=0):
            raise ValueError('nonunit conditional axis')
        for method in ('POINT','NORMAL'):
            with np.load(checked(run/f'artifacts/{method}_{i:03d}.npz'),allow_pickle=False) as f:
                axes=f['axes'];valid=f['axis_valid'];known=f['reference_known'];same=f['same_reference']
                n=f['neighbor_index'];weight=f['weights'].astype(float)
                angle=np.degrees(np.arccos(np.clip(abs(np.sum(axes*direction,axis=1)),0,1)))
                angle[~valid]=90.
                forward=np.degrees(np.arccos(np.clip(abs(direction[:,0]),0,1)))
                displacement=np.linalg.norm(f['centers']-center,axis=1)
                for category,mask in [('all',known),('same',known&same),('cross',known&~same)]:
                    if not mask.any():continue
                    a,b=np.nonzero(mask);w=weight[mask];w/=w.sum()
                    marginal=np.zeros(len(axes));np.add.at(marginal,a,w/2);np.add.at(marginal,n[mask],w/2)
                    if np.any((marginal>0)&(component<0)):raise ValueError('reference-free scored patch')
                    rows.append(dict(case=i,parent=entry['identity']['parent_id'],method=method,category=category,
                        absolute_error_deg=float(marginal@angle),
                        forward_constant_error_deg=float(marginal@forward),
                        valid_mass=float(marginal@valid),
                        reference_center_to_patch_distance_m=float(marginal@displacement)))
    summary={}
    for method in ('POINT','NORMAL'):
        summary[method]={}
        for category in ('all','same','cross'):
            selected=[r for r in rows if r['method']==method and r['category']==category]
            parents=sorted({r['parent'] for r in selected})
            perparent={p:{k:float(np.mean([r[k] for r in selected if r['parent']==p])) for k in
                ('absolute_error_deg','forward_constant_error_deg','valid_mass','reference_center_to_patch_distance_m')} for p in parents}
            summary[method][category]=dict(per_parent=perparent,macro={k:float(np.mean([d[k] for d in perparent.values()])) for k in next(iter(perparent.values()))})
    out=root/'docs/figures/gse_conditional_geometry_fit_v1/observation_axis_comparison/absolute_axis_diagnostic.json'
    with out.open('x') as f:json.dump(dict(status='POST_HOC_DIAGNOSTIC_NOT_NEW_ACCEPTANCE',
        meaning='Original conditional midpoint direction versus saved axes, unoriented angle; undefined90deg. Marginalize same original pair weights equally over endpoints. Robot-forward fixed baseline, no selection.',
        new_predictions=0,source_sha256=used,summary=summary,rows=rows),f,indent=2)
    print(json.dumps({m:{c:d['macro'] for c,d in v.items()} for m,v in summary.items()}))


if __name__=='__main__':main()
