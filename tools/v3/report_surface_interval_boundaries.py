"""Classify sealed boundary cases without modifying support states or labels."""
from _bootstrap import PROJECT_ROOT as ROOT
import hashlib,json
import numpy as np

RUN=ROOT/'results/gate3_semantics/gate3_20260911_gse_surface_interval_support_v1_seed0'


def main():
    lines=(RUN/'artifacts/evidence_sha256.txt').read_text().splitlines()
    for line in lines:
        h,p=line.split('  ',1)
        if hashlib.sha256((ROOT/p).read_bytes()).hexdigest()!=h:raise ValueError('seal drift')
    scope=json.loads((RUN/'config/data_card.json').read_text())['scope'];rows=[]
    for e in scope['entries']:
        for p in (e['residual'],e['construction']):
            if hashlib.sha256((ROOT/p).read_bytes()).hexdigest()!=scope['input_sha256'][p]:raise ValueError('input drift')
        with np.load(ROOT/e['residual'],allow_pickle=False) as z:a=z['records']
        with np.load(RUN/f"artifacts/window_{e['observation']:02d}.npz",allow_pickle=False) as z:state=z['row_state']
        primitives=json.loads((ROOT/e['construction']).read_text())['realized_primitives']
        count=dict(source_endpoint_bound=0,other_boundary_bound=0,zero_width_source_endpoint_bound=0)
        for k in np.unique(a[:,1]).astype(int):
            points=np.asarray(primitives[k]['centerline_xyz_m'])
            # Exact original arc accumulation, not pairwise sum or tolerance.
            end=np.cumsum(np.linalg.norm(np.diff(points,axis=0),axis=1))[-1]
            q=a[(a[:,1]==k)&(state==2)]
            at=(q[:,5]==0)|(q[:,6]==end)
            count['source_endpoint_bound']+=int(at.sum())
            count['other_boundary_bound']+=int((~at).sum())
            count['zero_width_source_endpoint_bound']+=int(((q[:,5]==q[:,6])&at).sum())
        rows.append(dict(observation=e['observation'],**count))
    result=dict(windows=rows,totals={k:sum(r[k] for r in rows) for k in count},
        seal_entries_verified=len(lines),labels_changed=0,original_states_changed=False,
        interpretation='A source endpoint is not necessarily a physical terminal. Strict-interior exclusion chiefly concerns source mesh endpoints, not ROI crop crossings. Keep endpoint support as a separate hypothesis; do not promote it to structural membership.')
    out=ROOT/'docs/figures/gse_membership_surface_residual_v1/interval_boundary_summary.json'
    with out.open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps(result['totals']))


if __name__=='__main__':main()
