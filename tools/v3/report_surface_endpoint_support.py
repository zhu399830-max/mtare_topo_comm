"""Describe source endpoint support from sealed interval artifacts only."""
from _bootstrap import PROJECT_ROOT as ROOT
import hashlib,json
import numpy as np
from mtare_topo.teacher.gse_surface_interval_support import describe_endpoint_support

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
        with np.load(RUN/f"artifacts/window_{e['observation']:02d}.npz",allow_pickle=False) as z:old_state=z['row_state']
        intervals={int(k):v for k,v in json.loads((RUN/f"artifacts/intervals_{e['observation']:02d}.json").read_text()).items()}
        primitives=json.loads((ROOT/e['construction']).read_text())['realized_primitives']
        lengths={k:float(np.cumsum(np.linalg.norm(np.diff(np.asarray(primitives[k]['centerline_xyz_m']),axis=0),axis=1))[-1]) for k in intervals}
        r=describe_endpoint_support(a,intervals,lengths)
        if not np.array_equal(r['row_state'],old_state):raise ValueError('historical state mutation')
        _,inverse,counts=np.unique(a[:,0],return_inverse=True,return_counts=True)
        single=counts[inverse]==1;endpoint=r['source_endpoint_interval_hypothesis']>=0
        interior=r['row_state']==3
        rows.append(dict(observation=e['observation'],returns=len(counts),
            single_source_interior=int((single&interior).sum()),
            single_source_endpoint_hypothesis=int((single&endpoint).sum()),
            single_source_either_hypothesis=int((single&(interior|endpoint)).sum()),
            roi_crop_contact_pairs=int(r['roi_crop_boundary_contact'].sum()),
            source_endpoint_unresolved_pairs=int(((r['source_endpoint_contact_bits']>0)&~endpoint).sum()),
            multi_source_returns=int((counts>1).sum())))
    totals={k:sum(r[k] for r in rows) for k in rows[0] if k!='observation'}
    result=dict(windows=rows,totals=totals,seal_entries_verified=len(lines),
        old_states_unchanged=True,point_labels_generated=0,structural_labels_generated=0,
        source='Saved original surface hypotheses, source lengths and saved ROI intervals; no point/surface recomputation',
        limitation='Single source and interval support do not by themselves prove physical terminal, structure membership, or trainable target qualification')
    out=ROOT/'docs/figures/gse_membership_surface_residual_v1/endpoint_support_summary.json'
    with out.open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps(totals))


def report_openings():
    """Link original opening evidence, without assuming a local surface return."""
    refs=ROOT/'results/gate3_semantics/gate3_20260911_gse_membership_fit_v1r1_seed0'
    seal={p:h for h,p in (line.split('  ',1) for line in (refs/'artifacts/evidence_sha256.txt').read_text().splitlines())}
    scope=json.loads((RUN/'config/data_card.json').read_text())['scope'];rows=[]
    for e in scope['entries']:
        i=e['observation'];path=f'artifacts/reference_{i:02d}.json';raw=(refs/path).read_bytes()
        if hashlib.sha256(raw).hexdigest()!=seal[path]:raise ValueError('reference drift')
        target=json.loads(raw)
        if target['source_binding']['source']['task']!=e['task'] or target['record']['source_frame_indices'][-1]!=e['frame']:
            raise ValueError('reference identity mismatch')
        for p in (e['residual'],e['construction']):
            if hashlib.sha256((ROOT/p).read_bytes()).hexdigest()!=scope['input_sha256'][p]:raise ValueError('input drift')
        primitives=json.loads((ROOT/e['construction']).read_text())['realized_primitives']
        lookup={p['primitive_id']:k for k,p in enumerate(primitives)}
        intervals=json.loads((RUN/f'artifacts/intervals_{i:02d}.json').read_text())
        with np.load(ROOT/e['residual'],allow_pickle=False) as z:
            a=z['records'];roi=set(z['roi_return_indices'].tolist())
        for j,o in enumerate(target['teacher_provenance']['openings']):
            k=lookup[o['primitive_id_teacher_only']];arc=o['reference_arc_m']
            spans=intervals.get(str(k),{}).get('intervals_m',[])
            covered=[n for n,(lo,hi) in enumerate(spans) if lo<=arc<=hi]
            rows.append(dict(observation=i,opening=j,source_index=k,interval_indices=covered,
                boundary_delta_m=min((min(abs(arc-lo),abs(arc-hi)) for lo,hi in spans),default=None),
                source_pairs_in_roi=int((a[:,1]==k).sum()),surface_witnesses=len(o['surface_ray_indices']),
                surface_witnesses_returning_in_roi=len(roi.intersection(o['surface_ray_indices'])),
                crossing_witnesses=len(o['crossing_ray_indices'])))
    result=dict(openings=rows,total=len(rows),exact_interval_links=sum(len(r['interval_indices'])==1 for r in rows),
        labels_changed=0,interpretation='Opening ray evidence need not end on a source surface inside the ROI. No source omission or floating boundary is repaired into a label.')
    out=ROOT/'docs/figures/gse_membership_surface_residual_v1/opening_interval_link_summary.json'
    with out.open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps({k:v for k,v in result.items() if k!='openings'}))


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--openings',action='store_true');args=p.parse_args()
    if args.openings:report_openings()
    else:main()
