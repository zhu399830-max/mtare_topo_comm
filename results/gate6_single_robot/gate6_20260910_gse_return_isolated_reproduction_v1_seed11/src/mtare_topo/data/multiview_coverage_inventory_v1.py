"""Prospective rank-zero route coverage, metadata only and label independent.

This inventory is not a data card or a license to read scans/generate labels.
It cannot establish positive/negative/unknown structure coverage.
"""
import json
from collections import Counter
from mtare_topo.governance_surface_material import read_pinned
from mtare_topo.governance_surface_selection import INVENTORY,SEAL_SHA256,PARENTS
from mtare_topo.governance_surface_input import SELECTION,SELECTION_SEAL
from .gse_surface_selection_v1 import select_parent,VARIANTS


def expand_interval(interval,split):
    records=interval['variants'];rows=[];reference=None
    if len(records)!=3 or {r['variant'] for r in records}!=set(VARIANTS):raise ValueError('three unique variants required')
    for v in VARIANTS:
        r=next(r for r in records if r['variant']==v)
        ids=r['source_sequence_ids'];frames=r['frame_rows'];indices=r['sequence_rows'];arcs=r['decision_arc_m']
        if not ids or not len(ids)==len(frames)==len(indices)==len(arcs):raise ValueError('nonempty aligned interval required')
        identity=(ids,frames)
        if reference is not None and identity!=reference:raise ValueError('variant correspondence differs')
        reference=identity
        for i,sid in enumerate(ids):
            if frames[i]!=list(range(frames[i][0],frames[i][0]+5)) or (i and frames[i]!=[f+1 for f in frames[i-1]]):
                raise ValueError('causal consecutive history required')
            rows.append(dict(task=r['task'],variant=v,split=split,traversal_id=interval['traversal_id'],
                source_sequence_id=sid,sequence_row=indices[i],frame_rows=frames[i],decision_index=i,
                decision_route_arc_m=arcs[i],label_status='NOT_EVALUATED',sampling_role='unfiltered_continuous_rank0_route'))
    return rows


def compile_inventory(root):
    opened={}
    def read(p,h):
        raw=read_pinned(root,p,h);opened[p]=h;return raw
    def sealed(run,h):
        raw=read(run+'/artifacts/evidence_sha256.txt',h)
        return {p:h for h,p in (line.split('  ',1) for line in raw.decode().splitlines())}
    ip=sealed(INVENTORY,SEAL_SHA256);bp=sealed(SELECTION,SELECTION_SEAL)
    p=SELECTION+'/artifacts/selection_manifest.json';old=json.loads(read(p,bp[p]))
    selected=[r for r in old['observations'] if r['edge_selection_rank']==0]
    if len(selected)!=210:raise ValueError('original rank0 population drift')
    rows=[];routes=[]
    for parent in PARENTS:
        previous=[r for r in selected if r['parent_id']==parent]
        if len(previous)!=3 or len({r['traversal_id'] for r in previous})!=1 or len({r['split'] for r in previous})!=1:
            raise ValueError('unique paired old route required')
        split=previous[0]['split'];p=INVENTORY+'/artifacts/'+parent+'_identity_intervals.json'
        report=json.loads(read(p,ip[p]))
        reselected=select_parent(report,split,edges_per_parent=1)
        # Check identity, not source labels or model output.
        for a in previous:
            b=next(r for r in reselected['observations'] if r['task']==a['task'])
            for field in ('traversal_id','source_sequence_id','frame_rows'):
                if a[field]!=b[field]:raise ValueError('old hash selection cannot be reproduced')
        route=next(r for r in report['intervals'] if r['traversal_id']==previous[0]['traversal_id'])
        expanded=expand_interval(route,split);rows.extend(expanded)
        arcs=route['variants'][0]['decision_arc_m']
        routes.append(dict(parent=parent,split=split,traversal_id=route['traversal_id'],logical_windows=len(arcs),
            arc_min_m=arcs[0],arc_max_m=arcs[-1],arc_span_m=arcs[-1]-arcs[0],
            continuous_route_evidence=False,structure_coverage_verified=False))
    counts=dict(parents=len(routes),physical_edges=len(routes),directed_traversals=len(routes),
        logical_windows=sum(r['logical_windows'] for r in routes),observations=len(rows),
        unique_variant_frames=len({(r['task'],f) for r in rows for f in r['frame_rows']}),
        splits=dict(Counter(r['split'] for r in rows)),parent_splits=dict(Counter(r['split'] for r in routes)),
        original_background_role_overlap=sum((r['task'],r['source_sequence_id']) in {(s['task'],s['source_sequence_id']) for s in selected} for r in rows))
    return dict(schema='multiview_coverage_inventory_v1',status='METADATA_CANDIDATE_NOT_DATA_CARD',
        policy='Original sealed hash rank0 edge and direction per parent, all existing causal windows, all3 variants, no target/score-based selection.',
        counts=counts,routes=routes,observations=rows,source_sha256=opened,
        positive_count=None,confirmed_background_count=None,unknown_count=None,
        restrictions=['C01_C07_only','original_parent_split','no_scan_teacher_or_checkpoint_payload',
                      'no_training_authority','not_strict_test','no_cross_traversal_stitching','background_sampling_role_is_not_negative_label'])
