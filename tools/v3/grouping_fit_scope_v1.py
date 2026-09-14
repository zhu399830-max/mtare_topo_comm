"""Metadata-only fixed16 sample selection from historical fit, no score selection."""
import json
from collections import Counter
from pathlib import Path
from bidirectional_paired_scope_v1 import compile_scope as original_scope
from mtare_topo.governance_surface_selection import digest
from mtare_topo.governance_surface_material import read_pinned

POP='results/gate3_semantics/gate3_20260909_gse_observed_detector_pilot_v1_seed0/config/population.json'


def compile_scope(root):
    from surface_features_v1 import sha
    root=Path(root);original=original_scope(root)
    h=sha(root/POP);metadata=json.loads(read_pinned(root,POP,h))
    fits=[r for r in metadata if r['split']=='fit']
    parents=sorted({r['source']['task'].split('__')[0] for r in fits})
    selected=[];taken=set()
    def add(rows,n):
        for row in sorted(rows,key=lambda r:digest(dict(seed=0,source=r['source']))):
            if len([r for r in selected if r in rows])>=n:return
            key=digest(row['source'])
            if key in taken:continue
            selected.append(row);taken.add(key)
    for parent in parents:
        add([r for r in fits if r['anchors']==1 and r['source']['task'].split('__')[0]==parent],2)
    add([r for r in fits if r['anchors']==1],12)
    for parent in parents:
        add([r for r in fits if r['anchors']==0 and r['source']['task'].split('__')[0]==parent],1)
    if len(selected)!=16 or sum(r['anchors'] for r in selected)!=12:raise ValueError('fixed16 selection not satisfied')
    selected=sorted(selected,key=lambda r:digest(r['source']))
    selected_rows=[r for r in original['observations'] if digest(r['source']) in taken]
    frames={(r['source']['task'],i) for r in selected for i in r['source']['frame_rows']}
    return dict(original_scope=original,selection=selected,selected_rows=selected_rows,
        historical_population=dict(path=POP,sha256=h),
        counts=dict(observations=16,positive_observations=12,without_positive=4,
            parent_maps=parents,unique_variant_frames=len(frames),raw_frame_occurrences=80,
            independent_physical_junctions=5),
        spacing='original five consecutive frames; identity-hash selection, not fixed inter-observation distance',
        split='fit-only interface diagnostic; all original860 scope metadata authorized, task-level incidental source decoding only; no cal target/features used',
        no_new_labels=True,no_protected_worlds=True)
