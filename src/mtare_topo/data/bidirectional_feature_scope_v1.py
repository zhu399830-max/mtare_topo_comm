"""Exact missing817 shared-encoder scope; metadata only, no label filtering."""
import hashlib
import json
from collections import Counter
from .gse_surface_feature_scope_v1 import TRAINING,CHECKPOINT_SHA
from mtare_topo.governance_surface_material import read_pinned

INVENTORY='configs/v3/gate3/bidirectional_feature_inventory_v1.json'
INVENTORY_SHA='39ae0f3ba770da12c3c8b94e50c6adf67a56220c35369436c5eccc88ca0d39d6'

def compile_feature_scope(root):
    raw=read_pinned(root,INVENTORY,INVENTORY_SHA);doc=json.loads(raw)
    groups={};reused=[];seen=set();splits=Counter();frames=set();parents=set()
    for r in doc['rows']:
        s=r['source'];k=(s['task'],s['source_sequence_id'],tuple(s['frame_rows']))
        if k in seen:raise ValueError('duplicate exact source')
        seen.add(k)
        parent=s['task'].split('__')[0];cohort=parent.rsplit('_',1)[-1]
        if (s['split']=='fit' and cohort not in {'C01','C02','C03','C04','C05','C06'}) or (s['split']=='calibration' and cohort!='C07') or s['split'] not in {'fit','calibration'}:
            raise ValueError('split violation')
        if r['feature_cache'] is not None:
            reused.append(dict(source=s,**r['feature_cache']));continue
        ref=r['input_reference'];p=ref['input_path']
        g=groups.setdefault(p,dict(task=s['task'],input_path=p,input_sha256=ref['input_sha256'],
                                 observation_count=ref['observation_count'],observations=[]))
        if (g['task'],g['input_sha256'],g['observation_count'])!=(s['task'],ref['input_sha256'],ref['observation_count']):raise ValueError('package binding conflict')
        g['observations'].append(dict(input_row=ref['input_row'],source_sequence_id=s['source_sequence_id'],frame_rows=s['frame_rows'],split=s['split']))
        splits[s['split']]+=1;frames.update((s['task'],f) for f in s['frame_rows']);parents.add(parent)
    for g in groups.values():
        g['observations'].sort(key=lambda r:r['input_row'])
        if [r['input_row'] for r in g['observations']]!=list(range(g['observation_count'])):raise ValueError('unexpected collateral rows')
    if len(seen)!=860 or len(reused)!=43 or len(groups)!=59 or splits!=dict(fit=266,calibration=551):raise ValueError('population drift')
    return dict(schema_version='bidirectional_feature_scope_v1',metadata_sha256={INVENTORY:INVENTORY_SHA},
                checkpoint=dict(path=TRAINING+'/artifacts/models/seed0/selected.pt',sha256=CHECKPOINT_SHA,seed=0,epoch=2),
                tasks=[groups[k] for k in sorted(groups)],reused_features=reused,
                counts=dict(observations=817,parents=len(parents),packages=59,unique_variant_frames=len(frames),splits=dict(splits),reused=43,incidental_rows=0),
                weights_read=False,scan_payloads_read=False,labels=0,training_steps=0,
                limitations=['historical positive-conditioned population','calibration not independent evaluation','no train updates','compact shared features only'])
