"""Metadata-only reuse inventory; no labels enter encoder input selection."""
import hashlib
import json
from pathlib import Path
from collections import Counter

ROOT=Path(__file__).resolve().parents[2]
BASE='results/gate3_semantics/'
INPUT_RUNS={
 'gate3_20260909_gse_v8_multiview_mechanism_v1_seed20260906':'2e5ddeef503cbe00fd851e2342d2480a4cd4230b831233202a809f04580cc9f6',
 'gate3_20260909_gse_v8_fit_opposite141_v1_seed20260906':'24eb64854830b947b578920975ec5461317b1c9c40d3d095918eb27a1ac4db8c',
 'gate3_20260909_gse_v8_calibration289_v1_seed20260906':'47cbc0c3d616eb7d0e16c94ecc3603ac4716c51cd672ecc7b464b45fa39ebc5f',
 'gate3_20260909_gse_v8_opposite289_v1_seed20260906':'af70ee9af26e01ea7ced1de90f3b9e1c21f6527688fdfbc8e09083b4571225f6'}
FEATURE_RUNS={
 'gate3_20260907_gse_surface_features_v1_seed0':'31bbe090558dc1c812bbf4aa6783cc15fad4618cbc095898128fede0b6f90698',
 'gate3_20260908_gse_supplement_features_v1_seed0':'b01dcff809acdd9d859a5fa461fefba507b0af6cc0f606fb8fffeaa5ba52ea6e',
 'gate3_20260909_gse_continuous_features_v1_seed0':'39f48ff6e49d9e543c1cd75b3d6678c830523002e8c658ab7c213310562b3d25'}
ENCODER='200f5c2fbe66d68961cf2aea06e21f747cb5b8d419536008491bb58d3641a8cb'

def compile_inventory():
    opened={}
    def read(p,h):
        raw=(ROOT/p).read_bytes()
        if hashlib.sha256(raw).hexdigest()!=h:raise ValueError('metadata drift '+p)
        opened[p]=h;return raw
    def manifest(name,h,file):
        run=BASE+name
        seal=read(run+'/artifacts/evidence_sha256.txt',h)
        pins={p:h for h,p in (l.split('  ',1) for l in seal.decode().splitlines())}
        p=run+'/'+file
        return json.loads(read(p,pins[p])),pins
    key=lambda s:(s['task'],s['source_sequence_id'],tuple(s['frame_rows']))
    rows={}
    for name,h in INPUT_RUNS.items():
        card,_=manifest(name,h,'config/data_card.json')
        for e in card['scope']['entries']:
            for r in e['input_references']:
                s=r['source'];k=key(s)
                if k in rows or s['split'] not in {'fit','calibration'}:raise ValueError('source/split collision')
                rows[k]=dict(source=s,input_reference=r,feature_cache=None)
    if Counter(r['source']['split'] for r in rows.values())!=dict(fit=282,calibration=578):raise ValueError('exact bidirectional population')
    for name,h in FEATURE_RUNS.items():
        doc,pins=manifest(name,h,'artifacts/feature_manifest.json')
        for f in doc['observations']:
            k=key(f['source'])
            if k not in rows:continue
            p=BASE+name+'/'+f['path']
            if f['frozen_encoder_state_sha256']!=ENCODER or pins[p]!=f['sha256']:raise ValueError('feature state/hash mismatch')
            ref=rows[k]['input_reference'];src=f['source']
            if src['input_file_sha256']!=ref['input_sha256'] or src['input_row']!=ref['input_row']:
                raise ValueError('cache source identity matches but container differs: equality check required')
            if rows[k]['feature_cache'] is not None:raise ValueError('duplicate feature source')
            rows[k]['feature_cache']=dict(path=p,sha256=f['sha256'],feature_entry=f)
    out=[rows[k] for k in sorted(rows)]
    return dict(status='METADATA_REUSE_CANDIDATES_NOT_GPU_AUTHORITY',rows=out,metadata_sha256=opened,
                encoder_state_sha256=ENCODER,counts=dict(observations=len(out),splits=dict(Counter(r['source']['split'] for r in out)),
                cached=sum(r['feature_cache'] is not None for r in out),missing=sum(r['feature_cache'] is None for r in out)),
                limitations=['no payload hashes checked yet','historical positive-conditioned development population','no independent development evaluation added','no labels or teacher candidates enter forward'],model_calls=0)

if __name__=='__main__':print(json.dumps(compile_inventory(),separators=(',',':')))
