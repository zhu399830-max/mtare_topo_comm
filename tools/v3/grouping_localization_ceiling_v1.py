"""Reduce existing checkpoint predictions; no inference, training or mask audit."""
from _bootstrap import PROJECT_ROOT as ROOT
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import numpy as np
from mtare_topo.evaluation.grouping_localization_ceiling_v1 import localization_funnel
from mtare_topo.governance_surface_selection import digest

PARENT='results/gate3_semantics/gate3_20260909_gse_grouping_center_supervision_v2_seed0'
PARENT_SEAL='a62d0798e3bfe2ee15d88d2f336ecc854dbd8d5416af4b97feb7175d14f5acba'
PRELIM='docs/figures/gse_graph/primitive_localization_ceiling_20260909.json'


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def reduce_step(run,step,reader=None):
    def read(p):return reader(p) if reader else p.read_bytes()
    scores=json.loads(read(run/'metrics'/f'evaluation_{step:04d}.json'))
    observations=[];counts=Counter()
    import io
    for row in scores['observations']:
        key=digest(row['source'])
        with np.load(io.BytesIO(read(run/'artifacts'/f'input_{key}.npz'))) as data:
            targets=data['target_positions_m'].copy()
        with np.load(io.BytesIO(read(run/'artifacts'/f'prediction_{step:04d}_{key}.npz'))) as data:
            f=localization_funnel(data['position_m'],data['presence_logits'],targets,row['scores']['1.0'])
        observations.append(dict(source=row['source'],funnel=f,actual=row['scores']['1.0']))
        counts.update({k:f[k] for k in ('reference_count','candidate_count','selected_count',
            'raw_one_to_one_tp','selected_one_to_one_tp','actual_tp','localization_deficit',
            'presence_deficit','coverage_or_matching_deficit')})
    result=dict(counts)
    for key in ('raw_one_to_one_tp','selected_one_to_one_tp','actual_tp'):
        result[key.replace('_tp','_recall')]=counts[key]/counts['reference_count'] if counts['reference_count'] else None
    return dict(step=step,summary=result,actual_summary=scores['summary'],observations=observations)


def reduce_parent():
    run=ROOT/PARENT;seal=run/'artifacts/evidence_sha256.txt'
    if sha(seal)!=PARENT_SEAL:raise ValueError('parent seal mismatch')
    index={p:h for h,p in (line.split('  ',1) for line in seal.read_text().splitlines())}
    reads={}
    def reader(p):
        data=p.read_bytes();key=str(p.relative_to(ROOT));h=hashlib.sha256(data).hexdigest()
        if index.get(key)!=h:raise ValueError('sealed prediction input drift')
        reads[key]=h;return data
    result=dict(kind='saved_checkpoint_prediction_reduction',parent=PARENT,parent_seal_sha256=PARENT_SEAL,
        model_inferences=0,optimizer_steps=0,mask_or_gradient_reaudits=0,
        steps=[reduce_step(run,s,reader) for s in range(0,1001,100)],source_reads_sha256=reads)
    path=ROOT/PRELIM
    with path.open('x') as out:json.dump(result,out,ensure_ascii=False,indent=2,allow_nan=False)
    print(json.dumps(dict(path=PRELIM,curve=[s['summary']|dict(step=s['step']) for s in result['steps']],
        final_misses=[dict(source=o['source'],reference=t) for o in result['steps'][-1]['observations'] for t in o['funnel']['references'] if not t['matched_actual']]),ensure_ascii=False))


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('--parent',action='store_true');a=p.parse_args()
    if a.parent:reduce_parent()
    else:p.error('--parent required')
