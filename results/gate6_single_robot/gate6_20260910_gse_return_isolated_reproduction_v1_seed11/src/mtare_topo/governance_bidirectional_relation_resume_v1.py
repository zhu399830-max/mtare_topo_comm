"""Single continuation of the sealed empty-selection scoring failure."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from mtare_topo import governance_bidirectional_relation_train_v1 as original

SCHEMA='v3_bidirectional_relation_resume_card_v1'
SLUG='gse_bidirectional_relation_resume_v1'
PARENT='results/gate3_semantics/gate3_20260909_gse_bidirectional_relation_train_v1_seed0'
SEAL_SHA='279a43842466fb4fdac274b54ef2e963e1d2aaecce5936fd50326b5fb4726bcd'
CHECKPOINT_SHA='9d9d82ecc82cbda5a44194235b6bda8d51f68d2089b551355be30dcfe80af671'
POLICY=deepcopy(original.POLICY)
POLICY.update(continuation_only=True,new_scheduled_batches={'c0':0,'c1':2000},
    restored_c0_updates=2000,scoring_policy='original_with_verified_empty_state_dtype_restoration')


def lineage(root):
    root=Path(root);parent=root/PARENT;seal=parent/'artifacts/evidence_sha256.txt'
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    if sha(seal)!=SEAL_SHA:raise ValueError('parent seal drift')
    entries=dict((p,h) for h,p in (line.split('  ',1) for line in seal.read_text().splitlines()))
    if len(entries)!=881:raise ValueError('parent evidence count drift')
    names=['checkpoints/c0_final.pt','checkpoints/initial.pt','config/schedule.json',
           'config/population.json','config/run_spec.json','config/data_card.json',
           'metrics/summary.json','RUN_STATE.json','logs/c0_updates.jsonl']
    pinned={PARENT+'/artifacts/evidence_sha256.txt':SEAL_SHA}
    for name in names:
        p=PARENT+'/'+name
        if p not in entries or sha(root/p)!=entries[p]:raise ValueError('parent artifact drift: '+name)
        pinned[p]=entries[p]
    if pinned[PARENT+'/checkpoints/c0_final.pt']!=CHECKPOINT_SHA:raise ValueError('C0 final drift')
    summary=json.loads((parent/'metrics/summary.json').read_text())
    if (summary['status']!='FAILED' or summary['phase']!='c0_final'
            or summary['updates']!={'c0':2000,'c1':0}):raise ValueError('not the fixed continuation boundary')
    updates=[json.loads(s) for s in (parent/'logs/c0_updates.jsonl').read_text().splitlines()]
    if len(updates)!=2000 or sum(r['optimizer_step'] for r in updates)!=2000:raise ValueError('C0 update history drift')
    old=json.loads((parent/'config/run_spec.json').read_text())
    # Training and model functions must still be the originals, not a new trial.
    for p,h in old['source_sha256'].items():
        if p.startswith('src/mtare_topo/representation/') and sha(root/p)!=h:
            raise ValueError('training/model implementation changed: '+p)
    return dict(parent_run=PARENT,parent_seal_sha256=SEAL_SHA,pinned=pinned,
                checkpoint_sha256=CHECKPOINT_SHA,c0_new_updates=0,c1_scheduled_batches=2000)


def validate_card(card):
    from mtare_topo.governance import ValidationReport
    if not isinstance(card,dict) or set(card)!= {'schema_version','card_id','operation','scope','scope_sha256','policy','approval','lineage'}:
        return ValidationReport(False,('closed continuation card required',))
    errors=[]
    if (card['schema_version'],card['card_id'])!=(SCHEMA,SLUG):errors.append('continuation identity mismatch')
    if card['policy']!=POLICY:errors.append('continuation policy drift')
    try:
        if card['lineage']!=lineage(Path(__file__).resolve().parents[2]):errors.append('lineage drift')
    except Exception as e:errors.append(str(e))
    translated=deepcopy(card);translated.pop('lineage')
    translated.update(schema_version=original.SCHEMA,card_id=original.SLUG,policy=original.POLICY)
    errors.extend(original.validate_card(translated).errors)
    return ValidationReport(not errors,tuple(errors))
