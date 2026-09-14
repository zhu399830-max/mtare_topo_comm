"""Shared frozen fit features; existing full-task archives decoded once each."""
from _bootstrap import PROJECT_ROOT as ROOT
import argparse, hashlib, json
from itertools import groupby
from ai_junction_pilot import sha, write
from export_new12_features import PYTHON
from mtare_topo.governance_conditional_fit_features import BINDING, LIMITS, validate_card

NAME='gse_conditional_fit_features_v1'
CARD=f'configs/v3/gate3/data_cards/{NAME}.json'
SPEC=f'configs/v3/gate3/{NAME}.json'
RUN=f'results/gate3_semantics/gate3_20260911_{NAME}_seed0'
ENCODER='configs/v3/gate3/data_cards/gse_new12_features_v1.json'


def task_students(root, entries):
    from mtare_topo.data.gse_membership_fit_reader import load_student_task_batch
    seen=set()
    for task, group in groupby(entries, key=lambda e:e['task']):
        if task in seen:raise ValueError('task must be contiguous')
        seen.add(task)
        yield from load_student_task_batch(root,list(group))


def freeze():
    b=json.loads((ROOT/BINDING).read_text())
    ck=json.loads((ROOT/ENCODER).read_text())['scope']['checkpoint']
    entries=[dict(e,parent=e['parent_id']) for e in b['entries']]
    s=dict(entries=entries,parents=60,frames=14400,checkpoint=ck,valid_returns_per_observation=None,
           training_steps=0,teacher_payload_reads=0,limits=LIMITS,
           execution_context='existing CUDA sidecar; original causal four-field student inputs',
           sampling='Original fixed seed20260906:16 physical edges per parent,one direction and window,three variants; no score selection',
           spacing='decision route arc and five ordered frame indices bound per entry; acquisition time/history spacing unknown',
           split_audit='C01-C06 fit only; old encoder trained here; no independent performance claim or C07-C10 payload',
           supervision='None; observation-only feature cache, not new conditional geometry labels')
    a=dict(status='APPROVED',approved_by='user-standing-development-authorization',approved_at='2026-09-11',
           authorized_gates=[3],authorized_operations=['data_export'],
           scope_sha256=hashlib.sha256(json.dumps(s,sort_keys=True).encode()).hexdigest(),
           scope='One common feature cache for fixed2880fit observations; zero teacher or optimizer',
           confirmation_reference='Active user-approved goal; PLAN authorizes exact common-feature export preparation; standing execution authorization')
    card=dict(schema_version='gse_conditional_fit_features_card_v1',scope=s,approval=a)
    if not validate_card(card).passed:raise ValueError('invalid fit scope')
    write(ROOT/CARD,card)
    pins={BINDING:sha(ROOT/BINDING),ENCODER:sha(ROOT/ENCODER),CARD:sha(ROOT/CARD),ck['path']:ck['sha256']}
    pins.update({e['student_path']:e['student_sha256'] for e in entries})
    sources={str(p.relative_to(ROOT)):sha(p) for folder in ('src/mtare_topo','tools/v3') for p in (ROOT/folder).rglob('*.py')}
    write(ROOT/SPEC,dict(schema_version='v3_run_spec_v1',gate=3,date='20260911',slug=NAME,seed=0,operation='data_export',data_card=CARD,user_authorization=a,
        command=['env','OPENBLAS_NUM_THREADS=1','OMP_NUM_THREADS=1','PYTHONPATH=src','CUBLAS_WORKSPACE_CONFIG=:4096:8',PYTHON,'tools/v3/export_conditional_fit_features.py','--execute'],
        question='Can all frozen fit observations share the unchanged observation/patch frontend without repeated source export?',
        method='Same encoder and deterministic0.5m/10m/k8 patches; each full task decoded once; retain all observed patches',
        baseline='One common cache for A/B/C; no new method scores',fallback='Seal failure, no sample dropping, retry or teacher substitution',
        acceptance_criteria=['2880 exact observations','Encoder unchanged','No teacher or optimizer','Every observed patch retained','Input hashes and frame identities match'],
        expected_evidence=['2880 compact NPZ, all-window counts, runtime environment, source snapshot, logs and SHA seal'],
        estimated_cost=dict(compute='2880 frozen encoder forwards; CPU patch pooling',host_ram_gb=4,gpu_vram_gb=28,disk_gb=20,wall_time_hours=1),
        input_sha256=pins,source_sha256=sources))
    print(json.dumps(dict(spec=SPEC,observations=2880,training_steps=0)))


if __name__=='__main__':
    p=argparse.ArgumentParser();g=p.add_mutually_exclusive_group(required=True)
    g.add_argument('--freeze',action='store_true');g.add_argument('--execute',action='store_true');a=p.parse_args()
    if a.freeze:freeze()
    else:
        from export_new12_features import execute
        raise SystemExit(execute(run_path=RUN,spec_path=SPEC,card_path=CARD,validator=validate_card,student_iterator=task_students))
