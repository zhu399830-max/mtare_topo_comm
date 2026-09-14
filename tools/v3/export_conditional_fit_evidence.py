"""Transport only missing source/pose evidence, retaining old student bindings."""
from _bootstrap import PROJECT_ROOT as ROOT
import argparse,hashlib,json,subprocess
from ai_junction_pilot import sha,write
from export_conditional_development_inputs import PYTHON
from mtare_topo.governance_conditional_fit_evidence import SOURCE,LIMITS,validate_card
NAME='gse_conditional_fit_evidence_v1'
CARD=f'configs/v3/gate3/data_cards/{NAME}.json';SPEC=f'configs/v3/gate3/{NAME}.json'
RUN=f'results/gate3_semantics/gate3_20260911_{NAME}_seed20260906'

def freeze():
    from mtare_topo.governance_surface_input import expected_tasks
    s=json.loads((ROOT/SOURCE).read_text());entries=s.pop('entries')
    s['observations']=[dict(e,parent=e['parent_id'],source_global_sequence_index=e['source_sequence_id']) for e in entries]
    tasks=expected_tasks();s['task_prefixes']={t:tasks[t] for t in s['task_files']}
    s['counts']=dict(fit=dict(parents=60,observations=2880,tasks=180,unique_variant_frames=14400))
    s['limits']=LIMITS
    a=dict(status='APPROVED',approved_by='user-standing-development-authorization',approved_at='2026-09-11',authorized_gates=[3],authorized_operations=['data_export'],
        scope_sha256=hashlib.sha256(json.dumps(s,sort_keys=True).encode()).hexdigest(),
        scope='Fixed2880fit observations; missing source/pose transport only; old student cache reused',
        confirmation_reference='Active approved goal and PLAN missing-evidence export; standing execution authority, no training')
    card=dict(schema_version='gse_conditional_fit_evidence_card_v1',scope=s,approval=a)
    assert validate_card(card).passed;write(ROOT/CARD,card)
    files=dict(s['input_sha256']);files[SOURCE]=sha(ROOT/SOURCE);files[CARD]=sha(ROOT/CARD)
    files.update({e['student_path']:e['student_sha256'] for e in entries})
    sources={str(p.relative_to(ROOT)):sha(p) for folder in ('src/mtare_topo','tools/v3') for p in (ROOT/folder).rglob('*.py')}
    freeze=subprocess.check_output([PYTHON,'-m','pip','freeze','--all'],text=True)
    write(ROOT/SPEC,dict(schema_version='v3_run_spec_v1',gate=3,date='20260911',slug=NAME,seed=20260906,operation='data_export',data_card=CARD,user_authorization=a,
        command=['env','OPENBLAS_NUM_THREADS=1','OMP_NUM_THREADS=1',PYTHON,'tools/v3/export_conditional_fit_evidence.py','--execute'],
        question='Can original source codes and poses align exactly with all cached fit student observations?',
        method='Original chunk decoder and verify_alignment; no new teacher or student export',baseline='Original sealed arrays and cached four-field student inputs',
        fallback='Seal failure; no replacement samples, pose tolerance widening or retry',
        estimated_cost=dict(compute='CPU180tasks;1.305GB padded chunk decode',host_ram_gb=4,gpu_vram_gb=0,disk_gb=2,wall_time_hours=.25),
        acceptance_criteria=['2880 exact identities','Original float64 pose and code validity retained','No student re-export','Zero labels or optimizer'],
        expected_evidence=['2880source NPZ,360documents,exact cache bindings,environment,logs,metrics and seal'],
        input_sha256=files,source_sha256=sources,sidecar_freeze=freeze,sidecar_freeze_sha256=hashlib.sha256(freeze.encode()).hexdigest()))
    print(SPEC)

if __name__=='__main__':
    p=argparse.ArgumentParser();g=p.add_mutually_exclusive_group(required=True);g.add_argument('--freeze',action='store_true');g.add_argument('--execute',action='store_true');a=p.parse_args()
    if a.freeze:freeze()
    else:
        from export_local_pair_pilot import execute
        raise SystemExit(execute(run_path=RUN,spec_path=SPEC,card_path=CARD,validator=validate_card,reuse_student_cache=True))
