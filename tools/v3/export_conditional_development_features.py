"""Same observation encoder/patch implementation for all frozen development rows."""
from _bootstrap import PROJECT_ROOT as ROOT
import argparse,hashlib,json
from ai_junction_pilot import sha,write
from export_new12_features import PYTHON
NAME='gse_conditional_development_features_v1'
CARD=f'configs/v3/gate3/data_cards/{NAME}.json';SPEC=f'configs/v3/gate3/{NAME}.json'
RUN=f'results/gate3_semantics/gate3_20260911_{NAME}_seed0'
INPUT='results/gate3_semantics/gate3_20260911_gse_conditional_development_inputs_v1_seed20260906'
MANIFEST='configs/v3/gate3/gse_conditional_development_manifest_v1.json'


def freeze():
    original=json.loads((ROOT/MANIFEST).read_text())
    rows=json.loads((ROOT/INPUT/'artifacts/manifest.json').read_text())
    lookup={(r['source']['task'],r['source']['source_global_sequence_index']):r for r in rows}
    seals={p:h for h,p in (l.split('  ',1) for l in (ROOT/INPUT/'artifacts/evidence_sha256.txt').read_text().splitlines())}
    entries=[];pins={MANIFEST:sha(ROOT/MANIFEST)}
    for e in original['entries']:
        r=lookup[(e['task'],e['source_sequence_id'])]
        assert r['source']['frame_rows']==e['frame_rows']
        path=INPUT+'/'+r['student_path']
        assert sha(ROOT/path)==r['student_sha256']==seals[path]
        entries.append(dict(e,parent=e['parent_id'],student_path=path,student_sha256=r['student_sha256']))
        pins[path]=r['student_sha256']
    ck=original['encoder_checkpoint'];pins[ck['path']]=ck['sha256']
    s=dict(entries=entries,parents=5,frames=1200,valid_returns_per_observation=None,
           checkpoint=ck,training_steps=0,teacher_payload_reads=0,
           split_audit=original['split_audit'],execution_context='existing frozen CUDA sidecar; no driver changes',
           limits=dict(wall_seconds=900,host_bytes=4*1024**3,gpu_bytes=28*1024**3,output_bytes=4*1024**3))
    a=dict(status='APPROVED',approved_by='user-standing-development-authorization',approved_at='2026-09-11',authorized_gates=[3],authorized_operations=['data_export'],
           scope_sha256=hashlib.sha256(json.dumps(s,sort_keys=True).encode()).hexdigest(),
           scope='One common frozen encoder/patch cache for exactly240development observations; no labels/updates',
           confirmation_reference='Active approved goal and PLAN: parent-held-out evaluation of frozen ABC, no refit')
    from mtare_topo.governance_conditional_development import validate_feature_card
    card=dict(schema_version='gse_conditional_development_features_card_v1',scope=s,approval=a)
    assert validate_feature_card(card).passed
    write(ROOT/CARD,card);pins[CARD]=sha(ROOT/CARD)
    sources={str(p.relative_to(ROOT)):sha(p) for folder in ('src/mtare_topo','tools/v3') for p in (ROOT/folder).rglob('*.py')}
    write(ROOT/SPEC,dict(schema_version='v3_run_spec_v1',gate=3,date='20260911',slug=NAME,seed=0,operation='data_export',data_card=CARD,user_authorization=a,
        command=['env','OPENBLAS_NUM_THREADS=1','OMP_NUM_THREADS=1','PYTHONPATH=src',PYTHON,'tools/v3/export_conditional_development_features.py','--execute'],
        question='Can the frozen development population use exactly the shared training-time observation/patch frontend?',
        method='Reuse same frozen seed0 epoch2 encoder, deterministic0.5m/10m patches, compact pooling; no teacher candidate filter',
        baseline='One cache shared by A/B/C; feature export not a model comparison',fallback='Seal failure; no dropping patches or replacing observations',
        acceptance_criteria=['All240frozen identities','Encoder state unchanged','Zero teacher payload and optimizer','All observed patches retained; report measured counts'],
        expected_evidence=['240NPZ caches, per-window log/counts, environment, immutable source snapshot and seal'],
        estimated_cost=dict(compute='240GPUencoder forwards and CPUpatch pooling',host_ram_gb=4,gpu_vram_gb=28,disk_gb=4,wall_time_hours=.25),
        input_sha256=pins,source_sha256=sources))
    print(json.dumps(dict(spec=SPEC,observations=len(entries),training_steps=0)))


if __name__=='__main__':
    p=argparse.ArgumentParser();g=p.add_mutually_exclusive_group(required=True)
    g.add_argument('--freeze',action='store_true');g.add_argument('--execute',action='store_true');a=p.parse_args()
    if a.freeze:freeze()
    else:
        from export_new12_features import execute
        from mtare_topo.governance_conditional_development import validate_feature_card
        raise SystemExit(execute(run_path=RUN,spec_path=SPEC,card_path=CARD,validator=validate_feature_card))
