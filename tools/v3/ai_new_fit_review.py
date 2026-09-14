"""Full-history AI first-pass review of the fixed new12, before reference reveal."""
from _bootstrap import PROJECT_ROOT as ROOT
import argparse,hashlib,json,shutil
from ai_junction_pilot import sha,write
NAME='gse_ai_new_fit_review_v1';CARD=f'configs/v3/gate3/data_cards/{NAME}.json';SPEC=f'configs/v3/gate3/{NAME}.json'
RUN=f'results/gate3_semantics/gate3_20260911_{NAME}_seed0'
SOURCE='results/gate3_semantics/gate3_20260911_gse_new_fit_indexed_review_v1r1_seed0'
PYTHON='/home/zeng-workstation/.local/share/mtare_topo_comm/envs/phase3_torch290_cu129_zarr2187_v1/bin/python'

def main(mode):
    run=ROOT/RUN
    if mode=='freeze':
        c=json.loads((ROOT/'configs/v3/gate3/data_cards/gse_new_fit_indexed_review_v1r1.json').read_text())
        m=json.loads((ROOT/SOURCE/'artifacts/review_manifest.json').read_text())
        c['scope']['indexed_bundles']={SOURCE+'/'+r['bundle']:r['sha256'] for r in m['observations']}
        c['scope']['valid_returns']=[r['valid_points'] if 'valid_points' in r else r['points'] for r in m['observations']]
        c['scope']['review_protocol']='View all five frames, full returns and local10m, preserve observable/uncertain distinctions; freeze first-pass judgments before reference reveal; no new samples or training'
        c['annotation']['input_bundle_contract']='Twelve indexed five-frame point bundles; no construction/reference payload in this run'
        c['annotation']['output_schema']='Per-case full-history observations, structural hypotheses, uncertainty and evidence frame indices; not precision ground truth'
        c['approval']['scope_sha256']=hashlib.sha256(json.dumps(c['scope'],sort_keys=True).encode()).hexdigest();write(ROOT/CARD,c)
        spec=dict(schema_version='v3_run_spec_v1',gate=3,date='20260911',slug=NAME,seed=0,operation='ai_annotation',data_card=CARD,user_authorization=c['approval'],
            command=['env','MPLCONFIGDIR=/tmp/gse-mpl','OPENBLAS_NUM_THREADS=1',PYTHON,'tools/v3/ai_new_fit_review.py','--begin'],
            question='Which observable layouts and uncertain structures occur in the fixed twelve new training observations?',
            method='Actual in-session AI review of all five frames before new reference payload reveal; explicit source-level uncertainty',baseline='Reference interpretation deferred; no model-ranking claim',fallback='Preserve uncertainty and all cases; no selection by review outcome or automatic training',
            acceptance_criteria=['12 cases/all five frames reviewed','Frame-linked actual AI response; no hidden-to-visible label conversion','Frozen judgments before reference reveal; no model inference or training'],
            expected_evidence=['Five-frame views, raw response, twelve case judgments, source hashes and seal'],estimated_cost=dict(compute='CPU views and current AI review',host_ram_gb=4,gpu_vram_gb=0,disk_gb=.1,wall_time_hours=.5),
            input_sha256={CARD:sha(ROOT/CARD),**c['scope']['indexed_bundles'],SOURCE+'/artifacts/review_manifest.json':sha(ROOT/SOURCE/'artifacts/review_manifest.json')},
            source_sha256={p:sha(ROOT/p) for p in ['tools/v3/ai_new_fit_review.py','tools/v3/view_ai_history_context.py','tools/v3/ai_junction_pilot.py']})
        write(ROOT/SPEC,spec);return
    s=json.loads((ROOT/SPEC).read_text());c=json.loads((ROOT/CARD).read_text())
    for p,h in {**s['input_sha256'],**s['source_sha256']}.items():
        if sha(ROOT/p)!=h:raise ValueError('drift '+p)
    state=json.loads((run/'RUN_STATE.json').read_text())['state']
    if mode=='begin':
        if state!='CREATED_NOT_EXECUTED':raise ValueError('fresh annotation required')
        write(run/'RUN_STATE.json',dict(state='RUNNING',activity='in-session annotation, not autonomous background training'),'w')
        for p in s['source_sha256']:shutil.copyfile(ROOT/p,run/'artifacts'/__import__('pathlib').Path(p).name)
        import view_ai_history_context as viewer
        viewer.SOURCE=SOURCE;viewer.OUT=RUN+'/previews';viewer.main();return
    if state!='RUNNING':raise ValueError('active annotation required')
    r=json.loads((run/'artifacts/first_pass.json').read_text())
    assert r['reference_payload_read'] is False and r['training_qualified'] is False and len(r['cases'])==12
    assert (run/'logs/raw_ai_response.md').stat().st_size>0
    for i,row in enumerate(r['cases']):
        assert row['case']==i and row['reviewed_slots']==[0,1,2,3,4] and row['evidence'] and row['complete_background'] is False
    write(run/'metrics/summary.json',dict(status='GATE_MIXED',reviewed_cases=12,reviewed_frames=60,reference_payload_read=False,training_qualified=False,training_steps=0,meaning='AI first-pass observations, not verified structural ground truth'))
    write(run/'RUN_STATE.json',dict(state='COMPLETED'),'w')
    with (run/'artifacts/evidence_sha256.txt').open('x') as f:
        for p in sorted(run.rglob('*')):
            if p.is_file() and p.name!='evidence_sha256.txt':f.write(sha(p)+'  '+str(p.relative_to(ROOT))+'\n')
    print((run/'metrics/summary.json').read_text())

if __name__=='__main__':
    p=argparse.ArgumentParser();g=p.add_mutually_exclusive_group(required=True)
    for m in ['freeze','begin','seal']:g.add_argument('--'+m,action='store_true')
    a=p.parse_args();main(next(m for m in ['freeze','begin','seal'] if getattr(a,m)))
