"""Freeze, begin and seal an in-session AI annotation (not model inference)."""
from _bootstrap import PROJECT_ROOT as ROOT
import argparse,hashlib,json,shutil,sys
CARD='configs/v3/gate3/data_cards/gse_ai_three_case_pilot_v1.json'
SPEC='configs/v3/gate3/gse_ai_three_case_pilot_v1.json'
RUN='results/gate3_semantics/gate3_20260911_gse_ai_three_case_pilot_v1_seed0'
REVISION=False
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,x,mode='x'):
    with p.open(mode) as f:json.dump(x,f,ensure_ascii=False,indent=2)
def main(mode):
    if REVISION and mode=='freeze':
        card=json.loads((ROOT/'configs/v3/gate3/data_cards/gse_ai_three_case_pilot_v1.json').read_text())
        folder='docs/figures/gse_supervision_acquisition_pilot_v1/'
        evidence=[folder+n for n in ['ai_surface_measurements.json','ai_ray_evidence.json','ai_boundary_components.json']]
        evidence.append('results/gate3_semantics/gate3_20260911_gse_ai_three_case_pilot_v1_seed0/artifacts/ai_proposals.json')
        card['scope']['measurement_evidence']={p:sha(ROOT/p) for p in evidence}
        p=folder+'ai_ray_evidence.png';card['scope']['images'][p]=sha(ROOT/p)
        card['approval']['scope_sha256']=hashlib.sha256(json.dumps(card['scope'],sort_keys=True).encode()).hexdigest()
        card['approval']['scope']='Same-three-case non-blind AI evidence revision; old labels unchanged; no training'
        write(ROOT/CARD,card)
    card=json.loads((ROOT/CARD).read_text());run=ROOT/RUN
    if mode=='freeze':
        inputs=dict(card['scope']['images']);inputs[CARD]=sha(ROOT/CARD)
        inputs.update(card['scope'].get('measurement_evidence',{}))
        for e in card['scope']['entries']:inputs[e['student_path']]=e['student_sha256']
        spec=dict(schema_version='v3_run_spec_v1',gate=3,date='20260911',slug='gse_ai_three_case_pilot_v2' if REVISION else 'gse_ai_three_case_pilot_v1',seed=0,
            operation='ai_annotation',data_card=CARD,user_authorization=card['approval'],
            command=['python3','tools/v3/ai_junction_pilot.py','--begin']+(['--evidence-revision'] if REVISION else []),
            question='Which visible channel geometry and opening hypotheses can AI annotate with traceable evidence?',
            method='In-session non-blind AI visual annotation; source images and raw response preserved; approximate and unknown fields explicit',
            baseline='Previously saved partial reference, for later discrepancy comparison only; not copied into proposals',
            fallback='Seal partial or uncertain proposals; no invented exact geometry or automatic training',
            acceptance_criteria=['All three observations addressed','Proposal linked to view and evidence','Unknown not negative; no full background claim','Non-blind AI authorship explicit'],
            expected_evidence=['Raw AI response, structured proposals, image, config, source and SHA seal'],
            estimated_cost=dict(compute='Current assistant visual reasoning, no GPU training',disk_gb=.02,wall_time_hours=.25),
            input_sha256=inputs,source_sha256={'tools/v3/ai_junction_pilot.py':sha(ROOT/'tools/v3/ai_junction_pilot.py')})
        write(ROOT/SPEC,spec);return
    spec=json.loads((ROOT/SPEC).read_text())
    for p,h in {**spec['input_sha256'],**spec['source_sha256']}.items():
        if sha(ROOT/p)!=h:raise ValueError('drift '+p)
    state=json.loads((run/'RUN_STATE.json').read_text())['state']
    if mode=='begin':
        if state!='CREATED_NOT_EXECUTED':raise ValueError('fresh annotation run required')
        write(run/'RUN_STATE.json',dict(state='RUNNING',activity='in-session AI annotation'),'w')
        for p in card['scope']['images']:shutil.copyfile(ROOT/p,run/'previews'/__import__('pathlib').Path(p).name)
        shutil.copyfile(ROOT/'tools/v3/ai_junction_pilot.py',run/'artifacts/runner_source.py')
        print('Ready for actual assistant raw response and structured proposals; no automatic label generation');return
    if state!='RUNNING':raise ValueError('annotation not active')
    labels=json.loads((run/'artifacts/ai_proposals.json').read_text())
    assert labels['blind'] is False and labels['human_gold'] is False and labels['training_qualified'] is False
    assert [x['task'] for x in labels['cases']]==[x['task'] for x in card['scope']['entries']]
    assert (run/'logs/raw_ai_response.md').stat().st_size>0
    for c in labels['cases']:
        assert c['complete_background'] is False and c['exact_center_xyz_m'] is None
        assert c['proposals'] and all(p['evidence'] and p['status']=='hypothesis' for p in c['proposals'])
    write(run/'metrics/summary.json',dict(status='GATE_MIXED',annotated_cases=3,
        proposal_count=sum(len(c['proposals']) for c in labels['cases']),training_qualified=False,
        meaning='AI proposals recorded, geometry quality not independently established',training_steps=0))
    write(run/'RUN_STATE.json',dict(state='COMPLETED'),'w')
    with (run/'artifacts/evidence_sha256.txt').open('x') as f:
        for p in sorted(run.rglob('*')):
            if p.is_file() and p.name!='evidence_sha256.txt':f.write(sha(p)+'  '+str(p.relative_to(ROOT))+'\n')
    print((run/'metrics/summary.json').read_text())
if __name__=='__main__':
    p=argparse.ArgumentParser();g=p.add_mutually_exclusive_group(required=True)
    for mode in ['freeze','begin','seal']:g.add_argument('--'+mode,action='store_true')
    p.add_argument('--evidence-revision',action='store_true');a=p.parse_args()
    if a.evidence_revision:
        REVISION=True
        CARD='configs/v3/gate3/data_cards/gse_ai_three_case_pilot_v2.json'
        SPEC='configs/v3/gate3/gse_ai_three_case_pilot_v2.json'
        RUN='results/gate3_semantics/gate3_20260911_gse_ai_three_case_pilot_v2_seed0'
    main(next(k for k in ['freeze','begin','seal'] if getattr(a,k)))
