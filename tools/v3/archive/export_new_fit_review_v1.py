"""Reuse indexed exporter for the explicitly approved fixed new12 population."""
from _bootstrap import PROJECT_ROOT as ROOT
import argparse,hashlib,json
import export_ai_indexed_review as exporter
NAME='gse_new_fit_indexed_review_v1'
CARD=f'configs/v3/gate3/data_cards/{NAME}.json';SPEC=f'configs/v3/gate3/{NAME}.json'
RUN=f'results/gate3_semantics/gate3_20260911_{NAME}_seed0'

def freeze():
    nompath='configs/v3/gate3/gse_new_fit_review_nomination_v1.json'
    nom=json.loads((ROOT/nompath).read_text());scope=nom['scope']
    scope.update(training_authorized=False,human_gold_count=0,blind=False,valid_returns=[None]*12)
    old=json.loads((ROOT/'configs/v3/gate3/data_cards/gse_ai_three_case_pilot_v1.json').read_text())
    annotation=old['annotation'];annotation['planned_ai_sample_count']=12;annotation['input_bundle_contract']='Twelve pinned fit-only five-frame observations; input preparation only in this execution'
    approval=dict(status='APPROVED',approved_by='user',approved_at='2026-09-11',authorized_operations=['ai_annotation'],authorized_gates=[3],confirmation_reference=nom['authorization']['confirmation_reference'],scope_sha256=hashlib.sha256(json.dumps(scope,sort_keys=True).encode()).hexdigest(),scope='New fixed12 fit-only annotation population; indexed input preparation, no training')
    card=dict(schema_version='gse_ai_fixed_fit_review_v1',scope=scope,annotation=annotation,approval=approval)
    from mtare_topo.governance import validate_data_card,validate_annotation_plan
    for r in (validate_data_card(card),validate_annotation_plan(card)):
        if not r.passed:raise ValueError(r.errors)
    exporter.write(ROOT/CARD,card)
    original=json.loads((ROOT/'configs/v3/gate3/gse_ai_indexed_review_v1r1.json').read_text())
    original.update(slug=NAME,data_card=CARD,user_authorization=approval,question='Prepare exact twelve newly nominated fit observations for AI annotation, excluding old parents',acceptance_criteria=['12 independent fit parents/60 original frames; zero model-score selection','Raw validity and original indices preserved; actual valid point count reported','No teacher payload, new labels or training'])
    original['command'][-2]='tools/v3/export_new_fit_review.py'
    original['estimated_cost'].update(disk_gb=.1,wall_time_hours=.05)
    original['input_sha256']={CARD:exporter.sha(ROOT/CARD),nompath:exporter.sha(ROOT/nompath),**{e['student_path']:e['student_sha256'] for e in scope['entries']}}
    files=set(original['source_sha256'])|{'tools/v3/export_new_fit_review.py','tools/v3/nominate_new_fit_review.py'}
    original['source_sha256']={p:exporter.sha(ROOT/p) for p in files}
    exporter.write(ROOT/SPEC,original);print(SPEC)

if __name__=='__main__':
    p=argparse.ArgumentParser();g=p.add_mutually_exclusive_group(required=True);g.add_argument('--freeze',action='store_true');g.add_argument('--execute',action='store_true');a=p.parse_args()
    if a.freeze:freeze()
    else:
        exporter.NAME=NAME;exporter.CARD=CARD;exporter.SPEC=SPEC;exporter.RUN=RUN
        raise SystemExit(exporter.execute())
