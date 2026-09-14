"""Bind the explicitly user-authorized three-case AI annotation scope."""
from _bootstrap import PROJECT_ROOT as ROOT
import hashlib,json


def main():
    original=ROOT/'configs/v3/gate3/data_cards/gse_acquisition_junction_review_v1.json'
    old=json.loads(original.read_text());entries=[]
    for e in old['entries']:
        entries.append(dict(task=e['source']['task'],parent=e['source']['parent'],split='fit',
            frame_rows=e['frame_rows'],traversal_index=e['source']['traversal_index'],
            student_path=e['student_path'],student_sha256=e['student_sha256']))
    image='docs/figures/gse_supervision_acquisition_pilot_v1/junction_observation_context.png'
    scope=dict(entries=entries,observations=3,independent_parents=3,frames=15,raw_ray_slots=172800,
        valid_returns=[51437,57147,54310],effective_annotation_windows=3,
        spacing=old['spacing'],selection=old['selection'],
        source_review_card_sha256=hashlib.sha256(original.read_bytes()).hexdigest(),
        images={image:hashlib.sha256((ROOT/image).read_bytes()).hexdigest()},
        blind=False,prior_teacher_exposure=True,human_gold_count=0,training_authorized=False,unknown_is_background=False,
        teacher_source='AI interpretation of saved causal observations; program reference only for later discrepancy review, not copied as AI answer',
        leakage='All three parents are fit-only C01/C02/C03; no C08-C10 reads or model prediction selection; non-blind and teacher-stratified selection disclosed',
        quality_checks=['Trace every proposal to view and visible surface evidence','Separate approximate geometry from exact labels','Explicit unknown relations and unreviewed space','No all-unmatched-as-background','Cross-view geometric consistency and reference disagreement logged, not silently corrected'])
    approval=dict(status='APPROVED',approved_by='user',approved_at='2026-09-11',authorized_gates=[3],authorized_operations=['ai_annotation'],
        scope='Non-blind AI-assisted proposals on the existing exact three junction cases; no training',
        confirmation_reference='User: 如果你的答案有问题我们本来智能体就足够聪明了你可以标注的呀; followed by continue active goal',
        scope_sha256=hashlib.sha256(json.dumps(scope,sort_keys=True).encode()).hexdigest())
    card=dict(schema_version='gse_ai_three_case_pilot_v1',scope=scope,approval=approval,
        annotation=dict(labeler_name='current Codex assistant',labeler_version='current session; exact model revision not exposed',
        prompt_reference='User-authorized geometry-to-structure annotation; evidence and unknowns retained',input_bundle_contract='scope entries and SHA-bound observation image',
        output_schema='AI proposals with observable geometry, relation, evidence, uncertainty; not full detection gold',
        conflict_policy='Keep AI proposal and reference separately; conflict becomes unresolved, not automatic overwrite',abstain_policy='Unknown remains unknown; no invented exact center',
        planned_ai_sample_count=3,planned_human_gold_count=0,strict_test_excluded=True,raw_responses_preserved=True))
    from mtare_topo.governance import validate_data_card,validate_annotation_plan
    for report in (validate_data_card(card),validate_annotation_plan(card)):
        if not report.passed:raise ValueError(report.errors)
    path=ROOT/'configs/v3/gate3/data_cards/gse_ai_three_case_pilot_v1.json'
    with path.open('x') as f:json.dump(card,f,ensure_ascii=False,indent=2)
    print(path)


if __name__=='__main__':main()
