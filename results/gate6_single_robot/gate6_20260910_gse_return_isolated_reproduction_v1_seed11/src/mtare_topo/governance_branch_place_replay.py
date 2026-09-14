from pathlib import Path

SCHEMA='v3_branch_place_replay_card_v1'
SLUG='gse_branch_place_replay_v1'
PARENT='results/gate6_single_robot/gate6_20260910_gse_live_geometry_execution_v3_seed11'
TRACE=PARENT+'/artifacts/geometry/live_geometry.jsonl'
POLICY=dict(direction_tolerance_deg=15,target_match_m=2,place_radius_m=4,support_observations=3)

def scope(root):
    entries=dict((r.split('  ',1)[1],r.split('  ',1)[0]) for r in (root/PARENT/'artifacts/evidence_sha256.txt').read_text().splitlines())
    return dict(trace=TRACE,sha256=entries[TRACE],worlds=['tunnel'],episodes=1,rows=292,
        geometry_observations=288,split='previously_used_development_episode',
        timing='all saved causal observations, no resampling; actual timestamps preserved',
        policy=POLICY,teacher_reads=0,new_control=False,new_training=False)

def validate_card(card):
    from .governance import ValidationReport
    a=card.get('approval',{});errors=[]
    if card.get('schema_version')!=SCHEMA or card.get('scope')!=scope(Path(__file__).resolve().parents[2]):errors.append('sealed episode scope drift')
    if a.get('status')!='APPROVED' or a.get('authorized_gates')!=[6] or a.get('authorized_operations')!=['audit']:errors.append('exact audit authority required')
    return ValidationReport(not errors,tuple(errors),())
