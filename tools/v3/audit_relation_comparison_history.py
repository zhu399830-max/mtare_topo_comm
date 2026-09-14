"""Read only three explicit development summaries and their frozen cards.

This is a scope audit, not a new experiment or full run-seal verification.
"""
import hashlib,json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
RUNS=(
 'gate3_20260908_gse_synthetic_fit_v2_seed0',
 'gate3_20260908_gse_synthetic_window_fit_v1_seed0',
 'gate3_20260908_gse_block_representation_fit_v2_seed0')


def main():
    records=[]
    for name in RUNS:
        folder=ROOT/'results/gate3_semantics'/name
        path=folder/'metrics/summary.json';payload=path.read_bytes();summary=json.loads(payload)
        passed=summary.get('fit_pass_by_method',summary.get('fit_pass'))
        assert isinstance(passed,dict) and len(passed)==3
        counts=summary.get('counts',{})
        steps=summary.get('optimizer_steps',{k:v['optimizer_steps'] for k,v in counts.items()})
        records.append(dict(run=name,summary_sha256=hashlib.sha256(payload).hexdigest(),
            status=summary['status'],fit_pass=passed,optimizer_steps=steps,
            scientific_gate_pass=summary['scientific_gate_pass']))
    # Exact frozen scope, not a new compile_scope call or sensor payload read.
    card=ROOT/'results/gate3_semantics'/RUNS[0]/'config/data_card.json'
    payload=card.read_bytes();data=json.loads(payload)
    scopes=[]
    def visit(value):
        if isinstance(value,dict):
            if 'unknown_fields' in value:scopes.append(value['unknown_fields'])
            for child in value.values():visit(child)
        elif isinstance(value,list):
            for child in value:visit(child)
    visit(data)
    assert scopes and any('membership' in x for x in scopes),'frozen scope unknown fields not found'
    print(json.dumps(dict(status='HISTORY_SCOPE_AUDIT_COMPLETE',runs=records,
        frozen_card_sha256=hashlib.sha256(payload).hexdigest(),unknown_fields=scopes,
        new_training_authorized_by_this_check=False,full_seals_reverified=False,
        conclusion='no successful fit among these runs; no direct membership supervision in first frozen scope',
        optimizer_steps=0,new_research_payloads=0),ensure_ascii=False))


if __name__=='__main__':main()
