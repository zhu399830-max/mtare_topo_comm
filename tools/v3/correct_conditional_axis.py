"""Once-only C correction; original initialization/schedule, changed axis loss."""
from _bootstrap import PROJECT_ROOT as ROOT
import argparse,copy,hashlib,json
from ai_junction_pilot import sha,write
import train_conditional_geometry as runner
NAME='gse_new12_axis_logit_correction_v1'
CARD=f'configs/v3/gate3/data_cards/{NAME}.json';SPEC=f'configs/v3/gate3/{NAME}.json'
RUN=f'results/gate3_semantics/gate3_20260911_{NAME}_seed0'
OLD=runner.RUN
LOSS='per observation: unordered component pair means, same/cross category equal means, axis soft-target logit loss + height L1/10m; four observations averaged'


def freeze():
    old=json.loads((ROOT/OLD/'config/run_spec.json').read_text())
    card=json.loads((ROOT/OLD/'config/data_card.json').read_text());s=copy.deepcopy(card['scope'])
    s.update(variants=['C'],axis_objective='conditional_axis_soft_target_logit_v1',loss=LOSS,
        original_initial=f'{OLD}/artifacts/initial.pt',original_schedule=f'{OLD}/artifacts/schedule.npy',
        estimand_change='conditional mean rather than conditional median; geometry reference is not connectivity probability')
    a=dict(status='APPROVED',approved_by='user-standing-evidence-based-correction-authorization',approved_at='2026-09-11',
        authorized_gates=[3],authorized_operations=['training'],scope_sha256=hashlib.sha256(json.dumps(s,sort_keys=True).encode()).hexdigest(),
        scope='One C2000 original-init correction; same fixed12 labels, weights, order and original evaluation',
        confirmation_reference='User confirmed conditional supervision and continuous evidence-based execution; PLAN fixes one numerical axis correction after pure tensor validation')
    pins={p:h for p,h in old['input_sha256'].items() if p!=old['data_card']}
    seal=ROOT/OLD/'artifacts/evidence_sha256.txt';known={p:h for h,p in (l.split('  ',1) for l in seal.read_text().splitlines())}
    pins[str(seal.relative_to(ROOT))]=sha(seal)
    for p in (s['original_initial'],s['original_schedule'],f'{OLD}/metrics/C_step0.json',f'{OLD}/metrics/C_step2000.json'):
        assert sha(ROOT/p)==known[p];pins[p]=known[p]
    for p,h in pins.items():assert sha(ROOT/p)==h,p
    write(ROOT/CARD,dict(schema_version='gse_conditional_axis_correction_card_v1',scope=s,approval=a));pins[CARD]=sha(ROOT/CARD)
    sources={str(p.relative_to(ROOT)):sha(p) for folder in ('src/mtare_topo','tools/v3') for p in (ROOT/folder).rglob('*.py')}
    write(ROOT/SPEC,dict(schema_version='v3_run_spec_v1',gate=3,date='20260911',slug=NAME,seed=0,operation='training',data_card=CARD,user_authorization=a,
        command=['env','OPENBLAS_NUM_THREADS=1','OMP_NUM_THREADS=1','CUBLAS_WORKSPACE_CONFIG=:4096:8','PYTHONPATH=src',runner.PYTHON,'tools/v3/correct_conditional_axis.py','--execute'],
        question='Does the single logit-loss correction remove constant-axis behavior under original C budget?',method=LOSS,
        baseline='Original C0/C2000 and constant axis1 height0; not corrected C vs old A/B geometry claim',
        fallback='Seal failure, no extra updates, labels or head changes; fit-only result, no graph expansion',
        acceptance_criteria=['Identical initial tensors and8000 schedule','Same known/unknown masks and original per-parent evaluation','2000 updates only; no best-checkpoint selection','Report same/cross errors and constant baseline, whether output still collapses'],
        expected_evidence=old['expected_evidence'],estimated_cost=dict(compute='One5090 C2000 fixed correction',host_ram_gb=32,gpu_vram_gb=28,disk_gb=30,wall_time_hours=12),
        input_sha256=pins,source_sha256=sources))


if __name__=='__main__':
    p=argparse.ArgumentParser();g=p.add_mutually_exclusive_group(required=True);g.add_argument('--freeze',action='store_true');g.add_argument('--execute',action='store_true');a=p.parse_args()
    if a.freeze:freeze()
    else:
        from mtare_topo.governance_conditional_geometry import validate_correction_card
        raise SystemExit(runner.execute(spec_path=SPEC,card_path=CARD,run_path=RUN,validator=validate_correction_card))
