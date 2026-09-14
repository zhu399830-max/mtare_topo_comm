"""Complete corrected A/B while reusing the sealed corrected C."""
from _bootstrap import PROJECT_ROOT as ROOT
import argparse,copy,hashlib,json
from ai_junction_pilot import sha,write
import train_conditional_geometry as runner
NAME='gse_new12_corrected_ab_v1'
CARD=f'configs/v3/gate3/data_cards/{NAME}.json';SPEC=f'configs/v3/gate3/{NAME}.json'
RUN=f'results/gate3_semantics/gate3_20260911_{NAME}_seed0'
C='results/gate3_semantics/gate3_20260911_gse_new12_axis_logit_correction_v1_seed0'


def freeze():
    old=json.loads((ROOT/C/'config/run_spec.json').read_text());card=json.loads((ROOT/C/'config/data_card.json').read_text())
    s=copy.deepcopy(card['scope']);s['variants']=['A','B'];s['reused_corrected_c']=C
    a=copy.deepcopy(card['approval']);a.update(scope_sha256=hashlib.sha256(json.dumps(s,sort_keys=True).encode()).hexdigest(),
        scope='Corrected A/B2000 each; same original init/schedule/targets/loss as sealed corrected C, no C rerun',
        confirmation_reference='User approved same-protocol A/B/C and standing execution; PLAN requires completing corrected A/B and reusing C')
    pins={p:h for p,h in old['input_sha256'].items() if p!=old['data_card']}
    seal=ROOT/C/'artifacts/evidence_sha256.txt';known={p:h for h,p in (l.split('  ',1) for l in seal.read_text().splitlines())}
    pins[str(seal.relative_to(ROOT))]=sha(seal)
    for p in (f'{C}/artifacts/C_final.pt',f'{C}/metrics/C_step2000.json',f'{C}/config/data_card.json'):
        assert sha(ROOT/p)==known[p];pins[p]=known[p]
    for p,h in pins.items():assert sha(ROOT/p)==h,p
    write(ROOT/CARD,dict(schema_version='gse_conditional_corrected_ab_card_v1',scope=s,approval=a));pins[CARD]=sha(ROOT/CARD)
    spec=copy.deepcopy(old);spec.update(slug=NAME,data_card=CARD,user_authorization=a,input_sha256=pins,
        command=['env','OPENBLAS_NUM_THREADS=1','OMP_NUM_THREADS=1','CUBLAS_WORKSPACE_CONFIG=:4096:8','PYTHONPATH=src',runner.PYTHON,'tools/v3/complete_conditional_pairing.py','--execute'],
        question='Same corrected objective: can geometry/relations improve fixed12 fit over observation-only?',
        baseline='A observation, B unary geometry, reused C messages/relations; same fit population, no independent validation claim',
        estimated_cost=dict(compute='One5090 A/B2000 each, no C rerun',host_ram_gb=32,gpu_vram_gb=28,disk_gb=30,wall_time_hours=12),
        source_sha256={str(p.relative_to(ROOT)):sha(p) for folder in ('src/mtare_topo','tools/v3') for p in (ROOT/folder).rglob('*.py')})
    write(ROOT/SPEC,spec)


if __name__=='__main__':
    p=argparse.ArgumentParser();g=p.add_mutually_exclusive_group(required=True);g.add_argument('--freeze',action='store_true');g.add_argument('--execute',action='store_true');a=p.parse_args()
    if a.freeze:freeze()
    else:
        from mtare_topo.governance_conditional_geometry import validate_corrected_ab_card
        raise SystemExit(runner.execute(spec_path=SPEC,card_path=CARD,run_path=RUN,validator=validate_corrected_ab_card))
