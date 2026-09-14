"""Reuse exact archived student/evidence transport for the frozen holdout."""
from _bootstrap import PROJECT_ROOT as ROOT
import argparse
import hashlib
import json
import subprocess
from ai_junction_pilot import sha, write

NAME = 'gse_conditional_development_inputs_v1'
CARD = f'configs/v3/gate3/data_cards/{NAME}.json'
SPEC = f'configs/v3/gate3/{NAME}_run.json'
RUN = f'results/gate3_semantics/gate3_20260911_{NAME}_seed20260906'
SOURCE = f'configs/v3/gate3/{NAME}.json'
PYTHON = '/home/zeng-workstation/.local/share/mtare_topo_comm/envs/phase3_torch290_cu129_zarr2187_v1/bin/python'


def freeze():
    s = json.loads((ROOT/SOURCE).read_text())
    entries = s.pop('entries')
    s['observations'] = [dict(e, parent=e['parent_id'], source_global_sequence_index=e['source_sequence_id']) for e in entries]
    s['counts'] = dict(development=dict(parents=5, observations=240, tasks=15, unique_variant_frames=1200))
    s['source_sha256'] = s['input_sha256']
    s['limits'] = dict(wall_seconds=900, host_bytes=4*1024**3, output_bytes=1024**3)
    a = dict(status='APPROVED', approved_by='user-standing-development-authorization', approved_at='2026-09-11',
             authorized_gates=[3], authorized_operations=['data_export'],
             scope_sha256=hashlib.sha256(json.dumps(s, sort_keys=True).encode()).hexdigest(),
             confirmation_reference='User confirms conditional construction references and standing execution of parent-held-out development validation; no training or strict-test access',
             scope='Exactly frozen240C07development observations, separate student/source arrays, no labels or training')
    card = dict(schema_version='gse_conditional_development_inputs_card_v1', scope=s, approval=a,
                limitations='C07 encoder-selection exposed. Independent relationship-head parents only, not unseen whole pipeline.')
    from mtare_topo.governance_conditional_development import validate_card
    assert validate_card(card).passed
    write(ROOT/CARD, card)
    files = dict(s['input_sha256']); files[SOURCE] = sha(ROOT/SOURCE)
    sources = {str(p.relative_to(ROOT)):sha(p) for folder in ('src/mtare_topo', 'tools/v3') for p in (ROOT/folder).rglob('*.py')}
    sources[CARD] = sha(ROOT/CARD)
    freeze = subprocess.check_output([PYTHON, '-m', 'pip', 'freeze', '--all'], text=True)
    spec = dict(schema_version='v3_run_spec_v1', gate=3, date='20260911', slug=NAME, seed=20260906,
                operation='data_export', data_card=CARD, user_authorization=a,
                command=['env','OPENBLAS_NUM_THREADS=1','OMP_NUM_THREADS=1',PYTHON,'tools/v3/export_conditional_development_inputs.py','--execute'],
                question='Can all240frozen relationship-head holdout observations be transported without student/teacher leakage?',
                method='Reuse archived exact chunk decoder and pose/causal-motion checks; no rerender, labels or model forward',
                baseline='Original archived arrays and immutable identity manifest',
                fallback='Seal failure; no replacement samples or conversion of unknown labels',
                estimated_cost=dict(compute='CPU15tasks;409160448 padded decoded bytes', host_ram_gb=4,gpu_vram_gb=0,disk_gb=1,wall_time_hours=.25),
                acceptance_criteria=['240exact observations across5parents/80edges/3variants','Original ranges/poses/motion/source alignment retained','Student files exclude absolute poses and source IDs','No training, no continuous-route claim'],
                expected_evidence=['Student and separate source files, full identity manifest, source hashes, environment, raw log, metrics, RUN_STATE and SHA seal'],
                input_sha256=files,source_sha256=sources,sidecar_freeze=freeze,
                sidecar_freeze_sha256=hashlib.sha256(freeze.encode()).hexdigest())
    write(ROOT/SPEC, spec)
    print(json.dumps(dict(spec=SPEC,observations=240,parents=5,frames=1200,decoded_bytes=s['decoded_padded_bytes'])))


if __name__ == '__main__':
    p=argparse.ArgumentParser();g=p.add_mutually_exclusive_group(required=True)
    g.add_argument('--freeze',action='store_true');g.add_argument('--execute',action='store_true');args=p.parse_args()
    if args.freeze:
        freeze()
    else:
        from export_local_pair_pilot import execute
        from mtare_topo.governance_conditional_development import validate_card
        raise SystemExit(execute(run_path=RUN,spec_path=SPEC,card_path=CARD,validator=validate_card))
