"""Bind existing sealed development inputs; no model or sensor asset export."""
from _bootstrap import PROJECT_ROOT as ROOT
import hashlib
import json

BASE='results/gate6_single_robot/gate6_20260910_gse_direct_isolated_reproduction_v1_seed11'
RUN=BASE+'/results/gate6_single_robot/gate6_20260910_gse_branch_geometry_direct_return_v1_seed11'


def sha(path):
    digest=hashlib.sha256()
    with path.open('rb') as source:
        for block in iter(lambda:source.read(8388608),b''):digest.update(block)
    return digest.hexdigest()


def main():
    spec=json.loads((ROOT/'configs/v3/gate3/primitive_relation_observable_failure_attribution_v1.json').read_text())
    checkpoint=next(p for p in spec['frozen_inputs'] if p.endswith('/models/seed0/selected.pt'))
    pins={}
    for name in ['artifacts/sensors.bag','artifacts/geometry/live_geometry.jsonl','artifacts/geometry/live_geometry_snapshot.json']:
        path=RUN+'/'+name;pins[path]=sha(ROOT/path)
    seal=dict((line.split('  ',1)[1],line.split('  ',1)[0]) for line in (ROOT/RUN/'artifacts/evidence_sha256.txt').read_text().splitlines())
    for path,h in pins.items():
        assert seal[path[len(BASE)+1:]]==h,'source seal mismatch'
    pins[checkpoint]=sha(ROOT/checkpoint)
    assert pins[checkpoint]==spec['frozen_inputs'][checkpoint],'checkpoint drift'
    windows=[];sources=set()
    for line in (ROOT/RUN/'artifacts/geometry/live_geometry.jsonl').open():
        row=json.loads(line);value=row.get('result')
        if value is None:continue
        g=value['geometry'];keys=g['source_frame_keys']
        if windows:assert keys[:-1]==windows[-1]['source_frame_keys'][1:],'sequence gap'
        sources.update(keys);windows.append(dict(timestamp=g['timestamp'],source_frame_keys=keys))
    assert len(windows)==290 and len(sources)==294
    scope=dict(worlds=['tunnel'],split='historically_used_development_not_unseen_test',
        independent_worlds=1,independent_trajectories=1,raw_frames=294,effective_five_frame_observations=290,
        sampling='all accepted causal windows of sealed independent v9 reproduction; no score selection',
        temporal_spacing_s='nominal 0.2; exact timestamps in windows',spatial_spacing_m='recorded motion; not uniform',
        windows=windows,source_sha256=pins,checkpoint=checkpoint,model_seed=0,
        checkpoint_selection='historical frozen selected checkpoint; seed0 fixed before inference',
        existence_threshold=.5,threshold_source='tools/v3/run_primitive_relation_observable_nonlearning_c07_v1.py',
        teacher_source=None,teacher_reads=0,training_steps=0,protected_world_reads=0,
        leakage_audit='development-only integration; no labels, GT identities, threshold fitting or checkpoint selection',
        comparison='same recorded poses, scans, current sector evidence and graph backend; model axes versus nonlearning fitted axes',
        inference_frames=290,full_graph_truth_available=False,
        metrics_scope='integration and observed-surface/temporal diagnostics; no semantic node-edge F1 or superiority without references')
    card=dict(schema_version='gse_learned_geometry_development_inputs_v1',scope=scope,
        approval=dict(status='APPROVED',approved_by='user-standing-development-authorization',
            confirmation_reference='User requests ongoing frozen model integration and bounded same-input comparison; no training or protected data',
            authorized_operations=['inference','topology_replay'],
            scope_sha256=hashlib.sha256(json.dumps(scope,sort_keys=True).encode()).hexdigest()))
    path=ROOT/'configs/v3/gate6/data_cards/gse_learned_geometry_development_inputs_v1.json'
    with path.open('x') as out:json.dump(card,out,indent=2)
    print(json.dumps(dict(card=str(path.relative_to(ROOT)),frames=294,observations=290,independent_routes=1,model_seed=0,teacher_reads=0)))


if __name__=='__main__':main()
