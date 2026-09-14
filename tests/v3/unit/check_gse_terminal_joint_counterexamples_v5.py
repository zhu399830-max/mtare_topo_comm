"""Actual synthetic counterexamples; no data population or file output."""
import json
import argparse
from check_gse_joint_producer_backend import main as scene_bundle
from check_gse_terminal_producer_backend import bundle as tunnel_bundle
from mtare_topo.teacher.gse_junction_interface_diagnostic_v1 import diagnose_observation
from mtare_topo.teacher.gse_joint_reference_targets_v4 import produce_joint_reference_targets as v4
from mtare_topo.teacher.gse_joint_reference_targets_v5 import produce_joint_reference_targets as v5

parser = argparse.ArgumentParser()
parser.add_argument('--corrected', action='store_true')
corrected = parser.parse_args().corrected
if corrected:
    from mtare_topo.teacher.gse_joint_reference_targets_v6 import produce_joint_reference_targets as v5


def check(name, b):
    raw = diagnose_observation(b)
    old, new = v4(b, raw), v5(b, raw)
    assert old['record']['anchors'] == new['record']['anchors']
    if name != 'window_reentry' or not corrected:
        assert old['record']['openings'] == new['record']['openings']
    else:
        assert old['record']['openings'] == [] and len(new['record']['openings']) == 1
    assert not new['full_training_gate_eligible']
    print(json.dumps(dict(scene=name, anchors=len(new['record']['anchors']),
        openings=len(new['record']['openings']), old_membership=old['record']['membership'],
        new_membership=new['record']['membership'],
        terminal_nodes=[t['node_id_teacher_only'] for t in new['teacher_provenance']['terminals']],
        terminal_relation_status=[d['status'] for d in new['teacher_provenance']['terminal_relations']])), flush=True)
    return old, new


old, new = check('terminal_before_junction', scene_bundle(return_bundle=True, terminal_branch=True))
start = new['teacher_provenance']['terminal_anchor_start']
assert new['teacher_provenance']['terminals'], 'must actually observe the terminal'
assert all(v is None for row in new['record']['membership'] for v in row[start:])
assert old['record']['membership'] == new['record']['membership']

old, new = check('layered', scene_bundle(return_bundle=True, scene='layered'))
assert old['record']['membership'] == new['record']['membership']
assert len(new['record']['anchors']) == 1

b = tunnel_bundle([[-3., 0., 0.], [12., 0., 0.], [12., 8., 0.], [0., 8., 0.], [-12., 8., 0.]])
b['source']['source_sequence_id'] = 2
old, new = check('window_reentry', b)
assert len(new['record']['anchors']) == 1
assert sum(v is True for row in new['record']['membership'] for v in row) == 1
for d in new['teacher_provenance']['terminal_relations']:
    if d['status'] == 'TWO_SIDED_SOURCE_WITNESS_ONLY':
        assert d['component']['source_arc_interval_m'][0] == 0.
print(json.dumps(dict(status='ACTUAL_TERMINAL_COUNTEREXAMPLES_PASS', corrected=corrected, scenes=3,
    frames_per_scene=5, rays_per_scene=57600, real_dataset_reads=0, optimizer_steps=0)))
