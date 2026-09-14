"""Actual synthetic scans, no real data or output files."""
import json
from check_gse_terminal_producer_backend import bundle
from mtare_topo.teacher.gse_junction_interface_diagnostic_v1 import diagnose_observation
from mtare_topo.teacher.gse_joint_reference_targets_v4 import produce_joint_reference_targets as v4
from mtare_topo.teacher.gse_joint_reference_targets_v5 import produce_joint_reference_targets as v5
from mtare_topo.data.gse_structure_review_v1 import canonical_sha

b = bundle([[-3., 0., 0.], [20., 0., 0.]])
b['source']['source_sequence_id'] = 0
raw = diagnose_observation(b)
old = v4(b, raw)
new = v5(b, raw)
assert len(new['record']['anchors']) == len(new['record']['openings']) == 1
assert old['record']['membership'] == [[None]]
assert new['record']['membership'] == [[True]], new['teacher_provenance']['terminal_relations']
assert old['record']['anchors'] == new['record']['anchors']
assert old['record']['openings'] == new['record']['openings']
assert new['target_record_sha256'] == canonical_sha(new['record'])
assert not new['full_training_gate_eligible']
assert new['teacher_provenance']['relations'][0]['anchor_index'] == 0
short = bundle()
short['source']['source_sequence_id'] = 1
short_result = v5(short, diagnose_observation(short))
assert len(short_result['record']['anchors']) == 2
assert short_result['record']['openings'] == []
assert short_result['record']['membership'] == []
print(json.dumps(dict(status='SYNTHETIC_TERMINAL_JOINT_PASS', frames=5, rays=57600,
    old_membership=old['record']['membership'], new_membership=new['record']['membership'],
    no_window_opening_scene_anchors=2, no_window_opening_scene_memberships=0,
    synthetic_scenes=2, real_dataset_reads=0, optimizer_steps=0, full_training_gate_eligible=False)))
