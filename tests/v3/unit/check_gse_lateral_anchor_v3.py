"""Real synthetic CSG scans: same failed T plus retained ambiguity guards."""
import json
from check_gse_joint_producer_backend import main
from mtare_topo.teacher.gse_junction_interface_diagnostic_v1 import diagnose_observation
from mtare_topo.teacher.gse_reference_anchor_targets_v2 import produce_junction_reference_targets as old
from mtare_topo.teacher.gse_reference_anchor_targets_v3 import produce_junction_reference_targets as new


def check(name, kwargs, expected_old, expected_new):
    bundle = main(return_bundle=True, **kwargs)
    raw = diagnose_observation(bundle)
    a = old(bundle, raw)
    b = new(bundle, raw, axial_spacing_m=.05, angular_segments=64, field_spacing_m=.01)
    assert len(a['record']['anchors']) == expected_old
    assert len(b['record']['anchors']) == expected_new
    old_positions = [x['position_m'] for x in a['record']['anchors']]
    new_positions = [x['position_m'] for x in b['record']['anchors']]
    assert all(p in new_positions for p in old_positions)
    assert not b['full_training_gate_eligible'] and not b['scientific_gate_pass']
    if expected_new:
        assert [x['node_id_teacher_only'] for x in b['teacher_provenance']] == ['junction0']
    print(json.dumps(dict(scene=name, old_anchors=expected_old, new_anchors=expected_new,
        lateral_entry_counts=[[len(v) for v in p['surface_entry_witness_ray_indices']] for p in b['teacher_provenance']],
        departure_counts=[[len(v) for v in p['surface_departure_witness_ray_indices']] for p in b['teacher_provenance']])), flush=True)


inside = dict(terminal_branch=True, terminal_length_m=8.,
    sensor_positions=[[-x, 0., .04] for x in (2.5,3.,3.5,4.,4.5)])
check('original_failed_terminal_T', inside, 0, 1)
check('same_T_stacked_layer', dict(inside, scene='layered'), 0, 1)
check('ordinary_original_T', {}, 1, 1)
check('ambiguous_overlapping_junctions', dict(scene='overlapping'), 0, 0)
print('FOUR_SYNTHETIC_LATERAL_ANCHOR_CHECKS_PASS')
