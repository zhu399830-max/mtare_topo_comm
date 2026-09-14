"""Same failing T, now through the entire joint reference producer."""
import json
from check_gse_joint_producer_backend import main
from check_gse_terminal_producer_backend import bundle as tube
from mtare_topo.teacher.gse_junction_interface_diagnostic_v1 import diagnose_observation
from mtare_topo.teacher.gse_joint_reference_targets_v7 import produce_joint_reference_targets as old
from mtare_topo.teacher.gse_joint_reference_targets_v8 import produce_joint_reference_targets as new
from mtare_topo.data.gse_structure_review_v1 import canonical_sha


def check(name, bundle, expected):
    bundle['source'].setdefault('source_sequence_id', 0)
    raw = diagnose_observation(bundle)
    before = old(bundle, raw)
    # Synthetic fixture field uses .01; real P1a uses .025. Never infer one
    # from the mesh axial spacing (.05).
    after = new(bundle, raw, axial_spacing_m=.05, angular_segments=64, field_spacing_m=.01)
    guarded = new(bundle, raw, axial_spacing_m=.05, angular_segments=64, field_spacing_m=.01,
                  qualify_cap_precision=True)
    assert guarded['record'] == after['record'], 'stable original synthetic targets must remain unchanged'
    assert set(guarded['terminal_exclusion_cap_precision']) == {'entering','leaving'}
    assert 'terminal_exclusion_cap_precision' not in after
    assert after['geometry_evidence_settings']['field_spacing_m'] == .01
    a, b = before['record'], after['record']
    assert a['openings'] == b['openings']
    assert a['score_region'] == b['score_region']
    assert a['source_frame_indices'] == b['source_frame_indices']
    assert before['source_binding'] == after['source_binding']
    assert after['target_record_sha256'] == canonical_sha(b)
    assert not after['full_training_gate_eligible'] and not after['scientific_gate_pass']
    for i, anchor in enumerate(a['anchors']):
        matches = [j for j, p in enumerate(b['anchors']) if p['position_m'] == anchor['position_m']]
        assert len(matches) == 1
        for oi, row in enumerate(a['membership']):
            if row[i] is not None:
                assert b['membership'][oi][matches[0]] is row[i]
    print(json.dumps(dict(scene=name, old_membership=a['membership'], new_membership=b['membership'],
        anchors=len(b['anchors']), negatives=len(after['teacher_provenance']['terminal_nonmembership']),
        precision_guarded_same_record=True)), flush=True)
    assert b['membership'] == expected


inside = dict(terminal_branch=True, terminal_length_m=8.,
    sensor_positions=[[-x,0.,.04] for x in (2.5,3.,3.5,4.,4.5)])
check('original_failed_terminal_T', main(return_bundle=True, **inside), [[True, False], [True, False]])
check('same_T_stacked_layer', main(return_bundle=True, scene='layered', **inside), [[True, False], [True, False]])
check('same_tunnel_positive', tube([[-3.,0.,0.],[15.,0.,0.]]), [[True]])
check('ambiguous_overlap', main(return_bundle=True, scene='overlapping'), [[], []])
print('FOUR_ACTUAL_SYNTHETIC_JOINT_V8_CHECKS_PASS')
