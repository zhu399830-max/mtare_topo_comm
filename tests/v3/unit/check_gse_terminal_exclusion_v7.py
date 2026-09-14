"""Actual synthetic scans only; no real source reads, labels, or files."""
import json
from copy import deepcopy
from check_gse_joint_producer_backend import main
from check_gse_terminal_producer_backend import bundle as tube
from mtare_topo.teacher.gse_junction_interface_diagnostic_v1 import diagnose_observation
from mtare_topo.teacher.gse_joint_reference_targets_v6 import produce_joint_reference_targets as old
from mtare_topo.teacher.gse_joint_reference_targets_v7 import produce_joint_reference_targets as new


def check(name, bundle, require_negative=False):
    bundle['source'].setdefault('source_sequence_id', 0)
    raw = diagnose_observation(bundle)
    before = old(bundle, raw)
    after = new(bundle, raw)
    a, b = deepcopy(before['record']), deepcopy(after['record'])
    ma, mb = a.pop('membership'), b.pop('membership')
    assert a == b
    changes = [(x, y) for ra, rb in zip(ma, mb) for x, y in zip(ra, rb) if x != y]
    assert all(x is None and y is False for x, y in changes)
    assert not after['full_training_gate_eligible']
    assert not after['scientific_gate_pass']
    print(json.dumps(dict(scene=name, before=ma, after=mb, new_negative_pairs=len(changes),
        junctions=len(after['teacher_provenance']['anchors']),
        terminals=len(after['teacher_provenance']['terminals']),
        unknown_anchors=after['unknown_candidates']['anchors'])), flush=True)
    if require_negative:
        assert changes, 'observed intervening junction should support reference exclusions'
    else:
        assert not changes


check('same_tunnel_no_intervening_junction', tube([[-3.,0.,0.],[15.,0.,0.]]))
check('stacked_junction_no_terminal', main(return_bundle=True, scene='layered'))
check('overlapping_ambiguous_junction', main(return_bundle=True, scene='overlapping'))
check('sensor_inside_terminal_branch', main(return_bundle=True, terminal_branch=True,
    terminal_length_m=8., sensor_positions=[[-x,0.,.04] for x in (2.5,3.,3.5,4.,4.5)]), True)
print('ACTUAL_SYNTHETIC_TERMINAL_EXCLUSION_CHECKS_PASS')
