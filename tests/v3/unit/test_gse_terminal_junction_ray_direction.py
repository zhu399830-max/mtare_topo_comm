"""Observation geometry regression, not a negative-membership teacher."""
import numpy as np

from mtare_topo.teacher.gse_directed_interface_evidence_v1 import directed_evidence


def test_sensor_in_terminal_branch_sees_cap_without_crossing_entrance():
    origin = 4.0
    entrance = 0.0
    cap = 8.0
    forward = 1.0
    assert (cap - origin) / forward == 4.0
    assert (entrance - origin) / forward < 0.0
    # No forward entrance hit is expected; it does not mean cap invisibility.
    assert (entrance - origin) / -1.0 == 4.0


def test_coincident_branch_interfaces_have_valid_evidence_without_strict_order():
    # Robot is inside +X terminal branch, looking through the junction into -X.
    interfaces = [
        dict(interface_id_teacher_only=0, node_id_teacher_only='junction',
             source_key_teacher_only='terminal_branch', inward_direction=[1., 0., 0.]),
        dict(interface_id_teacher_only=1, node_id_teacher_only='junction',
             source_key_teacher_only='opposite_branch', inward_direction=[-1., 0., 0.]),
    ]
    records = [dict(ray_index=0, interface_id_teacher_only=i, t=4., inside_roi=True)
               for i in (0, 1)]
    result = directed_evidence(records, interfaces, directions=np.array([[-1., 0., 0.]]),
        first_return=np.array([10.], dtype=np.float32), valid=np.array([True]),
        return_sources=[['opposite_branch']])
    assert result['interfaces'][0]['leaving_ray_indices'] == [0]
    assert result['interfaces'][1]['entering_ray_indices'] == [0]
    assert not records[0]['t'] < records[1]['t']
    assert result['qualified_labels'] == 0
