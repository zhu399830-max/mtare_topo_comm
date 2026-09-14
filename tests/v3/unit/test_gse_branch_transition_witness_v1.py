import numpy as np
import pytest

from mtare_topo.teacher.gse_branch_transition_witness_v1 import branch_transition_witness


def fixture():
    interfaces = [
        dict(interface_id_teacher_only=0, node_id_teacher_only='junction',
             source_key_teacher_only='terminal_branch', inward_direction=[1., 0., 0.]),
        dict(interface_id_teacher_only=1, node_id_teacher_only='junction',
             source_key_teacher_only='other_branch', inward_direction=[-1., 0., 0.]),
    ]
    records = [dict(ray_index=0, interface_id_teacher_only=i, t=4., inside_roi=True)
               for i in (0, 1)]
    kwargs = dict(leaving_interface=0, entering_interface=1,
        directions=np.array([[-1., 0., 0.]]), first_return=np.array([10.], dtype=np.float32),
        valid=np.array([True]), return_sources=[['other_branch']])
    return records, interfaces, kwargs


@pytest.mark.parametrize('parameters', [(3., 4.), (4., 4.), (5., 4.)])
def test_coincident_and_overlapping_interfaces_do_not_require_crossing_order(parameters):
    records, interfaces, kwargs = fixture()
    for row, t in zip(records, parameters):
        row['t'] = t
    result = branch_transition_witness(records, interfaces, **kwargs)
    assert result['ray_indices'] == [0]
    assert result['membership'] is None
    assert not result['physical_separation_certified']
    assert branch_transition_witness(records[::-1], interfaces[::-1], **kwargs) == result


@pytest.mark.parametrize('defect', ['occluded', 'on_return', 'invalid', 'outside', 'multisource', 'layer', 'parallel'])
def test_unsupported_or_cross_layer_evidence_cannot_supply_witness(defect):
    records, interfaces, kwargs = fixture()
    if defect == 'occluded':
        kwargs['first_return'][0] = 3.
    elif defect == 'on_return':
        kwargs['first_return'][0] = 4.
    elif defect == 'invalid':
        kwargs['valid'][0] = False
    elif defect == 'outside':
        records[0]['inside_roi'] = False
    elif defect == 'multisource':
        kwargs['return_sources'][0] = ['other_branch', 'stacked_branch']
    elif defect == 'layer':
        interfaces[1]['node_id_teacher_only'] = 'different_layer_junction'
    elif defect == 'parallel':
        interfaces[1]['inward_direction'] = [0., 1., 0.]
    result = branch_transition_witness(records, interfaces, **kwargs)
    assert result['ray_indices'] == []
    assert result['membership'] is None


def test_conflicting_repeated_crossings_are_not_cherry_picked():
    records, interfaces, kwargs = fixture()
    records += [dict(records[1], inside_roi=False, t=11.)]
    assert branch_transition_witness(records, interfaces, **kwargs)['ray_indices'] == []


def test_duplicate_triangle_hits_do_not_multiply_witness():
    records, interfaces, kwargs = fixture()
    assert branch_transition_witness(records + records, interfaces, **kwargs)['ray_indices'] == [0]


def test_interface_cannot_be_compared_with_itself():
    records, interfaces, kwargs = fixture()
    kwargs['entering_interface'] = 0
    with pytest.raises(ValueError):
        branch_transition_witness(records, interfaces, **kwargs)
