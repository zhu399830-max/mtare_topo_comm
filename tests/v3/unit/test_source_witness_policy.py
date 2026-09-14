import numpy as np
import pytest

from mtare_topo.teacher.source_witness_policy import source_witness_policy


def inputs():
    return dict(valid=np.array([1, 1, 1, 1, 0], dtype=bool),
                reported=np.array([[1,0], [1,0], [1,1], [0,1], [0,0]], dtype=bool),
                possible_surface=np.array([[1,0], [1,1], [1,1], [1,0], [0,0]], dtype=bool),
                uncertain_operands=np.zeros((5,2), dtype=bool),
                surface_uniqueness_certified=np.zeros(5, dtype=bool))


def test_unknown_does_not_remove_scan_or_become_certificate():
    data = inputs()
    before = {k: v.copy() for k, v in data.items()}
    policy = source_witness_policy(**data)
    assert policy.owners(mode='diagnostic').tolist() == [0,-1,-1,-1,-1]
    assert policy.owners(mode='certified').tolist() == [-1]*5
    for key in data:
        np.testing.assert_array_equal(data[key], before[key])
    data['reported'][:] = False
    assert policy.owners(mode='diagnostic')[0] == 0


def test_uncertainty_blocks_even_matching_singleton():
    data = inputs()
    data['uncertain_operands'][0,1] = True
    assert not source_witness_policy(**data).diagnostic_singleton.any()


def test_explicit_certificate_and_mode():
    data = inputs()
    data['surface_uniqueness_certified'][0] = True
    policy = source_witness_policy(**data)
    assert policy.owners(mode='certified')[0] == 0
    with pytest.raises(ValueError):
        policy.owners(mode='auto')
    with pytest.raises(ValueError):
        policy.diagnostic_singleton[0] = False
    data['surface_uniqueness_certified'][1] = True
    with pytest.raises(ValueError):
        source_witness_policy(**data)


def test_invalid_alignment_rejected():
    data = inputs()
    data['valid'] = data['valid'].astype(np.uint8)
    with pytest.raises(ValueError):
        source_witness_policy(**data)
    data = inputs()
    data['reported'][4,0] = True
    with pytest.raises(ValueError):
        source_witness_policy(**data)
