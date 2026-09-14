import numpy as np
import pytest
from mtare_topo.data.gse_frozen_observation_roi import validated_frozen_roi


def test_boundary_authority_is_frozen_student_not_world():
    # Numeric values recorded in the failed boundary example, no real payload.
    student = np.array([[9.999999869419094, 0, 0]], np.float64)
    world_relative = np.array([[10.00000023306973, 0, 0]], np.float64)
    assert not (np.linalg.norm(world_relative, axis=1) <= 10).any()
    result = validated_frozen_roi(student, np.array([1], np.uint8), np.array([0]))
    assert result.tolist() == [0] and not result.flags.writeable


def test_boundary_inclusive_without_radius_expansion():
    xyz = np.array([[10., 0, 0], [np.nextafter(10., np.inf), 0, 0], [0, 0, 0]])
    assert validated_frozen_roi(xyz, np.array([True, True, False]), np.array([0])).tolist() == [0]


def test_float32_keeps_original_norm_contract():
    xyz = np.array([[6, 8, 0], [0, 0, 0]], np.float32)
    assert validated_frozen_roi(xyz, np.ones(2, bool), np.arange(2)).tolist() == [0, 1]


@pytest.mark.parametrize('indices', [[], [0, 0], [1, 0], [0, 1, 2], [-1, 0]])
def test_missing_duplicate_reordered_or_extra_indices_rejected(indices):
    with pytest.raises(ValueError):
        validated_frozen_roi(np.zeros((2, 3)), np.ones(2, bool), np.array(indices, dtype=np.int64))


@pytest.mark.parametrize('defect', ['nan', 'mask', 'float_indices'])
def test_invalid_input_rejected(defect):
    xyz=np.zeros((1, 3));valid=np.ones(1, np.uint8);idx=np.array([0])
    if defect=='nan':xyz[0,0]=np.nan
    elif defect=='mask':valid[0]=2
    else:idx=idx.astype(float)
    with pytest.raises(ValueError):validated_frozen_roi(xyz,valid,idx)


def test_empty_observation():
    assert validated_frozen_roi(np.empty((0,3)),np.empty(0,bool),np.empty(0,np.int64)).size==0
