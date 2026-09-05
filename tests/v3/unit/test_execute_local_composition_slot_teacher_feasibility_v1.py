import numpy as np
import pytest

from execute_local_composition_slot_teacher_feasibility_v1 import (
    _attachments,
    _endpoint_overlap,
)


def test_neighbor_list_expands_to_symmetric_endpoint_clique():
    neighbors = np.full((1, 32, 2, 3), -1, dtype=np.int8)
    neighbors.reshape(1, 64, 3)[0, 0, :2] = (2, 4)
    neighbors.reshape(1, 64, 3)[0, 2, :2] = (0, 4)
    neighbors.reshape(1, 64, 3)[0, 4, :2] = (0, 2)
    value = _attachments(neighbors)
    assert value.shape == (1, 64, 64)
    assert value[0, 0, 2] and value[0, 2, 4] and value[0, 4, 0]
    assert np.array_equal(value, value.transpose(0, 2, 1))


def test_neighbor_index_outside_endpoint_capacity_fails():
    neighbors = np.full((1, 32, 2, 3), -1, dtype=np.int16)
    neighbors[0, 0, 0, 0] = 64
    with pytest.raises(ValueError, match="outside"):
        _attachments(neighbors)


def test_primitive_overlap_expands_to_all_four_endpoint_pairs():
    primitive = np.zeros((1, 32, 32), dtype=np.uint8)
    primitive[0, 1, 3] = primitive[0, 3, 1] = 1
    packed = np.packbits(primitive, axis=2, bitorder="little")
    endpoint = _endpoint_overlap(packed)
    assert endpoint.shape == (1, 64, 64)
    assert endpoint[0, 2:4, 6:8].all()
    assert endpoint[0, 6:8, 2:4].all()
