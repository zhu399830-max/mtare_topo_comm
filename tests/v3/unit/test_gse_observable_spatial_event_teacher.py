from __future__ import annotations

import numpy as np

from mtare_topo.teacher.gse_observable_spatial_event_teacher import (
    observable_teacher_contract,
    repack_observable_event_set,
)


def _arrays() -> dict[str, np.ndarray]:
    event_type = np.full((2, 16), -1, dtype=np.int8)
    relative = np.zeros((2, 16, 3), dtype=np.float32)
    distance = np.zeros((2, 16), dtype=np.float32)
    identity = np.full((2, 16), -1, dtype=np.int32)
    mask = np.zeros((2, 16), dtype=np.uint8)
    values = (
        (0, (10.0, 0.0, 0.0), 10),
        (1, (10.0, 0.0, 10.0), 11),
        (1, (10.0, 0.0, np.tan(np.deg2rad(15.0)) * 10.0), 12),
    )
    for slot, (event, xyz, event_identity) in enumerate(values):
        event_type[0, slot] = event
        relative[0, slot] = xyz
        distance[0, slot] = np.linalg.norm(xyz)
        identity[0, slot] = event_identity
        mask[0, slot] = 1
    return {
        "event_type_index": event_type,
        "event_relative_xyz_m": relative,
        "event_distance_m": distance,
        "event_identity_index": identity,
        "event_mask": mask,
        "set_cardinality": mask.sum(axis=1).astype(np.uint8),
    }


def test_vertical_fov_filter_is_inclusive_and_repack_preserves_order() -> None:
    result = repack_observable_event_set(_arrays())
    assert result.set_cardinality.tolist() == [2, 0]
    assert result.event_identity_index[0, :2].tolist() == [10, 12]
    assert result.retained_original_slot[0, :2].tolist() == [0, 2]
    assert result.event_type_index[0, 2] == -1
    assert result.event_identity_index[0, 2] == -1
    assert np.all(result.event_relative_xyz_m[0, 2:] == 0.0)
    assert result.removed_mask[0].sum() == 1


def test_filter_contract_forbids_model_driven_selection() -> None:
    contract = observable_teacher_contract()
    assert contract["vertical_fov_deg_inclusive"] == [-15.0, 15.0]
    assert "model_prediction" in contract["forbidden_filter_inputs"]
    assert "C10" in contract["forbidden_filter_inputs"]
