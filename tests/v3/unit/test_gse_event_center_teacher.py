import numpy as np

from mtare_topo.teacher.gse_event_center_teacher import (
    event_center_targets, local_event_center_vectors, traversal_tangents,
)


def test_reverse_traversal_changes_signed_offset_but_not_center():
    rows = [
        {"parent_id": "W", "traversal_id": "f", "sequence_index": 0, "event": "junction", "identity": "W:node:n"},
        {"parent_id": "W", "traversal_id": "f", "sequence_index": 1, "event": "junction", "identity": "W:node:n"},
        {"parent_id": "W", "traversal_id": "r", "sequence_index": 0, "event": "junction", "identity": "W:node:n"},
        {"parent_id": "W", "traversal_id": "r", "sequence_index": 1, "event": "junction", "identity": "W:node:n"},
    ]
    xyz = np.asarray([[0, 0, 0], [1, 0, 0], [2, 0, 0], [1, 0, 0]], dtype=float)
    target = event_center_targets(rows, xyz, {"W": {"n": (1.5, 0, 0)}})
    assert np.allclose(target["signed_center_offset_m"], [1.5, .5, .5, -.5])
    assert np.allclose(target["oracle_longitudinal_center_xyz_m"], (1.5, 0, 0))


def test_tangent_never_crosses_traversals():
    tangent = traversal_tangents(
        ["a", "b", "a", "b"], [0, 0, 1, 1],
        np.asarray([[0, 0, 0], [0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float),
    )
    assert np.allclose(tangent[[0, 2]], (1, 0, 0))
    assert np.allclose(tangent[[1, 3]], (0, 1, 0))


def test_singleton_corridor_uses_zero_sentinel_only():
    tangent = traversal_tangents(["short"], [0], np.asarray([[0, 0, 0]], dtype=float))
    assert np.allclose(tangent[0], 0.0)


def test_local_center_vector_reconstructs_sloped_objective_center():
    sensor = np.asarray([[1.0, 2.0, 3.0]])
    objective = np.asarray([[4.0, 5.0, 4.0]])
    tangent = np.asarray([[1.0, 1.0, .2]])
    result = local_event_center_vectors(sensor, objective, tangent, np.asarray([True]))
    reconstructed = sensor + np.einsum(
        "nij,ni->nj", result["route_local_basis"], result["local_center_vector_m"]
    )
    assert np.allclose(reconstructed, objective, atol=1e-6)
