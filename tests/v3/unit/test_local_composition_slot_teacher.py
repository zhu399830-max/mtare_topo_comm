import numpy as np
import pytest

from mtare_topo.evaluation.local_composition_slot_teacher import (
    decode_slot_attachment,
    decompose_observable_attachment,
    select_slot_capacity,
)


def _matrix():
    return np.zeros((64, 64), dtype=np.uint8)


def _clique(matrix, members):
    for left in members:
        for right in members:
            if left != right:
                matrix[left, right] = 1


def test_exchangeable_clusters_losslessly_decode_cliques_and_dustbin():
    attachment = _matrix(); _clique(attachment, (0, 3, 8)); _clique(attachment, (4, 7))
    observed = np.ones(64, dtype=np.uint8); overlap = _matrix(); overlap[1, 2] = overlap[2, 1] = 1
    value = decompose_observable_attachment(attachment, observed, overlap)
    assert value.cluster_sizes == (3, 2)
    assert value.labels[0] == value.labels[3] == value.labels[8]
    assert value.labels[4] == value.labels[7]
    assert value.labels[1] == -1
    assert value.is_clique_partition
    assert value.overlap_violations == 0
    assert np.array_equal(value.observed_attachment, value.reconstructed_attachment)


def test_hidden_endpoint_connection_is_unknown_not_negative_cluster_member():
    attachment = _matrix(); _clique(attachment, (0, 1, 2))
    observed = np.ones(64, dtype=np.uint8); observed[2] = 0
    value = decompose_observable_attachment(attachment, observed, _matrix())
    assert value.cluster_sizes == (2,)
    assert value.labels[2] == -1
    assert value.reconstructed_attachment[0, 1]
    assert not value.reconstructed_attachment[0, 2]


def test_nonclique_relation_is_detected_instead_of_silently_closed():
    attachment = _matrix()
    attachment[0, 1] = attachment[1, 0] = 1
    attachment[1, 2] = attachment[2, 1] = 1
    value = decompose_observable_attachment(attachment, np.ones(64), _matrix())
    assert not value.is_clique_partition
    assert not np.array_equal(value.observed_attachment, value.reconstructed_attachment)


def test_disconnected_overlap_inside_a_transitive_cluster_is_detected():
    attachment = _matrix(); _clique(attachment, (0, 2))
    overlap = _matrix(); overlap[0, 2] = overlap[2, 0] = 1
    value = decompose_observable_attachment(attachment, np.ones(64), overlap)
    assert value.overlap_violations == 1


def test_capacity_uses_fit_maximum_plus_frozen_25_percent_margin():
    assert select_slot_capacity(6)["selected_capacity"] == 8
    assert select_slot_capacity(8)["selected_capacity"] == 16
    assert select_slot_capacity(25)["selected_capacity"] == 32
    assert select_slot_capacity(26)["available"] is False


def test_invalid_asymmetric_relation_fails_closed():
    attachment = _matrix(); attachment[0, 1] = 1
    with pytest.raises(ValueError, match="symmetric"):
        decompose_observable_attachment(attachment, np.ones(64), _matrix())


def test_decode_rejects_invalid_label():
    labels = np.full(64, -1, dtype=np.int16); labels[0] = -2
    with pytest.raises(ValueError, match="dustbin"):
        decode_slot_attachment(labels)
