import numpy as np
import pytest

from mtare_topo.evaluation.primitive_connection_hypergraph_teacher import (
    audit_connection_cluster_batch,
    merge_connection_cluster_summaries,
    select_fit_only_cluster_capacity,
)


def _fixture():
    primitive = np.zeros((1, 6), dtype=np.uint8)
    primitive = np.pad(primitive, ((0, 0), (0, 26)))
    primitive[:, :6] = 1
    neighbor = np.full((1, 32, 2, 3), -1, dtype=np.int8)
    # Four endpoints from four distinct primitives share one composition.
    first = (1, 2, 4, 6)
    # Two endpoints from the remaining two primitives share another.
    second = (8, 10)
    for members in (first, second):
        for source in members:
            destinations = sorted(set(members) - {source})
            neighbor[0, source // 2, source % 2, :len(destinations)] = destinations
    observed = np.zeros((1, 32, 2), dtype=np.uint8)
    for endpoint in (1, 2, 4, 8, 10):
        observed[0, endpoint // 2, endpoint % 2] = 1
    return primitive, neighbor, observed


def test_clique_teacher_forms_unique_connection_clusters():
    primitive, neighbor, observed = _fixture()
    result = audit_connection_cluster_batch(primitive, neighbor, observed)
    assert result.passed
    assert result.physical_clusters == 2
    assert result.maximum_clusters_per_row == 2
    assert result.maximum_cluster_size == 4
    assert result.cluster_size_histogram[4] == 1
    assert result.cluster_size_histogram[2] == 1
    assert result.observable_multimember_clusters == 2
    assert result.observed_member_count_histogram[3] == 1
    assert result.observed_member_count_histogram[2] == 1
    assert result.undirected_attachment_labels == 7
    assert result.supervised_cluster_memberships == 6


def test_visible_crop_of_clique_remains_unambiguous_and_reports_singleton():
    primitive, neighbor, observed = _fixture()
    observed[:] = 0
    observed[0, 0, 1] = 1
    result = audit_connection_cluster_batch(primitive, neighbor, observed)
    assert result.passed
    assert result.observable_multimember_clusters == 0
    assert result.observed_singletons_from_physical_multimember == 1
    assert result.fully_hidden_physical_clusters == 1


def test_asymmetric_relation_fails_closed():
    primitive, neighbor, observed = _fixture()
    neighbor[0, 1, 0] = -1
    result = audit_connection_cluster_batch(primitive, neighbor, observed)
    assert not result.symmetric
    assert not result.clique_consistent
    assert not result.passed


def test_connected_non_clique_relation_fails_closed():
    primitive, neighbor, observed = _fixture()
    neighbor[:] = -1
    # Symmetric path 1--2--4, not a connection clique.
    neighbor[0, 0, 1, 0] = 2
    neighbor[0, 1, 0, :2] = (1, 4)
    neighbor[0, 2, 0, 0] = 2
    result = audit_connection_cluster_batch(primitive, neighbor, observed)
    assert not result.clique_consistent
    assert not result.passed


def test_inactive_destination_same_primitive_and_noncanonical_storage_fail():
    primitive, neighbor, observed = _fixture()
    inactive = neighbor.copy()
    inactive[0, 0, 0, 0] = 20
    assert not audit_connection_cluster_batch(primitive, inactive, observed).active_destination_only

    same = neighbor.copy()
    same[0, 0, 0, 0] = 1
    assert not audit_connection_cluster_batch(primitive, same, observed).cross_primitive_only

    noncanonical = neighbor.copy()
    noncanonical[0, 4, 0] = (-1, 10, -1)
    assert not audit_connection_cluster_batch(primitive, noncanonical, observed).canonical_padding


def test_invalid_observed_inactive_endpoint_is_rejected():
    primitive, neighbor, observed = _fixture()
    observed[0, 31, 1] = 1
    with pytest.raises(ValueError, match="inactive"):
        audit_connection_cluster_batch(primitive, neighbor, observed)


def test_summary_merge_is_additive_and_capacity_is_fit_only():
    value = audit_connection_cluster_batch(*_fixture())
    merged = merge_connection_cluster_summaries((value, value))
    assert merged.rows == 2
    assert merged.physical_clusters == 4
    assert merged.maximum_clusters_per_row == 2
    assert merged.cluster_size_histogram[4] == 2
    capacity = select_fit_only_cluster_capacity(6)
    assert capacity["required_capacity"] == 8
    assert capacity["selected_capacity"] == 8
    with pytest.raises(OverflowError):
        select_fit_only_cluster_capacity(32)


def test_audit_is_deterministic():
    inputs = _fixture()
    first = audit_connection_cluster_batch(*inputs)
    second = audit_connection_cluster_batch(*(value.copy() for value in inputs))
    assert first == second
