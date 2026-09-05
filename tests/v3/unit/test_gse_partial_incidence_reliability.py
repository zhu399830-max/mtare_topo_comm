import numpy as np
import pytest

from mtare_topo.evaluation.gse_partial_incidence_reliability import (
    incidence_commit_allowed,
    physical_edge_identity,
    physical_incidence_count,
    unanimous_seed_metric_support,
)


def test_directed_traversals_map_to_one_physical_edge() -> None:
    assert physical_edge_identity("world:edge_0001:d0") == "world:edge_0001"
    assert physical_incidence_count(("world:edge_0001:d0", "world:edge_0001:d1")) == 1


def test_junction_requires_route_diverse_physical_support() -> None:
    two = ("w:e0:d0", "w:e0:d1", "w:e1:d0")
    three = (*two, "w:e2:d1")
    assert not incidence_commit_allowed("junction", two, junction_support=3)
    assert incidence_commit_allowed("junction", three, junction_support=3)
    assert incidence_commit_allowed("terminal", ("w:e0:d0", "w:e0:d1"), junction_support=3)


def test_unanimous_metric_support_requires_all_three_seeds() -> None:
    centers = np.zeros((3, 2, 3), dtype=np.float32)
    centers[:, 1, 0] = (3.9, 4.1, 3.8)
    result = unanimous_seed_metric_support(centers, np.array([0]), np.array([1]))
    assert result.tolist() == [False]
    centers[1, 1, 0] = 4.0
    assert unanimous_seed_metric_support(centers, np.array([0]), np.array([1])).tolist() == [True]


def test_invalid_traversal_fails_closed() -> None:
    with pytest.raises(ValueError):
        physical_edge_identity("world:edge_0001")
