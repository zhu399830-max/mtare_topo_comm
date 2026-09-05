from __future__ import annotations

import numpy as np

from mtare_topo.semantics.range_exit_baseline import RangeExitBaseline
from mtare_topo.topology.online_topometric import OnlineTopometricGraph


ELEVATION = np.arange(-15.0, 16.0, 2.0, dtype=np.float32)


def test_range_exit_baseline_finds_two_separated_open_sectors() -> None:
    ranges = np.full((16, 720), 3.0, dtype=np.float32)
    valid = np.ones((16, 720), dtype=np.uint8)
    ranges[5:11, 710:720] = 25.0
    ranges[5:11, 0:11] = 25.0
    ranges[5:11, 350:371] = 20.0
    result = RangeExitBaseline().predict(ranges, valid, ELEVATION)
    assert result["branch_count"] == 2
    assert min(abs((value - 0.0 + 180) % 360 - 180) for value in result["headings_robot_deg"]) <= 3
    assert min(abs((value - 180.0 + 180) % 360 - 180) for value in result["headings_robot_deg"]) <= 3


def test_range_exit_baseline_ignores_invalid_far_values() -> None:
    ranges = np.full((16, 720), 50.0, dtype=np.float32)
    valid = np.zeros((16, 720), dtype=np.uint8)
    result = RangeExitBaseline().predict(ranges, valid, ELEVATION)
    assert result["branch_count"] == 0


def test_online_graph_creates_traversed_edge_and_exit_stubs() -> None:
    graph = OnlineTopometricGraph()
    graph.update((0, 0, 0), 0, (0, 180), 0)
    graph.update((7, 0, 0), 0, (0, 90, 180), 1)
    graph.update((8, 0, 0), 0, (0, 90, 180), 2)
    snapshot = graph.snapshot()
    assert snapshot["node_count"] == 2
    assert snapshot["edge_count"] == 1
    assert snapshot["nodes"][1]["role"] == "junction"
    assert "traversed" in snapshot["nodes"][0]["exit_stub_state"]


def test_online_graph_merges_revisited_structural_place() -> None:
    graph = OnlineTopometricGraph()
    graph.update((0, 0, 0), 0, (0, 90, 180), 0)
    graph.update((7, 0, 0), 0, (0, 180), 1)
    graph.update((14, 0, 0), 0, (0, 180), 2)
    graph.update((7, 0, 0), 180, (0, 180), 3)
    graph.update((1, 0, 0), 180, (0, 90, 180), 4)
    graph.update((0, 0, 0), 180, (180, 270, 0), 5)
    assert graph.snapshot()["node_count"] <= 3
