from __future__ import annotations

import pytest

from mtare_topo.topology.gse_hypothesis_graph import (
    SemanticHypothesisEvidence,
    SemanticHypothesisGraph,
    TraceCommitEvidence,
)


def _evidence(key: int, trace: str, event: str = "junction") -> SemanticHypothesisEvidence:
    return SemanticHypothesisEvidence(key, event, (float(key), 0.0, 0.0), .9, .1, f"exit{key}", f"geometry{key}", trace)


def test_hypothesis_needs_independent_approach_before_commit() -> None:
    graph = SemanticHypothesisGraph()
    hypothesis = graph.propose(_evidence(0, "forward"))
    graph.add_support(hypothesis, _evidence(1, "forward"))
    with pytest.raises(RuntimeError, match="independently supported"):
        graph.commit_node(hypothesis)
    graph.add_support(hypothesis, _evidence(2, "reverse"))
    assert graph.commit_node(hypothesis) == 0


def test_only_completed_trace_creates_verified_edge() -> None:
    graph = SemanticHypothesisGraph()
    nodes = []
    for offset in (0, 10):
        hypothesis = graph.propose(_evidence(offset, f"a{offset}"))
        graph.add_support(hypothesis, _evidence(offset + 1, f"b{offset}"))
        nodes.append((hypothesis, graph.commit_node(hypothesis)))
    assert graph.verified_edges == []
    edge = graph.record_trace(TraceCommitEvidence("trace", nodes[0][0], nodes[1][0], 5, 10.0, "depart", "arrive"))
    assert edge == 0 and graph.verified_edges[0]["execution_state"] == "verified"


def test_rejected_hypothesis_cannot_be_committed() -> None:
    graph = SemanticHypothesisGraph()
    hypothesis = graph.propose(_evidence(0, "forward", "terminal"))
    graph.reject(hypothesis, "reverse observation conflict")
    with pytest.raises(RuntimeError, match="already terminal"):
        graph.commit_node(hypothesis)
