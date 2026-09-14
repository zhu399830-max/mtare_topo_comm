"""Two-layer semantic hypotheses and execution-verified topology commits."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
import math
from typing import Any, Sequence

import numpy as np


class HypothesisState(str, Enum):
    PROVISIONAL = "provisional"
    SUPPORTED = "supported"
    COMMITTED = "committed"
    REJECTED = "rejected"


@dataclass(frozen=True)
class SemanticHypothesisEvidence:
    observation_key: int
    event: str
    xyz_m: tuple[float, float, float]
    confidence: float
    uncertainty: float
    exit_relation_digest: str
    geometry_digest: str
    approach_trace_id: str

    def __post_init__(self) -> None:
        xyz = np.asarray(self.xyz_m, dtype=np.float64)
        if (
            self.observation_key < 0 or self.event not in ("junction", "terminal")
            or xyz.shape != (3,) or not np.all(np.isfinite(xyz))
            or not math.isfinite(self.confidence) or not 0.0 <= self.confidence <= 1.0
            or not math.isfinite(self.uncertainty) or not 0.0 <= self.uncertainty <= 1.0
            or not self.exit_relation_digest or not self.geometry_digest or not self.approach_trace_id
        ):
            raise ValueError("semantic hypothesis evidence contract drift")


@dataclass(frozen=True)
class TraceCommitEvidence:
    trace_id: str
    from_hypothesis_id: int
    to_hypothesis_id: int
    frame_count: int
    distance_m: float
    departure_action_digest: str
    arrival_action_digest: str
    reverse_trace_id: str | None = None

    def __post_init__(self) -> None:
        if (
            not self.trace_id or self.from_hypothesis_id < 0 or self.to_hypothesis_id < 0
            or self.from_hypothesis_id == self.to_hypothesis_id or self.frame_count < 2
            or not math.isfinite(self.distance_m) or self.distance_m <= 0.0
            or not self.departure_action_digest or not self.arrival_action_digest
            or (self.reverse_trace_id is not None and not self.reverse_trace_id)
        ):
            raise ValueError("trace commit evidence contract drift")


class SemanticHypothesisGraph:
    """Keep unsafe proposals outside the verified graph until explicit commit.

    Association may group evidence into one hypothesis, but it cannot create a
    verified edge.  Only a completed physical trace supplied to
    :meth:`record_trace` can do that, and both endpoints must first be marked
    supported by independent semantic observations.
    """

    def __init__(self) -> None:
        self.hypotheses: list[dict[str, Any]] = []
        self.verified_nodes: list[dict[str, Any]] = []
        self.verified_edges: list[dict[str, Any]] = []
        self.decision_trace: list[dict[str, Any]] = []
        self._trace_by_id: dict[str, TraceCommitEvidence] = {}

    def propose(self, evidence: SemanticHypothesisEvidence) -> int:
        hypothesis_id = len(self.hypotheses)
        self.hypotheses.append({
            "id": hypothesis_id, "state": HypothesisState.PROVISIONAL.value,
            "event": evidence.event, "evidence": [asdict(evidence)],
            "verified_node_id": None, "rejection_reason": None,
        })
        self.decision_trace.append({"action": "propose", "hypothesis_id": hypothesis_id, "observation_key": evidence.observation_key})
        return hypothesis_id

    def add_support(self, hypothesis_id: int, evidence: SemanticHypothesisEvidence) -> None:
        hypothesis = self._open(hypothesis_id)
        if evidence.event != hypothesis["event"]:
            raise ValueError("support event conflicts with hypothesis")
        keys = {int(value["observation_key"]) for value in hypothesis["evidence"]}
        traces = {str(value["approach_trace_id"]) for value in hypothesis["evidence"]}
        if evidence.observation_key in keys:
            raise ValueError("support observation is duplicated")
        hypothesis["evidence"].append(asdict(evidence))
        if evidence.approach_trace_id not in traces:
            hypothesis["state"] = HypothesisState.SUPPORTED.value
        self.decision_trace.append({"action": "support", "hypothesis_id": hypothesis_id, "observation_key": evidence.observation_key, "state": hypothesis["state"]})

    def reject(self, hypothesis_id: int, reason: str) -> None:
        hypothesis = self._open(hypothesis_id)
        if not reason:
            raise ValueError("hypothesis rejection requires a reason")
        hypothesis["state"] = HypothesisState.REJECTED.value
        hypothesis["rejection_reason"] = str(reason)
        self.decision_trace.append({"action": "reject", "hypothesis_id": hypothesis_id, "reason": str(reason)})

    def commit_node(self, hypothesis_id: int) -> int:
        hypothesis = self._open(hypothesis_id)
        if hypothesis["state"] != HypothesisState.SUPPORTED.value:
            raise RuntimeError("only independently supported hypotheses may commit")
        node_id = len(self.verified_nodes)
        hypothesis["state"] = HypothesisState.COMMITTED.value
        hypothesis["verified_node_id"] = node_id
        self.verified_nodes.append({
            "id": node_id, "hypothesis_id": hypothesis_id, "event": hypothesis["event"],
            "evidence_count": len(hypothesis["evidence"]),
            "approach_trace_ids": sorted({value["approach_trace_id"] for value in hypothesis["evidence"]}),
        })
        self.decision_trace.append({"action": "commit_node", "hypothesis_id": hypothesis_id, "node_id": node_id})
        return node_id

    def record_trace(self, evidence: TraceCommitEvidence) -> int:
        if evidence.trace_id in self._trace_by_id:
            raise ValueError("physical trace is duplicated")
        left = self._committed(evidence.from_hypothesis_id)
        right = self._committed(evidence.to_hypothesis_id)
        if evidence.reverse_trace_id is not None and evidence.reverse_trace_id not in self._trace_by_id:
            raise RuntimeError("declared reverse trace has not been recorded")
        edge_id = len(self.verified_edges)
        self._trace_by_id[evidence.trace_id] = evidence
        self.verified_edges.append({
            "id": edge_id, "from": left["verified_node_id"], "to": right["verified_node_id"],
            "trace": asdict(evidence), "execution_state": "verified",
        })
        self.decision_trace.append({"action": "commit_edge", "edge_id": edge_id, "trace_id": evidence.trace_id})
        return edge_id

    def _open(self, hypothesis_id: int) -> dict[str, Any]:
        if not 0 <= int(hypothesis_id) < len(self.hypotheses):
            raise KeyError("unknown semantic hypothesis")
        hypothesis = self.hypotheses[int(hypothesis_id)]
        if hypothesis["state"] in (HypothesisState.COMMITTED.value, HypothesisState.REJECTED.value):
            raise RuntimeError("semantic hypothesis is already terminal")
        return hypothesis

    def _committed(self, hypothesis_id: int) -> dict[str, Any]:
        if not 0 <= int(hypothesis_id) < len(self.hypotheses):
            raise KeyError("unknown semantic hypothesis")
        hypothesis = self.hypotheses[int(hypothesis_id)]
        if hypothesis["state"] != HypothesisState.COMMITTED.value:
            raise RuntimeError("trace endpoint is not a committed node")
        return hypothesis


__all__ = ["HypothesisState", "SemanticHypothesisEvidence", "SemanticHypothesisGraph", "TraceCommitEvidence"]
