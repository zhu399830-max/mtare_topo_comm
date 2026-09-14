"""Leakage-safe replay from semantic proposals to execution-verified topology.

Teacher identities are carried only for the final scorer.  Association and
commit decisions consume event type, metric position, causal order, approach
trace identity, and a frozen pair-verifier decision.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class ProposalTrigger:
    row: int
    world: str
    order: int
    traversal_id: str
    sequence_index: int
    event: str
    confidence: float
    uncertainty: float
    xyz_m: tuple[float, float, float]
    teacher_identity: str | None = None
    position_uncertainty_m: float = 0.0

    def __post_init__(self) -> None:
        xyz = np.asarray(self.xyz_m, dtype=np.float64)
        if (
            self.row < 0 or not self.world or self.order < 0 or not self.traversal_id
            or self.sequence_index < 0 or self.event not in ("junction", "terminal")
            or not math.isfinite(self.confidence) or not 0.0 <= self.confidence <= 1.0
            or not math.isfinite(self.uncertainty) or not 0.0 <= self.uncertainty <= 1.0
            or not math.isfinite(self.position_uncertainty_m)
            or not 0.0 <= self.position_uncertainty_m <= 12.0
            or xyz.shape != (3,) or not np.all(np.isfinite(xyz))
        ):
            raise ValueError("proposal trigger contract drift")


def _f1(precision: float, recall: float) -> float:
    return 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0


def replay_trace_commits(
    triggers: Sequence[ProposalTrigger],
    accepted_pairs: Mapping[tuple[int, int], bool],
    *,
    association_valid_rows: set[int],
    distance_cap_m: float = 4.0,
    independent_traces_required: int = 2,
    commit_immediately: bool = False,
) -> dict:
    """Associate proposals and commit only independently supported hypotheses.

    A verifier may associate a new proposal with exactly one earlier
    hypothesis.  Zero or multiple accepted hypotheses fail closed and start a
    separate provisional hypothesis.  An ambiguous provisional hypothesis can
    be consolidated later only after independent traversal support and strict
    non-overlap of learned position-uncertainty intervals.  A node is committed
    after evidence from distinct physical traversals; one completed traversal
    can commit only its first-to-last semantic endpoint edge.
    """

    if distance_cap_m <= 0.0 or independent_traces_required < 1:
        raise ValueError("trace-commit configuration is invalid")
    ordered = sorted(triggers, key=lambda value: (value.world, value.order, value.row))
    if len({value.row for value in ordered}) != len(ordered):
        raise ValueError("proposal rows are duplicated")
    hypotheses: list[dict] = []
    row_to_hypothesis: dict[int, int] = {}
    decisions: list[dict] = []
    by_world: dict[str, list[int]] = {}

    for trigger in ordered:
        candidates: list[tuple[float, int]] = []
        if trigger.row in association_valid_rows:
            for hypothesis_id in by_world.get(trigger.world, []):
                hypothesis = hypotheses[hypothesis_id]
                if hypothesis["event"] != trigger.event:
                    continue
                accepted_distance = math.inf
                for evidence in hypothesis["evidence"]:
                    distance = float(np.linalg.norm(
                        np.asarray(trigger.xyz_m) - np.asarray(evidence.xyz_m)
                    ))
                    key = tuple(sorted((trigger.row, evidence.row)))
                    if (
                        evidence.row in association_valid_rows
                        and distance <= distance_cap_m + 1e-12
                        and bool(accepted_pairs.get(key, False))
                    ):
                        accepted_distance = min(accepted_distance, distance)
                if math.isfinite(accepted_distance):
                    candidates.append((accepted_distance, hypothesis_id))
        candidates.sort()
        if len(candidates) == 1:
            hypothesis_id = candidates[0][1]
            hypotheses[hypothesis_id]["evidence"].append(trigger)
            action = "associate"
        else:
            hypothesis_id = len(hypotheses)
            hypotheses.append({
                "id": hypothesis_id, "world": trigger.world, "event": trigger.event,
                "evidence": [trigger], "committed": False,
                "ambiguous_on_creation": len(candidates) > 1,
                "deferred_candidates": [value[1] for value in candidates],
                "merged_into": None,
            })
            by_world.setdefault(trigger.world, []).append(hypothesis_id)
            action = "propose_ambiguous" if candidates else "propose"
        row_to_hypothesis[trigger.row] = hypothesis_id
        hypothesis = hypotheses[hypothesis_id]
        independent = len({value.traversal_id for value in hypothesis["evidence"]})
        if not hypothesis["committed"] and (
            commit_immediately or independent >= independent_traces_required
        ):
            hypothesis["committed"] = True
            action += "+commit"
        decisions.append({
            "row": trigger.row, "action": action, "hypothesis_id": hypothesis_id,
            "accepted_hypothesis_candidates": [value[1] for value in candidates],
            "independent_traces": independent,
        })

    # Resolve only hypotheses explicitly rejected as ambiguous.  One
    # candidate's worst-case distance must be strictly below every alternative
    # candidate's best-case distance; otherwise the split remains provisional.
    for hypothesis in hypotheses:
        if (
            not hypothesis["committed"]
            or not hypothesis["ambiguous_on_creation"]
            or hypothesis["merged_into"] is not None
        ):
            continue
        candidate_intervals = []
        for candidate_id in hypothesis["deferred_candidates"]:
            candidate = hypotheses[candidate_id]
            if (
                not candidate["committed"] or candidate["merged_into"] is not None
                or candidate["world"] != hypothesis["world"]
                or candidate["event"] != hypothesis["event"]
            ):
                continue
            intervals = []
            for current in hypothesis["evidence"]:
                for prior in candidate["evidence"]:
                    key = tuple(sorted((current.row, prior.row)))
                    if not bool(accepted_pairs.get(key, False)):
                        continue
                    distance = float(np.linalg.norm(
                        np.asarray(current.xyz_m) - np.asarray(prior.xyz_m)
                    ))
                    radius = current.position_uncertainty_m + prior.position_uncertainty_m
                    intervals.append((max(0.0, distance - radius), distance + radius))
            if intervals:
                best = min(intervals, key=lambda value: (value[1], value[0]))
                candidate_intervals.append((best[0], best[1], candidate_id))
        if not candidate_intervals:
            continue
        candidate_intervals.sort(key=lambda value: (value[1], value[0], value[2]))
        best = candidate_intervals[0]
        alternative_lower = min(
            (value[0] for value in candidate_intervals[1:]), default=math.inf
        )
        if not best[1] < alternative_lower:
            continue
        winner = hypotheses[best[2]]
        winner["evidence"].extend(hypothesis["evidence"])
        winner["evidence"].sort(key=lambda value: (value.world, value.order, value.row))
        hypothesis["committed"] = False
        hypothesis["merged_into"] = winner["id"]
        for evidence in hypothesis["evidence"]:
            row_to_hypothesis[evidence.row] = winner["id"]
        decisions.append({
            "row": hypothesis["evidence"][-1].row,
            "action": "deferred_associate",
            "hypothesis_id": winner["id"],
            "merged_hypothesis_id": hypothesis["id"],
            "accepted_hypothesis_candidates": [value[2] for value in candidate_intervals],
            "independent_traces": len({value.traversal_id for value in winner["evidence"]}),
            "winning_distance_interval_m": [best[0], best[1]],
            "alternative_lower_bound_m": alternative_lower,
        })

    edges: list[dict] = []
    edge_keys: set[tuple[int, int]] = set()
    by_traversal: dict[str, list[ProposalTrigger]] = {}
    for trigger in ordered:
        by_traversal.setdefault(trigger.traversal_id, []).append(trigger)
    for traversal_id, values in sorted(by_traversal.items()):
        values.sort(key=lambda value: (value.sequence_index, value.row))
        endpoints: list[int] = []
        for value in values:
            hypothesis_id = row_to_hypothesis[value.row]
            if hypotheses[hypothesis_id]["committed"] and (
                not endpoints or endpoints[-1] != hypothesis_id
            ):
                endpoints.append(hypothesis_id)
        if len(endpoints) < 2:
            continue
        left, right = endpoints[0], endpoints[-1]
        if left == right:
            continue
        key = tuple(sorted((left, right)))
        if key in edge_keys:
            continue
        edge_keys.add(key)
        edges.append({
            "id": len(edges), "from_hypothesis": left, "to_hypothesis": right,
            "trace_id": traversal_id, "execution_state": "verified",
            "commit_contract": "completed_traversal_first_last_endpoints_v1",
        })
    return {
        "hypotheses": hypotheses, "edges": edges, "decision_trace": decisions,
        "row_to_hypothesis": row_to_hypothesis,
    }


def score_trace_replay(
    replay: Mapping[str, object],
    *,
    true_nodes: set[tuple[str, str]],
    true_trace_relations: set[tuple[tuple[str, str], tuple[str, str]]],
) -> dict:
    """Score committed nodes/edges; teacher values never affect replay."""

    committed = [value for value in replay["hypotheses"] if value["committed"]]
    predicted_nodes: list[tuple[str, str] | None] = []
    false_loop_merges = 0
    for hypothesis in committed:
        identities = {
            value.teacher_identity for value in hypothesis["evidence"]
            if value.teacher_identity is not None
        }
        if len(identities) == 1:
            predicted_nodes.append((hypothesis["world"], next(iter(identities))))
        else:
            predicted_nodes.append(None)
            if len(identities) > 1:
                false_loop_merges += 1
    unique_correct_nodes = {value for value in predicted_nodes if value in true_nodes}
    node_precision = len(unique_correct_nodes) / len(committed) if committed else 0.0
    node_recall = len(unique_correct_nodes) / len(true_nodes) if true_nodes else 0.0

    hypothesis_identity: dict[int, tuple[str, str] | None] = {}
    for hypothesis in committed:
        identities = {
            value.teacher_identity for value in hypothesis["evidence"]
            if value.teacher_identity is not None
        }
        hypothesis_identity[hypothesis["id"]] = (
            (hypothesis["world"], next(iter(identities))) if len(identities) == 1 else None
        )
    predicted_edges: set[tuple[tuple[str, str], tuple[str, str]]] = set()
    invalid_edges = 0
    for edge in replay["edges"]:
        left = hypothesis_identity.get(edge["from_hypothesis"])
        right = hypothesis_identity.get(edge["to_hypothesis"])
        if left is None or right is None or left == right:
            invalid_edges += 1
            continue
        predicted_edges.add(tuple(sorted((left, right))))
    correct_edges = predicted_edges & true_trace_relations
    edge_denominator = len(predicted_edges) + invalid_edges
    edge_precision = len(correct_edges) / edge_denominator if edge_denominator else 0.0
    edge_recall = len(correct_edges) / len(true_trace_relations) if true_trace_relations else 0.0
    false_loop_fraction = false_loop_merges / len(committed) if committed else 0.0
    node_f1 = _f1(node_precision, node_recall)
    edge_f1 = _f1(edge_precision, edge_recall)
    return {
        "true_nodes": len(true_nodes), "committed_nodes": len(committed),
        "correct_unique_nodes": len(unique_correct_nodes),
        "node_precision": node_precision, "node_recall": node_recall, "node_f1": node_f1,
        "false_loop_merges": false_loop_merges,
        "false_loop_merge_fraction": false_loop_fraction,
        "true_trace_relations": len(true_trace_relations),
        "committed_edges": edge_denominator, "correct_unique_edges": len(correct_edges),
        "edge_precision": edge_precision, "edge_recall": edge_recall, "edge_f1": edge_f1,
        "node_edge_macro_f1": (node_f1 + edge_f1) / 2.0,
    }


__all__ = ["ProposalTrigger", "replay_trace_commits", "score_trace_replay"]
