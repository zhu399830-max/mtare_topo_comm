"""Factorized decision-node graph with trace-verified geometric edges."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Mapping, Sequence

import numpy as np

from mtare_topo.evaluation.gse_causal_episode_metrics import extract_causal_event_triggers
from mtare_topo.semantics.geometric_semantics import EVENT_NAMES, StructuralEvent
from mtare_topo.topology.gse_graph import GSEGraphConfig, GeometrySemanticEventGraph


DECISION_EVENTS = frozenset((StructuralEvent.JUNCTION.value, StructuralEvent.TERMINAL.value))
DECISION_EVENT_INDICES = frozenset(EVENT_NAMES.index(value) for value in DECISION_EVENTS)


@dataclass(frozen=True)
class TraversedGeometrySample:
    arc_m: float
    local_axis: tuple[float, float, float]
    width_m: float
    height_m: float
    slope_deg: float
    curvature_per_m: float
    uncertainty: float

    def __post_init__(self) -> None:
        scalar = (
            self.arc_m, self.width_m, self.height_m, self.slope_deg,
            self.curvature_per_m, self.uncertainty,
        )
        if not all(math.isfinite(float(value)) for value in scalar):
            raise ValueError("geometry sample values must be finite")
        if self.arc_m < 0.0 or self.width_m <= 0.0 or self.height_m <= 0.0:
            raise ValueError("geometry sample arc/size is invalid")
        if self.curvature_per_m < 0.0 or not 0.0 <= self.uncertainty <= 1.0:
            raise ValueError("geometry sample curvature/uncertainty is invalid")
        axis = np.asarray(self.local_axis, dtype=np.float64)
        if axis.shape != (3,) or not np.all(np.isfinite(axis)) or np.linalg.norm(axis) <= 1e-8:
            raise ValueError("geometry sample axis must be finite and nonzero")
        axis /= np.linalg.norm(axis)
        object.__setattr__(self, "local_axis", tuple(float(value) for value in axis))

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["local_axis"] = list(self.local_axis)
        return result


class FactorizedConsensusMetricBackend:
    """Frozen k-of-three verifier contract with a metric-locality fail-closed gate."""

    def __init__(
        self,
        scores_by_pair: Mapping[tuple[int, int], Sequence[float]],
        *,
        seed_thresholds: Sequence[float],
        votes_required: int = 2,
        maximum_candidate_distance_m: float = 4.0,
    ) -> None:
        thresholds = np.asarray(seed_thresholds, dtype=np.float64)
        if thresholds.shape != (3,) or not np.all(np.isfinite(thresholds)):
            raise ValueError("three finite seed thresholds are required")
        if votes_required not in (1, 2, 3):
            raise ValueError("votes_required must be one, two or three")
        if not math.isfinite(maximum_candidate_distance_m) or maximum_candidate_distance_m <= 0.0:
            raise ValueError("maximum candidate distance must be positive")
        scores: dict[tuple[int, int], tuple[float, float, float]] = {}
        for pair, values in scores_by_pair.items():
            if len(pair) != 2 or int(pair[0]) == int(pair[1]):
                raise ValueError("association score pair is invalid")
            key = tuple(sorted((int(pair[0]), int(pair[1]))))
            score = np.asarray(values, dtype=np.float64)
            if score.shape != (3,) or not np.all(np.isfinite(score)):
                raise ValueError("association pair requires three finite scores")
            if key in scores:
                raise ValueError("association score pair is duplicated")
            scores[key] = tuple(float(value) for value in score)
        self._scores = scores
        self.seed_thresholds = tuple(float(value) for value in thresholds)
        self.votes_required = int(votes_required)
        self.maximum_candidate_distance_m = float(maximum_candidate_distance_m)

    @property
    def pair_count(self) -> int:
        return len(self._scores)

    def evaluate_pair(self, left: int, right: int, distance_m: float) -> dict[str, Any]:
        distance = float(distance_m)
        if not math.isfinite(distance) or distance < 0.0:
            raise ValueError("candidate distance must be finite and nonnegative")
        key = tuple(sorted((int(left), int(right))))
        if distance > self.maximum_candidate_distance_m + 1e-12:
            return {
                "seed_scores": [0.0, 0.0, 0.0], "seed_acceptance": [False] * 3,
                "ensemble_score": 0.0, "votes": 0, "votes_required": self.votes_required,
                "distance_m": distance, "distance_cap_m": self.maximum_candidate_distance_m,
                "accepted": False, "rejection_reason": "outside_frozen_metric_cap",
            }
        score = self._scores.get(key)
        if score is None:
            return {
                "seed_scores": [0.0, 0.0, 0.0], "seed_acceptance": [False] * 3,
                "ensemble_score": 0.0, "votes": 0, "votes_required": self.votes_required,
                "distance_m": distance, "distance_cap_m": self.maximum_candidate_distance_m,
                "accepted": False, "rejection_reason": "pair_score_unavailable",
            }
        acceptance = [value >= threshold for value, threshold in zip(score, self.seed_thresholds, strict=True)]
        votes = int(sum(acceptance))
        accepted = votes >= self.votes_required
        return {
            "seed_scores": list(score), "seed_acceptance": acceptance,
            "ensemble_score": float(np.mean(score)), "votes": votes,
            "votes_required": self.votes_required, "distance_m": distance,
            "distance_cap_m": self.maximum_candidate_distance_m,
            "accepted": accepted,
            "rejection_reason": None if accepted else "insufficient_frozen_seed_consensus",
        }


class FactorizedCausalDecisionBackend:
    """Frozen past-only episode triggers for junction/terminal node generation."""

    def __init__(
        self,
        *,
        global_sequence_index: Sequence[int],
        event_probability: np.ndarray,
        traversal_id: Sequence[str],
        sequence_index: Sequence[int],
        boundary_offset_m: Sequence[float],
        uncertainty: Sequence[float],
        structural_threshold: float = 0.986,
    ) -> None:
        keys = np.asarray(global_sequence_index, dtype=np.int64)
        probability = np.asarray(event_probability, dtype=np.float64)
        traversal = np.asarray(traversal_id, dtype=str)
        sequence = np.asarray(sequence_index, dtype=np.int64)
        boundary = np.asarray(boundary_offset_m, dtype=np.float64)
        uncertain = np.asarray(uncertainty, dtype=np.float64)
        count = len(keys)
        if (
            count == 0
            or len(np.unique(keys)) != count
            or probability.shape != (count, len(EVENT_NAMES))
            or traversal.shape != (count,)
            or sequence.shape != (count,)
            or boundary.shape != (count,)
            or uncertain.shape != (count,)
            or not np.all(np.isfinite(probability))
            or np.any(probability < 0.0)
            or not np.allclose(probability.sum(axis=1), 1.0, rtol=0.0, atol=1e-5)
        ):
            raise ValueError("factorized causal decision population drift")
        triggers = extract_causal_event_triggers(
            probability, traversal, sequence, boundary, uncertain,
            structural_threshold=float(structural_threshold),
        )
        trigger_rows = {
            int(trigger.row): trigger
            for trigger in triggers
            if trigger.predicted_event_index in DECISION_EVENT_INDICES
        }
        self._decision_by_key: dict[int, dict[str, Any]] = {}
        corridor = np.zeros(len(EVENT_NAMES), dtype=np.float64)
        corridor[EVENT_NAMES.index(StructuralEvent.CORRIDOR.value)] = 1.0
        for row, key in enumerate(keys):
            trigger = trigger_rows.get(row)
            accepted = trigger is not None
            self._decision_by_key[int(key)] = {
                "accepted": accepted,
                "mean_event_probability": probability[row].tolist() if accepted else corridor.tolist(),
                "structural_threshold": float(structural_threshold),
                "trigger_row": row if accepted else None,
                "trigger_event": EVENT_NAMES[trigger.predicted_event_index] if accepted else None,
                "rejection_reason": None if accepted else "no_frozen_decision_episode_trigger",
            }
        self.structural_threshold = float(structural_threshold)
        self.raw_structural_trigger_count = len(triggers)
        self.decision_trigger_count = len(trigger_rows)

    def evaluate_node(self, global_sequence_index: int) -> dict[str, Any]:
        try:
            decision = self._decision_by_key[int(global_sequence_index)]
        except KeyError as exc:
            raise RuntimeError("node key is absent from frozen causal decision outputs") from exc
        return dict(decision)


class FactorizedDecisionGraph(GeometrySemanticEventGraph):
    """Only action-changing events create nodes; geometry changes remain on edges."""

    def __init__(
        self,
        config: GSEGraphConfig,
        *,
        association_backend: FactorizedConsensusMetricBackend,
        node_generation_backend: Any | None = None,
    ) -> None:
        if not math.isclose(
            config.association_radius_m,
            association_backend.maximum_candidate_distance_m,
            rel_tol=0.0, abs_tol=1e-12,
        ):
            raise ValueError("graph radius must equal the frozen metric association cap")
        super().__init__(
            config,
            association_reason="factorized_consensus_metric",
            association_backend=association_backend,
            node_generation_backend=node_generation_backend,
            structural_events=DECISION_EVENTS,
            reject_multiple_backend_accepts=True,
        )

    def _append_trace_verified_edge(
        self, previous: int, target: int, frame_index: int, route_arc_m: float
    ) -> None:
        if len(self._trace) < 2:
            raise RuntimeError("factorized edge requires a multi-frame physical trace")
        start_arc = float(self._trace[0]["route_arc_m"])
        profile = [
            TraversedGeometrySample(
                arc_m=float(row["route_arc_m"]) - start_arc,
                local_axis=tuple(row["local_axis"]),
                width_m=float(row["width_m"]), height_m=float(row["height_m"]),
                slope_deg=float(row["slope_deg"]),
                curvature_per_m=float(row["curvature_per_m"]),
                uncertainty=float(row["uncertainty"]),
            ).to_dict()
            for row in self._trace
        ]
        super()._append_trace_verified_edge(previous, target, frame_index, route_arc_m)
        edge = next(
            item for item in self.edges
            if {int(item["from"]), int(item["to"])} == {int(previous), int(target)}
        )
        profiles = edge.setdefault("geometry_profiles", [])
        profiles.append(profile)
        if len(profiles) != int(edge["traversal_count"]):
            raise RuntimeError("edge geometry-profile/traversal count drift")
        edge["execution_state"] = "verified"
        edge["traversability_summary"] = {
            **edge["geometry"],
            "maximum_uncertainty": max(float(row["uncertainty"]) for row in profile),
        }


__all__ = [
    "DECISION_EVENTS", "FactorizedCausalDecisionBackend",
    "FactorizedConsensusMetricBackend",
    "FactorizedDecisionGraph", "TraversedGeometrySample",
]
