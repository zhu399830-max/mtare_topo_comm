"""Build a corrected GSE Teacher manifest from sealed causal change-points."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
import math
from typing import Any, Iterable, Mapping

from mtare_topo.teacher.gse_association_teacher import (
    AssociationTeacherExample,
    deterministic_association_pairs,
)


PROTECTED_NODE_EVENTS = frozenset({"junction", "terminal"})


@dataclass(frozen=True)
class SparseGlobalIndexAudit:
    """Evidence that a filtered split preserves immutable complete-export IDs."""

    count: int
    minimum: int | None
    maximum: int | None
    omitted_internal_index_count: int
    gap_block_count: int
    unique: bool
    strictly_increasing: bool
    exact_source_order: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def audit_preserved_sparse_global_indices(
    source_indices: Iterable[int],
    output_indices: Iterable[int],
) -> SparseGlobalIndexAudit:
    """Audit stable sparse IDs without treating them as compact array rows."""

    source = tuple(int(value) for value in source_indices)
    output = tuple(int(value) for value in output_indices)
    minimum = min(output) if output else None
    maximum = max(output) if output else None
    unique = len(set(output)) == len(output)
    strictly_increasing = all(right > left for left, right in zip(output, output[1:]))
    gap_block_count = sum(right > left + 1 for left, right in zip(output, output[1:]))
    omitted = 0
    if output and unique and strictly_increasing:
        omitted = int(maximum - minimum + 1 - len(output))
    return SparseGlobalIndexAudit(
        count=len(output),
        minimum=minimum,
        maximum=maximum,
        omitted_internal_index_count=omitted,
        gap_block_count=gap_block_count,
        unique=unique,
        strictly_increasing=strictly_increasing,
        exact_source_order=output == source,
    )


def association_signature(record: Mapping[str, Any]) -> tuple[float, float, float, float, float]:
    """Reproduce the sealed V1 finite, dimensionless association signature."""

    valid = record.get("width_m") is not None and record.get("height_m") is not None
    values = (
        float(record["width_m"]) / 60.0 if valid else 0.0,
        float(record["height_m"]) / 60.0 if valid else 0.0,
        abs(float(record["slope_deg"])) / 90.0,
        float(record["curvature_per_m"]) / math.pi,
        1.0 if valid else 0.0,
    )
    if not all(math.isfinite(value) for value in values):
        raise ValueError("association signature must be finite")
    return values


def apply_causal_change_point_labels(
    observations: Iterable[Mapping[str, Any]],
    causal_labels: Mapping[tuple[str, int], Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Replace all old transitions, then apply protected-priority causal labels."""

    corrected: list[dict[str, Any]] = []
    audit: list[dict[str, Any]] = []
    seen_observations: set[tuple[str, int]] = set()
    consumed_labels: set[tuple[str, int]] = set()
    for raw in observations:
        row = dict(raw)
        key = (str(row["traversal_id"]), int(row["sequence_index"]))
        if key in seen_observations:
            raise ValueError("Teacher observation keys must be unique")
        seen_observations.add(key)
        old_event = str(row["event"])
        old_identity = row.get("identity")
        if old_event == "geometry_transition":
            row["event"] = "corridor"
            row["identity"] = None
        label = causal_labels.get(key)
        if label is not None:
            consumed_labels.add(key)
            if str(label["parent_id"]) != str(row["parent_id"]):
                raise ValueError("causal label crosses a topology parent")
            if str(label["edge_id"]) != str(row["edge_id"]):
                raise ValueError("causal label crosses a physical edge")
            if old_event in PROTECTED_NODE_EVENTS:
                action = f"suppressed_by_{old_event}_priority"
            else:
                row["event"] = "geometry_transition"
                row["identity"] = str(label["identity"])
                action = "applied_geometry_transition"
            audit.append(
                {
                    "parent_id": row["parent_id"],
                    "traversal_id": row["traversal_id"],
                    "edge_id": row["edge_id"],
                    "sequence_index": row["sequence_index"],
                    "old_event": old_event,
                    "old_identity": old_identity,
                    "causal_identity": label["identity"],
                    "causal_identity_kind": label["identity_kind"],
                    "action": action,
                    "new_event": row["event"],
                    "new_identity": row["identity"],
                }
            )
        if row["event"] == "corridor" and row.get("identity") is not None:
            raise RuntimeError("corrected corridor observation retained an identity")
        if row["event"] != "corridor" and row.get("identity") is None:
            raise RuntimeError("corrected structural observation lacks an identity")
        corrected.append(row)
    missing = set(causal_labels) - consumed_labels
    if missing:
        raise RuntimeError(f"{len(missing)} causal labels lack a Teacher observation")
    return corrected, audit


def corrected_world_association_pairs(
    observations: Iterable[Mapping[str, Any]],
) -> tuple[dict[str, Any], ...]:
    """Regenerate deterministic pairs after the event/identity correction."""

    records = list(observations)
    parent_ids = {str(record["parent_id"]) for record in records}
    if len(parent_ids) != 1:
        raise ValueError("association pairs must be generated within one parent")
    examples = [
        AssociationTeacherExample(
            observation_id=str(record["observation_id"]),
            parent_id=str(record["parent_id"]),
            traversal_id=str(record["traversal_id"]),
            identity=str(record["identity"]),
            event=str(record["event"]),
            geometry_signature=association_signature(record),
        )
        for record in records
        if record["event"] != "corridor"
    ]
    return tuple(pair.to_dict() for pair in deterministic_association_pairs(examples))


def corrected_identity_summaries(
    observations: Iterable[Mapping[str, Any]],
    causal_identity_kind: Mapping[str, str],
) -> tuple[dict[str, Any], ...]:
    """Summarize node, turn and corrected change-point identities uniformly."""

    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in observations:
        if record["event"] != "corridor":
            groups[str(record["identity"])].append(record)
    summaries: list[dict[str, Any]] = []
    for identity, rows in sorted(groups.items()):
        events = {str(row["event"]) for row in rows}
        parents = {str(row["parent_id"]) for row in rows}
        if len(events) != 1 or len(parents) != 1:
            raise RuntimeError("one corrected identity crosses event or parent")
        event = next(iter(events))
        if event == "geometry_transition":
            identity_kind = causal_identity_kind.get(identity)
            if identity_kind not in {"degree_two_endpoint", "interior_edge"}:
                raise RuntimeError("corrected change-point identity kind is missing")
        elif event in PROTECTED_NODE_EVENTS:
            identity_kind = "tng_node"
        elif event == "turn":
            identity_kind = "edge_turn"
        else:
            raise RuntimeError(f"unexpected structural event: {event}")
        arcs = [float(row["canonical_edge_arc_m"]) for row in rows]
        summaries.append(
            {
                "parent_id": next(iter(parents)),
                "identity": identity,
                "event": event,
                "identity_kind": identity_kind,
                "observation_count": len(rows),
                "traversal_count": len({str(row["traversal_id"]) for row in rows}),
                "edge_count": len({str(row["edge_id"]) for row in rows}),
                "edge_ids": sorted({str(row["edge_id"]) for row in rows}),
                "canonical_arc_min_m": min(arcs),
                "canonical_arc_max_m": max(arcs),
                "canonical_arc_median_m": float(sorted(arcs)[len(arcs) // 2]),
            }
        )
    return tuple(summaries)


def corrected_manifest_summary(observations: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    rows = list(observations)
    events = Counter(str(row["event"]) for row in rows)
    identities: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        if row["event"] != "corridor":
            identities[str(row["event"])].add(str(row["identity"]))
    return {
        "observation_count": len(rows),
        "event_counts": dict(sorted(events.items())),
        "identity_counts": {event: len(values) for event, values in sorted(identities.items())},
        "structural_observation_count": sum(value for event, value in events.items() if event != "corridor"),
        "structural_identity_count": sum(len(values) for values in identities.values()),
    }


__all__ = [
    "PROTECTED_NODE_EVENTS",
    "SparseGlobalIndexAudit",
    "apply_causal_change_point_labels",
    "association_signature",
    "audit_preserved_sparse_global_indices",
    "corrected_identity_summaries",
    "corrected_manifest_summary",
    "corrected_world_association_pairs",
]
