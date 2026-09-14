"""Predeclared review nominations, NOT visible-structure labels or model inputs.

Pure loaded-data code. The caller must authenticate each construction, index
interval and selected teacher row. No files, scans, weights or poses are read.
"""
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import re

import numpy as np

from .gse_review_quota_assignment_v1 import ReviewNomination
from .gse_review_source_index_v1 import TraversalIndexInterval, VARIANTS


@dataclass(frozen=True)
class PrimitiveIdentity:
    primitive_id: str
    source_edge_id: str
    endpoint_nodes: tuple[str, str]


@dataclass(frozen=True)
class ConstructionIdentityProjection:
    primitives: tuple[PrimitiveIdentity, ...]
    node_degrees: tuple[tuple[str, int], ...]


@dataclass(frozen=True)
class OverlapRows:
    """Rows in the EXACT interval sequence order, never global-ID-as-offset."""
    sequence_rows: tuple[int, ...]
    primitive_index: np.ndarray
    primitive_mask: np.ndarray
    disconnected_overlap_packed: np.ndarray


@dataclass(frozen=True)
class NominationResult:
    nominations: tuple[ReviewNomination, ...]
    evidence: tuple[dict, ...]
    counts: dict
    training_labels_created: int = 0


def _name(value):
    if type(value) is not str or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.:-]*", value):
        raise ValueError("explicit construction identity string required")
    return value


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
        ensure_ascii=True, allow_nan=False).encode()).hexdigest()


def project_construction_identity(document) -> ConstructionIdentityProjection:
    """Check complete incidence/order, while deliberately ignoring shape values.

    A construction JSON necessarily contains geometric bytes too: a source card
    must disclose that read. This projection does not use shape to pick cases.
    """
    if type(document) is not dict or document.get("schema_version") != "primitive_relation_realized_construction_v1":
        raise ValueError("sealed P1a realized construction schema required")
    base = document.get("base_construction")
    if type(base) is not dict or base.get("schema_version") != "primitive_construction_graph_v1":
        raise ValueError("base construction schema required")
    raw = base.get("primitives")
    realized = document.get("realized_primitives")
    compositions = base.get("composition_operations")
    if any(type(x) is not list or not x for x in (raw, realized, compositions)):
        raise ValueError("complete primitive and composition lists required")
    projected, endpoints, edges, identities = [], {}, set(), set()
    for row in raw:
        if type(row) is not dict:
            raise ValueError("primitive object required")
        identity, edge = _name(row.get("primitive_id")), _name(row.get("source_edge_id"))
        if identity in identities or edge in edges:
            raise ValueError("one unique primitive per physical edge required")
        identities.add(identity); edges.add(edge)
        ends = row.get("endpoints")
        if type(ends) is not list or len(ends) != 2:
            raise ValueError("two ordered source-edge endpoints required")
        nodes = []
        for index, end in enumerate(ends):
            if type(end) is not dict or type(end.get("endpoint_index")) is not int or end["endpoint_index"] != index or end.get("primitive_id") != identity:
                raise ValueError("ordered endpoint identity drift")
            node = _name(end.get("node_id")); nodes.append(node)
            endpoints[(identity, index)] = node
        if nodes[0] == nodes[1]:
            raise ValueError("source traversal contract excludes self edges")
        projected.append(PrimitiveIdentity(identity, edge, tuple(nodes)))
    if any(type(row) is not dict for row in realized) or tuple(row.get("primitive_id") for row in realized) != tuple(p.primitive_id for p in projected):
        raise ValueError("realized/base ordered primitive identity drift")
    degrees, covered = {}, set()
    for row in compositions:
        if type(row) is not dict:
            raise ValueError("composition object required")
        node = _name(row.get("node_id")); members = row.get("member_endpoints")
        if node in degrees or type(members) is not list or not members or type(row.get("degree")) is not int or row["degree"] != len(members):
            raise ValueError("unique node with exact incidence degree required")
        degrees[node] = len(members)
        for member in members:
            if type(member) is not dict or type(member.get("endpoint_index")) is not int:
                raise ValueError("typed composition endpoint required")
            key = (_name(member.get("primitive_id")), member["endpoint_index"])
            if key in covered or endpoints.get(key) != node or member.get("node_id") != node:
                raise ValueError("duplicate, missing or inconsistent composition incidence")
            covered.add(key)
    if covered != set(endpoints):
        raise ValueError("construction omits source-edge endpoints")
    return ConstructionIdentityProjection(tuple(projected), tuple(sorted(degrees.items())))


def validate_paired_constructions(documents):
    if type(documents) is not dict or set(documents) != set(VARIANTS):
        raise ValueError("all three exact construction variants required")
    projections = tuple(project_construction_identity(documents[v]) for v in VARIANTS)
    if any(p != projections[0] for p in projections[1:]):
        raise ValueError("cross-variant construction identity/incidence/order drift")
    return projections[0]


def _overlap_witnesses(rows, expected_rows, construction, host_index):
    if type(rows) is not OverlapRows or rows.sequence_rows != expected_rows:
        raise ValueError("overlap rows must bind exact sequence row identities/order")
    q = len(expected_rows)
    for value, shape, dtype in ((rows.primitive_index, (q, 32), np.dtype("int32")),
            (rows.primitive_mask, (q, 32), np.dtype("uint8")),
            (rows.disconnected_overlap_packed, (q, 32, 4), np.dtype("uint8"))):
        if type(value) is not np.ndarray or value.shape != shape or value.dtype != dtype:
            raise ValueError("exact P1b slot dtype and interval shape required")
    idx, mask = rows.primitive_index, rows.primitive_mask
    if np.any((mask != 0) & (mask != 1)) or not np.array_equal(mask, (idx >= 0).astype(np.uint8)) or np.any(idx < -1) or np.any(idx >= len(construction.primitives)):
        raise ValueError("primitive index/mask/codebook bounds drift")
    matrices = np.unpackbits(rows.disconnected_overlap_packed, axis=2, count=32, bitorder="little")
    if not np.array_equal(matrices, matrices.transpose(0, 2, 1)) or np.any(np.diagonal(matrices, axis1=1, axis2=2)) or np.any(matrices & ~(mask[:, :, None] & mask[:, None, :])):
        raise ValueError("overlap must be symmetric, off diagonal and within active slots")
    result = []
    for sequence, indices in enumerate(idx):
        active = indices[mask[sequence].astype(bool)]
        if (not np.array_equal(indices[:len(active)], active)
                or (len(active) > 1 and np.any(np.diff(active) <= 0))):
            raise ValueError("P1b active primitive slots must be sorted unique then padded")
        witnesses = []
        for a, b in zip(*np.where(np.triu(matrices[sequence], 1))):
            first, second = int(indices[a]), int(indices[b])
            p, r = construction.primitives[first], construction.primitives[second]
            if set(p.endpoint_nodes) & set(r.endpoint_nodes):
                raise ValueError("disconnected overlap contradicts construction incidence")
            if host_index in (first, second):
                witnesses.append((p.primitive_id, r.primitive_id))
        result.append(tuple(witnesses))
    return tuple(result)


def nominate_interval(*, parent_id, split, interval, construction, overlap_by_variant):
    """Boundary-node, midpoint-corridor and host-overlap nominations.

    First/last full-traversal decisions only nominate the corresponding source
    endpoint. Midpoint means the lower median decision index, not a distance
    threshold or an inferred event. Alias must involve the actual host primitive
    in at least one paired realization and decision. None is a visibility label.
    """
    _name(parent_id)
    if not re.fullmatch(r"[A-Za-z0-9_]+_C0[1-7]", parent_id) or split not in ("fit", "calibration", "development") or (split == "fit") == parent_id.endswith("_C07"):
        raise ValueError("permitted parent and frozen split required")
    if type(interval) is not TraversalIndexInterval or type(construction) is not ConstructionIdentityProjection:
        raise ValueError("validated typed interval and construction required")
    if not interval.traversal_id.startswith(parent_id + ":") or not re.search(r":d[01]$", interval.traversal_id):
        raise ValueError("parent-bound directed traversal required")
    edge, direction = interval.traversal_id[len(parent_id) + 1:].rsplit(":d", 1)
    hosts = [(i, p) for i, p in enumerate(construction.primitives) if p.source_edge_id == edge]
    if len(hosts) != 1:
        raise ValueError("traversal must bind exactly one physical construction edge")
    host_index, host = hosts[0]
    if type(interval.variants) is not tuple or tuple(v.variant for v in interval.variants) != VARIANTS or interval.sequence_count < 21:
        raise ValueError("eligible full interval with exact three variants required")
    if type(overlap_by_variant) is not dict or set(overlap_by_variant) != set(VARIANTS):
        raise ValueError("all three variants' overlap rows required")
    q = interval.sequence_count
    witnesses = {}
    for v in interval.variants:
        if (v.task != parent_id + "__" + v.variant or len(v.sequence_rows) != q
                or v.source_sequence_ids != interval.variants[0].source_sequence_ids
                or any(type(values) is not tuple or len(values) != q for values in
                    (v.sequence_rows, v.source_sequence_ids, v.frame_rows, v.decision_arc_m))
                or any(type(i) is not int or i < 0 for i in v.sequence_rows)
                or any(type(i) is not int or i < 0 for i in v.source_sequence_ids)
                or any(b != a + 1 for values in (v.sequence_rows, v.source_sequence_ids)
                    for a, b in zip(values, values[1:]))
                or any(type(frames) is not tuple or len(frames) != 5
                    or any(type(i) is not int or i < 0 for i in frames)
                    or any(b != a + 1 for a, b in zip(frames, frames[1:])) for frames in v.frame_rows)
                or any(b[0] != a[0] + 1 for a, b in zip(v.frame_rows, v.frame_rows[1:]))
                or any(type(x) is not float or not math.isfinite(x) or x < 0 for x in v.decision_arc_m)
                or any(b <= a for a, b in zip(v.decision_arc_m, v.decision_arc_m[1:]))):
            raise ValueError("paired sequence identity/order drift")
        witnesses[v.variant] = _overlap_witnesses(overlap_by_variant[v.variant], v.sequence_rows, construction, host_index)
    degrees = dict(construction.node_degrees)
    start_node, end_node = host.endpoint_nodes[::1 if direction == "0" else -1]
    nominations, evidence = [], []
    for start, window in enumerate(interval.iter_windows()):
        clip = _digest({"parent": parent_id, "traversal": interval.traversal_id,
            "variants": [asdict(v) for v in window.variants]})
        base = dict(parent=parent_id, split=split, host_physical_edge_id=edge, clip_id=clip)
        for node, included, boundary in ((start_node, start == 0, "first_available_decision"),
                (end_node, start + 21 == q, "last_available_decision")):
            stratum = "terminal" if degrees[node] == 1 else "junction" if degrees[node] >= 3 else None
            if included and stratum is not None:
                nominations.append(ReviewNomination(**base, stratum=stratum, target_identity="node:" + node))
                evidence.append({"clip_id": clip, "stratum": stratum, "target_identity": "node:" + node,
                    "basis": boundary, "degree": degrees[node], "visible_label": None})
        if start <= (q - 1) // 2 < start + 21:
            nominations.append(ReviewNomination(**base, stratum="corridor", target_identity=edge))
            evidence.append({"clip_id": clip, "stratum": "corridor", "target_identity": edge,
                "basis": "contains_lower_median_traversal_decision", "visible_label": None})
        alias = [{"variant": v.variant, "sequence_row": v.sequence_rows[i], "pairs": witnesses[v.variant][i]}
            for v in interval.variants for i in range(start, start + 21) if witnesses[v.variant][i]]
        if alias:
            nominations.append(ReviewNomination(**base, stratum="alias", target_identity=edge))
            evidence.append({"clip_id": clip, "stratum": "alias", "target_identity": edge,
                "basis": "host_in_disconnected_azimuth_support_pair", "witnesses": alias, "visible_label": None})
    return NominationResult(tuple(nominations), tuple(evidence), {
        "eligible_windows": interval.window_count, "nominations": len(nominations),
        "unique_host_edges": 1, "independent_parent_maps": 1})
