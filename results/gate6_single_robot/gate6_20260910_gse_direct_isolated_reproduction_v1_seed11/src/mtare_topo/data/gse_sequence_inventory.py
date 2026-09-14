"""Read-only inventory for five-frame directed-traversal GSE sequences."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from mtare_topo.topology.continuous_trajectory import project_to_polyline


@dataclass(frozen=True)
class DirectedTraversalInventory:
    parent_id: str
    split: str
    traversal_id: str
    edge_id: str
    tunnel_id: str
    from_node_id: str
    to_node_id: str
    length_m: float
    start_spline_arc_m: float
    end_spline_arc_m: float
    start_connector_m: float
    end_connector_m: float
    sequence_count: int
    history_frames: int

    @property
    def unique_frame_count(self) -> int:
        return self.sequence_count + self.history_frames - 1 if self.sequence_count > 0 else 0

    @property
    def referenced_frame_count(self) -> int:
        return self.sequence_count * self.history_frames

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def causal_anchor_arcs(
    traversal_length_m: float,
    *,
    spacing_m: float = 1.0,
    history_frames: int = 5,
) -> np.ndarray:
    """Return current-frame arcs with four real past samples on one traversal."""

    length = float(traversal_length_m)
    spacing = float(spacing_m)
    if not math.isfinite(length) or length <= 0.0:
        raise ValueError("traversal_length_m must be positive and finite")
    if not math.isfinite(spacing) or spacing <= 0.0:
        raise ValueError("spacing_m must be positive and finite")
    if history_frames < 1:
        raise ValueError("history_frames must be positive")
    first = (int(history_frames) - 1) * spacing
    return np.arange(first, length, spacing, dtype=np.float64)


def deduplicated_frame_arcs_and_sequence_indices(
    traversal_length_m: float,
    *,
    spacing_m: float = 1.0,
    history_frames: int = 5,
    global_frame_offset: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return unique frame arcs and causal reference rows without duplication."""

    anchors = causal_anchor_arcs(
        traversal_length_m,
        spacing_m=spacing_m,
        history_frames=history_frames,
    )
    if not isinstance(global_frame_offset, int) or isinstance(global_frame_offset, bool) or global_frame_offset < 0:
        raise ValueError("global_frame_offset must be a nonnegative integer")
    unique_count = len(anchors) + history_frames - 1 if len(anchors) else 0
    frame_arcs = np.arange(unique_count, dtype=np.float64) * float(spacing_m)
    if len(anchors) == 0:
        return frame_arcs, anchors, np.empty((0, history_frames), dtype=np.int64)
    local = np.arange(history_frames, dtype=np.int64)[None, :] + np.arange(len(anchors), dtype=np.int64)[:, None]
    references = local + int(global_frame_offset)
    if not np.allclose(frame_arcs[history_frames - 1 :], anchors, rtol=0.0, atol=1e-12):
        raise RuntimeError("causal anchor and deduplicated frame arcs disagree")
    return frame_arcs, anchors, references


def enumerate_directed_traversals(
    *,
    parent_id: str,
    split: str,
    graph: Mapping[str, Any],
    spline_document: Mapping[str, Any],
    spacing_m: float = 1.0,
    history_frames: int = 5,
) -> list[DirectedTraversalInventory]:
    """Enumerate each physical graph edge once in each direction."""

    if split not in {"train", "validation"}:
        raise ValueError("GSE development inventory permits only train or validation")
    if str(parent_id).endswith("_C10"):
        raise ValueError("strict-test parent C10 is forbidden from GSE development inventory")
    nodes = {str(node["id"]): np.asarray(node["xyz"], dtype=np.float64) for node in graph["nodes"]}
    splines = {
        str(item["tunnel_id"]): np.asarray(item["points"], dtype=np.float64)
        for item in spline_document["tunnels"]
    }
    records: list[DirectedTraversalInventory] = []
    for edge in sorted(graph["edges"], key=lambda item: str(item["id"])):
        edge_id = str(edge["id"])
        node_ids = tuple(str(value) for value in edge.get("node_ids", ()))
        tunnel_ids = tuple(str(value) for value in edge.get("tunnel_ids", ()))
        if len(node_ids) != 2 or len(set(node_ids)) != 2:
            raise ValueError(f"edge {edge_id} must connect two distinct nodes")
        if len(tunnel_ids) != 1 or tunnel_ids[0] not in splines:
            raise ValueError(f"edge {edge_id} must reference one existing tunnel spline")
        tunnel_id = tunnel_ids[0]
        first_projection = project_to_polyline(nodes[node_ids[0]], splines[tunnel_id])
        second_projection = project_to_polyline(nodes[node_ids[1]], splines[tunnel_id])
        length = (
            first_projection.error_m
            + abs(second_projection.arc_m - first_projection.arc_m)
            + second_projection.error_m
        )
        if not math.isfinite(length) or length <= 0.0:
            raise ValueError(f"edge {edge_id} has invalid directed traversal length")
        for direction_index, (from_index, to_index) in enumerate(((0, 1), (1, 0))):
            start = first_projection if from_index == 0 else second_projection
            end = second_projection if to_index == 1 else first_projection
            anchors = causal_anchor_arcs(length, spacing_m=spacing_m, history_frames=history_frames)
            records.append(
                DirectedTraversalInventory(
                    parent_id=str(parent_id),
                    split=str(split),
                    traversal_id=f"{parent_id}:{edge_id}:d{direction_index}",
                    edge_id=edge_id,
                    tunnel_id=tunnel_id,
                    from_node_id=node_ids[from_index],
                    to_node_id=node_ids[to_index],
                    length_m=float(length),
                    start_spline_arc_m=float(start.arc_m),
                    end_spline_arc_m=float(end.arc_m),
                    start_connector_m=float(start.error_m),
                    end_connector_m=float(end.error_m),
                    sequence_count=int(len(anchors)),
                    history_frames=int(history_frames),
                )
            )
    if len(records) != 2 * len(graph["edges"]):
        raise RuntimeError("directed traversal inventory is not exactly two records per edge")
    identities = {record.traversal_id for record in records}
    if len(identities) != len(records):
        raise RuntimeError("directed traversal identities are not unique")
    return records


def inventory_from_registry(
    *,
    registry_path: Path,
    mesh_root: Path,
    spacing_m: float = 1.0,
    history_frames: int = 5,
) -> dict[str, Any]:
    """Read exactly the 80/10 development rows; never open C10 assets."""

    registry = json.loads(Path(registry_path).read_text(encoding="utf-8"))
    raw_rows = registry.get("rows")
    if not isinstance(raw_rows, list):
        raise ValueError("world registry rows are missing")
    row_schema = registry.get("sampling_contract", {}).get("row_schema")
    if not isinstance(row_schema, list) or not all(isinstance(name, str) for name in row_schema):
        raise ValueError("world registry row schema is missing")
    rows: list[dict[str, Any]] = []
    for raw_row in raw_rows:
        if isinstance(raw_row, Mapping):
            rows.append(dict(raw_row))
            continue
        if not isinstance(raw_row, Sequence) or isinstance(raw_row, (str, bytes)) or len(raw_row) != len(row_schema):
            raise ValueError("world registry row does not match its declared schema")
        rows.append(dict(zip(row_schema, raw_row, strict=True)))
    selected = [row for row in rows if row.get("split") in {"train", "validation"}]
    if len(selected) != 90:
        raise ValueError(f"expected exactly 90 train/validation parents, found {len(selected)}")
    if any(str(row["parent_id"]).endswith("_C10") for row in selected):
        raise ValueError("strict-test parent leaked into registry rows")
    records: list[DirectedTraversalInventory] = []
    world_summaries: list[dict[str, Any]] = []
    for row in sorted(selected, key=lambda item: str(item["parent_id"])):
        parent_id = str(row["parent_id"])
        split = str(row["split"])
        primary = Path(mesh_root) / parent_id / "primary"
        graph_path = primary / "graph.json"
        spline_path = primary / "splines.json"
        graph = json.loads(graph_path.read_text(encoding="utf-8"))
        splines = json.loads(spline_path.read_text(encoding="utf-8"))
        world_records = enumerate_directed_traversals(
            parent_id=parent_id,
            split=split,
            graph=graph,
            spline_document=splines,
            spacing_m=spacing_m,
            history_frames=history_frames,
        )
        records.extend(world_records)
        world_summaries.append(
            {
                "parent_id": parent_id,
                "split": split,
                "edge_count": len(graph["edges"]),
                "directed_traversal_count": len(world_records),
                "directed_length_m": float(sum(record.length_m for record in world_records)),
                "sequence_count": int(sum(record.sequence_count for record in world_records)),
                "unique_frame_count": int(sum(record.unique_frame_count for record in world_records)),
                "referenced_frame_count": int(sum(record.referenced_frame_count for record in world_records)),
            }
        )
    split_summaries: dict[str, dict[str, Any]] = {}
    for split in ("train", "validation"):
        split_records = [record for record in records if record.split == split]
        split_worlds = [world for world in world_summaries if world["split"] == split]
        split_summaries[split] = {
            "world_count": len(split_worlds),
            "edge_count": int(sum(world["edge_count"] for world in split_worlds)),
            "directed_traversal_count": len(split_records),
            "directed_length_m": float(sum(record.length_m for record in split_records)),
            "sequence_count": int(sum(record.sequence_count for record in split_records)),
            "unique_frame_count": int(sum(record.unique_frame_count for record in split_records)),
            "referenced_frame_count": int(sum(record.referenced_frame_count for record in split_records)),
        }
    return {
        "schema_version": "gse_sequence_inventory_v1",
        "spacing_m": float(spacing_m),
        "history_frames": int(history_frames),
        "history_span_m": float((history_frames - 1) * spacing_m),
        "strict_test_worlds_read": 0,
        "mtare_worlds_read": 0,
        "split_summaries": split_summaries,
        "totals": {
            "world_count": len(world_summaries),
            "edge_count": int(sum(world["edge_count"] for world in world_summaries)),
            "directed_traversal_count": len(records),
            "directed_length_m": float(sum(record.length_m for record in records)),
            "sequence_count": int(sum(record.sequence_count for record in records)),
            "unique_frame_count": int(sum(record.unique_frame_count for record in records)),
            "referenced_frame_count": int(sum(record.referenced_frame_count for record in records)),
        },
        "worlds": world_summaries,
        "traversals": [record.to_dict() for record in records],
    }


__all__ = [
    "DirectedTraversalInventory",
    "causal_anchor_arcs",
    "deduplicated_frame_arcs_and_sequence_indices",
    "enumerate_directed_traversals",
    "inventory_from_registry",
]
