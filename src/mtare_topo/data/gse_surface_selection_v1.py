"""Pure single-observation selection from the sealed causal identity inventory.

No scan, pose, teacher, model or filesystem access. A physical edge is the
source traversal identity without its d0/d1 suffix, never a tunnel ID.
"""
from collections import defaultdict
import hashlib
import json
import math
import re

SEED = 20260906
VARIANTS = ("ellipse", "rounded_rectangle", "c1_mixed")


def stable_rank(*parts):
    return hashlib.sha256(json.dumps([SEED, *parts], ensure_ascii=True,
                                    separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _ints(values, *, length=None, maximum=None):
    return (type(values) is list and (length is None or len(values) == length)
            and all(type(x) is int and x >= 0 and (maximum is None or x < maximum) for x in values))


def select_parent(report, split, *, edges_per_parent=16):
    """Validate all metadata, then choose a fixed hash-ranked edge/direction.

    Insufficient population is a hard failure; zero-window traversals remain
    counted but are not candidates. The caller separately binds source hashes.
    """
    if type(edges_per_parent) is not int or edges_per_parent < 1:
        raise ValueError("positive non-bool edge quota required")
    if type(report) is not dict or set(report) != {
        "parent_id", "partition", "intervals", "short_intervals", "counts",
        "continuity_requires_source_audit", "metadata_truth_requires_source_reader", "duration_s",
    }:
        raise ValueError("closed identity report schema required; no score/label inputs")
    parent = report["parent_id"]
    if type(parent) is not str or not re.fullmatch(r"S(?:0[1-9]|10)_[a-z0-9_]+_C0[1-7]", parent):
        raise ValueError("C01-C07 parent identity required")
    partition = "c07" if parent.endswith("_C07") else "fit"
    if report["partition"] != partition or split not in (("calibration", "development") if partition == "c07" else ("fit",)):
        raise ValueError("frozen parent split or partition drift")
    if (report["continuity_requires_source_audit"] is not True
            or report["metadata_truth_requires_source_reader"] is not True or report["duration_s"] is not None):
        raise ValueError("identity metadata does not prove physical continuity or duration")
    counts = report["counts"]
    if type(counts) is not dict or type(counts.get("variants")) is not dict or set(counts["variants"]) != set(VARIANTS):
        raise ValueError("exact three variants required")
    for value in counts["variants"].values():
        if (type(value) is not dict or set(value) != {"frames", "sequences"}
                or any(type(value[k]) is not int or value[k] < 0 for k in value)):
            raise ValueError("typed nonnegative full-shard bounds required")
    intervals = report["intervals"]
    if type(intervals) is not list or not intervals:
        raise ValueError("full traversal inventory required")
    by_edge, seen_t = defaultdict(list), set()
    seen_rows = {v: set() for v in VARIANTS}
    seen_ids = {v: set() for v in VARIANTS}
    seen_frames = {v: {} for v in VARIANTS}
    full_sequence_count, zero_windows = 0, 0
    for interval in intervals:
        if type(interval) is not dict or set(interval) != {"traversal_id", "variants"}:
            raise ValueError("closed traversal identity schema required")
        traversal = interval["traversal_id"]
        if (type(traversal) is not str or not traversal.startswith(parent + ":")
                or re.fullmatch(r"[A-Za-z0-9_.:-]+:d[01]", traversal) is None or traversal in seen_t):
            raise ValueError("unique source traversal identity with physical edge and direction required")
        seen_t.add(traversal)
        records = interval["variants"]
        if type(records) is not list or len(records) != 3:
            raise ValueError("three paired variant records required")
        lookup, shared_ids, shared_frames, length = {}, None, None, None
        for record in records:
            if type(record) is not dict or set(record) != {
                "variant", "task", "sequence_rows", "source_sequence_ids", "frame_rows", "decision_arc_m",
            }:
                raise ValueError("closed variant identity schema required")
            v = record["variant"]
            if v not in VARIANTS or v in lookup or record["task"] != parent + "__" + v:
                raise ValueError("variant identity/task mismatch")
            rows, ids, frames, arcs = (record[k] for k in ("sequence_rows", "source_sequence_ids", "frame_rows", "decision_arc_m"))
            if not _ints(rows, maximum=counts["variants"][v]["sequences"]):
                raise ValueError("sequence row bounds/type failure")
            n = len(rows)
            if (not _ints(ids, length=n, maximum=208228) or type(frames) is not list or len(frames) != n
                    or type(arcs) is not list or len(arcs) != n):
                raise ValueError("parallel metadata arrays differ or source identity outside sparse domain")
            if (any(b != a + 1 for a, b in zip(rows, rows[1:]))
                    or any(b != a + 1 for a, b in zip(ids, ids[1:]))
                    or len(set(rows)) != n or seen_rows[v].intersection(rows) or seen_ids[v].intersection(ids)):
                raise ValueError("duplicate/interleaved/gapped identity records")
            seen_rows[v].update(rows); seen_ids[v].update(ids)
            for j, frame in enumerate(frames):
                if (not _ints(frame, length=5, maximum=counts["variants"][v]["frames"])
                        or any(b != a + 1 for a, b in zip(frame, frame[1:]))
                        or (j and frame != [f + 1 for f in frames[j - 1]])):
                    raise ValueError("five causal consecutive rolling frame rows required")
                for f in frame:
                    if f in seen_frames[v] and seen_frames[v][f] != traversal:
                        raise ValueError("source frame crosses physical traversal boundary")
                    seen_frames[v][f] = traversal
            if (any(type(a) is not float or not math.isfinite(a) or a < 0 for a in arcs)
                    or any(b <= a for a, b in zip(arcs, arcs[1:]))):
                raise ValueError("finite ordered measured decision route arcs required")
            if length is None:
                length, shared_ids, shared_frames = n, ids, frames
            elif n != length or ids != shared_ids or frames != shared_frames:
                raise ValueError("paired variant source identities/history differ")
            lookup[v] = record
        if set(lookup) != set(VARIANTS):
            raise ValueError("missing paired variant")
        full_sequence_count += length
        if length:
            by_edge[traversal.rsplit(":", 1)[0]].append((traversal, lookup, length))
        else:
            zero_windows += 1
    if (counts.get("traversals") != len(intervals) or counts.get("logical_sequences") != full_sequence_count
            or any(len(seen_rows[v]) != counts["variants"][v]["sequences"] for v in VARIANTS)):
        raise ValueError("full inventory population mismatch")
    if len(by_edge) < edges_per_parent:
        raise ValueError(f"INSUFFICIENT_DISTINCT_EDGES:{parent}:{len(by_edge)}<{edges_per_parent}")
    ranked = sorted(by_edge, key=lambda edge: (stable_rank("surface_edge_v1", parent, edge), edge))[:edges_per_parent]
    selected = []
    for rank, edge in enumerate(ranked):
        traversal, records, n = min(by_edge[edge], key=lambda item: (stable_rank("surface_direction_v1", parent, item[0]), item[0]))
        mode = ("first", "middle", "last")[rank % 3]
        pos = {"first": 0, "middle": (n - 1) // 2, "last": n - 1}[mode]
        for v in VARIANTS:
            r = records[v]
            selected.append({"parent_id": parent, "split": split, "physical_edge_id": edge,
                "traversal_id": traversal, "variant": v, "task": r["task"], "edge_selection_rank": rank,
                "decision_position_policy": mode, "decision_index_in_traversal": pos,
                "available_decisions": n, "source_sequence_id": r["source_sequence_ids"][pos],
                "sequence_row": r["sequence_rows"][pos], "frame_rows": r["frame_rows"][pos],
                "decision_frame_row": r["frame_rows"][pos][-1], "decision_route_arc_m": r["decision_arc_m"][pos],
                "source_frame_count": counts["variants"][v]["frames"],
                "source_sequence_count": counts["variants"][v]["sequences"],
                "within_traversal_decision_spacing_m": [b - a for a, b in zip(r["decision_arc_m"], r["decision_arc_m"][1:])],
                "history_frame_spacing_m": None, "duration_s": None,
                "continuous_route_evidence": False, "labels_generated": False})
    return {"parent_id": parent, "split": split, "available_physical_edges": len(by_edge),
            "zero_window_traversals": zero_windows, "selected_physical_edges": len(ranked), "observations": selected}


def summarize_selection(parents):
    rows = [row for p in parents for row in p["observations"]]
    frames = {(r["task"], f) for r in rows for f in r["frame_rows"]}
    edges = {(r["parent_id"], r["physical_edge_id"]) for r in rows}
    return {"parents": len(parents), "physical_edge_units": len(edges), "observations": len(rows),
            "tasks": len({r["task"] for r in rows}), "unique_variant_source_frames": len(frames),
            "split_observations": {s: sum(r["split"] == s for r in rows) for s in ("fit", "calibration", "development")},
            "source_frame_references": 5 * len(rows), "independent_structure_count": None,
            "label_counts": None, "scan_frames_decoded": 0, "optimizer_steps": 0,
            "scientific_gate_pass": False, "training_eligible": False}
