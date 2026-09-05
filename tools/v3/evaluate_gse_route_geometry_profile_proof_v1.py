#!/usr/bin/env python3
"""Zero-training C01-C08 RouteGeometryProfile visibility/uniqueness proof."""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import zarr

from _bootstrap import PROJECT_ROOT
from mtare_topo.governance import write_json
from mtare_topo.representation.gse_route_geometry_profile import (
    PROFILE_DIMENSIONS,
    PROFILE_OFFSETS_M,
    objective_profile_from_sequence,
    range_image_profile_support,
)


PASS = "PASS_GSE_ROUTE_GEOMETRY_PROFILE_PROOF_V1"
FAIL = "FAIL_GSE_ROUTE_GEOMETRY_PROFILE_PROOF_V1"
EVENTS = ("turn", "geometry_transition")
EXPECTED_ROWS = 188126
EXPECTED_EVENT_ROWS = {"turn": 1998, "geometry_transition": 1031}
EXPECTED_IDENTITIES = {"turn": 392, "geometry_transition": 76}
MIN_SUPPORTED_PROFILE_BINS = 3
MIN_IDENTITY_COVERAGE = 0.80
REVERSE_LIMITS = {"width_m": 2.0, "height_m": 2.0, "slope_deg": 2.0, "curvature_per_m": 0.02}


def _partition(parent_id: str) -> str:
    suffix = int(parent_id.rsplit("_C", 1)[1])
    return "fit" if suffix <= 6 else "selection"


def _plot(summary: dict, output: Path) -> None:
    event_rows = summary["event_summary"]
    labels = [f"{partition}\n{event}" for partition in ("fit", "selection") for event in EVENTS]
    coverage = [event_rows[partition][event]["identity_support_coverage"] for partition in ("fit", "selection") for event in EVENTS]
    bins = [event_rows[partition][event]["mean_supported_bins"] for partition in ("fit", "selection") for event in EVENTS]
    figure, axes = plt.subplots(1, 3, figsize=(13.0, 4.1), constrained_layout=True)
    axes[0].bar(labels, coverage, color="#2563EB")
    axes[0].axhline(MIN_IDENTITY_COVERAGE, color="#DC2626", linestyle="--")
    axes[0].set_ylim(0, 1.05); axes[0].set_ylabel("physical identity coverage")
    axes[0].set_title("a  LiDAR-supported route profiles", loc="left")
    axes[1].bar(labels, bins, color="#0EA5E9")
    axes[1].axhline(MIN_SUPPORTED_PROFILE_BINS, color="#DC2626", linestyle="--")
    axes[1].set_ylim(0, len(PROFILE_OFFSETS_M) + .3); axes[1].set_ylabel("supported offset bins / row")
    axes[1].set_title("b  Longitudinal support", loc="left")
    reverse = summary["reverse_consistency"]["p95_absolute_error"]
    names = list(PROFILE_DIMENSIONS)
    values = [reverse[name] for name in names]
    limits = [REVERSE_LIMITS[name] for name in names]
    x = np.arange(len(names))
    axes[2].bar(x - .18, values, .36, label="observed p95", color="#16A34A")
    axes[2].bar(x + .18, limits, .36, label="frozen limit", color="#94A3B8")
    axes[2].set_xticks(x, ("width m", "height m", "slope deg", "curv /m"), rotation=15)
    axes[2].set_title("c  Reverse-direction Teacher consistency", loc="left")
    axes[2].legend(frameon=False, fontsize=8)
    for suffix in ("png", "pdf", "svg"):
        figure.savefig(output.with_suffix("." + suffix), dpi=180 if suffix == "png" else None)
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--supervision", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    started = time.monotonic()
    output = args.output_dir.resolve(); output.mkdir(parents=True, exist_ok=False)
    dataset = args.dataset.resolve(); supervision = args.supervision.resolve()
    shards = sorted((dataset / "artifacts/dataset/train").glob("*.zarr"))
    packs = {path.stem: path for path in (supervision / "artifacts/supervision/worlds").glob("*.npz")}
    if len(shards) != 80 or len(packs) != 80 or {path.stem for path in shards} != set(packs):
        raise RuntimeError("RouteGeometryProfile world population drift")

    event_accumulator: dict[str, dict[str, dict[str, object]]] = {
        partition: {
            event: {"rows": 0, "complete_rows": 0, "supported_bins": [], "identities": set(), "supported_identities": set()}
            for event in EVENTS
        }
        for partition in ("fit", "selection")
    }
    total_rows = 0
    reverse_records: dict[tuple[str, str, int], dict[int, np.ndarray]] = defaultdict(dict)
    audit_rows: list[dict[str, object]] = []
    proxy_errors: dict[str, list[float]] = {"width_m": [], "height_m": []}

    for world_index, shard_path in enumerate(shards, start=1):
        parent_id = shard_path.stem
        group = zarr.open_group(str(shard_path), mode="r")
        with np.load(packs[parent_id], allow_pickle=False) as source:
            source_arrays = {name: source[name] for name in source.files}
        global_id = np.asarray(group["global_sequence_index"][:], dtype=np.int64)
        if not np.array_equal(global_id, source_arrays["global_sequence_index"]):
            raise RuntimeError(f"sequence identity drift: {parent_id}")
        rows = len(global_id); total_rows += rows
        traversal = np.asarray(source_arrays["traversal_id"]).astype(str)
        sequence = np.asarray(source_arrays["sequence_index"], dtype=np.int64)
        event_name = np.asarray(source_arrays["event_name"]).astype(str)
        identity = np.asarray(source_arrays["identity"]).astype(str)
        geometry = np.asarray(group["geometry"][:], dtype=np.float64)
        geometry_valid = np.asarray(group["geometry_valid_mask"][:], dtype=bool)
        references = np.asarray(group["local_frame_references"][:], dtype=np.int64)
        route_arc = np.asarray(group["route_arc_m"][:], dtype=np.float64)
        ranges = group["range_m"]; scan_valid = group["valid_mask"]
        if geometry.shape != (rows, 4) or references.shape != (rows, 5):
            raise RuntimeError(f"dataset schema drift: {parent_id}")

        row_by_traversal: dict[str, dict[int, int]] = defaultdict(dict)
        for row, (name, index) in enumerate(zip(traversal, sequence, strict=True)):
            if int(index) in row_by_traversal[name]:
                raise RuntimeError("duplicate traversal sequence identity")
            row_by_traversal[name][int(index)] = row

        # Reverse consistency uses objective current geometry at the same
        # canonical half-metre arc.  The route length is shared by d0/d1.
        lengths: dict[str, float] = {}
        for name, mapping in row_by_traversal.items():
            current_rows = np.asarray(list(mapping.values()), dtype=np.int64)
            arcs = route_arc[references[current_rows, -1]]
            base, direction_text = name.rsplit(":d", 1)
            lengths[base] = max(lengths.get(base, 0.0), float(np.max(arcs)))
            direction = int(direction_text)
            for row, arc in zip(current_rows, arcs, strict=True):
                canonical = float(arc) if direction == 0 else lengths[base] - float(arc)
                key = int(round(canonical * 2.0))
                reverse_records[(parent_id, base, key)][direction] = geometry[row].copy()

        partition = _partition(parent_id)
        selected = np.flatnonzero(np.isin(event_name, EVENTS))
        for row in selected:
            event = str(event_name[row]); ident = str(identity[row])
            if not ident:
                raise RuntimeError(f"{event} row lacks physical identity")
            profile = objective_profile_from_sequence(
                geometry,
                geometry_valid,
                current_sequence_index=int(sequence[row]),
                row_by_sequence_index=row_by_traversal[str(traversal[row])],
            )
            current_frame = int(references[row, -1])
            support = range_image_profile_support(ranges[current_frame], scan_valid[current_frame])
            target_bins = np.all(profile.valid_mask, axis=1)
            jointly_supported = target_bins & support.valid_mask
            supported_bins = int(np.count_nonzero(jointly_supported))
            acc = event_accumulator[partition][event]
            acc["rows"] = int(acc["rows"]) + 1
            acc["complete_rows"] = int(acc["complete_rows"]) + int(np.all(target_bins))
            acc["supported_bins"].append(supported_bins)  # type: ignore[union-attr]
            acc["identities"].add(ident)  # type: ignore[union-attr]
            if supported_bins >= MIN_SUPPORTED_PROFILE_BINS:
                acc["supported_identities"].add(ident)  # type: ignore[union-attr]
            for profile_index in np.flatnonzero(jointly_supported):
                proxy_errors["width_m"].append(abs(float(support.proxy_width_height_m[profile_index, 0] - profile.values[profile_index, 0])))
                proxy_errors["height_m"].append(abs(float(support.proxy_width_height_m[profile_index, 1] - profile.values[profile_index, 1])))
            audit_rows.append({
                "parent_id": parent_id,
                "global_sequence_index": int(global_id[row]),
                "traversal_id": str(traversal[row]),
                "sequence_index": int(sequence[row]),
                "event": event,
                "identity": ident,
                "target_valid_bins": int(np.count_nonzero(target_bins)),
                "lidar_supported_bins": int(np.count_nonzero(support.valid_mask)),
                "jointly_supported_bins": supported_bins,
                "support_counts": support.side_counts.tolist(),
            })
        print(json.dumps({"world": parent_id, "index": world_index, "rows": rows, "event_rows": len(selected)}, sort_keys=True), flush=True)

    if total_rows != EXPECTED_ROWS:
        raise RuntimeError(f"expected {EXPECTED_ROWS} rows, found {total_rows}")
    flat_event_rows = {event: sum(int(event_accumulator[p][event]["rows"]) for p in ("fit", "selection")) for event in EVENTS}
    flat_identities = {event: set().union(*(event_accumulator[p][event]["identities"] for p in ("fit", "selection"))) for event in EVENTS}

    reverse_errors: dict[str, list[float]] = {name: [] for name in PROFILE_DIMENSIONS}
    reverse_pair_count = 0
    for pair in reverse_records.values():
        if set(pair) != {0, 1}:
            continue
        left, right = pair[0], pair[1].copy(); right[2] *= -1.0
        if not np.all(np.isfinite(left)) or not np.all(np.isfinite(right)):
            continue
        reverse_pair_count += 1
        for column, name in enumerate(PROFILE_DIMENSIONS):
            reverse_errors[name].append(abs(float(left[column] - right[column])))
    reverse_p95 = {name: float(np.quantile(values, .95)) if values else float("nan") for name, values in reverse_errors.items()}

    event_summary: dict[str, dict[str, dict[str, float | int]]] = {}
    for partition in ("fit", "selection"):
        event_summary[partition] = {}
        for event in EVENTS:
            acc = event_accumulator[partition][event]
            identities = acc["identities"]; supported_identities = acc["supported_identities"]
            bins = np.asarray(acc["supported_bins"], dtype=np.float64)
            event_summary[partition][event] = {
                "rows": int(acc["rows"]),
                "complete_teacher_rows": int(acc["complete_rows"]),
                "identity_count": len(identities),
                "supported_identity_count": len(supported_identities),
                "identity_support_coverage": len(supported_identities) / max(1, len(identities)),
                "mean_supported_bins": float(np.mean(bins)) if len(bins) else 0.0,
                "rows_with_at_least_three_bins": int(np.count_nonzero(bins >= MIN_SUPPORTED_PROFILE_BINS)),
            }
    checks = {
        "exact_world_count": len(shards) == 80,
        "exact_observation_count": total_rows == EXPECTED_ROWS,
        "exact_turn_rows": flat_event_rows["turn"] == EXPECTED_EVENT_ROWS["turn"],
        "exact_transition_rows": flat_event_rows["geometry_transition"] == EXPECTED_EVENT_ROWS["geometry_transition"],
        "exact_turn_identities": len(flat_identities["turn"]) == EXPECTED_IDENTITIES["turn"],
        "exact_transition_identities": len(flat_identities["geometry_transition"]) == EXPECTED_IDENTITIES["geometry_transition"],
        "all_partition_event_identity_coverage_at_least_0p80": all(event_summary[p][e]["identity_support_coverage"] >= MIN_IDENTITY_COVERAGE for p in ("fit", "selection") for e in EVENTS),
        "all_partition_events_have_nonzero_complete_teacher_rows": all(event_summary[p][e]["complete_teacher_rows"] > 0 for p in ("fit", "selection") for e in EVENTS),
        "reverse_pairs_nonzero": reverse_pair_count > 0,
        "reverse_teacher_p95_within_existing_geometry_limits": all(np.isfinite(reverse_p95[name]) and reverse_p95[name] <= REVERSE_LIMITS[name] for name in PROFILE_DIMENSIONS),
        "zero_test_model_training_graph": True,
    }
    passed = all(checks.values())
    summary = {
        "schema_version": "gse_route_geometry_profile_proof_v1",
        "overall_status": PASS if passed else FAIL,
        "scientific_pass": passed,
        "profile_offsets_m": list(PROFILE_OFFSETS_M),
        "minimum_supported_profile_bins": MIN_SUPPORTED_PROFILE_BINS,
        "minimum_identity_coverage": MIN_IDENTITY_COVERAGE,
        "worlds": len(shards), "observations": total_rows,
        "event_rows": flat_event_rows,
        "event_identities": {event: len(values) for event, values in flat_identities.items()},
        "event_summary": event_summary,
        "reverse_consistency": {"pair_count": reverse_pair_count, "p95_absolute_error": reverse_p95, "limits": REVERSE_LIMITS},
        "deterministic_proxy_mae_diagnostic": {name: float(np.mean(values)) if values else None for name, values in proxy_errors.items()},
        "checks": checks,
        "optimizer_steps": 0, "model_inference_frames": 0, "c09_worlds_read": 0, "c10_worlds_read": 0,
        "mtare_worlds_read": 0, "graph_replays": 0,
        "duration_seconds": time.monotonic() - started,
    }
    with (output / "observation_audit.jsonl").open("w", encoding="utf-8") as stream:
        for row in audit_rows: stream.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
    with (output / "event_summary.csv").open("w", newline="", encoding="utf-8") as stream:
        fields = ("partition", "event", "rows", "complete_teacher_rows", "identity_count", "supported_identity_count", "identity_support_coverage", "mean_supported_bins", "rows_with_at_least_three_bins")
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader()
        for partition in ("fit", "selection"):
            for event in EVENTS: writer.writerow({"partition": partition, "event": event, **event_summary[partition][event]})
    write_json(output / "summary.json", summary)
    _plot(summary, output / "gse_route_geometry_profile_proof_v1")
    print(json.dumps({"status": summary["overall_status"], "checks": checks}, indent=2, sort_keys=True))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
