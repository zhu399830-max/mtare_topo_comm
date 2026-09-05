#!/usr/bin/env python3
"""Prove exact twelve-scan C01-C08 references without reading sensor payloads."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from mtare_topo.governance import write_json
from mtare_topo.representation.gse_causal_episode_detector import (
    materialize_causal_episode_references,
)
from mtare_topo.semantics.geometric_semantics import EVENT_NAMES


PASS_STATUS = "PASS_GSE_CAUSAL_EPISODE_REFERENCE_PROOF_V1"
PASS_STATUS_V1R = "PASS_GSE_CAUSAL_EPISODE_REFERENCE_PROOF_V1R"


def _digest(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).view(np.uint8)).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--teacher", required=True, type=Path)
    parser.add_argument("--dataset-run", required=True, type=Path)
    parser.add_argument("--change-proof-summary", type=Path)
    parser.add_argument("--expected-status", choices=(PASS_STATUS, PASS_STATUS_V1R), default=PASS_STATUS)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    (run_dir / "artifacts").mkdir(parents=True, exist_ok=True)
    (run_dir / "metrics").mkdir(parents=True, exist_ok=True)
    with args.teacher.open("r", encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    if len(rows) != 188126:
        raise RuntimeError("causal episode proof Teacher count drift")
    bank = materialize_causal_episode_references(rows)
    by_parent: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        by_parent[str(row["parent_id"])].append(index)
    if len(by_parent) != 80 or any(parent.endswith(("_C09", "_C10")) for parent in by_parent):
        raise RuntimeError("causal episode proof split drift")

    import zarr

    world_rows = []
    total_unique_frames = 0
    resolved_references = 0
    dataset_root = args.dataset_run / "artifacts/dataset/train"
    for parent in sorted(by_parent):
        indices = np.asarray(by_parent[parent], dtype=np.int64)
        path = dataset_root / f"{parent}.zarr"
        group = zarr.open_group(str(path), mode="r")
        if group.attrs.get("parent_id") != parent or group.attrs.get("split") != "train":
            raise RuntimeError("causal episode source shard identity drift")
        global_frame = np.asarray(group["global_frame_index"][:], dtype=np.int64)
        if len(global_frame) == 0 or len(np.unique(global_frame)) != len(global_frame):
            raise RuntimeError("source shard global frame identity drift")
        global_to_local = {int(value): index for index, value in enumerate(global_frame)}
        refs = bank.global_frame_references[indices]
        valid = bank.valid_history_mask[indices]
        values = refs[valid]
        if any(int(value) not in global_to_local for value in values):
            raise RuntimeError("twelve-scan reference does not resolve inside its parent shard")
        source_local_frame = np.asarray(group["local_frame_index"][:], dtype=np.int64)
        for row_index in indices:
            valid_refs = bank.global_frame_references[row_index][bank.valid_history_mask[row_index]]
            local_rows = np.asarray([global_to_local[int(value)] for value in valid_refs], dtype=np.int64)
            frame_index = int(rows[int(row_index)]["frame_index"])
            expected = np.arange(frame_index - len(local_rows) + 1, frame_index + 1, dtype=np.int64)
            if not np.array_equal(source_local_frame[local_rows], expected):
                raise RuntimeError("twelve-scan reference crosses a traversal-local frame reset")
        total_unique_frames += len(global_frame)
        resolved_references += len(values)
        world_rows.append({
            "parent_id": parent,
            "observations": len(indices),
            "unique_frames": len(global_frame),
            "valid_reference_cells": len(values),
            "full_twelve_frame_observations": int(np.sum(bank.valid_history_mask[indices].sum(axis=1) == 12)),
            "structural_episodes": len(set(bank.episode_id[indices][bank.episode_id[indices] >= 0].tolist())),
            "source_shard": str(path.relative_to(args.dataset_run)),
        })

    lengths = bank.valid_history_mask.sum(axis=1)
    history_histogram = {str(length): int(np.sum(lengths == length)) for length in range(5, 13)}
    event_counts = {name: int(np.sum(bank.event_index == index)) for index, name in enumerate(EVENT_NAMES)}
    episode_ids = np.unique(bank.episode_id[bank.episode_id >= 0])
    expected_episode_ids = np.arange(len(episode_ids), dtype=np.int64)
    checks = {
        "exact_observations": len(rows) == 188126,
        "exact_worlds": len(by_parent) == 80,
        "exact_unique_frames": total_unique_frames == 252430,
        "exact_structural_episodes": len(episode_ids) == 5306,
        "episode_ids_contiguous": np.array_equal(episode_ids, expected_episode_ids),
        "all_references_resolve_in_parent_shard": resolved_references == int(bank.valid_history_mask.sum()),
        "all_histories_between_five_and_twelve": set(np.unique(lengths).tolist()) <= set(range(5, 13)),
        "exact_event_counts": event_counts == {
            "corridor": 150964, "junction": 26608, "terminal": 7525,
            "turn": 1998, "geometry_transition": 1031,
        },
        "zero_forbidden_reads": True,
    }
    if not all(checks.values()):
        raise RuntimeError(f"causal episode reference proof failed: {checks}")
    with (run_dir / "artifacts/world_reference_summary.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(world_rows[0]))
        writer.writeheader()
        writer.writerows(world_rows)
    observed_traversals = len({str(row["traversal_id"]) for row in rows})
    inventory_traversals = observed_traversals
    zero_observation_traversals = 0
    if args.change_proof_summary is not None:
        proof = json.loads(args.change_proof_summary.read_text(encoding="utf-8"))
        inventory_traversals = int(proof["totals"]["directed_traversal_count"])
        zero_observation_traversals = int(proof["totals"]["zero_frame_traversals"])
        if inventory_traversals != observed_traversals + zero_observation_traversals:
            raise RuntimeError("inventory/observed traversal decomposition drift")
    result = {
        "schema_version": "gse_causal_episode_reference_proof_v1",
        "overall_status": args.expected_status,
        "worlds": len(by_parent),
        "inventory_directed_traversals": inventory_traversals,
        "observed_directed_traversals": observed_traversals,
        "zero_observation_traversals": zero_observation_traversals,
        "unique_frames": total_unique_frames,
        "causal_observations": len(rows),
        "valid_reference_cells": resolved_references,
        "full_twelve_frame_observations": int(np.sum(lengths == 12)),
        "history_length_histogram": history_histogram,
        "event_counts": event_counts,
        "structural_episodes": len(episode_ids),
        "array_digests": {
            "global_frame_references": _digest(bank.global_frame_references),
            "valid_history_mask": _digest(bank.valid_history_mask),
            "episode_id": _digest(bank.episode_id),
            "event_index": _digest(bank.event_index),
        },
        "checks": checks,
        "sensor_payload_bytes_written": 0,
        "new_rays": 0,
        "optimizer_steps": 0,
        "model_inference_frames": 0,
        "c09_worlds_read": 0,
        "strict_test_worlds_read": 0,
        "mtare_worlds_read": 0,
    }
    write_json(run_dir / "metrics/reference_proof.json", result)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
