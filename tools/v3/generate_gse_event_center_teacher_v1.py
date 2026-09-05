#!/usr/bin/env python3
"""Generate the immutable C01--C08 signed event-center offset teacher."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from mtare_topo.teacher.gse_event_center_teacher import event_center_targets


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher", required=True, type=Path)
    parser.add_argument("--pair-cache", required=True, type=Path)
    parser.add_argument("--parent-manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    started = time.monotonic()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    rows = _read_jsonl(args.teacher.resolve())
    if len(rows) != 188_126:
        raise RuntimeError("event-center Teacher observation count drift")
    global_index = np.asarray([int(value["global_sequence_index"]) for value in rows])
    parent = np.asarray([str(value["parent_id"]) for value in rows])
    with np.load(args.pair_cache.resolve(), allow_pickle=False) as archive:
        if not np.array_equal(archive["compact_to_global_sequence_index"], global_index):
            raise RuntimeError("event-center pair-cache identity drift")
        sensor_xyz = archive["sensor_xyz_m"].astype(np.float64)
        partition = archive["partition_code"].astype(np.uint8)
    manifest = json.loads(args.parent_manifest.resolve().read_text(encoding="utf-8"))
    registry = {str(value["parent_id"]): value for value in manifest["parents"]}
    worlds = sorted(set(parent.tolist()))
    if len(worlds) != 80 or set(worlds) - set(registry):
        raise RuntimeError("event-center world registry drift")
    node_xyz = {}
    graph_hashes = {}
    for world in worlds:
        graph_path = Path(registry[world]["source_graph"])
        graph = json.loads(graph_path.read_text(encoding="utf-8"))
        node_xyz[world] = {str(value["id"]): value["xyz"] for value in graph["nodes"]}
        graph_hashes[world] = _sha(graph_path)
    target = event_center_targets(rows, sensor_xyz, node_xyz)
    valid = target["valid_mask"]
    fit = (partition == 0) & valid
    selection = (partition == 1) & valid
    identity = np.asarray([str(value["identity"]) for value in rows])
    event = np.asarray([str(value["event"]) for value in rows])
    if (
        int(np.sum(valid)) != 34_133 or int(np.sum(fit)) != 25_294
        or int(np.sum(selection)) != 8_839
        or len(set(identity[fit].tolist())) != 792
        or len(set(identity[selection].tolist())) != 274
    ):
        raise RuntimeError("event-center valid population drift")
    output = args.output_dir / "event_center_teacher.npz"
    np.savez_compressed(
        output, global_sequence_index=global_index, partition_code=partition,
        event=event, identity=identity, **target,
    )
    offset = target["signed_center_offset_m"]
    residual = target["transverse_residual_m"]
    summary = {
        "schema_version": "gse_event_center_teacher_v1",
        "status": "PASS_GSE_EVENT_CENTER_TEACHER_V1",
        "worlds": 80, "causal_observations": len(rows),
        "valid_decision_rows": int(np.sum(valid)),
        "fit_decision_rows": int(np.sum(fit)),
        "selection_decision_rows": int(np.sum(selection)),
        "fit_decision_identities": len(set(identity[fit].tolist())),
        "selection_decision_identities": len(set(identity[selection].tolist())),
        "event_rows": {
            "junction": int(np.sum(valid & (event == "junction"))),
            "terminal": int(np.sum(valid & (event == "terminal"))),
        },
        "signed_offset_m_percentiles": np.percentile(offset[valid], [0, 1, 10, 25, 50, 75, 90, 99, 100]).tolist(),
        "transverse_residual_m_percentiles": np.percentile(residual[valid], [0, 50, 90, 99, 100]).tolist(),
        "teacher_sha256": _sha(args.teacher.resolve()),
        "pair_cache_sha256": _sha(args.pair_cache.resolve()),
        "parent_manifest_sha256": _sha(args.parent_manifest.resolve()),
        "graph_sha256": graph_hashes,
        "teacher_archive_sha256": _sha(output),
        "duration_seconds": time.monotonic() - started,
        "c09_worlds_read": 0, "c10_worlds_read": 0, "mtare_worlds_read": 0,
        "optimizer_steps": 0, "model_inference_frames": 0,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
