"""Compile existing sealed metadata to exact shared encoder export scope.

Reads manifests/seals and historical selection metadata only. Does not read
NPZ payloads, weights or test worlds, and does not create/authorize a run.
"""
import hashlib
import json
from pathlib import Path
from collections import Counter, defaultdict

from mtare_topo.governance_surface_material import read_pinned

INPUT = "results/gate3_semantics/gate3_20260907_gse_surface_input_export_v1r_seed20260906"
SELECTION = "results/gate3_semantics/gate3_20260907_gse_surface_identity_selection_v1_seed20260906"
TRAINING = "results/gate3_semantics/gate3_20260902_primitive_relation_observable_three_seed_training_v1_seed0"
HISTORICAL_SPEC = "configs/v3/gate3/primitive_composition_anchor_model_readiness_v2.json"
INPUT_SEAL = "5e7c150959fb4eebfe837ea05015c0de8aa27b6e8fa03991c15c2e98a323c88b"
SELECTION_SEAL = "0284a664605b0c9fa3e5d8a3a56c4bc90e4b1a78861c16aedc24a4f1941602e2"
CHECKPOINT_SHA = "8d5d2e0ec9779c28b068d0c7db80aeed38ebdb22dff5b781c26234d9001287fb"


def compile_feature_scope(root):
    root = Path(root).resolve(strict=True)
    opened = {}

    def read(path, expected=None):
        # Unpinned historical metadata is recorded, not treated as immutable
        # authority. The checkpoint digest/epoch are separately fixed below.
        if expected is None:
            target = root / path
            if target.resolve(strict=True) != target:
                raise ValueError("metadata path escape")
            raw = target.read_bytes()
        else:
            raw = read_pinned(root, path, expected)
        opened[path] = hashlib.sha256(raw).hexdigest()
        return raw

    def seal(run, expected):
        raw = read(run + "/artifacts/evidence_sha256.txt", expected)
        result = {}
        for line in raw.decode().splitlines():
            digest, path = line.split("  ", 1)
            if path in result:
                raise ValueError("duplicate sealed path")
            result[path] = digest
        return result

    input_seal = seal(INPUT, INPUT_SEAL)
    selection_seal = seal(SELECTION, SELECTION_SEAL)
    ip, sp = INPUT + "/artifacts/input_manifest.json", SELECTION + "/artifacts/selection_manifest.json"
    inputs = json.loads(read(ip, input_seal[ip]))
    selection = json.loads(read(sp, selection_seal[sp]))
    if (inputs["schema_version"] != "gse_surface_sensor_inputs_v1" or
            selection["schema_version"] != "gse_surface_identity_selection_v1" or
            inputs["observations"] != selection["observations"] or
            inputs["selection"]["sha256"] != selection_seal[sp]):
        raise ValueError("sealed input/selection mismatch")
    checkpoint_path = TRAINING + "/artifacts/models/seed0/selected.pt"
    old_spec = json.loads(read(HISTORICAL_SPEC))
    if old_spec["frozen_inputs"].get(checkpoint_path) != CHECKPOINT_SHA:
        raise ValueError("historical checkpoint identity drift")
    summary_path = TRAINING + "/metrics/summary.json"
    summary = json.loads(read(summary_path, old_spec["frozen_inputs"][summary_path]))
    seeds = summary["evaluation"]["c07"]["seeds"]
    seed0 = [r for r in seeds if r["seed"] == 0]
    if len(seed0) != 1 or seed0[0]["selected_epoch"] != 2:
        raise ValueError("historical seed0 selected epoch drift")
    shards = {r["task"]: r for r in inputs["task_shards"]}
    grouped = defaultdict(list)
    for source in inputs["observations"]:
        grouped[source["task"]].append(source)
    if len(shards) != 210 or set(shards) != set(grouped):
        raise ValueError("exact210 task shards required")
    tasks, frames, edges, parents, splits = [], set(), set(), {}, Counter()
    for task in sorted(grouped):
        rows = grouped[task]
        shard = shards[task]
        path = INPUT + "/artifacts/inputs/" + task + ".npz"
        if (len(rows) != 16 or shard["path"] != "artifacts/inputs/" + task + ".npz" or
                input_seal[path] != shard["sha256"]):
            raise ValueError("exact shard source binding drift")
        for row in rows:
            parent, variant = task.split("__")
            if (row["parent_id"] != parent or row["variant"] != variant or
                    parent[-3:] not in {"C01", "C02", "C03", "C04", "C05", "C06", "C07"}):
                raise ValueError("development source scope drift")
            if parent in parents and parents[parent] != row["split"]:
                raise ValueError("parent crosses split")
            parents[parent] = row["split"]
            edges.add((parent, row["physical_edge_id"]))
            frames.update((task, f) for f in row["frame_rows"])
            splits[row["split"]] += 1
        tasks.append({"task": task, "input_path": path, "input_sha256": input_seal[path],
                      "observations": [dict(input_row=i, **row) for i, row in enumerate(rows)]})
    if (len(frames), len(edges), len(parents), dict(splits)) != (
            16800, 1120, 70, {"fit": 2880, "calibration": 240, "development": 240}):
        raise ValueError("fixed population drift")
    return {"schema_version": "gse_surface_feature_source_plan_v1", "metadata_sha256": opened,
            "checkpoint": {"path": checkpoint_path, "sha256": CHECKPOINT_SHA, "seed": 0, "epoch": 2},
            "tasks": tasks, "counts": {"parents": 70, "tasks": 210, "physical_edges": 1120,
                "observations": 3360, "source_frames": 16800, "splits": dict(splits)},
            "compact_sensor_context_bytes_float32": 3360 * 900 * 128 * 4,
            "weights_read": False, "scan_payloads_read": False, "labels": 0, "training_steps": 0}
