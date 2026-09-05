#!/usr/bin/env python3
"""Formal C07 non-learning baseline on observable endpoint relations."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import resource
import subprocess
import sys
import time
import traceback

import numpy as np
import scipy
import torch
import zarr

from _bootstrap import PROJECT_ROOT
from mtare_topo.data.primitive_relation_observable_batches import (
    ObservablePrimitiveRelationBatchLoader,
)
from mtare_topo.evaluation.primitive_relation_metrics import (
    BinaryCounts, THRESHOLD_GRID, align_for_evaluation, existence_sweep,
    finalize_geometry_totals, geometry_batch_totals, merge_geometry_totals,
    nonlearning_batch_to_torch, relation_sweeps, temporal_batch_metrics,
)
from mtare_topo.governance import load_json, validate_data_card, write_json
from mtare_topo.representation.primitive_relation_observable_training import (
    observable_numpy_batch_to_torch,
)
from mtare_topo.semantics.primitive_relation_nonlearning import (
    RobustPrimitiveRelationBaseline,
)
from run_primitive_relation_nonlearning_c07_v1 import _plot
from run_primitive_relation_nonlearning_readiness_v1 import _seal, _sha, _tree_hash


RUN_ID = "gate3_20260902_primitive_relation_observable_nonlearning_c07_v1_seed0"
PASS = "PASS_PRIMITIVE_RELATION_OBSERVABLE_NONLEARNING_C07_V1"
FAIL = "FAIL_PRIMITIVE_RELATION_OBSERVABLE_NONLEARNING_C07_V1"
CARD_STATUS = "APPROVED_FOR_ONE_IMMUTABLE_PRIMITIVE_RELATION_OBSERVABLE_NONLEARNING_C07_V1"
PYTHON = "/home/zeng-workstation/.local/share/mtare_topo_comm/envs/phase3_torch290_cu129_zarr2187_v1/bin/python"
P1A = PROJECT_ROOT / "results/gate3_semantics/gate3_20260830_primitive_relation_p1a_lossless_storage_corrective_v1_seed0"
P1B = PROJECT_ROOT / "results/gate3_semantics/gate3_20260830_primitive_relation_p1b_teacher_materialization_v1r_seed0"
SIDECAR = PROJECT_ROOT / "results/gate3_semantics/gate3_20260902_primitive_attachment_observability_sidecar_v1_seed0"
READINESS = PROJECT_ROOT / "results/gate3_semantics/gate3_20260902_primitive_relation_observable_readiness_v1r_seed0"
EXPECTED_ROWS = 64_644
EXPECTED_TASKS = 30
EXPECTED_POSITIVE_ATTACHMENTS = 442_936
BATCH_SIZE = 8


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    args = parser.parse_args()
    spec = load_json(args.spec.resolve())
    run = args.run_dir.resolve()
    started = time.monotonic()
    overall, error = FAIL, None
    before: dict[str, str] = {}
    try:
        if (
            run.name != RUN_ID
            or load_json(run / "RUN_STATE.json").get("state")
            != "CREATED_NOT_EXECUTED"
        ):
            raise RuntimeError("observable non-learning C07 executes exactly once")
        card = load_json(PROJECT_ROOT / spec["data_card"])
        validation = validate_data_card(card)
        if not validation.passed or card.get("status") != CARD_STATUS:
            raise RuntimeError(f"observable baseline Data Card invalid: {validation.errors}")
        for relative, expected in spec["frozen_inputs"].items():
            before[relative] = _sha(PROJECT_ROOT / relative)
            if before[relative] != expected:
                raise RuntimeError(f"observable baseline input drift: {relative}")
        for record in spec["frozen_tools"].values():
            if _sha(PROJECT_ROOT / record["path"]) != record["sha256"]:
                raise RuntimeError(f"observable baseline tool drift: {record['path']}")
        readiness = load_json(READINESS / "metrics/summary.json")
        if (
            not readiness.get("scientific_pass")
            or readiness.get("decision")
            != "ALLOW_OBSERVABLE_RELATION_THREE_SEED_TRAINING_SPEC"
        ):
            raise RuntimeError("observable relation readiness prerequisite drift")
        environment = {
            "python": sys.version.split()[0], "executable": sys.executable,
            "numpy": np.__version__, "scipy": scipy.__version__,
            "torch": torch.__version__, "zarr": zarr.__version__,
        }
        expected_environment = {
            "python": "3.13.5", "executable": PYTHON,
            "numpy": "2.1.3", "scipy": "1.15.3",
            "torch": "2.9.0+cu129", "zarr": "2.18.7",
        }
        if environment != expected_environment:
            raise RuntimeError(f"observable baseline environment drift: {environment}")
        write_json(run / "config/source_integrity_before.json", before)
        write_json(run / "config/environment.json", {
            "versions": environment, "platform": platform.platform(),
            "device": "cpu",
        })
        write_json(run / "RUN_STATE.json", {
            "schema_version": "v3_run_state_v1", "run_id": RUN_ID,
            "state": "RUNNING",
        })
        tests = subprocess.run(
            [
                sys.executable, "-m", "pytest", "-q",
                "tests/v3/unit/test_primitive_relation_nonlearning.py",
                "tests/v3/unit/test_primitive_relation_metrics.py",
                "tests/v3/unit/test_primitive_relation_observable_batches.py",
            ],
            cwd=PROJECT_ROOT,
            env={**os.environ, "PYTHONPATH": f"{PROJECT_ROOT / 'src'}:{PROJECT_ROOT}"},
            text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            timeout=300, check=False,
        )
        (run / "logs/unit_tests.log").write_text(tests.stdout, encoding="utf-8")
        if tests.returncode or "17 passed" not in tests.stdout:
            raise RuntimeError("expected exactly 17 observable baseline tests")

        sensor_manifest = {
            value["task_id"]: value
            for value in load_json(P1A / "artifacts/task_manifest.json")["tasks"]
        }
        teacher_manifest = {
            value["task_id"]: value
            for value in load_json(P1B / "artifacts/task_manifest.json")["tasks"]
        }
        sidecar_manifest = {
            value["task_id"]: value
            for value in load_json(SIDECAR / "artifacts/task_manifest.json")["tasks"]
        }
        sensor_root = P1A / "artifacts/dataset/c07"
        teacher_root = P1B / "artifacts/teacher/c07"
        sidecar_root = SIDECAR / "artifacts/endpoint_observability/c07"
        loader = ObservablePrimitiveRelationBatchLoader(
            sensor_root, teacher_root, sidecar_root,
        )
        if len(loader) != EXPECTED_ROWS or len(loader.task_names) != EXPECTED_TASKS:
            raise RuntimeError("observable baseline C07 population drift")
        if any("_C07__" not in name for name in loader.task_names):
            raise RuntimeError("observable baseline C07 task split drift")

        threshold_index = int(np.argmin(np.abs(THRESHOLD_GRID - 0.5)))
        primitive = BinaryCounts()
        attachment = BinaryCounts()
        overlap = BinaryCounts()
        temporal_presence = BinaryCounts()
        temporal_correct = temporal_total = 0
        geometry = None
        method = RobustPrimitiveRelationBaseline()
        task_summaries = []
        rows = predictions = 0
        for task_position, name in enumerate(loader.task_names):
            task_id = name.removesuffix(".zarr")
            sensor_path = sensor_root / name
            teacher_path = teacher_root / name
            sidecar_path = sidecar_root / name
            hashes = {
                "sensor_tree_sha256": _tree_hash(sensor_path),
                "teacher_tree_sha256": _tree_hash(teacher_path),
                "sidecar_tree_sha256": _tree_hash(sidecar_path),
            }
            expected = {
                "sensor_tree_sha256": sensor_manifest[task_id]["corrected_shard_tree_sha256"],
                "teacher_tree_sha256": teacher_manifest[task_id]["shard_tree_sha256"],
                "sidecar_tree_sha256": sidecar_manifest[task_id]["sidecar_tree_sha256"],
            }
            if hashes != expected:
                raise RuntimeError(f"observable baseline C07 shard drift: {task_id}")
            task_started = time.monotonic()
            task_rows = task_predictions = 0
            length = loader._lengths[name]
            for start in range(0, length, BATCH_SIZE):
                indices = np.arange(
                    start, min(start + BATCH_SIZE, length), dtype=np.int64,
                )
                numpy_batch = loader._read(name, indices)
                hard = [
                    method.predict(
                        numpy_batch.base.range_valid[index],
                        numpy_batch.base.relative_translation_current_sensor_m[index],
                        numpy_batch.base.relative_yaw_current_sensor_deg[index],
                    )
                    for index in range(len(indices))
                ]
                prediction = nonlearning_batch_to_torch(
                    hard, device=torch.device("cpu"),
                )
                target_batch = observable_numpy_batch_to_torch(
                    numpy_batch, device=torch.device("cpu"),
                )
                aligned = align_for_evaluation(prediction, target_batch.targets)
                primitive += existence_sweep(prediction, aligned)[threshold_index]
                relation = relation_sweeps(
                    prediction, aligned, existence_threshold=.5,
                )
                attachment += relation["attachment"][threshold_index]
                overlap += relation["disconnected_overlap"][threshold_index]
                temporal = temporal_batch_metrics(
                    prediction, aligned, existence_threshold=.5,
                )
                temporal_presence += temporal["presence"]
                temporal_correct += int(temporal["correspondence_correct"])
                temporal_total += int(temporal["correspondence_total"])
                geometry = merge_geometry_totals(
                    geometry,
                    geometry_batch_totals(
                        prediction, aligned, existence_threshold=.5,
                    ),
                )
                batch_predictions = sum(
                    int(value.primitive_mask.sum()) for value in hard
                )
                rows += len(indices)
                predictions += batch_predictions
                task_rows += len(indices)
                task_predictions += batch_predictions
            task_summary = {
                "task_id": task_id, "rows": task_rows,
                "predicted_primitives": task_predictions,
                "seconds": time.monotonic() - task_started, **hashes,
            }
            task_summaries.append(task_summary)
            write_json(
                run / "metrics" / f"task_{task_position:02d}.json",
                task_summary,
            )
            print(json.dumps({
                "completed_tasks": task_position + 1, **task_summary,
            }), flush=True)
        if rows != EXPECTED_ROWS or geometry is None:
            raise RuntimeError(f"observable baseline completed row drift: {rows}")
        if attachment.true_positive + attachment.false_negative != EXPECTED_POSITIVE_ATTACHMENTS:
            raise RuntimeError("observable attachment target population drift")
        payload = {
            "schema_version": "primitive_relation_observable_nonlearning_c07_v1",
            "overall_status": PASS, "scientific_pass": True,
            "partition": "c07", "independent_parent_worlds": 10,
            "paired_geometry_tasks": EXPECTED_TASKS, "rows": rows,
            "source_independent_sequences": 21_548,
            "causal_frame_references": rows * 5,
            "config": method.config.to_dict(),
            "exit_config": method.exit_baseline.config.to_dict(),
            "primitive_detection": primitive.to_dict(),
            "attachment": attachment.to_dict(),
            "observable_attachment_target_positives": EXPECTED_POSITIVE_ATTACHMENTS,
            "hidden_attachment_targets_excluded": 82_141,
            "disconnected_overlap": overlap.to_dict(),
            "temporal_presence": temporal_presence.to_dict(),
            "temporal_correspondence_accuracy": temporal_correct / temporal_total,
            "temporal_correspondence_correct": temporal_correct,
            "temporal_correspondence_total": temporal_total,
            "geometry": finalize_geometry_totals(geometry),
            "predicted_primitives": predictions,
            "task_summaries": task_summaries,
            "optimizer_steps": 0, "model_inference_frames": 0,
            "checkpoint_writes": 0, "c08_rows_read": 0,
            "c09_c10_worlds_read": 0, "graph_replays": 0,
            "mtare_worlds_read": 0,
            "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "duration_seconds": time.monotonic() - started, "error": None,
            "decision": "FREEZE_OBSERVABLE_NONLEARNING_C07_BASELINE_FOR_P2",
        }
        _plot(payload, run / "previews/observable_nonlearning_c07_baseline")
        write_json(run / "metrics/summary.json", payload)
        after = {relative: _sha(PROJECT_ROOT / relative) for relative in before}
        if after != before:
            raise RuntimeError("observable baseline frozen sources changed")
        write_json(run / "config/source_integrity_after.json", after)
        overall = PASS
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        (run / "logs/failure_traceback.log").write_text(
            traceback.format_exc(), encoding="utf-8",
        )
        write_json(run / "metrics/summary.json", {
            "schema_version": "primitive_relation_observable_nonlearning_c07_v1",
            "overall_status": FAIL, "scientific_pass": False, "error": error,
            "optimizer_steps": 0, "model_inference_frames": 0,
            "c08_rows_read": 0, "c09_c10_worlds_read": 0,
            "graph_replays": 0, "duration_seconds": time.monotonic() - started,
        })
    write_json(run / "RUN_STATE.json", {
        "schema_version": "v3_run_state_v1", "run_id": RUN_ID,
        "state": "COMPLETED" if error is None else "FAILED",
        "overall_status": overall, "error": error,
        "duration_seconds": time.monotonic() - started,
    })
    entries = _seal(run)
    print(json.dumps({
        "overall_status": overall, "error": error, "evidence_files": entries,
    }, indent=2))
    return 0 if overall == PASS and error is None else 2


if __name__ == "__main__":
    raise SystemExit(main())
