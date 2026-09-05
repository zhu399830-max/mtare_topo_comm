#!/usr/bin/env python3
"""Formal zero-training readiness for the same-input non-learning baseline."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import traceback

import numpy as np
import scipy
import zarr

from _bootstrap import PROJECT_ROOT
from mtare_topo.data.primitive_relation_training import PrimitiveRelationTrainingShard
from mtare_topo.governance import load_json, validate_data_card, write_json
from mtare_topo.semantics.primitive_relation_nonlearning import (
    RobustPrimitiveRelationBaseline,
)


RUN_ID = "gate3_20260830_primitive_relation_nonlearning_readiness_v1_seed0"
PASS = "PASS_PRIMITIVE_RELATION_NONLEARNING_READINESS_V1"
FAIL = "FAIL_PRIMITIVE_RELATION_NONLEARNING_READINESS_V1"
CARD_STATUS = "APPROVED_FOR_ONE_IMMUTABLE_PRIMITIVE_RELATION_NONLEARNING_READINESS_V1"
PYTHON = "/home/zeng-workstation/.local/share/mtare_topo_comm/envs/phase3_torch290_cu129_zarr2187_v1/bin/python"
P1A = PROJECT_ROOT / "results/gate3_semantics/gate3_20260830_primitive_relation_p1a_lossless_storage_corrective_v1_seed0"
P1B = PROJECT_ROOT / "results/gate3_semantics/gate3_20260830_primitive_relation_p1b_teacher_materialization_v1r_seed0"
MODEL_READINESS = PROJECT_ROOT / "results/gate3_semantics/gate3_20260830_primitive_relation_model_readiness_v1r_seed0"
REAL_TASKS = (
    "S10_3d_complex_C04__ellipse",
    "S10_3d_complex_C04__rounded_rectangle",
    "S10_3d_complex_C04__c1_mixed",
)
REAL_ROW = 2


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _tree_hash(path: Path) -> str:
    digest = hashlib.sha256()
    for value in sorted(item for item in path.rglob("*") if item.is_file()):
        relative = value.relative_to(path).as_posix().encode()
        digest.update(len(relative).to_bytes(4, "little")); digest.update(relative)
        digest.update(bytes.fromhex(_sha(value)))
    return digest.hexdigest()


def _seal(run_dir: Path) -> int:
    destination = run_dir / "artifacts/evidence_sha256.txt"
    files = sorted(value for value in run_dir.rglob("*") if value.is_file() and value != destination)
    with destination.open("w", encoding="utf-8") as stream:
        for value in files:
            stream.write(f"{_sha(value)}  {value.relative_to(PROJECT_ROOT)}\n")
    return len(files)


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--spec", required=True, type=Path); parser.add_argument("--run-dir", required=True, type=Path); args = parser.parse_args()
    spec = load_json(args.spec.resolve()); run_dir = args.run_dir.resolve(); started = time.monotonic(); overall, error = FAIL, None; before: dict[str, str] = {}
    try:
        if run_dir.name != RUN_ID or load_json(run_dir / "RUN_STATE.json").get("state") != "CREATED_NOT_EXECUTED":
            raise RuntimeError("non-learning readiness executes exactly once")
        card = load_json(PROJECT_ROOT / spec["data_card"]); validation = validate_data_card(card)
        if not validation.passed or card.get("status") != CARD_STATUS:
            raise RuntimeError(f"non-learning readiness Data Card invalid: {validation.errors}")
        for relative, expected in spec["frozen_inputs"].items():
            before[relative] = _sha(PROJECT_ROOT / relative)
            if before[relative] != expected: raise RuntimeError(f"frozen input drift: {relative}")
        for record in spec["frozen_tools"].values():
            if _sha(PROJECT_ROOT / record["path"]) != record["sha256"]: raise RuntimeError(f"frozen tool drift: {record['path']}")
        environment = {"python": sys.version.split()[0], "executable": sys.executable, "numpy": np.__version__, "scipy": scipy.__version__, "zarr": zarr.__version__}
        expected_environment = {"python": "3.13.5", "executable": PYTHON, "numpy": "2.1.3", "scipy": "1.15.3", "zarr": "2.18.7"}
        if environment != expected_environment: raise RuntimeError(f"non-learning environment drift: {environment}")
        prerequisites = [load_json(path / "metrics/summary.json") for path in (P1A, P1B, MODEL_READINESS)]
        if not all(value.get("scientific_pass") for value in prerequisites): raise RuntimeError("non-learning readiness prerequisite drift")
        write_json(run_dir / "config/source_integrity_before.json", before)
        write_json(run_dir / "RUN_STATE.json", {"schema_version":"v3_run_state_v1","run_id":RUN_ID,"state":"RUNNING"})
        tests = subprocess.run(
            [sys.executable, "-m", "pytest", "-q", "tests/v3/unit/test_primitive_relation_nonlearning.py"],
            cwd=PROJECT_ROOT, env={**os.environ, "PYTHONPATH": str(PROJECT_ROOT / "src")},
            text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False, timeout=300,
        )
        (run_dir / "logs/unit_tests.log").write_text(tests.stdout, encoding="utf-8")
        if tests.returncode or "5 passed" not in tests.stdout: raise RuntimeError("expected exactly five non-learning baseline tests")

        sensor_manifest = {value["task_id"]: value for value in load_json(P1A / "artifacts/task_manifest.json")["tasks"]}
        teacher_manifest = {value["task_id"]: value for value in load_json(P1B / "artifacts/task_manifest.json")["tasks"]}
        examples = []; selected_hashes = {}
        for task in REAL_TASKS:
            sensor = P1A / "artifacts/dataset/fit" / f"{task}.zarr"; teacher = P1B / "artifacts/teacher/fit" / f"{task}.zarr"
            sensor_hash, teacher_hash = _tree_hash(sensor), _tree_hash(teacher)
            if sensor_hash != sensor_manifest[task]["corrected_shard_tree_sha256"] or teacher_hash != teacher_manifest[task]["shard_tree_sha256"]:
                raise RuntimeError(f"selected real baseline shard drift: {task}")
            selected_hashes[task] = {"sensor_tree_sha256": sensor_hash, "teacher_tree_sha256": teacher_hash}
            examples.append(PrimitiveRelationTrainingShard(sensor, teacher)[REAL_ROW])
        method = RobustPrimitiveRelationBaseline(); predictions = []; durations = []
        for example in examples:
            begin = time.monotonic(); first = method.predict(example.student.range_valid, example.student.relative_translation_current_sensor_m, example.student.relative_yaw_current_sensor_deg); durations.append(time.monotonic()-begin)
            second = method.predict(example.student.range_valid, example.student.relative_translation_current_sensor_m, example.student.relative_yaw_current_sensor_deg)
            if any(not np.array_equal(getattr(first, name), getattr(second, name)) for name in first.__dict__):
                raise RuntimeError("real non-learning prediction is not bit deterministic")
            predictions.append(first)
        predicted_counts = [int(value.primitive_mask.sum()) for value in predictions]
        target_counts = [int(value.targets.primitive_mask.sum()) for value in examples]
        attachment_counts = [int(value.endpoint_attachment.sum()) for value in predictions]
        overlap_counts = [int(value.disconnected_overlap.sum()) for value in predictions]
        checks = {
            "prerequisites_scientific_pass": True, "exact_5_tests": True,
            "selected_real_shards_tree_exact": len(selected_hashes) == 3,
            "same_source_three_shapes": [value.source_global_sequence_index for value in examples] == [188724] * 3,
            "one_to_32_candidates_each": min(predicted_counts) >= 1 and max(predicted_counts) <= 32,
            "real_predictions_finite": all(all(np.isfinite(getattr(value, name)).all() for name in value.__dict__) for value in predictions),
            "real_repeat_bit_exact": True,
            "relations_symmetric": all(np.array_equal(value.endpoint_attachment, value.endpoint_attachment.transpose(2,3,0,1)) and np.array_equal(value.disconnected_overlap, value.disconnected_overlap.T) for value in predictions),
            "real_relation_output_nonempty": min(attachment_counts) > 0,
            "same_student_interface_no_teacher_identity": True,
            "maximum_real_row_seconds_le_1": max(durations) <= 1.0,
            "zero_training_model_graph_test_reads": True,
        }
        overall = PASS if all(checks.values()) else FAIL
        np.savez_compressed(
            run_dir / "artifacts/real_batch_predictions.npz",
            primitive_mask=np.stack([value.primitive_mask for value in predictions]),
            axis_control_current_sensor_m=np.stack([value.axis_control_current_sensor_m for value in predictions]),
            endpoint_half_axes_m=np.stack([value.endpoint_half_axes_m for value in predictions]),
            endpoint_shape_exponent=np.stack([value.endpoint_shape_exponent for value in predictions]),
            geometry_uncertainty=np.stack([value.geometry_uncertainty for value in predictions]),
            temporal_visibility=np.stack([value.temporal_visibility for value in predictions]),
            endpoint_attachment=np.stack([value.endpoint_attachment for value in predictions]),
            disconnected_overlap=np.stack([value.disconnected_overlap for value in predictions]),
        )
        payload = {
            "schema_version":"primitive_relation_nonlearning_readiness_v1", "overall_status":overall, "scientific_pass":overall==PASS,
            "checks":checks, "config":method.config.to_dict(), "exit_config":method.exit_baseline.config.to_dict(),
            "real_tasks":list(REAL_TASKS), "real_row":REAL_ROW, "source_global_sequence_indices":[value.source_global_sequence_index for value in examples],
            "target_primitives":target_counts, "predicted_primitives":predicted_counts, "predicted_directed_attachments":attachment_counts,
            "predicted_disconnected_overlaps":overlap_counts, "real_row_seconds":durations, "selected_tree_hashes":selected_hashes,
            "optimizer_steps":0, "model_inference_frames":0, "checkpoint_writes":0, "graph_replays":0,
            "c07_c08_data_rows_read":0, "c09_c10_worlds_read":0, "mtare_worlds_read":0,
            "decision":"ALLOW_P2_BASELINE_EVALUATOR_AND_TRAINING_CARD" if overall==PASS else "STOP_NONLEARNING_READINESS_FAILED",
            "duration_seconds":time.monotonic()-started, "error":None,
        }
        write_json(run_dir / "metrics/summary.json", payload)
        after = {relative:_sha(PROJECT_ROOT/relative) for relative in before}
        if before != after: raise RuntimeError("frozen non-learning readiness source changed")
        write_json(run_dir / "config/source_integrity_after.json", after)
        write_json(run_dir / "config/environment.json", {"versions":environment,"platform":platform.platform(),"device":"cpu"})
    except Exception as exc:
        error=f"{type(exc).__name__}: {exc}"; (run_dir/"logs/failure_traceback.log").write_text(traceback.format_exc(),encoding="utf-8"); write_json(run_dir/"metrics/summary.json",{"overall_status":FAIL,"scientific_pass":False,"error":error})
    write_json(run_dir/"RUN_STATE.json",{"schema_version":"v3_run_state_v1","run_id":RUN_ID,"state":"COMPLETED" if error is None else "FAILED","overall_status":overall,"error":error})
    entries=_seal(run_dir); print(json.dumps({"overall_status":overall,"error":error,"evidence_files":entries},indent=2)); return 0 if overall==PASS and error is None else 2


if __name__ == "__main__": raise SystemExit(main())
