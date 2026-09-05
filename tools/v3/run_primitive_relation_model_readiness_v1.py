#!/usr/bin/env python3
"""Formal zero-training readiness audit for the primitive-relation model."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import traceback

import numpy as np
import numcodecs
import scipy
import torch
import zarr

from _bootstrap import PROJECT_ROOT
from mtare_topo.data.primitive_relation_training import PrimitiveRelationTrainingShard
from mtare_topo.governance import load_json, validate_data_card, write_json
from mtare_topo.representation.primitive_relation_losses import (
    PrimitiveRelationLossTargets,
    primitive_relation_losses,
)
from mtare_topo.representation.primitive_relation_model import (
    MAXIMUM_SLOTS,
    PrimitiveRelationNet,
    register_causal_lidar_points,
)


RUN_ID = "gate3_20260830_primitive_relation_model_readiness_v1_seed0"
PASS = "PASS_PRIMITIVE_RELATION_MODEL_READINESS_V1"
FAIL = "FAIL_PRIMITIVE_RELATION_MODEL_READINESS_V1"
P1A = PROJECT_ROOT / "results/gate3_semantics/gate3_20260830_primitive_relation_p1a_lossless_storage_corrective_v1_seed0"
P1B = PROJECT_ROOT / "results/gate3_semantics/gate3_20260830_primitive_relation_p1b_teacher_materialization_v1r_seed0"
REAL_TASKS = (
    "S10_3d_complex_C04__ellipse",
    "S10_3d_complex_C04__rounded_rectangle",
    "S10_3d_complex_C04__c1_mixed",
)
REAL_ROW = 2
EXPECTED_PARAMETERS = 1_969_516
EXPECTED_PYTHON = "/home/zeng-workstation/.local/share/mtare_topo_comm/envs/phase3_torch290_cu129_zarr2187_v1/bin/python"
EXPECTED_TESTS = 29
CARD_STATUS = "APPROVED_FOR_ONE_IMMUTABLE_PRIMITIVE_RELATION_MODEL_READINESS_V1"
SCHEMA_VERSION = "primitive_relation_model_readiness_v1"


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
        digest.update(len(relative).to_bytes(4, "little")); digest.update(relative); digest.update(bytes.fromhex(_sha(value)))
    return digest.hexdigest()


def _seal(run_dir: Path) -> int:
    destination = run_dir / "artifacts/evidence_sha256.txt"
    files = sorted(value for value in run_dir.rglob("*") if value.is_file() and value != destination)
    with destination.open("w", encoding="utf-8") as stream:
        for value in files: stream.write(f"{_sha(value)}  {value.relative_to(PROJECT_ROOT)}\n")
    return len(files)


def _stack(examples, section: str, name: str, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(np.stack([getattr(getattr(value, section), name) for value in examples])).to(device)


def _loss_targets(examples, device: torch.device) -> PrimitiveRelationLossTargets:
    return PrimitiveRelationLossTargets(
        _stack(examples, "targets", "primitive_mask", device).float(),
        _stack(examples, "targets", "axis_control_current_sensor_m", device),
        _stack(examples, "targets", "endpoint_half_axes_m", device),
        _stack(examples, "targets", "endpoint_shape_exponent", device),
        _stack(examples, "targets", "temporal_visibility", device).float(),
        _stack(examples, "targets", "endpoint_attachment", device).float(),
        _stack(examples, "targets", "disconnected_overlap", device).float(),
    )


def _prediction_permutation_error(reference, changed, permutation: torch.Tensor) -> float:
    errors = []
    for name in ("existence_logits", "axis_control_current_sensor_m", "endpoint_half_axes_m", "endpoint_shape_exponent", "endpoint_descriptor", "geometry_uncertainty"):
        errors.append(torch.max(torch.abs(getattr(changed, name) - getattr(reference, name)[:, permutation])))
    errors.append(torch.max(torch.abs(changed.temporal_presence_logits - reference.temporal_presence_logits[:, :, permutation])))
    expected_attachment = reference.endpoint_attachment_logits[:, permutation][:, :, :, permutation]
    expected_overlap = reference.disconnected_overlap_logits[:, permutation][:, :, permutation]
    expected_temporal = torch.cat((reference.temporal_correspondence_logits[..., :32][:, :, permutation][:, :, :, permutation], reference.temporal_correspondence_logits[..., 32:][:, :, permutation]), dim=-1)
    errors.extend((
        torch.max(torch.abs(changed.endpoint_attachment_logits - expected_attachment)),
        torch.max(torch.abs(changed.disconnected_overlap_logits - expected_overlap)),
        torch.max(torch.abs(changed.temporal_correspondence_logits - expected_temporal)),
    ))
    return float(torch.stack(errors).max().detach().cpu())


def _target_permutation(target: PrimitiveRelationLossTargets, permutation: torch.Tensor) -> PrimitiveRelationLossTargets:
    return PrimitiveRelationLossTargets(
        target.primitive_mask[:, permutation], target.axis_control_current_sensor_m[:, permutation],
        target.endpoint_half_axes_m[:, permutation], target.endpoint_shape_exponent[:, permutation],
        target.temporal_visibility[:, :, permutation],
        target.endpoint_attachment[:, permutation][:, :, :, permutation],
        target.disconnected_overlap[:, permutation][:, :, permutation],
    )


def _target_reversal(target: PrimitiveRelationLossTargets, slot: int) -> PrimitiveRelationLossTargets:
    axis = target.axis_control_current_sensor_m.clone(); axes = target.endpoint_half_axes_m.clone(); exponent = target.endpoint_shape_exponent.clone(); attachment = target.endpoint_attachment.clone()
    axis[:, slot] = axis[:, slot].flip(1); axes[:, slot] = axes[:, slot].flip(1); exponent[:, slot] = exponent[:, slot].flip(1)
    attachment[:, slot] = attachment[:, slot].flip(1); attachment[:, :, :, slot] = attachment[:, :, :, slot].flip(3)
    return PrimitiveRelationLossTargets(target.primitive_mask, axis, axes, exponent, target.temporal_visibility, attachment, target.disconnected_overlap)


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--spec", required=True, type=Path); parser.add_argument("--run-dir", required=True, type=Path); args = parser.parse_args()
    run_dir = args.run_dir.resolve(); spec = load_json(args.spec.resolve()); started = time.monotonic(); overall, error = FAIL, None
    before: dict[str, str] = {}
    try:
        if run_dir.name != RUN_ID or load_json(run_dir / "RUN_STATE.json").get("state") != "CREATED_NOT_EXECUTED": raise RuntimeError("readiness executes exactly once")
        if spec.get("operation") != "audit" or spec.get("gate") != 3 or spec.get("user_authorization", {}).get("status") != "APPROVED": raise RuntimeError("readiness scope mismatch")
        card = load_json(PROJECT_ROOT / spec["data_card"])
        card_validation = validate_data_card(card)
        if not card_validation.passed or card.get("status") != CARD_STATUS:
            raise RuntimeError(f"readiness Data Card invalid: {card_validation.errors}")
        for relative, expected in spec["frozen_inputs"].items():
            before[relative] = _sha(PROJECT_ROOT / relative)
            if before[relative] != expected: raise RuntimeError(f"frozen input drift: {relative}")
        for record in spec["frozen_tools"].values():
            if _sha(PROJECT_ROOT / record["path"]) != record["sha256"]: raise RuntimeError(f"frozen tool drift: {record['path']}")
        environment = {
            "python": sys.version.split()[0], "executable": sys.executable,
            "numpy": np.__version__, "scipy": scipy.__version__,
            "torch": torch.__version__, "cuda": torch.version.cuda,
            "zarr": zarr.__version__, "numcodecs": numcodecs.__version__,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        }
        expected_environment = {
            "python": "3.13.5", "executable": EXPECTED_PYTHON,
            "numpy": "2.1.3", "scipy": "1.15.3",
            "torch": "2.9.0+cu129", "cuda": "12.9",
            "zarr": "2.18.7", "numcodecs": "0.15.1",
            "gpu": "NVIDIA GeForce RTX 5090 D",
        }
        if environment != expected_environment:
            raise RuntimeError(f"frozen environment drift: {environment}")
        write_json(run_dir / "config/source_integrity_before.json", before)
        write_json(run_dir / "RUN_STATE.json", {
            "schema_version": "v3_run_state_v1", "run_id": RUN_ID, "state": "RUNNING",
        })
        p1a_summary, p1b_summary = load_json(P1A / "metrics/summary.json"), load_json(P1B / "metrics/summary.json")
        if not p1a_summary.get("scientific_pass") or not p1b_summary.get("scientific_pass") or p1b_summary.get("sequences") != 564378: raise RuntimeError("P1 source qualification drift")
        p1a_manifest = {value["task_id"]: value for value in load_json(P1A / "artifacts/task_manifest.json")["tasks"]}
        p1b_manifest = {value["task_id"]: value for value in load_json(P1B / "artifacts/task_manifest.json")["tasks"]}
        selected_hashes = {}
        examples = []
        for task in REAL_TASKS:
            sensor = P1A / "artifacts/dataset/fit" / f"{task}.zarr"; teacher = P1B / "artifacts/teacher/fit" / f"{task}.zarr"
            sensor_hash, teacher_hash = _tree_hash(sensor), _tree_hash(teacher)
            if sensor_hash != p1a_manifest[task]["corrected_shard_tree_sha256"] or teacher_hash != p1b_manifest[task]["shard_tree_sha256"]: raise RuntimeError(f"selected real shard drift: {task}")
            selected_hashes[task] = {"sensor_tree_sha256": sensor_hash, "teacher_tree_sha256": teacher_hash}
            examples.append(PrimitiveRelationTrainingShard(sensor, teacher)[REAL_ROW])

        tests = subprocess.run(
            [sys.executable, "-m", "pytest", "-q", "tests/v3/unit/test_swept_superellipse_field.py", "tests/v3/unit/test_geometry_variant_contract.py", "tests/v3/unit/test_primitive_relation_model.py", "tests/v3/unit/test_primitive_relation_training.py"],
            cwd=PROJECT_ROOT, env={**os.environ, "PYTHONPATH": str(PROJECT_ROOT / "src")}, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False,
        )
        (run_dir / "logs/unit_tests.log").write_text(tests.stdout, encoding="utf-8")
        if tests.returncode or f"{EXPECTED_TESTS} passed" not in tests.stdout:
            raise RuntimeError(f"expected exactly {EXPECTED_TESTS} model/geometry/reader tests")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device.type != "cuda": raise RuntimeError("frozen Torch 2.9 CUDA sidecar/GPU is required")
        if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
            raise RuntimeError("CUBLAS_WORKSPACE_CONFIG must be frozen to :4096:8")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.manual_seed(20260830); torch.cuda.manual_seed_all(20260830); torch.use_deterministic_algorithms(True)
        model = PrimitiveRelationNet().to(device)
        parameter_count = sum(value.numel() for value in model.parameters())
        if parameter_count != EXPECTED_PARAMETERS: raise RuntimeError(f"model parameter drift: {parameter_count}")
        range_valid = _stack(examples, "student", "range_valid", device)
        translation = _stack(examples, "student", "relative_translation_current_sensor_m", device)
        yaw = _stack(examples, "student", "relative_yaw_current_sensor_deg", device)
        targets = _loss_targets(examples, device)
        model.train(); prediction = model(range_valid, translation, yaw); losses = primitive_relation_losses(prediction, targets, range_valid); losses["total"].backward()
        gradients = [value.grad for value in model.parameters()]
        finite_gradients = all(value is not None and bool(torch.isfinite(value).all()) for value in gradients)
        nonzero_gradients = all(value is not None and float(value.abs().max()) > 0.0 for value in gradients)
        gradient_tensors = sum(value is not None for value in gradients)
        model.zero_grad(set_to_none=True); model.eval()
        permutation = torch.randperm(MAXIMUM_SLOTS, generator=torch.Generator().manual_seed(17)).to(device)
        with torch.no_grad():
            reference = model(range_valid, translation, yaw); repeat = model(range_valid, translation, yaw); changed = model(range_valid, translation, yaw, query_permutation=permutation)
        determinism_error = max(float(torch.max(torch.abs(getattr(reference, name) - getattr(repeat, name))).cpu()) for name in reference.__dict__)
        query_permutation_error = _prediction_permutation_error(reference, changed, permutation)
        reference_losses = primitive_relation_losses(reference, targets, range_valid)
        target_permutation = torch.cat((torch.tensor((2, 0, 1), device=device), torch.arange(3, 32, device=device)))
        permuted_losses = primitive_relation_losses(reference, _target_permutation(targets, target_permutation), range_valid)
        reversed_losses = primitive_relation_losses(reference, _target_reversal(targets, 0), range_valid)
        target_permutation_loss_error = max(
            abs(
                float(reference_losses[name].detach().cpu())
                - float(permuted_losses[name].detach().cpu())
            )
            for name in reference_losses
        )
        endpoint_reversal_loss_error = max(
            abs(
                float(reference_losses[name].detach().cpu())
                - float(reversed_losses[name].detach().cpu())
            )
            for name in reference_losses
        )

        points, valid = register_causal_lidar_points(range_valid[:1], translation[:1], yaw[:1]); shift = 72; angle = math.radians(36.0)
        rotated_range = range_valid[:1].roll(shift, dims=-1); rotated_translation = translation[:1].clone(); x, y = translation[:1, ..., 0].clone(), translation[:1, ..., 1].clone()
        rotated_translation[..., 0] = math.cos(angle) * x - math.sin(angle) * y; rotated_translation[..., 1] = math.sin(angle) * x + math.cos(angle) * y
        rotated_points, rotated_valid = register_causal_lidar_points(rotated_range, rotated_translation, yaw[:1])
        expected = points.clone(); x, y = points[..., 0].clone(), points[..., 1].clone(); expected[..., 0] = math.cos(angle) * x - math.sin(angle) * y; expected[..., 1] = math.sin(angle) * x + math.cos(angle) * y
        registration_rotation_error = float(torch.max(torch.abs(rotated_points - expected.roll(shift, dims=-2))).cpu())
        registration_mask_exact = bool(torch.equal(rotated_valid, valid.roll(shift, dims=-1)))

        loss_values = {name: float(value.detach().cpu()) for name, value in losses.items()}
        active = [int(value.targets.primitive_mask.sum()) for value in examples]
        attachments = [int(value.targets.endpoint_attachment.sum()) for value in examples]
        overlaps = [int(value.targets.disconnected_overlap.sum()) for value in examples]
        dustbins = [int((value.targets.temporal_destination == 32).sum()) for value in examples]
        checks = {
            "p1a_p1b_scientific_pass": True,
            "selected_real_shards_tree_exact": len(selected_hashes) == 3,
            f"exact_{EXPECTED_TESTS}_tests": True,
            "exact_1969516_parameters": parameter_count == EXPECTED_PARAMETERS,
            "real_three_shape_rows_same_source": [value.source_global_sequence_index for value in examples] == [188724] * 3,
            "real_batch_relation_classes_nonempty": min(attachments) > 0 and min(overlaps) > 0 and min(dustbins) > 0,
            "real_batch_capacity_le_32": max(active) <= 32,
            "six_losses_and_total_finite": len(loss_values) == 7 and all(math.isfinite(value) for value in loss_values.values()),
            "all_parameter_gradients_finite_nonzero": finite_gradients and nonzero_gradients and gradient_tensors == len(list(model.parameters())),
            "determinism_error_zero": determinism_error == 0.0,
            "query_permutation_error_le_2e5": query_permutation_error <= 2e-5,
            "target_permutation_loss_error_le_1e6": target_permutation_loss_error <= 1e-6,
            "endpoint_reversal_loss_error_le_1e6": endpoint_reversal_loss_error <= 1e-6,
            "registration_rotation_error_le_2e5": registration_rotation_error <= 2e-5 and registration_mask_exact,
            "relation_logits_symmetric": bool(torch.equal(
                reference.endpoint_attachment_logits,
                reference.endpoint_attachment_logits.permute(0, 3, 4, 1, 2),
            )) and bool(torch.equal(
                reference.disconnected_overlap_logits,
                reference.disconnected_overlap_logits.transpose(1, 2),
            )),
            "zero_optimizer_inference_selection_graph_test_reads": True,
        }
        overall = PASS if all(checks.values()) else FAIL
        payload = {
            "schema_version": SCHEMA_VERSION, "overall_status": overall,
            "scientific_pass": overall == PASS, "checks": checks, "parameters": parameter_count,
            "gradient_tensors": gradient_tensors, "losses": loss_values,
            "real_tasks": list(REAL_TASKS), "real_row": REAL_ROW,
            "source_global_sequence_indices": [value.source_global_sequence_index for value in examples],
            "active_primitives": active, "directed_attachments": attachments, "disconnected_overlaps": overlaps, "temporal_dustbins": dustbins,
            "selected_tree_hashes": selected_hashes, "determinism_error": determinism_error,
            "query_permutation_error": query_permutation_error, "target_permutation_loss_error": target_permutation_loss_error,
            "endpoint_reversal_loss_error": endpoint_reversal_loss_error, "registration_rotation_error_m": registration_rotation_error,
            "device": str(device), "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated(),
            "optimizer_steps": 0, "trained_inference_frames": 0, "checkpoint_selection_steps": 0,
            "c07_c08_data_rows_read": 0, "c09_c10_worlds_read": 0, "graph_replays": 0, "mtare_worlds_read": 0,
            "decision": "ALLOW_P2_THREE_SEED_TRAINING_SPEC" if overall == PASS else "STOP_MODEL_READINESS_FAILED",
            "duration_seconds": time.monotonic() - started, "error": None,
        }
        write_json(run_dir / "metrics/summary.json", payload)
        write_json(run_dir / "artifacts/real_batch_contract.json", {key: payload[key] for key in ("real_tasks", "real_row", "source_global_sequence_indices", "active_primitives", "directed_attachments", "disconnected_overlaps", "temporal_dustbins", "selected_tree_hashes")})
        after = {relative: _sha(PROJECT_ROOT / relative) for relative in before}
        if before != after:
            raise RuntimeError("frozen readiness source changed during execution")
        write_json(run_dir / "config/source_integrity_after.json", after)
        write_json(run_dir / "config/environment.json", {
            "versions": environment, "platform": platform.platform(),
            "deterministic_algorithms": True, "tf32": False,
            "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"],
        })
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"; (run_dir / "logs/failure_traceback.log").write_text(traceback.format_exc(), encoding="utf-8"); write_json(run_dir / "metrics/summary.json", {"overall_status": FAIL, "scientific_pass": False, "error": error})
    write_json(run_dir / "RUN_STATE.json", {"schema_version": "v3_run_state_v1", "run_id": RUN_ID, "state": "COMPLETED" if error is None else "FAILED", "overall_status": overall, "error": error})
    entries = _seal(run_dir); print(json.dumps({"overall_status": overall, "error": error, "evidence_files": entries}, indent=2)); return 0 if overall == PASS and error is None else 2


if __name__ == "__main__": raise SystemExit(main())
