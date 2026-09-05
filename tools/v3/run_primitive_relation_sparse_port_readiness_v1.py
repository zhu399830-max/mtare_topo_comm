#!/usr/bin/env python3
"""Formal zero-training readiness for the sparse-port relation corrective."""

from __future__ import annotations

import argparse
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
from mtare_topo.representation.primitive_relation_sparse_port_losses import (
    sparse_cardinality_objective,
    sparse_port_relation_losses,
)
from mtare_topo.representation.primitive_relation_sparse_port_model import (
    SparsePortRelationNet,
    invariant_endpoint_pair_geometry,
    outward_endpoint_tangents,
)
from run_primitive_relation_model_readiness_v1 import (
    P1A,
    P1B,
    REAL_ROW,
    REAL_TASKS,
    _loss_targets,
    _seal,
    _sha,
    _stack,
    _target_permutation,
    _target_reversal,
    _tree_hash,
)


RUN_ID = "gate3_20260831_primitive_relation_sparse_port_readiness_v1_seed0"
PASS = "PASS_PRIMITIVE_RELATION_SPARSE_PORT_READINESS_V1"
FAIL = "FAIL_PRIMITIVE_RELATION_SPARSE_PORT_READINESS_V1"
CARD_STATUS = "APPROVED_FOR_ONE_IMMUTABLE_PRIMITIVE_RELATION_SPARSE_PORT_READINESS_V1"
SCHEMA_VERSION = "primitive_relation_sparse_port_readiness_v1"
EXPECTED_PARAMETERS = 2_629_870
EXPECTED_PARAMETER_TENSORS = 193
EXPECTED_TESTS = 42
EXPECTED_PYTHON = "/home/zeng-workstation/.local/share/mtare_topo_comm/envs/phase3_torch290_cu129_zarr2187_v1/bin/python"
ATTRIBUTION = PROJECT_ROOT / "results/gate3_semantics/gate3_20260831_primitive_relation_v1_failure_attribution_v1r2_seed0"


def _prediction_permutation_error(
    reference,
    changed,
    permutation: torch.Tensor,
) -> float:
    errors: list[torch.Tensor] = []
    for name in (
        "existence_logits", "axis_control_current_sensor_m",
        "endpoint_half_axes_m", "endpoint_shape_exponent",
        "endpoint_descriptor", "geometry_uncertainty",
    ):
        errors.append(torch.max(torch.abs(
            getattr(changed, name) - getattr(reference, name)[:, permutation]
        )))
    errors.append(torch.max(torch.abs(
        changed.temporal_presence_logits
        - reference.temporal_presence_logits[:, :, permutation]
    )))
    for name in (
        "endpoint_attachment_logits", "endpoint_attachment_uncertainty",
    ):
        expected = getattr(reference, name)[:, permutation][:, :, :, permutation]
        errors.append(torch.max(torch.abs(getattr(changed, name) - expected)))
    for name in (
        "disconnected_overlap_logits", "disconnected_overlap_uncertainty",
    ):
        expected = getattr(reference, name)[:, permutation][:, :, permutation]
        errors.append(torch.max(torch.abs(getattr(changed, name) - expected)))
    expected_temporal = torch.cat((
        reference.temporal_correspondence_logits[..., :32]
        [:, :, permutation][:, :, :, permutation],
        reference.temporal_correspondence_logits[..., 32:][:, :, permutation],
    ), dim=-1)
    errors.append(torch.max(torch.abs(
        changed.temporal_correspondence_logits - expected_temporal
    )))
    return float(torch.stack(errors).max().detach().cpu())


def _pair_geometry_rotation_error(targets) -> float:
    axis = targets.axis_control_current_sensor_m[:, :5]
    endpoint = axis[:, :, (0, 2)].reshape(len(axis), 10, 3)
    tangent = outward_endpoint_tangents(axis).reshape(len(axis), 10, 3)
    half_axes = targets.endpoint_half_axes_m[:, :5].reshape(len(axis), 10, 2)
    exponent = targets.endpoint_shape_exponent[:, :5].reshape(len(axis), 10)
    descriptor = torch.arange(
        len(axis) * 10 * 32, device=axis.device, dtype=axis.dtype,
    ).reshape(len(axis), 10, 32)
    reference = invariant_endpoint_pair_geometry(
        endpoint, tangent, half_axes, exponent, descriptor,
    )
    angle = math.radians(53.0)
    rotation = torch.tensor(
        ((math.cos(angle), -math.sin(angle), 0.0),
         (math.sin(angle), math.cos(angle), 0.0),
         (0.0, 0.0, 1.0)),
        device=axis.device,
        dtype=axis.dtype,
    )
    changed = invariant_endpoint_pair_geometry(
        endpoint @ rotation.T,
        tangent @ rotation.T,
        half_axes,
        exponent,
        descriptor,
    )
    return float(torch.max(torch.abs(reference - changed)).detach().cpu())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    spec = load_json(args.spec.resolve())
    started = time.monotonic()
    overall, error = FAIL, None
    before: dict[str, str] = {}
    try:
        state = load_json(run_dir / "RUN_STATE.json")
        if run_dir.name != RUN_ID or state.get("state") != "CREATED_NOT_EXECUTED":
            raise RuntimeError("sparse-port readiness executes exactly once")
        authorization = spec.get("user_authorization", {})
        if (
            spec.get("operation") != "audit"
            or spec.get("gate") != 3
            or authorization.get("status") != "APPROVED"
        ):
            raise RuntimeError("sparse-port readiness scope mismatch")
        card = load_json(PROJECT_ROOT / spec["data_card"])
        validation = validate_data_card(card)
        if not validation.passed or card.get("status") != CARD_STATUS:
            raise RuntimeError(f"sparse-port Data Card invalid: {validation.errors}")
        for relative, expected in spec["frozen_inputs"].items():
            before[relative] = _sha(PROJECT_ROOT / relative)
            if before[relative] != expected:
                raise RuntimeError(f"frozen input drift: {relative}")
        for record in spec["frozen_tools"].values():
            if _sha(PROJECT_ROOT / record["path"]) != record["sha256"]:
                raise RuntimeError(f"frozen tool drift: {record['path']}")

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
            "schema_version": "v3_run_state_v1", "run_id": RUN_ID,
            "state": "RUNNING",
        })

        p1a_summary = load_json(P1A / "metrics/summary.json")
        p1b_summary = load_json(P1B / "metrics/summary.json")
        attribution = load_json(ATTRIBUTION / "metrics/attribution/summary.json")
        if not (
            p1a_summary.get("scientific_pass")
            and p1b_summary.get("scientific_pass")
            and p1b_summary.get("sequences") == 564378
            and attribution.get("overall_status")
            == "PASS_PRIMITIVE_RELATION_V1_FAILURE_ATTRIBUTION_V1"
            and attribution.get("diagnosis")
            == "RELATION_HEAD_FAILS_EVEN_WITH_PROPOSAL_ORACLE"
            and attribution.get("oracle_gate_seeds") == 0
        ):
            raise RuntimeError("sparse-port corrective trigger drift")

        p1a_manifest = {
            value["task_id"]: value
            for value in load_json(P1A / "artifacts/task_manifest.json")["tasks"]
        }
        p1b_manifest = {
            value["task_id"]: value
            for value in load_json(P1B / "artifacts/task_manifest.json")["tasks"]
        }
        selected_hashes, examples = {}, []
        for task in REAL_TASKS:
            sensor = P1A / "artifacts/dataset/fit" / f"{task}.zarr"
            teacher = P1B / "artifacts/teacher/fit" / f"{task}.zarr"
            sensor_hash, teacher_hash = _tree_hash(sensor), _tree_hash(teacher)
            if (
                sensor_hash != p1a_manifest[task]["corrected_shard_tree_sha256"]
                or teacher_hash != p1b_manifest[task]["shard_tree_sha256"]
            ):
                raise RuntimeError(f"selected real shard drift: {task}")
            selected_hashes[task] = {
                "sensor_tree_sha256": sensor_hash,
                "teacher_tree_sha256": teacher_hash,
            }
            examples.append(PrimitiveRelationTrainingShard(sensor, teacher)[REAL_ROW])

        tests = subprocess.run(
            [
                sys.executable, "-m", "pytest", "-q",
                "tests/v3/unit/test_swept_superellipse_field.py",
                "tests/v3/unit/test_geometry_variant_contract.py",
                "tests/v3/unit/test_primitive_relation_model.py",
                "tests/v3/unit/test_primitive_relation_training.py",
                "tests/v3/unit/test_primitive_relation_sparse_port.py",
            ],
            cwd=PROJECT_ROOT,
            env={**os.environ, "PYTHONPATH": str(PROJECT_ROOT / "src")},
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        (run_dir / "logs/unit_tests.log").write_text(tests.stdout, encoding="utf-8")
        if tests.returncode or f"{EXPECTED_TESTS} passed" not in tests.stdout:
            raise RuntimeError(f"expected exactly {EXPECTED_TESTS} sparse-port tests")

        if not torch.cuda.is_available():
            raise RuntimeError("frozen Torch 2.9 CUDA sidecar/GPU is required")
        if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
            raise RuntimeError("CUBLAS_WORKSPACE_CONFIG must be :4096:8")
        device = torch.device("cuda")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
        torch.manual_seed(20260831)
        torch.cuda.manual_seed_all(20260831)
        torch.use_deterministic_algorithms(True)
        torch.cuda.reset_peak_memory_stats()

        model = SparsePortRelationNet().to(device)
        parameters = sum(value.numel() for value in model.parameters())
        parameter_tensors = len(list(model.parameters()))
        if parameters != EXPECTED_PARAMETERS or parameter_tensors != EXPECTED_PARAMETER_TENSORS:
            raise RuntimeError(
                f"sparse-port parameter drift: {parameters}/{parameter_tensors}"
            )
        range_valid = _stack(examples, "student", "range_valid", device)
        translation = _stack(
            examples, "student", "relative_translation_current_sensor_m", device,
        )
        yaw = _stack(examples, "student", "relative_yaw_current_sensor_deg", device)
        targets = _loss_targets(examples, device)

        model.train()
        prediction = model(range_valid, translation, yaw)
        losses = sparse_port_relation_losses(prediction, targets, range_valid)
        losses["total"].backward()
        gradients = [value.grad for value in model.parameters()]
        finite_gradients = all(
            value is not None and bool(torch.isfinite(value).all())
            for value in gradients
        )
        nonzero_gradients = all(
            value is not None and float(value.abs().max()) > 0.0
            for value in gradients
        )
        gradient_tensors = sum(value is not None for value in gradients)

        model.zero_grad(set_to_none=True)
        model.eval()
        permutation = torch.randperm(
            32, generator=torch.Generator().manual_seed(17), device="cpu",
        ).to(device)
        with torch.no_grad():
            reference = model(range_valid, translation, yaw)
            repeat = model(range_valid, translation, yaw)
            changed = model(
                range_valid, translation, yaw, query_permutation=permutation,
            )
        determinism_error = max(
            float(torch.max(torch.abs(
                getattr(reference, name) - getattr(repeat, name)
            )).cpu())
            for name in reference.__dict__
        )
        query_permutation_error = _prediction_permutation_error(
            reference, changed, permutation,
        )
        reference_losses = sparse_port_relation_losses(
            reference, targets, range_valid,
        )
        target_permutation = torch.cat((
            torch.tensor((2, 0, 1), device=device),
            torch.arange(3, 32, device=device),
        ))
        permuted_losses = sparse_port_relation_losses(
            reference, _target_permutation(targets, target_permutation), range_valid,
        )
        reversed_losses = sparse_port_relation_losses(
            reference, _target_reversal(targets, 0), range_valid,
        )
        target_permutation_loss_error = max(
            abs(float(reference_losses[name]) - float(permuted_losses[name]))
            for name in reference_losses
        )
        endpoint_reversal_loss_error = max(
            abs(float(reference_losses[name]) - float(reversed_losses[name]))
            for name in reference_losses
        )
        pair_rotation_error = _pair_geometry_rotation_error(targets)

        synthetic_logits = torch.zeros(1, 32, device=device, requires_grad=True)
        synthetic_mask = torch.zeros(1, 32, device=device, dtype=torch.bool)
        synthetic_mask[:, :3] = True
        cardinality = sparse_cardinality_objective(synthetic_logits, synthetic_mask)
        cardinality["total"].backward()
        sparse_matched_gradient = float(synthetic_logits.grad[:, :3].max().cpu())
        sparse_redundant_gradient = float(synthetic_logits.grad[:, 3:].min().cpu())

        loss_values = {
            name: float(value.detach().cpu()) for name, value in losses.items()
        }
        active = [int(value.targets.primitive_mask.sum()) for value in examples]
        attachments = [int(value.targets.endpoint_attachment.sum()) for value in examples]
        overlaps = [int(value.targets.disconnected_overlap.sum()) for value in examples]
        dustbins = [int((value.targets.temporal_destination == 32).sum()) for value in examples]
        attachment_symmetric = bool(torch.equal(
            reference.endpoint_attachment_logits,
            reference.endpoint_attachment_logits.permute(0, 3, 4, 1, 2),
        ))
        overlap_symmetric = bool(torch.equal(
            reference.disconnected_overlap_logits,
            reference.disconnected_overlap_logits.transpose(1, 2),
        ))
        uncertainty_valid = all(bool(((value >= 0.0) & (value <= 1.0)).all()) for value in (
            reference.endpoint_attachment_uncertainty,
            reference.disconnected_overlap_uncertainty,
        ))
        peak_gpu = int(torch.cuda.max_memory_allocated())
        checks = {
            "p1a_p1b_scientific_pass": True,
            "v1_attribution_exact_trigger": True,
            "selected_real_shards_tree_exact": len(selected_hashes) == 3,
            f"exact_{EXPECTED_TESTS}_tests": True,
            "exact_2629870_parameters_193_tensors": (
                parameters == EXPECTED_PARAMETERS
                and parameter_tensors == EXPECTED_PARAMETER_TENSORS
            ),
            "real_three_shape_rows_same_source": (
                [value.source_global_sequence_index for value in examples]
                == [188724] * 3
            ),
            "real_batch_relation_classes_nonempty": (
                min(attachments) > 0 and min(overlaps) > 0 and min(dustbins) > 0
            ),
            "real_batch_capacity_le_32": max(active) <= 32,
            "six_losses_and_total_finite": (
                len(loss_values) == 7
                and all(math.isfinite(value) for value in loss_values.values())
            ),
            "all_parameter_gradients_finite_nonzero": (
                finite_gradients and nonzero_gradients
                and gradient_tensors == parameter_tensors
            ),
            "sparse_cardinality_gradient_direction": (
                sparse_matched_gradient < 0.0 and sparse_redundant_gradient > 0.0
            ),
            "determinism_error_zero": determinism_error == 0.0,
            "query_permutation_error_le_3e5": query_permutation_error <= 3e-5,
            "target_permutation_loss_error_le_1e6": target_permutation_loss_error <= 1e-6,
            "endpoint_reversal_loss_error_le_1e6": endpoint_reversal_loss_error <= 1e-6,
            "pair_geometry_rotation_error_le_2e6": pair_rotation_error <= 2e-6,
            "relation_logits_symmetric": attachment_symmetric and overlap_symmetric,
            "relation_uncertainty_in_unit_interval": uncertainty_valid,
            "peak_cuda_allocation_le_4gib": peak_gpu <= 4 * 1024**3,
            "zero_optimizer_checkpoint_selection_graph_test_reads": True,
        }
        overall = PASS if all(checks.values()) else FAIL
        payload = {
            "schema_version": SCHEMA_VERSION,
            "overall_status": overall,
            "scientific_pass": overall == PASS,
            "checks": checks,
            "parameters": parameters,
            "parameter_tensors": parameter_tensors,
            "gradient_tensors": gradient_tensors,
            "losses": loss_values,
            "sparse_cardinality": {
                name: float(value.detach().cpu()) for name, value in cardinality.items()
            },
            "sparse_matched_gradient_max": sparse_matched_gradient,
            "sparse_redundant_gradient_min": sparse_redundant_gradient,
            "real_tasks": list(REAL_TASKS), "real_row": REAL_ROW,
            "source_global_sequence_indices": [
                value.source_global_sequence_index for value in examples
            ],
            "active_primitives": active,
            "directed_attachments": attachments,
            "disconnected_overlaps": overlaps,
            "temporal_dustbins": dustbins,
            "selected_tree_hashes": selected_hashes,
            "determinism_error": determinism_error,
            "query_permutation_error": query_permutation_error,
            "target_permutation_loss_error": target_permutation_loss_error,
            "endpoint_reversal_loss_error": endpoint_reversal_loss_error,
            "pair_geometry_rotation_error": pair_rotation_error,
            "peak_gpu_memory_bytes": peak_gpu,
            "optimizer_steps": 0, "checkpoint_writes": 0,
            "checkpoint_selection_steps": 0,
            "c07_c08_data_rows_read": 0, "c09_c10_worlds_read": 0,
            "graph_replays": 0, "mtare_worlds_read": 0,
            "decision": (
                "ALLOW_SPARSE_PORT_THREE_SEED_TRAINING_SPEC"
                if overall == PASS else "STOP_SPARSE_PORT_READINESS_FAILED"
            ),
            "duration_seconds": time.monotonic() - started,
            "error": None,
        }
        write_json(run_dir / "metrics/summary.json", payload)
        write_json(run_dir / "artifacts/real_batch_contract.json", {
            key: payload[key] for key in (
                "real_tasks", "real_row", "source_global_sequence_indices",
                "active_primitives", "directed_attachments",
                "disconnected_overlaps", "temporal_dustbins",
                "selected_tree_hashes",
            )
        })
        after = {relative: _sha(PROJECT_ROOT / relative) for relative in before}
        if before != after:
            raise RuntimeError("frozen sparse-port source changed during execution")
        write_json(run_dir / "config/source_integrity_after.json", after)
        write_json(run_dir / "config/environment.json", {
            "versions": environment,
            "platform": platform.platform(),
            "deterministic_algorithms": True,
            "tf32": {"matmul": False, "cudnn": False},
            "float32_matmul_precision": "highest",
            "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"],
        })
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        (run_dir / "logs/failure_traceback.log").write_text(
            traceback.format_exc(), encoding="utf-8",
        )
        write_json(run_dir / "metrics/summary.json", {
            "overall_status": FAIL, "scientific_pass": False, "error": error,
        })
    write_json(run_dir / "RUN_STATE.json", {
        "schema_version": "v3_run_state_v1", "run_id": RUN_ID,
        "state": "COMPLETED" if error is None else "FAILED",
        "overall_status": overall, "error": error,
        "duration_seconds": time.monotonic() - started,
    })
    entries = _seal(run_dir)
    print(json.dumps({
        "overall_status": overall, "error": error, "evidence_files": entries,
    }, indent=2))
    return 0 if overall == PASS and error is None else 2


if __name__ == "__main__":
    raise SystemExit(main())
