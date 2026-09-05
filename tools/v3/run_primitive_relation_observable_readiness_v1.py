#!/usr/bin/env python3
"""Formal zero-training readiness for observable primitive relations."""

from __future__ import annotations

import argparse
from dataclasses import replace
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

import numcodecs
import numpy as np
import scipy
import torch
import zarr

from _bootstrap import PROJECT_ROOT
from mtare_topo.data.primitive_relation_observable_batches import (
    ObservablePrimitiveRelationBatchLoader,
)
from mtare_topo.governance import load_json, validate_data_card, write_json
from mtare_topo.representation.primitive_relation_losses import (
    PrimitiveRelationLossTargets,
)
from mtare_topo.representation.primitive_relation_observable_initialization import (
    initialize_observable_relation_from_checkpoint,
)
from mtare_topo.representation.primitive_relation_observable_model import (
    safe_attachment_score,
)
from mtare_topo.representation.primitive_relation_observable_training import (
    observable_numpy_batch_to_torch,
    observable_primitive_relation_losses,
)
from mtare_topo.representation.primitive_relation_sparse_port_losses import (
    sparse_port_relation_losses,
)
from run_primitive_relation_model_readiness_v1 import _seal, _sha, _tree_hash


RUN_ID = "gate3_20260902_primitive_relation_observable_readiness_v1r_seed0"
PASS = "PASS_PRIMITIVE_RELATION_OBSERVABLE_READINESS_V1R"
FAIL = "FAIL_PRIMITIVE_RELATION_OBSERVABLE_READINESS_V1R"
CARD_STATUS = "APPROVED_FOR_ONE_IMMUTABLE_PRIMITIVE_RELATION_OBSERVABLE_READINESS_V1R"
EXPECTED_PYTHON = "/home/zeng-workstation/.local/share/mtare_topo_comm/envs/phase3_torch290_cu129_zarr2187_v1/bin/python"
EXPECTED_TESTS = 42
EXPECTED_PARAMETERS = 2_635_631
EXPECTED_PARAMETER_TENSORS = 197
EXPECTED_TRAINABLE_PARAMETERS = 742_149
EXPECTED_TRAINABLE_TENSORS = 64
TASK = "S01_flat_tree_small_C01__c1_mixed"
TASK_ZARR = f"{TASK}.zarr"
ROW = 5

P1A = PROJECT_ROOT / "results/gate3_semantics/gate3_20260830_primitive_relation_p1a_lossless_storage_corrective_v1_seed0"
P1B = PROJECT_ROOT / "results/gate3_semantics/gate3_20260830_primitive_relation_p1b_teacher_materialization_v1r_seed0"
SIDECAR = PROJECT_ROOT / "results/gate3_semantics/gate3_20260902_primitive_attachment_observability_sidecar_v1_seed0"
AUDIT = PROJECT_ROOT / "results/gate3_semantics/gate3_20260902_primitive_attachment_teacher_observability_v1_seed0"
V2 = PROJECT_ROOT / "results/gate3_semantics/gate3_20260901_primitive_relation_sparse_port_three_seed_training_v1r_seed0"
ATTRIBUTION = PROJECT_ROOT / "results/gate3_semantics/gate3_20260902_primitive_relation_sparse_port_failure_attribution_evidence_corrective_v1_seed0"


def _prediction_permutation_error(reference, changed, permutation: torch.Tensor) -> float:
    errors: list[torch.Tensor] = []
    for name in (
        "existence_logits", "axis_control_current_sensor_m",
        "endpoint_half_axes_m", "endpoint_shape_exponent",
        "endpoint_descriptor", "geometry_uncertainty",
        "endpoint_evidence_logits",
    ):
        errors.append(torch.max(torch.abs(
            getattr(changed, name) - getattr(reference, name)[:, permutation]
        )))
    errors.append(torch.max(torch.abs(
        changed.temporal_presence_logits
        - reference.temporal_presence_logits[:, :, permutation]
    )))
    for name in ("endpoint_attachment_logits", "endpoint_attachment_uncertainty"):
        expected = getattr(reference, name)[:, permutation][:, :, :, permutation]
        errors.append(torch.max(torch.abs(getattr(changed, name) - expected)))
    for name in ("disconnected_overlap_logits", "disconnected_overlap_uncertainty"):
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


def _target_permutation(
    target: PrimitiveRelationLossTargets,
    permutation: torch.Tensor,
) -> PrimitiveRelationLossTargets:
    return PrimitiveRelationLossTargets(
        target.primitive_mask[:, permutation],
        target.axis_control_current_sensor_m[:, permutation],
        target.endpoint_half_axes_m[:, permutation],
        target.endpoint_shape_exponent[:, permutation],
        target.temporal_visibility[:, :, permutation],
        target.endpoint_attachment[:, permutation][:, :, :, permutation],
        target.disconnected_overlap[:, permutation][:, :, permutation],
        target.endpoint_observed[:, permutation],
    )


def _target_reversal(
    target: PrimitiveRelationLossTargets,
    slot: int,
) -> PrimitiveRelationLossTargets:
    axis = target.axis_control_current_sensor_m.clone()
    axes = target.endpoint_half_axes_m.clone()
    exponent = target.endpoint_shape_exponent.clone()
    attachment = target.endpoint_attachment.clone()
    observed = target.endpoint_observed.clone()
    axis[:, slot] = axis[:, slot].flip(1)
    axes[:, slot] = axes[:, slot].flip(1)
    exponent[:, slot] = exponent[:, slot].flip(1)
    attachment[:, slot] = attachment[:, slot].flip(1)
    attachment[:, :, :, slot] = attachment[:, :, :, slot].flip(3)
    observed[:, slot] = observed[:, slot].flip(1)
    return replace(
        target, axis_control_current_sensor_m=axis,
        endpoint_half_axes_m=axes, endpoint_shape_exponent=exponent,
        endpoint_attachment=attachment, endpoint_observed=observed,
    )


def _state_digest(model, trainable: bool) -> str:
    digest = hashlib.sha256()
    parameter_flags = {name: value.requires_grad for name, value in model.named_parameters()}
    for name, value in sorted(model.state_dict().items()):
        if parameter_flags.get(name) != trainable:
            continue
        array = value.detach().cpu().contiguous().numpy()
        digest.update(name.encode("utf-8"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


def _hidden_positive_pair(targets: PrimitiveRelationLossTargets) -> tuple[int, int, int, int]:
    attachment = targets.endpoint_attachment[0].bool()
    observed = targets.endpoint_observed[0].bool()
    for source in range(32):
        for source_endpoint in range(2):
            for destination in range(source + 1, 32):
                for destination_endpoint in range(2):
                    if (
                        bool(attachment[source, source_endpoint, destination, destination_endpoint])
                        and not bool(observed[source, source_endpoint] and observed[destination, destination_endpoint])
                    ):
                        return source, source_endpoint, destination, destination_endpoint
    raise RuntimeError("selected real row lacks a hidden positive attachment")


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
            raise RuntimeError("observable readiness executes exactly once")
        if (
            spec.get("operation") != "audit" or spec.get("gate") != 3
            or spec.get("user_authorization", {}).get("status") != "APPROVED"
        ):
            raise RuntimeError("observable readiness scope mismatch")
        card = load_json(PROJECT_ROOT / spec["data_card"])
        validation = validate_data_card(card)
        if not validation.passed or card.get("status") != CARD_STATUS:
            raise RuntimeError(f"observable readiness Data Card invalid: {validation.errors}")
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

        source_checks = {
            "p1a": load_json(P1A / "metrics/summary.json").get("scientific_pass") is True,
            "p1b": load_json(P1B / "metrics/summary.json").get("scientific_pass") is True,
            "sidecar": load_json(SIDECAR / "metrics/summary.json").get("scientific_pass") is True,
            "audit_diagnosis": load_json(AUDIT / "metrics/summary.json").get("diagnosis")
            == "WINDOW_LEVEL_HIDDEN_ENDPOINT_SUPERVISION_CORRECTABLE_BY_OBSERVABILITY_MASK",
            "v2_formal_fail": load_json(V2 / "metrics/summary.json").get("scientific_pass") is False,
            "v2_attribution": load_json(ATTRIBUTION / "metrics/attribution/summary.json").get("diagnosis")
            == "RELATION_SCORE_FAILS_EVEN_WITH_PROPOSAL_ORACLE",
        }
        if not all(source_checks.values()):
            raise RuntimeError(f"observable corrective trigger drift: {source_checks}")

        tests = subprocess.run(
            [
                sys.executable, "-m", "pytest", "-q",
                "tests/v3/unit/test_primitive_attachment_observability.py",
                "tests/v3/unit/test_primitive_attachment_observability_sidecar.py",
                "tests/v3/unit/test_primitive_relation_observable_batches.py",
                "tests/v3/unit/test_primitive_relation_observable_model.py",
                "tests/v3/unit/test_primitive_relation_observable_initialization.py",
                "tests/v3/unit/test_primitive_relation_sparse_port.py",
                "tests/v3/unit/test_primitive_relation_model.py",
                "tests/v3/unit/test_primitive_relation_metrics.py",
            ],
            cwd=PROJECT_ROOT,
            env={**os.environ, "PYTHONPATH": f"{PROJECT_ROOT / 'src'}:{PROJECT_ROOT}"},
            text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            check=False,
        )
        (run_dir / "logs/unit_tests.log").write_text(tests.stdout, encoding="utf-8")
        if tests.returncode or f"{EXPECTED_TESTS} passed" not in tests.stdout:
            raise RuntimeError(f"expected exactly {EXPECTED_TESTS} observable relation tests")

        p1a_manifest = {
            value["task_id"]: value
            for value in load_json(P1A / "artifacts/task_manifest.json")["tasks"]
        }
        p1b_manifest = {
            value["task_id"]: value
            for value in load_json(P1B / "artifacts/task_manifest.json")["tasks"]
        }
        sidecar_manifest = {
            value["task_id"]: value
            for value in load_json(SIDECAR / "artifacts/task_manifest.json")["tasks"]
        }
        sensor_path = P1A / "artifacts/dataset/fit" / TASK_ZARR
        teacher_path = P1B / "artifacts/teacher/fit" / TASK_ZARR
        sidecar_path = SIDECAR / "artifacts/endpoint_observability/fit" / TASK_ZARR
        selected_hashes = {
            "sensor": _tree_hash(sensor_path), "teacher": _tree_hash(teacher_path),
            "sidecar": _tree_hash(sidecar_path),
        }
        if selected_hashes != {
            "sensor": p1a_manifest[TASK]["corrected_shard_tree_sha256"],
            "teacher": p1b_manifest[TASK]["shard_tree_sha256"],
            "sidecar": sidecar_manifest[TASK]["sidecar_tree_sha256"],
        }:
            raise RuntimeError("selected real shard tree hash drift")

        transfer_reports = []
        for seed in (0, 1, 2):
            checkpoint = V2 / f"artifacts/models/seed{seed}/selected.pt"
            initialized, report = initialize_observable_relation_from_checkpoint(
                checkpoint, initialization_seed=202609020 + seed,
            )
            transfer_reports.append({
                **report.__dict__, "seed": seed,
                "frozen_state_sha256": _state_digest(initialized, False),
                "reset_state_sha256": _state_digest(initialized, True),
            })
            del initialized

        loader = ObservablePrimitiveRelationBatchLoader(
            P1A / "artifacts/dataset/fit", P1B / "artifacts/teacher/fit",
            SIDECAR / "artifacts/endpoint_observability/fit",
        )
        numpy_batch = loader._read(TASK_ZARR, np.asarray([ROW], dtype=np.int64))
        if (
            int(numpy_batch.base.primitive_mask.sum()) != 11
            or int(numpy_batch.endpoint_observed.sum()) != 17
            or int(numpy_batch.base.endpoint_attachment.sum()) != 22
            or int(numpy_batch.base.source_global_sequence_index[0]) != 5
        ):
            raise RuntimeError("selected real observable row drift")

        if not torch.cuda.is_available():
            raise RuntimeError("frozen CUDA environment is required")
        if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
            raise RuntimeError("CUBLAS_WORKSPACE_CONFIG must be :4096:8")
        device = torch.device("cuda")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
        torch.use_deterministic_algorithms(True)
        torch.manual_seed(202609020)
        torch.cuda.manual_seed_all(202609020)
        torch.cuda.reset_peak_memory_stats()
        model, initialization = initialize_observable_relation_from_checkpoint(
            V2 / "artifacts/models/seed0/selected.pt",
            initialization_seed=202609020,
        )
        model = model.to(device)
        batch = observable_numpy_batch_to_torch(numpy_batch, device=device)

        parameters = sum(value.numel() for value in model.parameters())
        parameter_tensors = len(list(model.parameters()))
        trainable_parameters = sum(
            value.numel() for value in model.parameters() if value.requires_grad
        )
        trainable_tensors = sum(value.requires_grad for value in model.parameters())
        if (
            parameters != EXPECTED_PARAMETERS
            or parameter_tensors != EXPECTED_PARAMETER_TENSORS
            or trainable_parameters != EXPECTED_TRAINABLE_PARAMETERS
            or trainable_tensors != EXPECTED_TRAINABLE_TENSORS
        ):
            raise RuntimeError("observable relation parameter boundary drift")

        model.train()
        prediction = model(
            batch.range_valid, batch.relative_translation_current_sensor_m,
            batch.relative_yaw_current_sensor_deg,
        )
        losses = observable_primitive_relation_losses(
            prediction, batch.targets, batch.range_valid,
        )
        sparse_losses = sparse_port_relation_losses(
            prediction, batch.targets, batch.range_valid,
        )
        losses["total"].backward()
        trainable_gradients = [
            value.grad for value in model.parameters() if value.requires_grad
        ]
        frozen_gradients = [
            value.grad for value in model.parameters() if not value.requires_grad
        ]
        trainable_finite_nonzero = all(
            value is not None and bool(torch.isfinite(value).all())
            and float(value.abs().max()) > 0.0
            for value in trainable_gradients
        )
        frozen_gradients_absent = all(value is None for value in frozen_gradients)

        model.zero_grad(set_to_none=True)
        model.eval()
        permutation = torch.randperm(
            32, generator=torch.Generator().manual_seed(17), device="cpu",
        ).to(device)
        with torch.no_grad():
            reference = model(
                batch.range_valid, batch.relative_translation_current_sensor_m,
                batch.relative_yaw_current_sensor_deg,
            )
            repeat = model(
                batch.range_valid, batch.relative_translation_current_sensor_m,
                batch.relative_yaw_current_sensor_deg,
            )
            changed = model(
                batch.range_valid, batch.relative_translation_current_sensor_m,
                batch.relative_yaw_current_sensor_deg,
                query_permutation=permutation,
            )
        determinism_error = max(
            float(torch.max(torch.abs(
                getattr(reference, name) - getattr(repeat, name)
            )).cpu()) for name in reference.__dict__
        )
        query_permutation_error = _prediction_permutation_error(
            reference, changed, permutation,
        )
        reference_losses = observable_primitive_relation_losses(
            reference, batch.targets, batch.range_valid,
        )
        target_permutation = torch.cat((
            torch.tensor((2, 0, 1), device=device),
            torch.arange(3, 32, device=device),
        ))
        permuted_losses = observable_primitive_relation_losses(
            reference, _target_permutation(batch.targets, target_permutation),
            batch.range_valid,
        )
        reversed_losses = observable_primitive_relation_losses(
            reference, _target_reversal(batch.targets, 0), batch.range_valid,
        )
        target_permutation_loss_error = max(
            abs(float(reference_losses[name]) - float(permuted_losses[name]))
            for name in reference_losses
        )
        endpoint_reversal_loss_error = max(
            abs(float(reference_losses[name]) - float(reversed_losses[name]))
            for name in reference_losses
        )

        hidden = _hidden_positive_pair(batch.targets)
        changed_attachment = batch.targets.endpoint_attachment.clone()
        source, source_endpoint, destination, destination_endpoint = hidden
        changed_attachment[0, source, source_endpoint, destination, destination_endpoint] = 0
        changed_attachment[0, destination, destination_endpoint, source, source_endpoint] = 0
        hidden_changed = replace(
            batch.targets, endpoint_attachment=changed_attachment,
        )
        hidden_losses = observable_primitive_relation_losses(
            reference, hidden_changed, batch.range_valid,
        )
        hidden_label_loss_error = max(
            abs(float(reference_losses[name]) - float(hidden_losses[name]))
            for name in reference_losses
        )
        safe_score = safe_attachment_score(reference)
        safe_score_valid = bool(
            torch.isfinite(safe_score).all()
            and ((safe_score >= 0.0) & (safe_score <= 1.0)).all()
        )
        peak_gpu = int(torch.cuda.max_memory_allocated())
        loss_values = {
            name: float(value.detach().cpu()) for name, value in losses.items()
        }
        checks = {
            "qualified_corrective_sources": all(source_checks.values()),
            f"exact_{EXPECTED_TESTS}_tests": True,
            "selected_real_shards_tree_exact": True,
            "three_checkpoint_geometry_transfer_exact": all(
                value["transferred_tensors"] == 133
                and value["reset_relation_tensors"] == 64
                for value in transfer_reports
            ),
            "exact_parameter_and_freeze_boundary": (
                parameters == EXPECTED_PARAMETERS
                and parameter_tensors == EXPECTED_PARAMETER_TENSORS
                and trainable_parameters == EXPECTED_TRAINABLE_PARAMETERS
                and trainable_tensors == EXPECTED_TRAINABLE_TENSORS
                and initialization.frozen_parameters == 1_893_482
            ),
            "real_row_contains_supported_and_hidden_relations": (
                hidden_label_loss_error == 0.0
            ),
            "six_families_and_total_finite": (
                len(loss_values) == 7
                and all(math.isfinite(value) for value in loss_values.values())
            ),
            "sparse_cardinality_objective_preserved": torch.equal(
                losses["primitive_set_parameters"],
                sparse_losses["primitive_set_parameters"],
            ),
            "all_trainable_gradients_finite_nonzero": trainable_finite_nonzero,
            "all_frozen_gradients_absent": frozen_gradients_absent,
            "determinism_error_zero": determinism_error == 0.0,
            "query_permutation_error_le_3e5": query_permutation_error <= 3e-5,
            "target_permutation_loss_error_le_1e6": target_permutation_loss_error <= 1e-6,
            "endpoint_reversal_loss_error_le_1e6": endpoint_reversal_loss_error <= 1e-6,
            "hidden_positive_label_is_loss_invariant": hidden_label_loss_error == 0.0,
            "safe_attachment_score_finite_unit_interval": safe_score_valid,
            "peak_cuda_allocation_le_4gib": peak_gpu <= 4 * 1024**3,
            "zero_optimizer_checkpoint_selection_graph_test_reads": True,
        }
        overall = PASS if all(checks.values()) else FAIL
        payload = {
            "schema_version": "primitive_relation_observable_readiness_v1r",
            "overall_status": overall, "scientific_pass": overall == PASS,
            "checks": checks, "source_checks": source_checks,
            "parameters": parameters, "parameter_tensors": parameter_tensors,
            "trainable_parameters": trainable_parameters,
            "trainable_tensors": trainable_tensors,
            "frozen_parameters": parameters - trainable_parameters,
            "frozen_tensors": parameter_tensors - trainable_tensors,
            "transfer_reports": transfer_reports,
            "losses": loss_values,
            "selected_task": TASK, "selected_row": ROW,
            "source_global_sequence_index": int(
                numpy_batch.base.source_global_sequence_index[0]
            ),
            "active_primitives": int(numpy_batch.base.primitive_mask.sum()),
            "observed_endpoints": int(numpy_batch.endpoint_observed.sum()),
            "directed_attachment_entries": int(
                numpy_batch.base.endpoint_attachment.sum()
            ),
            "hidden_positive_pair_toggled": list(hidden),
            "selected_tree_hashes": selected_hashes,
            "determinism_error": determinism_error,
            "query_permutation_error": query_permutation_error,
            "target_permutation_loss_error": target_permutation_loss_error,
            "endpoint_reversal_loss_error": endpoint_reversal_loss_error,
            "hidden_label_loss_error": hidden_label_loss_error,
            "safe_attachment_score_min": float(safe_score.min().cpu()),
            "safe_attachment_score_max": float(safe_score.max().cpu()),
            "peak_gpu_memory_bytes": peak_gpu,
            "optimizer_steps": 0, "checkpoint_writes": 0,
            "checkpoint_selection_steps": 0, "c07_rows_read": 0,
            "c08_rows_read": 0, "c09_c10_worlds_read": 0,
            "graph_replays": 0, "mtare_worlds_read": 0,
            "decision": (
                "ALLOW_OBSERVABLE_RELATION_THREE_SEED_TRAINING_SPEC"
                if overall == PASS else "STOP_OBSERVABLE_RELATION_READINESS_FAILED"
            ),
            "duration_seconds": time.monotonic() - started, "error": None,
        }
        write_json(run_dir / "metrics/summary.json", payload)
        write_json(run_dir / "artifacts/checkpoint_transfer.json", {
            "reports": transfer_reports,
        })
        write_json(run_dir / "artifacts/real_batch_contract.json", {
            key: payload[key] for key in (
                "selected_task", "selected_row", "source_global_sequence_index",
                "active_primitives", "observed_endpoints",
                "directed_attachment_entries", "hidden_positive_pair_toggled",
                "selected_tree_hashes",
            )
        })
        after = {relative: _sha(PROJECT_ROOT / relative) for relative in before}
        if before != after:
            raise RuntimeError("frozen observable readiness source changed")
        write_json(run_dir / "config/source_integrity_after.json", after)
        write_json(run_dir / "config/environment.json", {
            "versions": environment, "platform": platform.platform(),
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
