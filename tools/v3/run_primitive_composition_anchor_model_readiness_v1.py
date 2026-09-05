#!/usr/bin/env python3
"""Formal zero-training readiness for the O(E) composition-anchor model."""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import time
import traceback

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numcodecs
import numpy as np
import scipy
import torch
import zarr

from _bootstrap import PROJECT_ROOT
from mtare_topo.data.primitive_relation_observable_batches import (
    ObservablePrimitiveRelationBatchLoader,
)
from mtare_topo.evaluation.primitive_composition_anchor_teacher import (
    construction_endpoint_anchor_table,
    current_sensor_anchor_targets,
)
from mtare_topo.governance import load_json, validate_data_card, write_json
from mtare_topo.representation.primitive_composition_anchor_model import (
    COMPOSITION_ANCHOR_OUTPUTS_PER_ENDPOINT,
    FrozenObservableCompositionAnchorNet,
)
from mtare_topo.representation.primitive_composition_anchor_training import (
    align_composition_anchor_targets,
    composition_anchor_training_losses,
)
from mtare_topo.representation.primitive_relation_losses import match_primitives
from mtare_topo.representation.primitive_relation_observable_model import (
    ObservableSparsePortRelationNet,
)
from mtare_topo.representation.primitive_relation_observable_training import (
    observable_numpy_batch_to_torch,
)


RUN_ID = "gate3_20260903_primitive_composition_anchor_model_readiness_v2_seed0"
PASS = "PASS_PRIMITIVE_COMPOSITION_ANCHOR_MODEL_READINESS_V2"
FAIL = "FAIL_PRIMITIVE_COMPOSITION_ANCHOR_MODEL_READINESS_V2"
CARD_STATUS = "APPROVED_FOR_ONE_IMMUTABLE_PRIMITIVE_COMPOSITION_ANCHOR_MODEL_READINESS_V2"
PYTHON = "/home/zeng-workstation/.local/share/mtare_topo_comm/envs/phase3_torch290_cu129_zarr2187_v1/bin/python"
TASK = "S01_flat_tree_small_C01__c1_mixed"
TASK_ZARR = f"{TASK}.zarr"
ROW = 5
EXPECTED_TESTS = 20
EXPECTED_BACKBONE_PARAMETERS = 2_635_631
EXPECTED_HEAD_PARAMETERS = 22_278
P1A = PROJECT_ROOT / "results/gate3_semantics/gate3_20260830_primitive_relation_p1a_lossless_storage_corrective_v1_seed0"
P1B = PROJECT_ROOT / "results/gate3_semantics/gate3_20260830_primitive_relation_p1b_teacher_materialization_v1r_seed0"
SIDECAR = PROJECT_ROOT / "results/gate3_semantics/gate3_20260902_primitive_attachment_observability_sidecar_v1_seed0"
TRAINING = PROJECT_ROOT / "results/gate3_semantics/gate3_20260902_primitive_relation_observable_three_seed_training_v1_seed0"
ANCHOR_TEACHER = PROJECT_ROOT / "results/gate3_semantics/gate3_20260903_primitive_composition_anchor_teacher_readiness_v1_seed0"
DIAGNOSTIC = PROJECT_ROOT / "results/gate3_semantics/gate3_20260903_primitive_predicted_geometry_association_diagnostic_v1_seed0"


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _tree_hash(path: Path) -> str:
    digest = hashlib.sha256()
    for value in sorted(item for item in path.rglob("*") if item.is_file()):
        relative = value.relative_to(path).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(4, "little"))
        digest.update(relative)
        digest.update(bytes.fromhex(_sha(value)))
    return digest.hexdigest()


def _seal(run: Path) -> int:
    target = run / "artifacts/evidence_sha256.txt"
    files = sorted(path for path in run.rglob("*") if path.is_file() and path != target)
    target.write_text("".join(
        f"{_sha(path)}  {path.relative_to(PROJECT_ROOT)}\n" for path in files
    ), encoding="utf-8")
    return len(files)


def _load_backbone(path: Path) -> ObservableSparsePortRelationNet:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if (
        checkpoint.get("schema_version") != "primitive_relation_observable_checkpoint_v1"
        or int(checkpoint.get("seed", -1)) != 0
        or checkpoint.get("training_contract", {}).get("teacher_attachment_validity")
        != "dual_endpoint_observed"
    ):
        raise RuntimeError("composition-anchor backbone checkpoint identity drift")
    model = ObservableSparsePortRelationNet()
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    return model


def _rotate_yaw(value: torch.Tensor, degrees: float) -> torch.Tensor:
    angle = torch.deg2rad(torch.as_tensor(degrees, dtype=value.dtype, device=value.device))
    zero, one = torch.zeros_like(angle), torch.ones_like(angle)
    rotation = torch.stack((
        torch.stack((torch.cos(angle), -torch.sin(angle), zero)),
        torch.stack((torch.sin(angle), torch.cos(angle), zero)),
        torch.stack((zero, zero, one)),
    ))
    return torch.einsum("...j,kj->...k", value, rotation)


def _query_permutation_metrics(reference, changed, permutation: torch.Tensor) -> tuple[float, float]:
    reference_endpoint = reference.primitive.axis_control_current_sensor_m[:, :, (0, 2)]
    changed_endpoint = changed.primitive.axis_control_current_sensor_m[:, :, (0, 2)]
    reference_correction = (
        reference.composition.anchor_current_sensor_m - reference_endpoint
    )[:, permutation]
    changed_correction = changed.composition.anchor_current_sensor_m - changed_endpoint
    errors = [
        torch.max(torch.abs(
            changed_correction - reference_correction
        )),
        torch.max(torch.abs(
            changed.composition.residual_local_m
            - reference.composition.residual_local_m[:, permutation]
        )),
        torch.max(torch.abs(
            changed.composition.scale_m - reference.composition.scale_m[:, permutation]
        )),
    ]
    endpoint_permutation = (
        permutation[:, None] * 2
        + torch.arange(2, device=permutation.device)[None]
    ).reshape(-1)
    expected = reference.composition.compatibility_logits[:, endpoint_permutation][
        :, :, endpoint_permutation
    ]
    errors.append(torch.max(torch.abs(changed.composition.compatibility_logits - expected)))
    incremental = float(torch.stack(errors).detach().cpu().max())
    absolute_anchor = float(torch.max(torch.abs(
        changed.composition.anchor_current_sensor_m
        - reference.composition.anchor_current_sensor_m[:, permutation]
    )).detach().cpu())
    return incremental, absolute_anchor


def _plot(
    raw_endpoint: np.ndarray,
    initial_anchor: np.ndarray,
    target_anchor: np.ndarray,
    observed: np.ndarray,
    output: Path,
) -> None:
    raw = raw_endpoint.reshape(64, 3)[observed.reshape(64)]
    initial = initial_anchor.reshape(64, 3)[observed.reshape(64)]
    target = target_anchor.reshape(64, 3)[observed.reshape(64)]
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.2))
    for axis, dimensions, labels in (
        (axes[0], (0, 1), ("x [m]", "y [m]")),
        (axes[1], (0, 2), ("x [m]", "z [m]")),
    ):
        axis.scatter(raw[:, dimensions[0]], raw[:, dimensions[1]], s=22, label="frozen predicted endpoint")
        axis.scatter(initial[:, dimensions[0]], initial[:, dimensions[1]], s=18, label="untrained anchor head")
        axis.scatter(target[:, dimensions[0]], target[:, dimensions[1]], s=30, marker="x", label="construction target")
        axis.set_xlabel(labels[0]); axis.set_ylabel(labels[1]); axis.axis("equal")
    axes[0].legend(fontsize=7)
    fig.suptitle("Composition-anchor model readiness: real C01 row, no training")
    fig.tight_layout()
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(output / f"primitive_composition_anchor_model_readiness.{suffix}", dpi=180)
    plt.close(fig)


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
    after: dict[str, str] = {}
    payload: dict = {}
    try:
        if run.name != RUN_ID or load_json(run / "RUN_STATE.json").get("state") != "CREATED_NOT_EXECUTED":
            raise RuntimeError("composition-anchor model readiness executes exactly once")
        card = load_json(PROJECT_ROOT / spec["data_card"])
        validation = validate_data_card(card)
        if not validation.passed or card.get("status") != CARD_STATUS:
            raise RuntimeError(f"composition-anchor model Data Card invalid: {validation.errors}")
        for relative, expected in spec["frozen_inputs"].items():
            path = PROJECT_ROOT / relative
            before[relative] = _sha(path)
            if before[relative] != expected:
                raise RuntimeError(f"composition-anchor frozen input drift: {relative}")
        for record in spec["frozen_tools"].values():
            if _sha(PROJECT_ROOT / record["path"]) != record["sha256"]:
                raise RuntimeError(f"composition-anchor tool drift: {record['path']}")

        versions = {
            "python": sys.version.split()[0], "executable": sys.executable,
            "numpy": np.__version__, "scipy": scipy.__version__,
            "torch": torch.__version__, "cuda": torch.version.cuda,
            "zarr": zarr.__version__, "numcodecs": numcodecs.__version__,
            "matplotlib": matplotlib.__version__,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        }
        expected_versions = {
            "python": "3.13.5", "executable": PYTHON,
            "numpy": "2.1.3", "scipy": "1.15.3",
            "torch": "2.9.0+cu129", "cuda": "12.9",
            "zarr": "2.18.7", "numcodecs": "0.15.1",
            "matplotlib": "3.10.0", "gpu": "NVIDIA GeForce RTX 5090 D",
        }
        if versions != expected_versions:
            raise RuntimeError(f"composition-anchor environment drift: {versions}")
        write_json(run / "config/source_integrity_before.json", before)
        write_json(run / "RUN_STATE.json", {
            "schema_version": "v3_run_state_v1", "run_id": RUN_ID, "state": "RUNNING",
        })

        source_checks = {
            "p1a_complete": load_json(P1A / "RUN_STATE.json").get("state") == "COMPLETED",
            "p1b_complete": load_json(P1B / "RUN_STATE.json").get("state") == "COMPLETED",
            "sidecar_complete": load_json(SIDECAR / "RUN_STATE.json").get("state") == "COMPLETED",
            "anchor_teacher_pass": load_json(
                ANCHOR_TEACHER / "metrics/teacher_readiness/summary.json"
            ).get("scientific_pass") is True,
            "predicted_geometry_requires_anchor": load_json(
                DIAGNOSTIC / "metrics/diagnostic/summary.json"
            ).get("decision") == "ALLOW_OE_COMPOSITION_ANCHOR_RESIDUAL_UNCERTAINTY_MODEL_READINESS",
            "c08_still_unread": load_json(DIAGNOSTIC / "metrics/summary.json").get("c08_rows_read") == 0,
        }
        if not all(source_checks.values()):
            raise RuntimeError(f"composition-anchor readiness trigger drift: {source_checks}")

        tests = subprocess.run(
            [
                PYTHON, "-m", "pytest", "-q",
                "tests/v3/unit/test_primitive_composition_anchor_model.py",
                "tests/v3/unit/test_primitive_composition_anchor_teacher.py",
                "tests/v3/unit/test_primitive_relation_observable_model.py",
            ],
            cwd=PROJECT_ROOT,
            env={**os.environ, "PYTHONPATH": f"{PROJECT_ROOT / 'src'}:{PROJECT_ROOT}"},
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, check=False,
        )
        (run / "logs/unit_tests.log").write_text(tests.stdout, encoding="utf-8")
        if tests.returncode or f"{EXPECTED_TESTS} passed" not in tests.stdout:
            raise RuntimeError(f"expected exactly {EXPECTED_TESTS} composition-anchor tests")

        sensor_path = P1A / "artifacts/dataset/fit" / TASK_ZARR
        teacher_path = P1B / "artifacts/teacher/fit" / TASK_ZARR
        sidecar_path = SIDECAR / "artifacts/endpoint_observability/fit" / TASK_ZARR
        construction_path = P1A / "artifacts/constructions/fit" / f"{TASK}.json"
        selected_hashes = {
            "sensor": _tree_hash(sensor_path),
            "teacher": _tree_hash(teacher_path),
            "sidecar": _tree_hash(sidecar_path),
            "construction": _sha(construction_path),
        }
        if selected_hashes != spec.get("selected_real_data_hashes"):
            raise RuntimeError("composition-anchor selected real data hash drift")
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
            raise RuntimeError("composition-anchor selected real row drift")
        construction = load_json(construction_path)
        table = construction_endpoint_anchor_table(construction)
        teacher_group = zarr.open_group(str(teacher_path), mode="r")
        frame_row = int(np.asarray(teacher_group["frame_row"][ROW], dtype=np.int64)[-1])
        sensor_group = zarr.open_group(str(sensor_path), mode="r")
        target_numpy = current_sensor_anchor_targets(
            primitive_index=numpy_batch.base.primitive_index,
            primitive_mask=numpy_batch.base.primitive_mask,
            anchor_world_m=table.anchor_world_m,
            sensor_xyz_m=np.asarray(sensor_group["sensor_xyz_m"].oindex[[frame_row]], dtype=np.float64),
            yaw_deg=np.asarray(sensor_group["yaw_deg"].oindex[[frame_row]], dtype=np.float64),
        ).astype(np.float32)
        target_numpy[~numpy_batch.base.primitive_mask.astype(bool)] = 0.0

        if not torch.cuda.is_available() or os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
            raise RuntimeError("composition-anchor readiness requires frozen deterministic CUDA")
        device = torch.device("cuda")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
        torch.use_deterministic_algorithms(True)
        torch.manual_seed(202609030)
        torch.cuda.manual_seed_all(202609030)
        torch.cuda.reset_peak_memory_stats()
        backbone = _load_backbone(TRAINING / "artifacts/models/seed0/selected.pt")
        model = FrozenObservableCompositionAnchorNet(backbone).to(device).train()
        batch = observable_numpy_batch_to_torch(numpy_batch, device=device)
        target_anchor = torch.from_numpy(target_numpy).to(device=device)

        backbone_parameters = sum(value.numel() for value in model.backbone.parameters())
        head_parameters = sum(value.numel() for value in model.anchor_head.parameters())
        trainable_parameters = sum(value.numel() for value in model.parameters() if value.requires_grad)
        if (
            backbone_parameters != EXPECTED_BACKBONE_PARAMETERS
            or head_parameters != EXPECTED_HEAD_PARAMETERS
            or trainable_parameters != EXPECTED_HEAD_PARAMETERS
        ):
            raise RuntimeError("composition-anchor parameter boundary drift")

        prediction = model(
            batch.range_valid, batch.relative_translation_current_sensor_m,
            batch.relative_yaw_current_sensor_deg,
        )
        assignments = match_primitives(prediction.primitive, batch.targets)
        aligned = align_composition_anchor_targets(target_anchor, batch.targets, assignments)
        axis = prediction.primitive.axis_control_current_sensor_m
        segment_vector = torch.stack((
            axis[:, :, 0] - axis[:, :, 1], axis[:, :, 2] - axis[:, :, 1],
        ), dim=2)
        degenerate_endpoint = torch.linalg.vector_norm(segment_vector, dim=-1) < 1e-4
        matched_endpoint = aligned.primitive_mask[:, :, None].expand(-1, -1, 2)
        degenerate_endpoints = int(degenerate_endpoint.sum())
        degenerate_matched_endpoints = int((degenerate_endpoint & matched_endpoint).sum())
        degenerate_observed_endpoints = int((degenerate_endpoint & aligned.endpoint_observed).sum())
        radial_degenerate_endpoint = (
            torch.linalg.vector_norm(axis[:, :, (0, 2), :2], dim=-1) < 1e-4
        )
        radial_degenerate_observed_endpoints = int((
            radial_degenerate_endpoint & aligned.endpoint_observed
        ).sum())
        losses = composition_anchor_training_losses(prediction.composition, aligned)
        losses["total"].backward()
        head_gradients = [value.grad for value in model.anchor_head.parameters()]
        backbone_gradients = [value.grad for value in model.backbone.parameters()]
        head_gradients_finite_nonzero = all(
            value is not None and bool(torch.isfinite(value).all())
            and float(value.abs().max()) > 0.0
            for value in head_gradients
        )
        backbone_gradients_absent = all(value is None for value in backbone_gradients)

        observed_flat = aligned.endpoint_observed.reshape(1, 64)
        attachment_flat = aligned.attachment.reshape(1, 64, 64).bool()
        overlap_flat = aligned.disconnected_overlap.bool().repeat_interleave(2, 1).repeat_interleave(2, 2)
        endpoint_index = torch.arange(64, device=device)
        upper = (endpoint_index[:, None] // 2 != endpoint_index[None, :] // 2) & torch.triu(
            torch.ones(64, 64, dtype=torch.bool, device=device), diagonal=1,
        )
        eligible = observed_flat[:, :, None] & observed_flat[:, None, :] & upper[None]
        positive_pairs = int((eligible & attachment_flat).sum())
        overlap_hard_negatives = int((eligible & overlap_flat & ~attachment_flat).sum())

        model.zero_grad(set_to_none=True)
        model.eval()
        permutation = torch.randperm(32, generator=torch.Generator().manual_seed(47)).to(device)
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
                getattr(reference.composition, name) - getattr(repeat.composition, name)
            )).cpu())
            for name in reference.composition.__dict__
        )
        permutation_error, absolute_query_anchor_error = _query_permutation_metrics(
            reference, changed, permutation,
        )
        rotated_primitive = replace(
            reference.primitive,
            axis_control_current_sensor_m=_rotate_yaw(
                reference.primitive.axis_control_current_sensor_m, 61.0,
            ),
        )
        with torch.no_grad():
            rotated = model.anchor_head(rotated_primitive)
        yaw_anchor_error = float(torch.max(torch.abs(
            rotated.anchor_current_sensor_m
            - _rotate_yaw(reference.composition.anchor_current_sensor_m, 61.0)
        )).cpu())
        yaw_scale_error = float(torch.max(torch.abs(
            rotated.scale_m - reference.composition.scale_m
        )).cpu())
        yaw_compatibility_error = float(torch.max(torch.abs(
            rotated.compatibility_logits - reference.composition.compatibility_logits
        )).cpu())
        peak_gpu = int(torch.cuda.max_memory_allocated())

        output_dir = run / "previews"
        _plot(
            reference.primitive.axis_control_current_sensor_m[:, :, (0, 2)].cpu().numpy()[0],
            reference.composition.anchor_current_sensor_m.cpu().numpy()[0],
            aligned.anchor_current_sensor_m.cpu().numpy()[0],
            aligned.endpoint_observed.cpu().numpy()[0],
            output_dir,
        )
        student_parameters = set(inspect.signature(model.forward).parameters)
        forbidden = {"world", "node_id", "primitive_id", "construction", "absolute_pose"}
        output_values = 64 * COMPOSITION_ANCHOR_OUTPUTS_PER_ENDPOINT
        pair_values = (64 * 64 - 32 * 4) // 2
        checks = {
            "qualified_sources": all(source_checks.values()),
            f"exact_{EXPECTED_TESTS}_tests": True,
            "real_fit_row_and_construction_targets_exact": True,
            "exact_backbone_and_head_parameter_boundary": (
                backbone_parameters == EXPECTED_BACKBONE_PARAMETERS
                and head_parameters == EXPECTED_HEAD_PARAMETERS
                and trainable_parameters == EXPECTED_HEAD_PARAMETERS
            ),
            "real_row_has_positive_and_overlap_hard_negative": (
                positive_pairs > 0 and overlap_hard_negatives > 0
            ),
            "polar_frame_covers_observed_degenerate_axis_endpoint": (
                degenerate_endpoints > 0
                and degenerate_matched_endpoints > 0
                and degenerate_observed_endpoints > 0
                and radial_degenerate_observed_endpoints == 0
            ),
            "six_losses_finite": len(losses) == 6 and all(
                bool(torch.isfinite(value)) for value in losses.values()
            ),
            "all_anchor_head_gradients_finite_nonzero": head_gradients_finite_nonzero,
            "all_frozen_backbone_gradients_absent": backbone_gradients_absent,
            "determinism_error_zero": determinism_error == 0.0,
            "query_permutation_error_le_3e5": permutation_error <= 3e-5,
            "absolute_query_anchor_error_le_1e4_m": absolute_query_anchor_error <= 1e-4,
            "conditional_yaw_equivariance_le_3e5": max(
                yaw_anchor_error, yaw_scale_error, yaw_compatibility_error,
            ) <= 3e-5,
            "student_forward_has_no_teacher_identity": not bool(student_parameters & forbidden),
            "linear_output_7p75x_smaller_than_pair": (
                output_values == 256 and pair_values == 1984
                and pair_values / output_values == 7.75
            ),
            "peak_cuda_allocation_le_4gib": peak_gpu <= 4 * 1024**3,
            "zero_optimizer_checkpoint_c07_c08_graph_planner": True,
        }
        overall = PASS if all(checks.values()) else FAIL
        payload = {
            "schema_version": "primitive_composition_anchor_model_readiness_v1",
            "overall_status": overall,
            "scientific_pass": overall == PASS,
            "checks": checks,
            "source_checks": source_checks,
            "losses": {name: float(value.detach().cpu()) for name, value in losses.items()},
            "backbone_parameters": backbone_parameters,
            "head_parameters": head_parameters,
            "trainable_parameters": trainable_parameters,
            "learned_output_values_per_row": output_values,
            "independent_pair_values_per_row": pair_values,
            "output_reduction_ratio": pair_values / output_values,
            "selected_task": TASK,
            "selected_row": ROW,
            "source_global_sequence_index": int(numpy_batch.base.source_global_sequence_index[0]),
            "active_primitives": int(numpy_batch.base.primitive_mask.sum()),
            "observed_endpoints": int(numpy_batch.endpoint_observed.sum()),
            "positive_pairs": positive_pairs,
            "overlap_hard_negatives": overlap_hard_negatives,
            "degenerate_endpoints": degenerate_endpoints,
            "degenerate_matched_endpoints": degenerate_matched_endpoints,
            "degenerate_observed_endpoints": degenerate_observed_endpoints,
            "radial_degenerate_observed_endpoints": radial_degenerate_observed_endpoints,
            "selected_hashes": selected_hashes,
            "determinism_error": determinism_error,
            "query_permutation_error": permutation_error,
            "absolute_query_anchor_error_m": absolute_query_anchor_error,
            "yaw_anchor_error": yaw_anchor_error,
            "yaw_scale_error": yaw_scale_error,
            "yaw_compatibility_error": yaw_compatibility_error,
            "peak_gpu_memory_bytes": peak_gpu,
            "optimizer_steps": 0,
            "checkpoint_writes": 0,
            "c07_rows_read": 0,
            "c08_rows_read": 0,
            "c09_c10_worlds_read": 0,
            "graph_replays": 0,
            "mtare_worlds_read": 0,
            "decision": (
                "ALLOW_COMPOSITION_ANCHOR_TARGET_SIDECAR_AND_HEAD_ONLY_TRAINING_SPEC"
                if overall == PASS
                else "STOP_COMPOSITION_ANCHOR_MODEL_READINESS_FAILED"
            ),
            "duration_seconds": time.monotonic() - started,
            "error": None,
        }
        write_json(run / "metrics/summary.json", payload)
        write_json(run / "artifacts/real_batch_contract.json", {
            key: payload[key] for key in (
                "selected_task", "selected_row", "source_global_sequence_index",
                "active_primitives", "observed_endpoints", "positive_pairs",
                "overlap_hard_negatives", "degenerate_endpoints",
                "degenerate_matched_endpoints", "degenerate_observed_endpoints",
                "radial_degenerate_observed_endpoints",
                "selected_hashes",
            )
        })
        write_json(run / "artifacts/figure_source.json", {
            "schema_version": "primitive_composition_anchor_model_readiness_figure_v1",
            "scope": "UNTRAINED_READINESS_ONLY_NOT_PERFORMANCE",
            "selected_task": TASK,
            "selected_row": ROW,
            "raw_predicted_endpoint": reference.primitive.axis_control_current_sensor_m[
                :, :, (0, 2)
            ].cpu().numpy()[0].tolist(),
            "untrained_anchor": reference.composition.anchor_current_sensor_m.cpu().numpy()[0].tolist(),
            "construction_target": aligned.anchor_current_sensor_m.cpu().numpy()[0].tolist(),
            "endpoint_observed": aligned.endpoint_observed.cpu().numpy()[0].astype(np.uint8).tolist(),
        })
        write_json(run / "config/environment.json", {
            "versions": versions,
            "platform": platform.platform(),
            "deterministic_algorithms": True,
            "tf32": {"matmul": False, "cudnn": False},
            "float32_matmul_precision": "highest",
            "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"],
        })
        after = {relative: _sha(PROJECT_ROOT / relative) for relative in before}
        if after != before:
            raise RuntimeError("composition-anchor frozen source changed")
        write_json(run / "config/source_integrity_after.json", after)
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        (run / "logs/failure_traceback.log").write_text(traceback.format_exc(), encoding="utf-8")
        write_json(run / "metrics/summary.json", {
            "schema_version": "primitive_composition_anchor_model_readiness_v1",
            "overall_status": FAIL, "scientific_pass": False, "error": error,
            "optimizer_steps": 0, "c07_rows_read": 0, "c08_rows_read": 0,
            "graph_replays": 0, "mtare_worlds_read": 0,
        })
    write_json(run / "RUN_STATE.json", {
        "schema_version": "v3_run_state_v1", "run_id": RUN_ID,
        "state": "COMPLETED" if error is None else "FAILED",
        "overall_status": overall, "error": error,
        "duration_seconds": time.monotonic() - started,
    })
    entries = _seal(run)
    print(json.dumps({
        "overall_status": overall, "error": error,
        "decision": payload.get("decision"), "evidence_files": entries,
    }, indent=2))
    return 0 if overall == PASS and error is None else 2


if __name__ == "__main__":
    raise SystemExit(main())
