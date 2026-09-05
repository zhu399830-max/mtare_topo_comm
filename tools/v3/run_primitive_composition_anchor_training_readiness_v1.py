#!/usr/bin/env python3
"""Formal batch-128 zero-optimizer readiness for anchor-head training."""

from __future__ import annotations

import argparse
import hashlib
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
import torch
import zarr

from _bootstrap import PROJECT_ROOT
from mtare_topo.data.primitive_composition_anchor_batches import (
    CompositionAnchorBatchLoader,
)
from mtare_topo.governance import load_json, validate_data_card, write_json
from train_primitive_composition_anchor_v1 import (
    EVALUATION_BATCH_SIZE,
    EXPECTED_BACKBONE_PARAMETERS,
    EXPECTED_C07_ROWS,
    EXPECTED_C07_TASKS,
    EXPECTED_FIT_ROWS,
    EXPECTED_FIT_TASKS,
    EXPECTED_HEAD_PARAMETERS,
    EXPECTED_HEAD_TENSORS,
    TRAINING_BATCH_SIZE,
    batch_losses,
    load_frozen_model,
)
from train_primitive_relation_model_v1 import _configure_determinism, _process_memory_bytes


RUN_ID = "gate3_20260903_primitive_composition_anchor_training_readiness_v1_seed0"
PASS = "PASS_PRIMITIVE_COMPOSITION_ANCHOR_TRAINING_READINESS_V1"
FAIL = "FAIL_PRIMITIVE_COMPOSITION_ANCHOR_TRAINING_READINESS_V1"
CARD_STATUS = "APPROVED_FOR_ONE_IMMUTABLE_PRIMITIVE_COMPOSITION_ANCHOR_TRAINING_READINESS_V1"
PYTHON = "/home/zeng-workstation/.local/share/mtare_topo_comm/envs/phase3_torch290_cu129_zarr2187_v1/bin/python"
P1A = PROJECT_ROOT / "results/gate3_semantics/gate3_20260830_primitive_relation_p1a_lossless_storage_corrective_v1_seed0"
P1B = PROJECT_ROOT / "results/gate3_semantics/gate3_20260830_primitive_relation_p1b_teacher_materialization_v1r_seed0"
OBSERVABILITY = PROJECT_ROOT / "results/gate3_semantics/gate3_20260902_primitive_attachment_observability_sidecar_v1_seed0"
SOURCE_MODELS = PROJECT_ROOT / "results/gate3_semantics/gate3_20260902_primitive_relation_observable_three_seed_training_v1_seed0"
ANCHORS = PROJECT_ROOT / "results/gate3_semantics/gate3_20260903_primitive_composition_anchor_target_sidecar_v1r_seed0"
MODEL_READINESS = PROJECT_ROOT / "results/gate3_semantics/gate3_20260903_primitive_composition_anchor_model_readiness_v2_seed0"
DIAGNOSTIC = PROJECT_ROOT / "results/gate3_semantics/gate3_20260903_primitive_predicted_geometry_association_diagnostic_v1_seed0"
EXPECTED_TESTS = 33


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _seal(run: Path) -> int:
    target = run / "artifacts/evidence_sha256.txt"
    files = sorted(path for path in run.rglob("*") if path.is_file() and path != target)
    target.write_text("".join(
        f"{_sha(path)}  {path.relative_to(PROJECT_ROOT)}\n" for path in files
    ), encoding="utf-8")
    return len(files)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    args = parser.parse_args()
    spec = load_json(args.spec.resolve())
    run = args.run_dir.resolve()
    started = time.monotonic()
    overall, error = FAIL, None
    checks: dict[str, bool] = {}
    before: dict[str, str] = {}
    payload: dict = {}
    try:
        if run.name != RUN_ID or load_json(run / "RUN_STATE.json").get("state") != "CREATED_NOT_EXECUTED":
            raise RuntimeError("composition-anchor training readiness executes exactly once")
        card = load_json(PROJECT_ROOT / spec["data_card"])
        validation = validate_data_card(card)
        if not validation.passed or card.get("status") != CARD_STATUS:
            raise RuntimeError(f"composition-anchor readiness Data Card invalid: {validation.errors}")
        for relative, expected in spec["frozen_inputs"].items():
            before[relative] = _sha(PROJECT_ROOT / relative)
            if before[relative] != expected:
                raise RuntimeError(f"composition-anchor readiness input drift: {relative}")
        for record in spec["frozen_tools"].values():
            if _sha(PROJECT_ROOT / record["path"]) != record["sha256"]:
                raise RuntimeError(f"composition-anchor readiness tool drift: {record['path']}")
        environment = {
            "python": sys.version.split()[0], "executable": sys.executable,
            "numpy": np.__version__, "torch": torch.__version__,
            "cuda": torch.version.cuda, "zarr": zarr.__version__,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        }
        expected_environment = {
            "python": "3.13.5", "executable": PYTHON,
            "numpy": "2.1.3", "torch": "2.9.0+cu129",
            "cuda": "12.9", "zarr": "2.18.7",
            "gpu": "NVIDIA GeForce RTX 5090 D",
        }
        if environment != expected_environment:
            raise RuntimeError(f"composition-anchor readiness environment drift: {environment}")
        prerequisites = {
            "anchor_sidecar_pass": load_json(ANCHORS / "metrics/summary.json").get("scientific_pass") is True,
            "model_readiness_pass": load_json(MODEL_READINESS / "metrics/summary.json").get("scientific_pass") is True,
            "diagnostic_requires_anchor": load_json(
                DIAGNOSTIC / "metrics/diagnostic/summary.json"
            ).get("decision") == "ALLOW_OE_COMPOSITION_ANCHOR_RESIDUAL_UNCERTAINTY_MODEL_READINESS",
        }
        if not all(prerequisites.values()):
            raise RuntimeError(f"composition-anchor readiness prerequisite drift: {prerequisites}")
        checkpoint_hashes = {}
        for seed in range(3):
            path = SOURCE_MODELS / f"artifacts/models/seed{seed}/selected.pt"
            model, checkpoint = load_frozen_model(path, seed=seed)
            if int(checkpoint["seed"]) != seed:
                raise RuntimeError("composition-anchor source seed drift")
            checkpoint_hashes[f"seed{seed}"] = _sha(path)
            del model
        write_json(run / "config/environment.json", {
            "versions": environment, "platform": platform.platform(),
        })
        write_json(run / "config/source_integrity_before.json", before)
        write_json(run / "config/source_checkpoint_hashes.json", checkpoint_hashes)
        write_json(run / "RUN_STATE.json", {
            "schema_version": "v3_run_state_v1", "run_id": RUN_ID, "state": "RUNNING",
        })
        env = os.environ.copy()
        env["PYTHONPATH"] = f"{PROJECT_ROOT / 'src'}:{PROJECT_ROOT / 'tools/v3'}:{PROJECT_ROOT}"
        env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        tests = [
            "tests/v3/unit/test_primitive_composition_anchor_batches.py",
            "tests/v3/unit/test_primitive_composition_anchor_sidecar.py",
            "tests/v3/unit/test_primitive_composition_anchor_teacher.py",
            "tests/v3/unit/test_primitive_composition_anchor_model.py",
            "tests/v3/unit/test_evaluate_primitive_composition_anchor_three_seed_v1.py",
        ]
        with (run / "logs/00_unit_tests.log").open("w", encoding="utf-8") as stream:
            tests_run = subprocess.run(
                [PYTHON, "-m", "pytest", "-q", *tests], cwd=PROJECT_ROOT, env=env,
                stdout=stream, stderr=subprocess.STDOUT, text=True, timeout=600, check=False,
            )
        test_log = (run / "logs/00_unit_tests.log").read_text(encoding="utf-8")
        if tests_run.returncode or f"{EXPECTED_TESTS} passed" not in test_log:
            raise RuntimeError(f"expected exactly {EXPECTED_TESTS} composition-anchor tests")

        fit = CompositionAnchorBatchLoader(
            P1A / "artifacts/dataset/fit", P1B / "artifacts/teacher/fit",
            OBSERVABILITY / "artifacts/endpoint_observability/fit",
            ANCHORS / "artifacts/materialized/anchor_targets/fit",
        )
        c07 = CompositionAnchorBatchLoader(
            P1A / "artifacts/dataset/c07", P1B / "artifacts/teacher/c07",
            OBSERVABILITY / "artifacts/endpoint_observability/c07",
            ANCHORS / "artifacts/materialized/anchor_targets/c07",
        )
        if (
            len(fit) != EXPECTED_FIT_ROWS or len(fit.task_names) != EXPECTED_FIT_TASKS
            or len(c07) != EXPECTED_C07_ROWS or len(c07.task_names) != EXPECTED_C07_TASKS
        ):
            raise RuntimeError("composition-anchor readiness loader population drift")
        _configure_determinism(0)
        device = torch.device("cuda")
        numpy_batch = next(fit.iter_epoch(
            batch_size=TRAINING_BATCH_SIZE, seed=0, epoch=0, shuffle=False,
        ))
        model, _ = load_frozen_model(
            SOURCE_MODELS / "artifacts/models/seed0/selected.pt", seed=0,
        )
        model = model.to(device).train()
        trainable = [value for value in model.parameters() if value.requires_grad]
        frozen = [value for value in model.parameters() if not value.requires_grad]
        torch.cuda.reset_peak_memory_stats(device)
        losses, prediction, _ = batch_losses(model, numpy_batch, device=device)
        first_anchor = prediction.composition.anchor_current_sensor_m.detach().clone()
        first_scale = prediction.composition.scale_m.detach().clone()
        first_compatibility = prediction.composition.compatibility_logits.detach().clone()
        losses["total"].backward()
        gradient_checks = {
            "all_head_gradients_finite_nonzero": all(
                value.grad is not None and bool(torch.isfinite(value.grad).all())
                and bool((value.grad != 0).any()) for value in trainable
            ),
            "all_backbone_gradients_absent": all(value.grad is None for value in frozen),
        }
        model.zero_grad(set_to_none=True)
        with torch.no_grad():
            repeated_losses, repeated_prediction, _ = batch_losses(
                model, numpy_batch, device=device,
            )
        repeat_errors = {
            "anchor": float(torch.max(torch.abs(
                repeated_prediction.composition.anchor_current_sensor_m - first_anchor
            )).cpu()),
            "scale": float(torch.max(torch.abs(
                repeated_prediction.composition.scale_m - first_scale
            )).cpu()),
            "compatibility": float(torch.max(torch.abs(
                repeated_prediction.composition.compatibility_logits - first_compatibility
            )).cpu()),
            "loss": max(
                abs(float(repeated_losses[name].cpu()) - float(losses[name].detach().cpu()))
                for name in losses
            ),
        }
        peak_cuda = int(torch.cuda.max_memory_allocated(device))
        process_memory = int(_process_memory_bytes())
        peak_rss_kib = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        payload = {
            "rows_per_forward": len(numpy_batch.base.base.range_valid),
            "model_forward_rows": 2 * len(numpy_batch.base.base.range_valid),
            "training_batch_size": TRAINING_BATCH_SIZE,
            "evaluation_batch_size": EVALUATION_BATCH_SIZE,
            "fit_rows": len(fit), "fit_tasks": len(fit.task_names),
            "c07_rows": len(c07), "c07_tasks": len(c07.task_names),
            "backbone_parameters": sum(value.numel() for value in model.backbone.parameters()),
            "head_parameters": sum(value.numel() for value in trainable),
            "head_parameter_tensors": len(trainable),
            "losses": {name: float(value.detach().cpu()) for name, value in losses.items()},
            "gradient_checks": gradient_checks,
            "repeat_maximum_errors": repeat_errors,
            "peak_cuda_allocated_bytes": peak_cuda,
            "gpu_process_memory_bytes": process_memory,
            "peak_host_rss_kib": peak_rss_kib,
            "optimizer_steps": 0, "checkpoint_writes": 0,
            "c08_rows_read": 0, "c09_c10_worlds_read": 0,
        }
        checks = {
            "exact_33_tests": True,
            "three_source_checkpoints_bound": len(checkpoint_hashes) == 3,
            "exact_fit_population": len(fit) == EXPECTED_FIT_ROWS and len(fit.task_names) == EXPECTED_FIT_TASKS,
            "exact_c07_population_without_model_forward": len(c07) == EXPECTED_C07_ROWS and len(c07.task_names) == EXPECTED_C07_TASKS,
            "batch128_two_forwards": payload["model_forward_rows"] == 256,
            "exact_backbone_parameters": payload["backbone_parameters"] == EXPECTED_BACKBONE_PARAMETERS,
            "exact_head_parameters": payload["head_parameters"] == EXPECTED_HEAD_PARAMETERS,
            "exact_head_tensors": payload["head_parameter_tensors"] == EXPECTED_HEAD_TENSORS,
            **gradient_checks,
            "all_losses_finite": all(np.isfinite(value) for value in payload["losses"].values()),
            "repeat_exact": max(repeat_errors.values()) == 0.0,
            "cuda_allocation_below_16gib": peak_cuda <= 16 * 1024**3,
            "gpu_process_below_16gib": process_memory <= 16 * 1024**3,
            "host_rss_below_4gib": peak_rss_kib <= 4 * 1024**2,
            "zero_optimizer_checkpoint_c08_graph_mtare": True,
        }
        overall = PASS if all(checks.values()) else FAIL
        after = {relative: _sha(PROJECT_ROOT / relative) for relative in before}
        checks["all_frozen_inputs_unchanged"] = after == before
        if not checks["all_frozen_inputs_unchanged"]:
            overall = FAIL
        write_json(run / "config/source_integrity_after.json", after)
        write_json(run / "metrics/summary.json", {
            "schema_version": "primitive_composition_anchor_training_readiness_v1",
            "overall_status": overall, "scientific_pass": overall == PASS,
            "checks": checks, "probe": payload,
            "optimizer_steps": 0, "checkpoint_writes": 0,
            "model_forward_rows": 256, "c07_model_forward_rows": 0,
            "c08_rows_read": 0, "c09_c10_worlds_read": 0,
            "graph_replays": 0, "mtare_worlds_read": 0,
            "peak_rss_kib": peak_rss_kib,
            "duration_seconds": time.monotonic() - started,
            "decision": "ALLOW_COMPOSITION_ANCHOR_HEAD_ONLY_THREE_SEED_TRAINING_CARD"
                if overall == PASS else "STOP_COMPOSITION_ANCHOR_TRAINING_READINESS_FAILED",
            "error": None,
        })
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        (run / "logs/failure_traceback.log").write_text(traceback.format_exc(), encoding="utf-8")
        write_json(run / "metrics/summary.json", {
            "overall_status": FAIL, "scientific_pass": False, "checks": checks,
            "optimizer_steps": 0, "checkpoint_writes": 0,
            "model_forward_rows": payload.get("model_forward_rows", 0),
            "c07_model_forward_rows": 0, "c08_rows_read": 0,
            "c09_c10_worlds_read": 0, "graph_replays": 0, "mtare_worlds_read": 0,
            "error": error,
        })
    write_json(run / "RUN_STATE.json", {
        "schema_version": "v3_run_state_v1", "run_id": RUN_ID,
        "state": "COMPLETED" if error is None else "FAILED",
        "overall_status": overall, "error": error,
    })
    entries = _seal(run)
    print(json.dumps({"overall_status": overall, "error": error, "evidence_files": entries}, indent=2))
    return 0 if error is None else 2


if __name__ == "__main__":
    raise SystemExit(main())
