#!/usr/bin/env python3
"""Formal real-batch zero-training readiness for the endpoint relation metric."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
import resource
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
from mtare_topo.data.primitive_relation_observable_batches import ObservablePrimitiveRelationBatchLoader
from mtare_topo.governance import load_json, validate_data_card, write_json
from mtare_topo.representation.primitive_endpoint_relation_metric_decoding import decode_complete_link_clusters
from mtare_topo.representation.primitive_endpoint_relation_metric_model import (
    FrozenObservableEndpointRelationMetricNet,
    endpoint_relation_pair_score,
)
from mtare_topo.representation.primitive_endpoint_relation_metric_training import endpoint_relation_metric_losses
from mtare_topo.representation.primitive_local_composition_slot_training import align_local_composition_slot_targets
from mtare_topo.representation.primitive_relation_losses import match_primitives
from mtare_topo.representation.primitive_relation_observable_training import observable_numpy_batch_to_torch
from run_primitive_local_composition_slot_readiness_v1 import (
    _load_backbone, _process_memory_bytes, _rotate_yaw, _seal, _sha,
    _state_hash, _tree_hash,
)


RUN_ID = "gate3_20260904_primitive_endpoint_relation_metric_readiness_v1_seed0"
PASS = "PASS_PRIMITIVE_ENDPOINT_RELATION_METRIC_READINESS_V1"
FAIL = "FAIL_PRIMITIVE_ENDPOINT_RELATION_METRIC_READINESS_V1"
CARD_STATUS = "APPROVED_FOR_ONE_IMMUTABLE_PRIMITIVE_ENDPOINT_RELATION_METRIC_READINESS_V1"
PYTHON = "/home/zeng-workstation/.local/share/mtare_topo_comm/envs/phase3_torch290_cu129_zarr2187_v1/bin/python"
TASK = "S01_flat_tree_small_C01__c1_mixed.zarr"
BATCH_ROWS = 128
EXPECTED_TESTS = 17
EXPECTED_BACKBONE_PARAMETERS = 2_635_631
EXPECTED_HEAD_PARAMETERS = 426_818
EXPECTED_HEAD_TENSORS = 34
P1A = PROJECT_ROOT / "results/gate3_semantics/gate3_20260830_primitive_relation_p1a_lossless_storage_corrective_v1_seed0"
P1B = PROJECT_ROOT / "results/gate3_semantics/gate3_20260830_primitive_relation_p1b_teacher_materialization_v1r_seed0"
OBS = PROJECT_ROOT / "results/gate3_semantics/gate3_20260902_primitive_attachment_observability_sidecar_v1_seed0"
SOURCE_MODELS = PROJECT_ROOT / "results/gate3_semantics/gate3_20260902_primitive_relation_observable_three_seed_training_v1_seed0"
ATTRIBUTION = PROJECT_ROOT / "results/gate3_semantics/gate3_20260904_primitive_local_composition_slot_failure_attribution_v1_seed0"


def _plot(score: np.ndarray, target: np.ndarray, observed: np.ndarray, output: Path) -> dict:
    selected = np.flatnonzero(observed)[:24]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8))
    first = axes[0].imshow(score[np.ix_(selected, selected)], vmin=0, vmax=1, cmap="viridis")
    axes[0].set_title("untrained learned relation score")
    axes[1].imshow(target[np.ix_(selected, selected)], vmin=0, vmax=1, cmap="Greys")
    axes[1].set_title("construction composition relation")
    fig.colorbar(first, ax=axes[0], fraction=.046); fig.suptitle("Endpoint relation metric readiness — fixed real batch")
    fig.tight_layout()
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(output / f"primitive_endpoint_relation_metric_readiness.{suffix}", dpi=180)
    plt.close(fig)
    return {"endpoint_indices": selected.tolist(), "note": "Zero-training implementation evidence, not performance."}


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--spec", type=Path, required=True); parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args(); spec = load_json(args.spec.resolve()); run = args.run_dir.resolve()
    started = time.monotonic(); overall = FAIL; error = None; checks = {}; payload = {}; before = {}
    try:
        if run.name != RUN_ID or load_json(run / "RUN_STATE.json").get("state") != "CREATED_NOT_EXECUTED":
            raise RuntimeError("endpoint relation metric readiness executes exactly once")
        card = load_json(PROJECT_ROOT / spec["data_card"]); validation = validate_data_card(card)
        if not validation.passed or card.get("status") != CARD_STATUS:
            raise RuntimeError(f"endpoint relation metric Data Card invalid: {validation.errors}")
        for relative, expected in spec["frozen_inputs"].items():
            before[relative] = _sha(PROJECT_ROOT / relative)
            if before[relative] != expected: raise RuntimeError(f"frozen input drift: {relative}")
        for record in spec["frozen_tools"].values():
            if _sha(PROJECT_ROOT / record["path"]) != record["sha256"]: raise RuntimeError(f"frozen tool drift: {record['path']}")
        versions = {
            "python": sys.version.split()[0], "executable": sys.executable,
            "numpy": np.__version__, "scipy": scipy.__version__, "torch": torch.__version__,
            "cuda": torch.version.cuda, "zarr": zarr.__version__, "numcodecs": numcodecs.__version__,
            "matplotlib": matplotlib.__version__, "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        }
        expected = {
            "python": "3.13.5", "executable": PYTHON, "numpy": "2.1.3", "scipy": "1.15.3",
            "torch": "2.9.0+cu129", "cuda": "12.9", "zarr": "2.18.7", "numcodecs": "0.15.1",
            "matplotlib": "3.10.0", "gpu": "NVIDIA GeForce RTX 5090 D",
        }
        if versions != expected or os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8": raise RuntimeError(f"environment drift: {versions}")
        attribution = load_json(ATTRIBUTION / "metrics/summary.json")
        if attribution.get("decision") != "SLOT_IDENTITY_AND_CONFIDENCE_NOT_CROSS_SEED_SAFE" or attribution.get("scientific_pass") is not True:
            raise RuntimeError("endpoint relation metric readiness requires exact attribution PASS")
        write_json(run / "config/source_integrity_before.json", before); write_json(run / "config/environment.json", versions)
        write_json(run / "RUN_STATE.json", {"schema_version": "v3_run_state_v1", "run_id": RUN_ID, "state": "RUNNING"})
        tests = subprocess.run([
            PYTHON, "-m", "pytest", "-q",
            "tests/v3/unit/test_primitive_endpoint_relation_metric.py",
            "tests/v3/unit/test_primitive_local_composition_slot_training.py",
            "tests/v3/unit/test_primitive_relation_observable_model.py",
        ], cwd=PROJECT_ROOT, env={**os.environ, "PYTHONPATH": f"{PROJECT_ROOT / 'src'}:{PROJECT_ROOT}"}, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, check=False)
        (run / "logs/unit_tests.log").write_text(tests.stdout, encoding="utf-8")
        if tests.returncode or f"{EXPECTED_TESTS} passed" not in tests.stdout: raise RuntimeError(f"expected exactly {EXPECTED_TESTS} tests")
        fit = ObservablePrimitiveRelationBatchLoader(P1A / "artifacts/dataset/fit", P1B / "artifacts/teacher/fit", OBS / "artifacts/endpoint_observability/fit")
        c07 = ObservablePrimitiveRelationBatchLoader(P1A / "artifacts/dataset/c07", P1B / "artifacts/teacher/c07", OBS / "artifacts/endpoint_observability/c07")
        population = {"fit_rows": len(fit), "fit_tasks": len(fit.task_names), "c07_rows": len(c07), "c07_tasks": len(c07.task_names)}
        if population != {"fit_rows": 426552, "fit_tasks": 180, "c07_rows": 64644, "c07_tasks": 30}: raise RuntimeError(f"population drift: {population}")
        hashes = {"sensor": _tree_hash(P1A / "artifacts/dataset/fit" / TASK), "teacher": _tree_hash(P1B / "artifacts/teacher/fit" / TASK), "observability": _tree_hash(OBS / "artifacts/endpoint_observability/fit" / TASK)}
        if hashes != spec["selected_real_data_hashes"]: raise RuntimeError("selected real batch drift")
        numpy_batch = fit._read(TASK, np.arange(BATCH_ROWS, dtype=np.int64))
        torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision("highest"); torch.use_deterministic_algorithms(True)
        torch.manual_seed(202609041); torch.cuda.manual_seed_all(202609041); device = torch.device("cuda")
        model = FrozenObservableEndpointRelationMetricNet(_load_backbone(SOURCE_MODELS / "artifacts/models/seed0/selected.pt")).to(device).train()
        batch = observable_numpy_batch_to_torch(numpy_batch, device=device)
        head_parameters = [value for value in model.relation_head.parameters() if value.requires_grad]
        frozen_parameters = list(model.backbone.parameters())
        parameter_counts = {"backbone": sum(value.numel() for value in frozen_parameters), "head": sum(value.numel() for value in head_parameters), "head_tensors": len(head_parameters)}
        if parameter_counts != {"backbone": EXPECTED_BACKBONE_PARAMETERS, "head": EXPECTED_HEAD_PARAMETERS, "head_tensors": EXPECTED_HEAD_TENSORS}: raise RuntimeError(f"parameter drift: {parameter_counts}")
        backbone_before = _state_hash(model.backbone); torch.cuda.reset_peak_memory_stats(device)
        prediction = model(batch.range_valid, batch.relative_translation_current_sensor_m, batch.relative_yaw_current_sensor_deg)
        assignments = match_primitives(prediction.primitive, batch.targets)
        targets = align_local_composition_slot_targets(batch.targets, assignments)
        losses = endpoint_relation_metric_losses(prediction.relation, targets); losses["total"].backward(); torch.cuda.synchronize()
        gradient_checks = {
            "all_head_gradients_finite_nonzero": all(value.grad is not None and bool(torch.isfinite(value.grad).all()) and bool((value.grad != 0).any()) for value in head_parameters),
            "all_backbone_gradients_absent": all(value.grad is None for value in frozen_parameters),
        }
        peak_allocated = int(torch.cuda.max_memory_allocated(device)); peak_reserved = int(torch.cuda.max_memory_reserved(device)); process_memory = _process_memory_bytes(); peak_rss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        model.zero_grad(set_to_none=True); model.eval(); primitive = prediction.primitive
        permutation = torch.randperm(32, generator=torch.Generator().manual_seed(47), device="cpu").to(device)
        endpoint_permutation = (permutation[:, None] * 2 + torch.arange(2, device=device)[None]).reshape(-1)
        with torch.no_grad():
            reference = model.relation_head(primitive); repeat = model.relation_head(primitive)
            permuted_model = model(batch.range_valid, batch.relative_translation_current_sensor_m, batch.relative_yaw_current_sensor_deg, query_permutation=permutation)
            rotated = model.relation_head(replace(primitive, axis_control_current_sensor_m=_rotate_yaw(primitive.axis_control_current_sensor_m, 34.0)))
        permuted = permuted_model.relation
        repeat_error = max(float(torch.max(torch.abs(getattr(reference, name) - getattr(repeat, name))).cpu()) for name in reference.__dict__)
        expected_pair = reference.pair_logits[:, endpoint_permutation][:, :, endpoint_permutation]
        permutation_errors = {
            "embedding": float(torch.max(torch.abs(permuted.embedding - reference.embedding[:, endpoint_permutation])).cpu()),
            "pair_logits": float(torch.max(torch.abs(permuted.pair_logits - expected_pair)).cpu()),
        }
        yaw_errors = {name: float(torch.max(torch.abs(getattr(rotated, name) - getattr(reference, name))).cpu()) for name in reference.__dict__}
        score = endpoint_relation_pair_score(replace(prediction, relation=reference)); active = torch.sigmoid(primitive.existence_logits).repeat_interleave(2, 1) >= .5
        decoded1 = decode_complete_link_clusters(score[:4], active[:4], confidence_threshold=.5); decoded2 = decode_complete_link_clusters(score[:4], active[:4], confidence_threshold=.5)
        target_relation = ((targets.labels[:, :, None] >= 0) & (targets.labels[:, :, None] == targets.labels[:, None, :]))
        target_counts = {"clusters": int(targets.cluster_count.sum()), "clustered_endpoints": int((targets.labels >= 0).sum()), "dustbin_endpoints": int((targets.labels == -1).sum()), "overlap_upper_pairs": int(torch.triu(targets.disconnected_overlap, 1).sum())}
        payload = {
            "population": population, "batch_rows": BATCH_ROWS, "model_forward_rows": BATCH_ROWS,
            "parameter_counts": parameter_counts, "target_counts": target_counts,
            "losses": {name: float(value.detach().cpu()) for name, value in losses.items()}, "gradient_checks": gradient_checks,
            "repeat_maximum_error": repeat_error, "permutation_errors": permutation_errors, "yaw_errors": yaw_errors,
            "pair_score_finite_symmetric": bool(torch.isfinite(score).all()) and torch.equal(score, score.transpose(1, 2)),
            "complete_link_repeat_exact": torch.equal(decoded1[0], decoded2[0]) and torch.equal(decoded1[1], decoded2[1]),
            "backbone_state_before": backbone_before, "backbone_state_after": _state_hash(model.backbone),
            "peak_cuda_allocated_bytes": peak_allocated, "peak_cuda_reserved_bytes": peak_reserved,
            "gpu_process_memory_bytes": process_memory, "peak_host_rss_kib": peak_rss,
            "optimizer_steps": 0, "checkpoint_writes": 0, "c07_model_forward_rows": 0,
            "c08_rows_read": 0, "graph_replays": 0, "mtare_worlds_read": 0,
        }
        checks = {
            "exact_tests": True, "attribution_trigger_bound": True, "exact_population": population == {"fit_rows": 426552, "fit_tasks": 180, "c07_rows": 64644, "c07_tasks": 30},
            "fixed_real_batch128": BATCH_ROWS == 128, "real_batch_has_positive_dustbin_overlap": all(target_counts[key] > 0 for key in target_counts),
            "exact_parameter_boundary": parameter_counts == {"backbone": EXPECTED_BACKBONE_PARAMETERS, "head": EXPECTED_HEAD_PARAMETERS, "head_tensors": EXPECTED_HEAD_TENSORS},
            **gradient_checks, "all_losses_finite": all(np.isfinite(value) for value in payload["losses"].values()),
            "repeat_bit_exact": repeat_error == 0, "primitive_permutation_equivariant": max(permutation_errors.values()) <= 3e-5,
            "yaw_invariant_relation": max(yaw_errors.values()) <= 3e-5, "pair_score_finite_symmetric": payload["pair_score_finite_symmetric"],
            "complete_link_deterministic": payload["complete_link_repeat_exact"], "frozen_backbone_unchanged": payload["backbone_state_before"] == payload["backbone_state_after"],
            "resource_caps": max(peak_allocated, peak_reserved, process_memory) <= 16 * 1024**3 and peak_rss <= 4 * 1024**2,
            "zero_optimizer_checkpoint_c07_c08_graph_mtare": True,
        }
        figure = _plot(score[0].detach().cpu().numpy(), target_relation[0].detach().cpu().numpy(), targets.endpoint_supervised[0].detach().cpu().numpy(), run / "previews")
        write_json(run / "metrics/figure_source.json", figure)
        after = {relative: _sha(PROJECT_ROOT / relative) for relative in before}; checks["all_frozen_inputs_unchanged"] = after == before
        overall = PASS if all(checks.values()) else FAIL; write_json(run / "config/source_integrity_after.json", after)
        write_json(run / "metrics/summary.json", {
            "schema_version": "primitive_endpoint_relation_metric_readiness_v1", "overall_status": overall,
            "scientific_pass": overall == PASS, "checks": checks, "probe": payload,
            "decision": "ALLOW_PRIMITIVE_ENDPOINT_RELATION_METRIC_THREE_SEED_TRAINING_DATA_CARD" if overall == PASS else "STOP_PRIMITIVE_ENDPOINT_RELATION_METRIC_READINESS_FAILED",
            "duration_seconds": time.monotonic() - started, "error": None,
        })
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"; (run / "logs/failure_traceback.log").write_text(traceback.format_exc(), encoding="utf-8")
        write_json(run / "metrics/summary.json", {"schema_version": "primitive_endpoint_relation_metric_readiness_v1", "overall_status": FAIL, "scientific_pass": False, "checks": checks, "probe": payload, "decision": "STOP_PRIMITIVE_ENDPOINT_RELATION_METRIC_READINESS_SYSTEM_FAILURE", "duration_seconds": time.monotonic() - started, "error": error})
    write_json(run / "RUN_STATE.json", {"schema_version": "v3_run_state_v1", "run_id": RUN_ID, "state": "COMPLETED" if error is None else "FAILED", "overall_status": overall, "error": error, "duration_seconds": time.monotonic() - started})
    entries = _seal(run); print(json.dumps({"overall_status": overall, "error": error, "scientific_pass": overall == PASS, "evidence_files": entries}, indent=2)); return 0 if error is None else 2


if __name__ == "__main__": raise SystemExit(main())
