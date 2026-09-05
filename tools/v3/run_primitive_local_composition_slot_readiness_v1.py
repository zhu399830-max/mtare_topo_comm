#!/usr/bin/env python3
"""Formal zero-training readiness for the local composition-slot model."""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
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
from mtare_topo.data.primitive_relation_observable_batches import (
    ObservablePrimitiveRelationBatchLoader,
)
from mtare_topo.governance import load_json, validate_data_card, write_json
from mtare_topo.representation.primitive_local_composition_slot_model import (
    COMPOSITION_SLOT_COUNT,
    ENDPOINT_COUNT,
    FrozenObservableLocalCompositionSlotNet,
    composition_slot_relation_probability,
    local_composition_slot_safe_score,
)
from mtare_topo.representation.primitive_local_composition_slot_training import (
    align_local_composition_slot_targets,
    local_composition_slot_losses,
)
from mtare_topo.representation.primitive_relation_losses import match_primitives
from mtare_topo.representation.primitive_relation_observable_model import (
    ObservableSparsePortRelationNet,
)
from mtare_topo.representation.primitive_relation_observable_training import (
    observable_numpy_batch_to_torch,
)


RUN_ID = "gate3_20260903_primitive_local_composition_slot_readiness_v1_seed0"
PASS = "PASS_PRIMITIVE_LOCAL_COMPOSITION_SLOT_READINESS_V1"
FAIL = "FAIL_PRIMITIVE_LOCAL_COMPOSITION_SLOT_READINESS_V1"
CARD_STATUS = "APPROVED_FOR_ONE_IMMUTABLE_PRIMITIVE_LOCAL_COMPOSITION_SLOT_READINESS_V1"
PYTHON = "/home/zeng-workstation/.local/share/mtare_topo_comm/envs/phase3_torch290_cu129_zarr2187_v1/bin/python"
TASK = "S01_flat_tree_small_C01__c1_mixed.zarr"
BATCH_ROWS = 128
EXPECTED_TESTS = 26
EXPECTED_FIT_ROWS = 426_552
EXPECTED_C07_ROWS = 64_644
EXPECTED_FIT_TASKS = 180
EXPECTED_C07_TASKS = 30
EXPECTED_BACKBONE_PARAMETERS = 2_635_631
EXPECTED_HEAD_PARAMETERS = 1_001_507
EXPECTED_HEAD_TENSORS = 81
P1A = PROJECT_ROOT / "results/gate3_semantics/gate3_20260830_primitive_relation_p1a_lossless_storage_corrective_v1_seed0"
P1B = PROJECT_ROOT / "results/gate3_semantics/gate3_20260830_primitive_relation_p1b_teacher_materialization_v1r_seed0"
OBS = PROJECT_ROOT / "results/gate3_semantics/gate3_20260902_primitive_attachment_observability_sidecar_v1_seed0"
SOURCE_MODELS = PROJECT_ROOT / "results/gate3_semantics/gate3_20260902_primitive_relation_observable_three_seed_training_v1_seed0"
TEACHER_FEASIBILITY = PROJECT_ROOT / "results/gate3_semantics/gate3_20260903_local_composition_slot_teacher_feasibility_v1_seed0"


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


def _state_hash(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        encoded = name.encode("utf-8")
        array = value.detach().cpu().contiguous().numpy()
        digest.update(len(encoded).to_bytes(4, "little")); digest.update(encoded)
        digest.update(str(array.dtype).encode("ascii")); digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
        digest.update(array.tobytes())
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
        raise RuntimeError("local composition-slot backbone checkpoint identity drift")
    model = ObservableSparsePortRelationNet()
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    return model


def _process_memory_bytes() -> int:
    text = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
        text=True,
    )
    for line in text.splitlines():
        fields = [value.strip() for value in line.split(",")]
        if len(fields) == 2 and int(fields[0]) == os.getpid():
            return int(fields[1]) * 1024 * 1024
    raise RuntimeError("local composition-slot CUDA process is absent from nvidia-smi")


def _rotate_yaw(value: torch.Tensor, degrees: float) -> torch.Tensor:
    angle = torch.deg2rad(torch.as_tensor(degrees, dtype=value.dtype, device=value.device))
    zero, one = torch.zeros_like(angle), torch.ones_like(angle)
    rotation = torch.stack((
        torch.stack((torch.cos(angle), -torch.sin(angle), zero)),
        torch.stack((torch.sin(angle), torch.cos(angle), zero)),
        torch.stack((zero, zero, one)),
    ))
    return torch.einsum("...j,kj->...k", value, rotation)


def _plot(
    probability: np.ndarray,
    labels: np.ndarray,
    supervised: np.ndarray,
    output: Path,
) -> dict:
    endpoint = np.flatnonzero(supervised)[:24]
    target = np.zeros((len(endpoint), COMPOSITION_SLOT_COUNT + 1), dtype=np.float32)
    for row, item in enumerate(endpoint):
        label = int(labels[item])
        target[row, label if label >= 0 else COMPOSITION_SLOT_COUNT] = 1.0
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.0), sharey=True)
    first = axes[0].imshow(probability[endpoint].T, aspect="auto", origin="lower", vmin=0.0, vmax=max(0.05, float(probability[endpoint].max())))
    axes[0].set_title("untrained predicted assignment probability")
    axes[0].set_xlabel("observable endpoint"); axes[0].set_ylabel("32 slots + dustbin")
    fig.colorbar(first, ax=axes[0], fraction=0.046)
    axes[1].imshow(target.T, aspect="auto", origin="lower", vmin=0.0, vmax=1.0, cmap="Greys")
    axes[1].set_title("matched construction-program target")
    axes[1].set_xlabel("observable endpoint")
    fig.suptitle("Local composition-slot readiness — fixed C01 batch, zero training")
    fig.tight_layout()
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(output / f"primitive_local_composition_slot_readiness.{suffix}", dpi=180)
    plt.close(fig)
    return {
        "endpoint_indices": endpoint.tolist(),
        "prediction_probability": probability[endpoint].tolist(),
        "teacher_one_hot": target.tolist(),
        "note": "Implementation evidence only; untrained probabilities are not a performance result.",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    args = parser.parse_args()
    spec = load_json(args.spec.resolve()); run = args.run_dir.resolve()
    started = time.monotonic(); overall, error = FAIL, None
    before: dict[str, str] = {}; after: dict[str, str] = {}; payload: dict = {}; checks: dict = {}
    try:
        if run.name != RUN_ID or load_json(run / "RUN_STATE.json").get("state") != "CREATED_NOT_EXECUTED":
            raise RuntimeError("local composition-slot readiness executes exactly once")
        card = load_json(PROJECT_ROOT / spec["data_card"])
        validation = validate_data_card(card)
        if not validation.passed or card.get("status") != CARD_STATUS:
            raise RuntimeError(f"local composition-slot Data Card invalid: {validation.errors}")
        for relative, expected in spec["frozen_inputs"].items():
            before[relative] = _sha(PROJECT_ROOT / relative)
            if before[relative] != expected:
                raise RuntimeError(f"local composition-slot frozen input drift: {relative}")
        for record in spec["frozen_tools"].values():
            if _sha(PROJECT_ROOT / record["path"]) != record["sha256"]:
                raise RuntimeError(f"local composition-slot tool drift: {record['path']}")
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
            raise RuntimeError(f"local composition-slot environment drift: {versions}")
        if not torch.cuda.is_available() or os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
            raise RuntimeError("local composition-slot readiness requires frozen deterministic CUDA")
        write_json(run / "config/source_integrity_before.json", before)
        write_json(run / "config/environment.json", versions)
        write_json(run / "RUN_STATE.json", {
            "schema_version": "v3_run_state_v1", "run_id": RUN_ID, "state": "RUNNING",
        })

        feasibility = load_json(TEACHER_FEASIBILITY / "metrics/teacher_feasibility/summary.json")
        source_checks = {
            "teacher_feasibility_pass": feasibility.get("scientific_pass") is True,
            "teacher_capacity_32": feasibility.get("capacity", {}).get("selected_capacity") == 32,
            "teacher_c07_overflow_zero": feasibility.get("capacity", {}).get("c07_overflow_rows") == 0,
            "source_model_run_completed": load_json(SOURCE_MODELS / "RUN_STATE.json").get("state") == "COMPLETED",
            "p1a_completed": load_json(P1A / "RUN_STATE.json").get("state") == "COMPLETED",
            "p1b_completed": load_json(P1B / "RUN_STATE.json").get("state") == "COMPLETED",
            "observability_completed": load_json(OBS / "RUN_STATE.json").get("state") == "COMPLETED",
        }
        if not all(source_checks.values()):
            raise RuntimeError(f"local composition-slot readiness trigger drift: {source_checks}")

        tests = subprocess.run(
            [
                PYTHON, "-m", "pytest", "-q",
                "tests/v3/unit/test_primitive_local_composition_slot_model.py",
                "tests/v3/unit/test_primitive_local_composition_slot_training.py",
                "tests/v3/unit/test_local_composition_slot_teacher.py",
                "tests/v3/unit/test_primitive_relation_observable_model.py",
            ],
            cwd=PROJECT_ROOT,
            env={**os.environ, "PYTHONPATH": f"{PROJECT_ROOT / 'src'}:{PROJECT_ROOT}"},
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, check=False,
        )
        (run / "logs/unit_tests.log").write_text(tests.stdout, encoding="utf-8")
        if tests.returncode or f"{EXPECTED_TESTS} passed" not in tests.stdout:
            raise RuntimeError(f"expected exactly {EXPECTED_TESTS} local composition-slot tests")

        fit = ObservablePrimitiveRelationBatchLoader(
            P1A / "artifacts/dataset/fit", P1B / "artifacts/teacher/fit",
            OBS / "artifacts/endpoint_observability/fit",
        )
        c07 = ObservablePrimitiveRelationBatchLoader(
            P1A / "artifacts/dataset/c07", P1B / "artifacts/teacher/c07",
            OBS / "artifacts/endpoint_observability/c07",
        )
        population = {
            "fit_rows": len(fit), "fit_tasks": len(fit.task_names),
            "c07_rows": len(c07), "c07_tasks": len(c07.task_names),
        }
        if population != {
            "fit_rows": EXPECTED_FIT_ROWS, "fit_tasks": EXPECTED_FIT_TASKS,
            "c07_rows": EXPECTED_C07_ROWS, "c07_tasks": EXPECTED_C07_TASKS,
        }:
            raise RuntimeError(f"local composition-slot population drift: {population}")
        selected_hashes = {
            "sensor": _tree_hash(P1A / "artifacts/dataset/fit" / TASK),
            "teacher": _tree_hash(P1B / "artifacts/teacher/fit" / TASK),
            "observability": _tree_hash(OBS / "artifacts/endpoint_observability/fit" / TASK),
        }
        if selected_hashes != spec.get("selected_real_data_hashes"):
            raise RuntimeError("local composition-slot selected real data hash drift")
        numpy_batch = fit._read(TASK, np.arange(BATCH_ROWS, dtype=np.int64))

        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
        torch.use_deterministic_algorithms(True)
        torch.manual_seed(202609031); torch.cuda.manual_seed_all(202609031)
        device = torch.device("cuda")
        model = FrozenObservableLocalCompositionSlotNet(
            _load_backbone(SOURCE_MODELS / "artifacts/models/seed0/selected.pt"),
        ).to(device).train()
        backbone_before = _state_hash(model.backbone)
        batch = observable_numpy_batch_to_torch(numpy_batch, device=device)
        head_parameters = [value for value in model.composition_head.parameters() if value.requires_grad]
        frozen_parameters = [value for value in model.backbone.parameters()]
        parameter_counts = {
            "backbone": sum(value.numel() for value in model.backbone.parameters()),
            "head": sum(value.numel() for value in head_parameters),
            "head_tensors": len(head_parameters),
        }
        if parameter_counts != {
            "backbone": EXPECTED_BACKBONE_PARAMETERS,
            "head": EXPECTED_HEAD_PARAMETERS,
            "head_tensors": EXPECTED_HEAD_TENSORS,
        }:
            raise RuntimeError(f"local composition-slot parameter boundary drift: {parameter_counts}")

        torch.cuda.reset_peak_memory_stats(device)
        prediction = model(
            batch.range_valid,
            batch.relative_translation_current_sensor_m,
            batch.relative_yaw_current_sensor_deg,
        )
        assignments = match_primitives(prediction.primitive, batch.targets)
        targets = align_local_composition_slot_targets(batch.targets, assignments)
        losses = local_composition_slot_losses(prediction.composition, targets)
        losses["total"].backward()
        torch.cuda.synchronize()
        gradient_checks = {
            "all_head_gradients_finite_nonzero": all(
                value.grad is not None and bool(torch.isfinite(value.grad).all())
                and bool((value.grad != 0).any()) for value in head_parameters
            ),
            "all_backbone_gradients_absent": all(value.grad is None for value in frozen_parameters),
        }
        peak_allocated = int(torch.cuda.max_memory_allocated(device))
        peak_reserved = int(torch.cuda.max_memory_reserved(device))
        process_memory = _process_memory_bytes()
        peak_rss_kib = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)

        model.zero_grad(set_to_none=True); model.eval()
        permutation = torch.randperm(
            COMPOSITION_SLOT_COUNT,
            generator=torch.Generator().manual_seed(47),
            device=device,
        )
        with torch.no_grad():
            reference = model.composition_head(prediction.primitive)
            repeat = model.composition_head(prediction.primitive)
            permuted = model.composition_head(
                prediction.primitive, composition_slot_permutation=permutation,
            )
            rotated_primitive = replace(
                prediction.primitive,
                axis_control_current_sensor_m=_rotate_yaw(
                    prediction.primitive.axis_control_current_sensor_m, 34.0,
                ),
            )
            rotated = model.composition_head(rotated_primitive)
        repeat_error = max(
            float(torch.max(torch.abs(getattr(reference, name) - getattr(repeat, name))).cpu())
            for name in reference.__dict__
        )
        slot_permutation_errors = {
            "assignment": float(torch.max(torch.abs(
                permuted.assignment_logits[..., :COMPOSITION_SLOT_COUNT]
                - reference.assignment_logits[..., permutation]
            )).cpu()),
            "dustbin": float(torch.max(torch.abs(
                permuted.assignment_logits[..., COMPOSITION_SLOT_COUNT]
                - reference.assignment_logits[..., COMPOSITION_SLOT_COUNT]
            )).cpu()),
            "presence": float(torch.max(torch.abs(
                permuted.slot_presence_logits - reference.slot_presence_logits[:, permutation]
            )).cpu()),
            "relation": float(torch.max(torch.abs(
                composition_slot_relation_probability(permuted)
                - composition_slot_relation_probability(reference)
            )).cpu()),
        }
        reference_losses = local_composition_slot_losses(reference, targets)
        permuted_losses = local_composition_slot_losses(permuted, targets)
        loss_permutation_error = max(
            abs(float(reference_losses[name].cpu()) - float(permuted_losses[name].cpu()))
            for name in reference_losses
        )
        yaw_errors = {
            "assignment": float(torch.max(torch.abs(
                rotated.assignment_logits - reference.assignment_logits
            )).cpu()),
            "presence": float(torch.max(torch.abs(
                rotated.slot_presence_logits - reference.slot_presence_logits
            )).cpu()),
        }
        safe_score = local_composition_slot_safe_score(
            replace(prediction, composition=reference),
        )
        probability = torch.softmax(reference.assignment_logits, dim=-1)
        target_counts = {
            "cluster_count_min": int(targets.cluster_count.min()),
            "cluster_count_max": int(targets.cluster_count.max()),
            "clusters": int(targets.cluster_count.sum()),
            "supervised_endpoints": int(targets.endpoint_supervised.sum()),
            "clustered_endpoints": int((targets.labels >= 0).sum()),
            "dustbin_endpoints": int((targets.labels == -1).sum()),
            "overlap_upper_pairs": int(torch.triu(targets.disconnected_overlap, diagonal=1).sum()),
        }
        backbone_after = _state_hash(model.backbone)
        payload = {
            "population": population,
            "batch_rows": BATCH_ROWS,
            "model_forward_rows": BATCH_ROWS,
            "c07_model_forward_rows": 0,
            "parameter_counts": parameter_counts,
            "target_counts": target_counts,
            "losses": {name: float(value.detach().cpu()) for name, value in losses.items()},
            "gradient_checks": gradient_checks,
            "repeat_maximum_error": repeat_error,
            "slot_permutation_errors": slot_permutation_errors,
            "loss_permutation_error": loss_permutation_error,
            "yaw_errors": yaw_errors,
            "safe_score_symmetric": torch.equal(safe_score, safe_score.transpose(1, 2)),
            "safe_score_finite": bool(torch.isfinite(safe_score).all()),
            "backbone_state_before": backbone_before,
            "backbone_state_after": backbone_after,
            "peak_cuda_allocated_bytes": peak_allocated,
            "peak_cuda_reserved_bytes": peak_reserved,
            "gpu_process_memory_bytes": process_memory,
            "peak_host_rss_kib": peak_rss_kib,
            "optimizer_steps": 0,
            "checkpoint_writes": 0,
            "c08_rows_read": 0,
            "c09_c10_worlds_read": 0,
            "graph_replays": 0,
            "mtare_worlds_read": 0,
        }
        checks = {
            "exact_26_tests": True,
            "source_prerequisites_bound": all(source_checks.values()),
            "exact_fit_c07_population": population == {
                "fit_rows": EXPECTED_FIT_ROWS, "fit_tasks": EXPECTED_FIT_TASKS,
                "c07_rows": EXPECTED_C07_ROWS, "c07_tasks": EXPECTED_C07_TASKS,
            },
            "fixed_real_batch128": payload["batch_rows"] == 128,
            "teacher_cluster_capacity_within_32": target_counts["cluster_count_max"] <= 32,
            "real_batch_has_clusters_dustbin_overlap": (
                target_counts["clusters"] > 0
                and target_counts["dustbin_endpoints"] > 0
                and target_counts["overlap_upper_pairs"] > 0
            ),
            "exact_parameter_boundary": parameter_counts == {
                "backbone": EXPECTED_BACKBONE_PARAMETERS,
                "head": EXPECTED_HEAD_PARAMETERS,
                "head_tensors": EXPECTED_HEAD_TENSORS,
            },
            **gradient_checks,
            "all_losses_finite": all(np.isfinite(value) for value in payload["losses"].values()),
            "repeat_bit_exact": repeat_error == 0.0,
            "slot_permutation_equivariant": max(slot_permutation_errors.values()) <= 3e-6,
            "slot_matched_loss_permutation_invariant": loss_permutation_error <= 3e-6,
            "yaw_invariant_local_relation": max(yaw_errors.values()) <= 3e-5,
            "safe_score_finite_symmetric": payload["safe_score_symmetric"] and payload["safe_score_finite"],
            "frozen_backbone_unchanged": backbone_before == backbone_after,
            "cuda_allocated_below_16gib": peak_allocated <= 16 * 1024**3,
            "cuda_reserved_below_16gib": peak_reserved <= 16 * 1024**3,
            "gpu_process_below_16gib": process_memory <= 16 * 1024**3,
            "host_rss_below_4gib": peak_rss_kib <= 4 * 1024**2,
            "zero_optimizer_checkpoint_c08_graph_mtare": True,
        }
        figure_source = _plot(
            probability[0].detach().cpu().numpy(),
            targets.labels[0].detach().cpu().numpy(),
            targets.endpoint_supervised[0].detach().cpu().numpy(),
            run / "previews",
        )
        write_json(run / "metrics/figure_source.json", figure_source)
        overall = PASS if all(checks.values()) else FAIL
        after = {relative: _sha(PROJECT_ROOT / relative) for relative in before}
        checks["all_frozen_inputs_unchanged"] = after == before
        if not checks["all_frozen_inputs_unchanged"]:
            overall = FAIL
        write_json(run / "config/source_integrity_after.json", after)
        write_json(run / "metrics/summary.json", {
            "schema_version": "primitive_local_composition_slot_readiness_v1",
            "overall_status": overall,
            "scientific_pass": overall == PASS,
            "checks": checks,
            "probe": payload,
            "decision": "ALLOW_PRIMITIVE_LOCAL_COMPOSITION_SLOT_THREE_SEED_TRAINING_DATA_CARD"
                if overall == PASS else "STOP_PRIMITIVE_LOCAL_COMPOSITION_SLOT_READINESS_FAILED",
            "duration_seconds": time.monotonic() - started,
            "error": None,
        })
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        (run / "logs/failure_traceback.log").write_text(traceback.format_exc(), encoding="utf-8")
        write_json(run / "metrics/summary.json", {
            "schema_version": "primitive_local_composition_slot_readiness_v1",
            "overall_status": FAIL, "scientific_pass": False,
            "checks": checks, "probe": payload,
            "optimizer_steps": 0, "checkpoint_writes": 0,
            "model_forward_rows": payload.get("model_forward_rows", 0),
            "c07_model_forward_rows": 0, "c08_rows_read": 0,
            "c09_c10_worlds_read": 0, "graph_replays": 0, "mtare_worlds_read": 0,
            "decision": "STOP_PRIMITIVE_LOCAL_COMPOSITION_SLOT_READINESS_SYSTEM_FAILURE",
            "duration_seconds": time.monotonic() - started, "error": error,
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
        "scientific_pass": overall == PASS,
        "evidence_files": entries,
    }, indent=2))
    return 0 if error is None else 2


if __name__ == "__main__":
    raise SystemExit(main())
