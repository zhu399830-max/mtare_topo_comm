#!/usr/bin/env python3
"""Tiny-overfit gate for the permutation-invariant structural-node student."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
import zarr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import execute_gse_structural_node_evidence_funnel_v1 as evidence
from mtare_topo.data.primitive_relation_observable_batches import ObservablePrimitiveRelationBatchLoader
from mtare_topo.representation.gse_structural_node_evidence import pair_distance, safest_nonempty_threshold
from mtare_topo.representation.gse_structural_node_student import (
    StructuralNodeAggregationHead,
    structural_node_student_inputs,
    structural_node_student_losses,
)
from mtare_topo.representation.primitive_relation_observable_model import ObservableSparsePortRelationNet
from mtare_topo.representation.primitive_relation_observable_training import observable_numpy_batch_to_torch
from train_primitive_relation_model_v1 import _configure_determinism


FAMILIES = tuple(f"S{index:02d}" for index in range(1, 11))
OBSERVATIONS_PER_FAMILY = 18
OBSERVATIONS = 180
NODES = 100
STEPS = 500
LEARNING_RATE = 3e-3
WEIGHT_DECAY = 0.0


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _state_digest(module: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        array = value.detach().cpu().contiguous().numpy()
        digest.update(name.encode()); digest.update(str(array.dtype).encode())
        digest.update(np.asarray(array.shape, dtype="<i8").tobytes()); digest.update(array.tobytes())
    return digest.hexdigest()


def _selected_rows(construction_path: Path, teacher_path: Path, targets: dict[str, str]) -> list[dict]:
    construction = json.loads(construction_path.read_text(encoding="utf-8"))
    degree = {str(row["node_id"]): int(row["degree"]) for row in construction["base_construction"]["composition_operations"]}
    observations = evidence._task_observations(construction_path, teacher_path, targets)
    by_traversal = {str(row["traversal"]): row for row in observations}
    group = zarr.open_group(str(teacher_path), mode="r")
    traversal = list(group.attrs["traversal_ids"])
    terminal_rows = []
    start = 0
    while start < len(traversal):
        end = start + 1
        while end < len(traversal) and traversal[end] == traversal[start]:
            end += 1
        observation = dict(by_traversal[str(traversal[start])])
        observation["row_index"] = end - 1
        terminal_rows.append(observation)
        start = end
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in terminal_rows:
        grouped[str(row["node"])].append(row)
    selected: list[dict] = []
    terminals = [node for node in sorted(grouped) if degree[node] == 1][:2]
    branches = [node for node in sorted(grouped) if degree[node] >= 3][:2]
    for node in terminals:
        selected.extend(grouped[node][:1])
    for node in branches:
        selected.extend(grouped[node][:2])
    for node in [value for value in sorted(grouped) if degree[value] == 2]:
        if len(selected) >= OBSERVATIONS_PER_FAMILY:
            break
        selected.extend(grouped[node][:2])
    if (
        len(selected) != OBSERVATIONS_PER_FAMILY
        or len({row["node"] for row in selected}) != 10
        or sum(row["degree"] == 1 for row in selected) != 2
        or sum(row["degree"] >= 3 for row in selected) != 4
        or len({row["traversal"] for row in selected}) != len(selected)
    ):
        raise RuntimeError("tiny-overfit family balance contract drift")
    return sorted(selected, key=lambda row: int(row["row_index"]))


def _pair_metrics(descriptor: np.ndarray, identity: np.ndarray, task: np.ndarray) -> dict:
    left, right = np.triu_indices(len(descriptor), 1)
    candidate = task[left] == task[right]
    left, right = left[candidate], right[candidate]
    distance = pair_distance(descriptor[left], descriptor[right])
    target = identity[left] == identity[right]
    safe = safest_nonempty_threshold(distance, target, minimum_precision=.99)
    return {"candidate_pairs": len(distance), "positive_pairs": int(np.sum(target)), "safe_operating_point": safe}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sensor-root", required=True, type=Path)
    parser.add_argument("--teacher-root", required=True, type=Path)
    parser.add_argument("--observability-root", required=True, type=Path)
    parser.add_argument("--construction-root", required=True, type=Path)
    parser.add_argument("--traversal-manifest", required=True, type=Path)
    parser.add_argument("--source-checkpoint", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--steps", type=int, default=STEPS)
    args = parser.parse_args()
    if args.steps != STEPS:
        raise ValueError("tiny-overfit step count drift")
    output = args.output_dir.resolve()
    if output.exists():
        raise RuntimeError("tiny-overfit output exists; overwrite is forbidden")
    output.mkdir(parents=True)
    started = time.monotonic(); _configure_determinism(0)
    if not torch.cuda.is_available():
        raise RuntimeError("tiny-overfit requires the frozen CUDA sidecar")
    device = torch.device("cuda")
    torch.cuda.reset_peak_memory_stats(device)
    checkpoint = torch.load(args.source_checkpoint.resolve(), map_location="cpu", weights_only=False)
    if checkpoint.get("schema_version") != "primitive_relation_observable_checkpoint_v1" or int(checkpoint.get("seed", -1)) != 0:
        raise RuntimeError("tiny-overfit source checkpoint drift")
    backbone = ObservableSparsePortRelationNet(); backbone.load_state_dict(checkpoint["model_state_dict"], strict=True)
    backbone.eval().to(device)
    for parameter in backbone.parameters(): parameter.requires_grad_(False)
    backbone_before = _state_digest(backbone)
    loader = ObservablePrimitiveRelationBatchLoader(args.sensor_root, args.teacher_root, args.observability_root)
    targets = evidence._traversal_targets(args.traversal_manifest.resolve())
    feature_rows = []; confidence_rows = []; target_rows = []; degrees = []
    identities = []; tasks = []; manifest = []
    repeated_cache_error = 0.0
    for family in FAMILIES:
        candidates = sorted(args.construction_root.glob(f"{family}_*_C01__c1_mixed.json"))
        if len(candidates) != 1:
            raise RuntimeError(f"tiny-overfit task selection drift for {family}")
        construction_path = candidates[0]
        task = construction_path.stem
        teacher_path = args.teacher_root / f"{task}.zarr"
        selected = _selected_rows(construction_path, teacher_path, targets)
        indices = np.asarray([row["row_index"] for row in selected], dtype=np.int64)
        numpy_batch = loader._read(f"{task}.zarr", indices)
        batch = observable_numpy_batch_to_torch(numpy_batch, device=device)
        with torch.no_grad():
            prediction = backbone(batch.range_valid, batch.relative_translation_current_sensor_m, batch.relative_yaw_current_sensor_deg)
            features, confidence = structural_node_student_inputs(prediction)
            if family == "S01":
                repeat = backbone(batch.range_valid, batch.relative_translation_current_sensor_m, batch.relative_yaw_current_sensor_deg)
                repeat_features, repeat_confidence = structural_node_student_inputs(repeat)
                repeated_cache_error = max(float(torch.max(torch.abs(features - repeat_features))), float(torch.max(torch.abs(confidence - repeat_confidence))))
        feature_rows.append(features.cpu().numpy()); confidence_rows.append(confidence.cpu().numpy())
        target_rows.extend(np.asarray(row["descriptor"]["causal_five_frame"], dtype=np.float32) for row in selected)
        degrees.extend(int(row["degree"]) for row in selected)
        identities.extend(f"{task}:{row['node']}" for row in selected)
        tasks.extend([task] * len(selected))
        manifest.extend({"family": family, "task": task, "row_index": int(row["row_index"]), "source_global_sequence_index": int(numpy_batch.base.source_global_sequence_index[offset]), "traversal": row["traversal"], "target_node_scoring_only": row["node"], "degree": int(row["degree"])} for offset, row in enumerate(selected))
    features_np = np.concatenate(feature_rows); confidence_np = np.concatenate(confidence_rows)
    targets_np = np.stack(target_rows); degree_np = np.asarray(degrees, dtype=np.int64)
    identity_np = np.asarray(identities); task_np = np.asarray(tasks)
    if len(features_np) != OBSERVATIONS or len(set(identity_np.tolist())) != NODES or repeated_cache_error != 0.0:
        raise RuntimeError("tiny-overfit frozen feature population/determinism drift")
    np.savez_compressed(output / "frozen_tiny_features.npz", endpoint_features=features_np, endpoint_confidence=confidence_np, target_descriptor=targets_np, target_degree=degree_np, source_global_sequence_index=np.asarray([row["source_global_sequence_index"] for row in manifest], dtype=np.int64))
    (output / "sample_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    endpoint_features = torch.from_numpy(features_np).to(device)
    endpoint_confidence = torch.from_numpy(confidence_np).to(device)
    target_descriptor = torch.from_numpy(targets_np).to(device)
    target_degree = torch.from_numpy(degree_np).to(device)
    ids = {value: index for index, value in enumerate(sorted(set(identity_np.tolist())))}
    identity_index = torch.as_tensor([ids[value] for value in identity_np], device=device)
    same_node = identity_index[:, None] == identity_index[None, :]
    torch.manual_seed(2_026_090_401)
    head = StructuralNodeAggregationHead().to(device)
    parameters = [value for value in head.parameters() if value.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    def evaluate() -> tuple[dict, np.ndarray, np.ndarray]:
        head.eval()
        with torch.no_grad():
            prediction = head(endpoint_features, endpoint_confidence)
            losses = structural_node_student_losses(prediction, target_descriptor, target_degree, same_node)
            descriptor = prediction.descriptor.detach().cpu().numpy()
            degree_prediction = torch.argmax(prediction.degree_logits, dim=1).cpu().numpy() + 1
        return ({name: float(value) for name, value in losses.items()}, descriptor, degree_prediction)

    initial_loss, initial_descriptor, initial_degree = evaluate()
    initial_pair = _pair_metrics(initial_descriptor, identity_np, task_np)
    history = []
    all_gradients_finite = True
    for step in range(1, STEPS + 1):
        head.train(); optimizer.zero_grad(set_to_none=True)
        prediction = head(endpoint_features, endpoint_confidence)
        losses = structural_node_student_losses(prediction, target_descriptor, target_degree, same_node)
        losses["total"].backward()
        all_gradients_finite &= all(parameter.grad is not None and bool(torch.isfinite(parameter.grad).all()) for parameter in parameters)
        torch.nn.utils.clip_grad_norm_(parameters, 1.0); optimizer.step()
        if step == 1 or step % 25 == 0:
            history.append({"step": step, **{name: float(value.detach()) for name, value in losses.items()}})
    final_loss, final_descriptor, final_degree = evaluate()
    repeat_loss, repeat_descriptor, repeat_degree = evaluate()
    final_pair = _pair_metrics(final_descriptor, identity_np, task_np)
    geometry_rmse = float(np.sqrt(np.mean(np.square(final_descriptor[:, :-1] - targets_np[:, :-1]))))
    uncertainty_rmse = float(np.sqrt(np.mean(np.square(final_descriptor[:, -1] - targets_np[:, -1]))))
    degree_accuracy = float(np.mean(final_degree == degree_np))
    loss_reduction = 1.0 - final_loss["total"] / initial_loss["total"]
    safe = final_pair["safe_operating_point"]
    checks = {
        "exact_180_observations_100_nodes": len(features_np) == OBSERVATIONS and len(set(identity_np.tolist())) == NODES,
        "ten_families_18_each": all(tasks.count(next(task for task in tasks if task.startswith(family + "_"))) == OBSERVATIONS_PER_FAMILY for family in FAMILIES),
        "frozen_feature_repeat_exact": repeated_cache_error == 0.0,
        "all_head_gradients_finite": all_gradients_finite,
        "total_loss_reduction_ge_0_95": loss_reduction >= .95,
        "geometry_rmse_le_0_01": geometry_rmse <= .01,
        "degree_accuracy_ge_0_99": degree_accuracy >= .99,
        "relation_safe_nonempty": bool(safe["found"]),
        "relation_precision_ge_0_99": float(safe["precision"]) >= .99,
        "relation_recall_ge_0_95": float(safe["recall"]) >= .95,
        "final_repeat_exact": np.array_equal(final_descriptor, repeat_descriptor) and np.array_equal(final_degree, repeat_degree) and final_loss == repeat_loss,
        "backbone_frozen_unchanged": _state_digest(backbone) == backbone_before,
    }
    passed = all(checks.values())
    torch.save({"schema_version": "gse_structural_node_tiny_overfit_checkpoint_v1", "head_state_dict": head.state_dict(), "source_checkpoint": str(args.source_checkpoint.resolve()), "source_checkpoint_sha256": _sha(args.source_checkpoint.resolve()), "steps": STEPS, "learning_rate": LEARNING_RATE, "weight_decay": WEIGHT_DECAY, "checks": checks}, output / "tiny_overfit_head.pt")
    summary = {
        "schema_version": "gse_structural_node_tiny_overfit_v1",
        "status": "PASS_GSE_STRUCTURAL_NODE_TINY_OVERFIT_V1" if passed else "FAIL_GSE_STRUCTURAL_NODE_TINY_OVERFIT_V1",
        "scientific_pass": passed, "checks": checks,
        "population": {"families": 10, "tasks": 10, "observations": OBSERVATIONS, "nodes": NODES, "degree_counts": {str(value): int(np.sum(degree_np == value)) for value in (1, 2, 3, 4)}, "candidate_pairs": final_pair["candidate_pairs"], "positive_pairs": final_pair["positive_pairs"]},
        "training": {"steps": STEPS, "learning_rate": LEARNING_RATE, "weight_decay": WEIGHT_DECAY, "trainable_parameters": sum(value.numel() for value in parameters), "initial_losses": initial_loss, "final_losses": final_loss, "total_loss_reduction": loss_reduction, "geometry_rmse": geometry_rmse, "uncertainty_rmse": uncertainty_rmse, "degree_accuracy": degree_accuracy, "initial_pair_metrics": initial_pair, "final_pair_metrics": final_pair, "history": history, "all_gradients_finite": all_gradients_finite},
        "frozen_cache": {"shape": list(features_np.shape), "confidence_shape": list(confidence_np.shape), "repeat_max_abs_error": repeated_cache_error, "source_checkpoint_sha256": _sha(args.source_checkpoint.resolve())},
        "leakage": {"forward_inputs": ["frozen_endpoint_geometry_features", "frozen_endpoint_confidence"], "teacher_node_identity_forward": False, "absolute_pose_forward": False, "world_identity_forward": False, "future_frame_forward": False},
        "optimizer_steps": STEPS, "model_inference_sequences": OBSERVATIONS + OBSERVATIONS_PER_FAMILY,
        "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "peak_cuda_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        "c07_rows_read": 0, "c08_c09_c10_worlds_read": 0, "graph_replays": 0, "mtare_runs": 0,
        "duration_seconds": time.monotonic() - started,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))
    steps = [row["step"] for row in history]
    for name in ("total", "geometry", "degree", "relation"):
        axes[0].plot(steps, [row[name] for row in history], label=name)
    axes[0].set_yscale("log"); axes[0].set_xlabel("optimizer step"); axes[0].set_ylabel("loss")
    axes[0].set_title("Tiny-overfit convergence"); axes[0].grid(alpha=.25); axes[0].legend(frameon=False)
    axes[1].scatter(targets_np[:, :-1].ravel(), final_descriptor[:, :-1].ravel(), s=3, alpha=.25)
    lo = float(min(targets_np[:, :-1].min(), final_descriptor[:, :-1].min()))
    hi = float(max(targets_np[:, :-1].max(), final_descriptor[:, :-1].max()))
    axes[1].plot((lo, hi), (lo, hi), color="black", linewidth=1)
    axes[1].set_xlabel("Teacher node evidence"); axes[1].set_ylabel("student prediction")
    axes[1].set_title(f"Geometry RMSE = {geometry_rmse:.4f}"); axes[1].grid(alpha=.25)
    figure.tight_layout()
    for suffix in ("png", "pdf", "svg"):
        figure.savefig(output / f"tiny_overfit_diagnostic.{suffix}", dpi=180)
    plt.close(figure)
    print(json.dumps(summary, sort_keys=True))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
