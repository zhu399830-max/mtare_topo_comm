#!/usr/bin/env python3
"""Corrective tiny-overfit for the structural-node dual readout."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from mtare_topo.representation.gse_structural_node_dual_readout import (
    StructuralNodeDualReadout,
    association_distance,
    structural_node_dual_losses,
)
from mtare_topo.representation.gse_structural_node_evidence import (
    safest_nonempty_threshold,
)
from train_primitive_relation_model_v1 import _configure_determinism


OBSERVATIONS = 180
NODES = 100
POSITIVE_PAIRS = 80
STEPS = 500
LEARNING_RATE = 2e-3
FINAL_LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.0
PASS = "PASS_GSE_STRUCTURAL_NODE_TINY_OVERFIT_V1R"
FAIL = "FAIL_GSE_STRUCTURAL_NODE_TINY_OVERFIT_V1R"


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _pair_metrics(embedding: np.ndarray, identity: np.ndarray, task: np.ndarray) -> dict:
    normalized = embedding / np.maximum(np.linalg.norm(embedding, axis=1, keepdims=True), 1e-12)
    left, right = np.triu_indices(len(normalized), 1)
    candidate = task[left] == task[right]
    left, right = left[candidate], right[candidate]
    distance = np.clip(1.0 - np.sum(normalized[left] * normalized[right], axis=1), 0.0, 2.0)
    target = identity[left] == identity[right]
    safe = safest_nonempty_threshold(distance, target, minimum_precision=.99)
    return {
        "candidate_pairs": int(len(distance)),
        "positive_pairs": int(np.sum(target)),
        "same_distance_quantiles": [float(value) for value in np.quantile(distance[target], [0, .5, .9, 1])],
        "different_distance_quantiles": [float(value) for value in np.quantile(distance[~target], [0, .1, .5, 1])],
        "safe_operating_point": safe,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--steps", type=int, default=STEPS)
    args = parser.parse_args()
    if args.steps != STEPS:
        raise ValueError("dual-readout tiny-overfit step count drift")
    output = args.output_dir.resolve()
    if output.exists():
        raise RuntimeError("dual-readout output exists; overwrite is forbidden")
    output.mkdir(parents=True)
    started = time.monotonic()
    _configure_determinism(0)
    if not torch.cuda.is_available():
        raise RuntimeError("dual-readout tiny-overfit requires the frozen CUDA sidecar")
    device = torch.device("cuda")
    torch.cuda.reset_peak_memory_stats(device)

    cache_path = args.cache.resolve()
    manifest_path = args.manifest.resolve()
    cache_sha256_before = _sha(cache_path)
    manifest_sha256_before = _sha(manifest_path)
    cache = np.load(cache_path)
    feature_np = np.asarray(cache["endpoint_features"], dtype=np.float32)
    confidence_np = np.asarray(cache["endpoint_confidence"], dtype=np.float32)
    target_np = np.asarray(cache["target_descriptor"], dtype=np.float32)
    degree_np = np.asarray(cache["target_degree"], dtype=np.int64)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if len(manifest) != OBSERVATIONS or feature_np.shape[0] != OBSERVATIONS:
        raise RuntimeError("dual-readout frozen population drift")
    identity_np = np.asarray([
        f"{row['task']}:{row['target_node_scoring_only']}" for row in manifest
    ])
    task_np = np.asarray([row["task"] for row in manifest])
    family_np = np.asarray([row["family"] for row in manifest])
    if (
        len(set(identity_np.tolist())) != NODES
        or len(set(task_np.tolist())) != 10
        or len(set(family_np.tolist())) != 10
        or any(int(np.sum(family_np == family)) != 18 for family in sorted(set(family_np)))
    ):
        raise RuntimeError("dual-readout manifest contract drift")

    feature = torch.from_numpy(feature_np).to(device)
    confidence = torch.from_numpy(confidence_np).to(device)
    target = torch.from_numpy(target_np).to(device)
    degree = torch.from_numpy(degree_np).to(device)
    identity_lookup = {value: index for index, value in enumerate(sorted(set(identity_np)))}
    task_lookup = {value: index for index, value in enumerate(sorted(set(task_np)))}
    identity_index = torch.as_tensor(
        [identity_lookup[value] for value in identity_np],
        dtype=torch.long,
        device=device,
    )
    task_index = torch.as_tensor(
        [task_lookup[value] for value in task_np],
        dtype=torch.long,
        device=device,
    )
    same_node = identity_index[:, None] == identity_index[None, :]
    candidate = task_index[:, None] == task_index[None, :]
    positive_count = int(torch.triu(same_node & candidate, diagonal=1).sum().item())
    if positive_count != POSITIVE_PAIRS:
        raise RuntimeError("dual-readout positive-pair count drift")

    torch.manual_seed(2_026_090_402)
    model = StructuralNodeDualReadout().to(device)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=STEPS, eta_min=FINAL_LEARNING_RATE,
    )

    def evaluate() -> tuple[dict[str, float], np.ndarray, np.ndarray, np.ndarray]:
        model.eval()
        with torch.no_grad():
            prediction = model(feature, confidence)
            losses = structural_node_dual_losses(prediction, target, degree, same_node, candidate)
            geometry = prediction.geometry.cpu().numpy()
            association = prediction.association.cpu().numpy()
            degree_prediction = torch.argmax(prediction.degree_logits, dim=1).cpu().numpy() + 1
        return (
            {name: float(value) for name, value in losses.items()},
            geometry,
            association,
            degree_prediction,
        )

    initial_loss, initial_geometry, initial_association, initial_degree = evaluate()
    initial_pair = _pair_metrics(initial_association, identity_np, task_np)
    history = []
    all_gradients_finite = True
    for step in range(1, STEPS + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        prediction = model(feature, confidence)
        losses = structural_node_dual_losses(prediction, target, degree, same_node, candidate)
        losses["total"].backward()
        all_gradients_finite &= all(
            parameter.grad is not None and bool(torch.isfinite(parameter.grad).all())
            for parameter in parameters
        )
        torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        optimizer.step()
        scheduler.step()
        if step == 1 or step % 25 == 0:
            history.append({
                "step": step,
                "learning_rate": float(scheduler.get_last_lr()[0]),
                **{name: float(value.detach()) for name, value in losses.items()},
            })

    final_loss, final_geometry, final_association, final_degree = evaluate()
    repeat_loss, repeat_geometry, repeat_association, repeat_degree = evaluate()
    final_pair = _pair_metrics(final_association, identity_np, task_np)
    geometry_rmse = float(np.sqrt(np.mean(np.square(final_geometry[:, :-1] - target_np[:, :-1]))))
    uncertainty_rmse = float(np.sqrt(np.mean(np.square(final_geometry[:, -1] - target_np[:, -1]))))
    degree_accuracy = float(np.mean(final_degree == degree_np))
    loss_reduction = 1.0 - final_loss["total"] / initial_loss["total"]
    safe = final_pair["safe_operating_point"]
    checks = {
        "exact_180_observations_100_nodes": len(feature_np) == OBSERVATIONS and len(set(identity_np)) == NODES,
        "ten_families_18_each": all(int(np.sum(family_np == family)) == 18 for family in sorted(set(family_np))),
        "positive_pairs_80": positive_count == POSITIVE_PAIRS,
        "all_gradients_finite": all_gradients_finite,
        "total_loss_reduction_ge_0_95": loss_reduction >= .95,
        "geometry_rmse_le_0_01": geometry_rmse <= .01,
        "degree_accuracy_ge_0_99": degree_accuracy >= .99,
        "relation_safe_nonempty": bool(safe["found"]),
        "relation_precision_ge_0_99": float(safe["precision"]) >= .99,
        "relation_recall_ge_0_95": float(safe["recall"]) >= .95,
        "final_repeat_exact": (
            np.array_equal(final_geometry, repeat_geometry)
            and np.array_equal(final_association, repeat_association)
            and np.array_equal(final_degree, repeat_degree)
            and final_loss == repeat_loss
        ),
        "source_cache_unchanged": (
            _sha(cache_path) == cache_sha256_before
            and _sha(manifest_path) == manifest_sha256_before
        ),
    }
    passed = all(checks.values())
    checkpoint = {
        "schema_version": "gse_structural_node_tiny_overfit_checkpoint_v1r",
        "model_state_dict": model.state_dict(),
        "source_cache": str(cache_path),
        "source_cache_sha256": cache_sha256_before,
        "source_manifest_sha256": manifest_sha256_before,
        "steps": STEPS,
        "learning_rate": LEARNING_RATE,
        "final_learning_rate": FINAL_LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "checks": checks,
    }
    torch.save(checkpoint, output / "tiny_overfit_dual_readout.pt")
    summary = {
        "schema_version": "gse_structural_node_tiny_overfit_v1r",
        "status": PASS if passed else FAIL,
        "scientific_pass": passed,
        "checks": checks,
        "population": {
            "families": 10,
            "tasks": 10,
            "observations": OBSERVATIONS,
            "nodes": NODES,
            "positive_pairs": positive_count,
            "candidate_pairs": final_pair["candidate_pairs"],
        },
        "training": {
            "steps": STEPS,
            "learning_rate": LEARNING_RATE,
            "final_learning_rate": FINAL_LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "trainable_parameters": sum(parameter.numel() for parameter in parameters),
            "initial_losses": initial_loss,
            "final_losses": final_loss,
            "total_loss_reduction": loss_reduction,
            "geometry_rmse": geometry_rmse,
            "uncertainty_rmse": uncertainty_rmse,
            "degree_accuracy": degree_accuracy,
            "initial_pair_metrics": initial_pair,
            "final_pair_metrics": final_pair,
            "history": history,
            "all_gradients_finite": all_gradients_finite,
        },
        "frozen_source": {
            "cache_sha256": cache_sha256_before,
            "manifest_sha256": manifest_sha256_before,
            "feature_shape": list(feature_np.shape),
            "confidence_shape": list(confidence_np.shape),
        },
        "leakage": {
            "forward_inputs": ["frozen_endpoint_geometry_features", "frozen_endpoint_confidence"],
            "node_identity_forward": False,
            "task_identity_forward": False,
            "absolute_pose_forward": False,
            "future_frame_forward": False,
        },
        "optimizer_steps": STEPS,
        "model_inference_sequences": 0,
        "c07_rows_read": 0,
        "c08_c09_c10_worlds_read": 0,
        "graph_replays": 0,
        "mtare_runs": 0,
        "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "peak_cuda_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        "duration_seconds": time.monotonic() - started,
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )

    figure, axes = plt.subplots(1, 3, figsize=(14.4, 4.0))
    steps = [row["step"] for row in history]
    for name in ("total", "geometry", "degree", "relation"):
        axes[0].plot(steps, [row[name] for row in history], label=name)
    axes[0].set_yscale("log")
    axes[0].set_xlabel("optimizer step")
    axes[0].set_ylabel("loss")
    axes[0].set_title("Dual-readout convergence")
    axes[0].grid(alpha=.25)
    axes[0].legend(frameon=False)
    axes[1].scatter(target_np[:, :-1].ravel(), final_geometry[:, :-1].ravel(), s=3, alpha=.25)
    lo = float(min(target_np[:, :-1].min(), final_geometry[:, :-1].min()))
    hi = float(max(target_np[:, :-1].max(), final_geometry[:, :-1].max()))
    axes[1].plot((lo, hi), (lo, hi), color="black", linewidth=1)
    axes[1].set_xlabel("Teacher structural geometry")
    axes[1].set_ylabel("student geometry")
    axes[1].set_title(f"Geometry RMSE = {geometry_rmse:.4f}")
    axes[1].grid(alpha=.25)
    same_q = final_pair["same_distance_quantiles"]
    diff_q = final_pair["different_distance_quantiles"]
    axes[2].boxplot([same_q, diff_q], tick_labels=["same node", "different node"])
    axes[2].set_ylabel("association cosine distance")
    axes[2].set_title(f"Association P/R = {safe['precision']:.3f}/{safe['recall']:.3f}")
    axes[2].grid(alpha=.25)
    figure.tight_layout()
    for suffix in ("png", "pdf", "svg"):
        figure.savefig(output / f"tiny_overfit_dual_readout.{suffix}", dpi=180)
    plt.close(figure)
    print(json.dumps(summary, sort_keys=True))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
