#!/usr/bin/env python3
"""Train three frozen-backbone explicit causal geometry-delta event heads."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import random
import time
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F

from _bootstrap import PROJECT_ROOT
from mtare_topo.representation.gse_causal_geometry_delta import (
    CausalGeometryDeltaEventHead,
    GEOMETRY_DELTA_NAMES,
    GEOMETRY_DELTA_SCALE,
    causal_geometry_delta_loss,
)
from mtare_topo.representation.gse_corrected_causal_event import (
    OLD_DIRECTIONAL_ENSEMBLE_MACRO_F1,
    corrected_causal_event_gate,
)
from mtare_topo.representation.gse_directional_structural_event import (
    directional_identity_balanced_weights,
    directional_structural_event_loss,
)
from mtare_topo.representation.gse_graph import GeometrySemanticEventNet
from mtare_topo.representation.gse_rare_event_corrective import evaluate_rare_event_corrective
import train_gse_corrected_causal_event_v1 as corrected


PASS_STATUS = "PASS_GSE_CAUSAL_GEOMETRY_DELTA_MULTITASK_TRAINING_V1"
FAIL_STATUS = "FAIL_GSE_CAUSAL_GEOMETRY_DELTA_MULTITASK_TRAINING_V1"
TEACHER = PROJECT_ROOT / (
    "results/gate2_representation/"
    "gate2_20260827_gse_corrected_causal_teacher_manifest_v1r_seed0"
)
OLD_DIRECTIONAL = PROJECT_ROOT / (
    "results/gate3_semantics/"
    "gate3_20260826_gse_directional_structural_event_training_v1_seed0"
)
EXPECTED_VALID = (92845, 29920)
FROZEN_ENSEMBLE_SELECTION_NORMALIZED_MAE = 0.5871666431474375


def _seed(value: int) -> None:
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    torch.cuda.manual_seed_all(value)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False


def _load_delta_targets(arrays: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    row_by_global = {
        int(value): index for index, value in enumerate(arrays["global_sequence_index"])
    }
    names = ("width_m", "height_m", "slope_deg", "curvature_per_m")
    geometry = np.full((len(row_by_global), 4), np.nan, dtype=np.float64)
    valid = np.zeros(len(row_by_global), dtype=np.bool_)
    traversal: list[str | None] = [None] * len(row_by_global)
    sequence = np.full(len(row_by_global), -1, dtype=np.int64)
    with (TEACHER / "artifacts/teacher_observations.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            record = json.loads(line)
            try:
                row = row_by_global[int(record["global_sequence_index"])]
            except KeyError as exc:
                raise RuntimeError("delta Teacher global identity drift") from exc
            values = [record.get(name) for name in names]
            if bool(record.get("geometry_valid")) and all(value is not None for value in values):
                geometry[row] = np.asarray(values, dtype=np.float64)
                valid[row] = bool(np.all(np.isfinite(geometry[row])))
            traversal[row] = str(record["traversal_id"])
            sequence[row] = int(record["sequence_index"])
    if any(value is None for value in traversal) or np.any(sequence < 0):
        raise RuntimeError("delta Teacher does not cover the compact population")
    lookup = {(str(traversal[row]), int(sequence[row])): row for row in range(len(sequence))}
    if len(lookup) != len(sequence):
        raise RuntimeError("duplicate traversal/sequence key in delta Teacher")
    delta = np.full_like(geometry, np.nan)
    delta_valid = np.zeros(len(sequence), dtype=np.bool_)
    for row in range(len(sequence)):
        previous = lookup.get((str(traversal[row]), int(sequence[row]) - 4))
        if previous is not None and valid[row] and valid[previous]:
            if arrays["partition"][row] != arrays["partition"][previous]:
                raise RuntimeError("causal delta crosses a development partition")
            delta[row] = geometry[row] - geometry[previous]
            delta_valid[row] = True
    counts = tuple(int(np.sum(delta_valid & (arrays["partition"] == code))) for code in (0, 1))
    if counts != EXPECTED_VALID or not np.all(np.isfinite(delta[delta_valid])):
        raise RuntimeError(f"geometry-valid causal delta population drift: {counts}")
    return delta.astype(np.float32), delta_valid


def _batch_student(dataset, population_rows: np.ndarray, dataset_rows: np.ndarray, device) -> torch.Tensor:
    examples = [dataset[int(dataset_rows[int(row)])] for row in population_rows]
    return torch.from_numpy(np.stack([item["student"] for item in examples])).to(
        device, non_blocking=True
    )


def _forward(backbone, head, student) -> dict[str, torch.Tensor]:
    with torch.no_grad():
        causal = backbone.encode_causal_features(student)
        baseline_logits = backbone.event_head(causal["context"])
    return head(causal["azimuth_sequence"], causal["context"], baseline_logits)


def _delta_metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, Any]:
    error = np.abs(np.asarray(prediction, dtype=np.float64) - np.asarray(target, dtype=np.float64))
    if error.ndim != 2 or error.shape[1] != 4 or not np.all(np.isfinite(error)):
        raise RuntimeError("invalid geometry-delta metric population")
    component = {
        name: {"mae": float(np.mean(error[:, index])), "scale": GEOMETRY_DELTA_SCALE[index]}
        for index, name in enumerate(GEOMETRY_DELTA_NAMES)
    }
    normalized = error / np.asarray(GEOMETRY_DELTA_SCALE, dtype=np.float64)
    return {
        "observations": len(error),
        "component": component,
        "normalized_mae": float(np.mean(normalized)),
        "normalized_rmse": float(np.sqrt(np.mean(np.square(normalized)))),
    }


def _frozen_delta_baseline(
    verifier_run: Path,
    arrays: dict[str, np.ndarray],
    delta_target: np.ndarray,
    delta_valid: np.ndarray,
) -> tuple[dict[str, Any], list[np.ndarray], np.ndarray]:
    valid_selection = np.where(delta_valid & (arrays["partition"] == 1))[0]
    predictions = []
    for seed in (0, 1, 2):
        feature = np.load(
            verifier_run / f"artifacts/models/seed{seed}/frozen_observation_features.npy",
            mmap_mode="r",
        )
        geometry = np.asarray(feature[:, 8:12], dtype=np.float64) * np.asarray(
            (30.0, 30.0, 45.0, 0.1), dtype=np.float64
        )
        # The exact predecessor map is recoverable from target-valid compact rows
        # only through Teacher sequence identity; rebuild once deterministically.
        predictions.append(geometry)
    teacher_rows = []
    with (TEACHER / "artifacts/teacher_observations.jsonl").open(encoding="utf-8") as stream:
        teacher_rows = [json.loads(line) for line in stream]
    lookup = {
        (str(record["traversal_id"]), int(record["sequence_index"])): row
        for row, record in enumerate(teacher_rows)
    }
    seed_delta: list[np.ndarray] = []
    for geometry in predictions:
        value = np.full((len(valid_selection), 4), np.nan, dtype=np.float64)
        for output_row, compact_row in enumerate(valid_selection):
            record = teacher_rows[int(compact_row)]
            previous = lookup[(str(record["traversal_id"]), int(record["sequence_index"]) - 4)]
            value[output_row] = geometry[compact_row] - geometry[previous]
        seed_delta.append(value.astype(np.float32))
    metrics = {str(seed): _delta_metrics(value, delta_target[valid_selection]) for seed, value in enumerate(seed_delta)}
    ensemble = np.mean(np.stack(seed_delta).astype(np.float64), axis=0).astype(np.float32)
    metrics["ensemble"] = _delta_metrics(ensemble, delta_target[valid_selection])
    if not math.isclose(
        metrics["ensemble"]["normalized_mae"],
        FROZEN_ENSEMBLE_SELECTION_NORMALIZED_MAE,
        rel_tol=0.0,
        abs_tol=1e-9,
    ):
        raise RuntimeError("sealed frozen delta baseline drift")
    return metrics, seed_delta, ensemble


def _slice_event_outputs(output: dict[str, torch.Tensor], rows: slice) -> dict[str, torch.Tensor]:
    return {
        "structural_logit": output["structural_logit"][rows],
        "conditional_event_logits": output["conditional_event_logits"][rows],
    }


def _evaluate(
    dataset,
    arrays,
    rows,
    backbone,
    head,
    device,
    batch_size,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    dataset.set_epoch(0)
    dataset.augment_azimuth = False
    head.eval()
    probability = []
    delta = []
    with torch.inference_mode():
        for start in range(0, len(rows), batch_size):
            batch_rows = rows[start : start + batch_size]
            student = _batch_student(dataset, batch_rows, arrays["dataset_row"], device)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                output = _forward(backbone, head, student)
            probability.append(output["event_probability"].float().cpu().numpy())
            delta.append(output["geometry_delta"].float().cpu().numpy())
    probability_array = np.concatenate(probability).astype(np.float32)
    delta_array = np.concatenate(delta).astype(np.float32)
    event_metrics = evaluate_rare_event_corrective(
        probability_array,
        arrays["event"][rows],
        arrays["identity"][rows],
        arrays["family"][rows],
    )
    valid = arrays["delta_valid"][rows]
    geometry_metrics = _delta_metrics(delta_array[valid], arrays["delta_target"][rows][valid])
    return {"event": event_metrics, "geometry_delta": geometry_metrics}, probability_array, delta_array


def _ranking(metrics: dict[str, Any], epoch: int) -> tuple[Any, ...]:
    event = metrics["event"]
    gate = corrected_causal_event_gate(event)
    coverage = event.get("identity_coverage") or {}
    return (
        bool(gate["passed"]),
        int(coverage.get("geometry_transition", {}).get("covered_identities", 0)),
        -float(metrics["geometry_delta"]["normalized_mae"]),
        float(event.get("event", {}).get("macro_f1", 0.0)),
        -int(epoch),
    )


def train(args: argparse.Namespace) -> dict[str, Any]:
    started = time.monotonic()
    dataset_run = Path(args.dataset_run).resolve()
    training_run = Path(args.training_run).resolve()
    verifier_run = Path(args.verifier_run).resolve()
    output_dir = Path(args.output_dir).resolve()
    for path in (dataset_run, training_run, verifier_run, output_dir.parent):
        path.relative_to(PROJECT_ROOT)
    if output_dir.exists():
        raise RuntimeError("geometry-delta multitask output already exists")
    output_dir.mkdir(parents=True)
    dataset, arrays = corrected._load_population(dataset_run, verifier_run)
    delta_target, delta_valid = _load_delta_targets(arrays)
    arrays["delta_target"] = delta_target
    arrays["delta_valid"] = delta_valid
    fit_rows = np.where(arrays["partition"] == 0)[0]
    selection_rows = np.where(arrays["partition"] == 1)[0]
    regression_rows = np.where((arrays["partition"] == 0) & delta_valid)[0]
    weights = directional_identity_balanced_weights(
        arrays["event"][fit_rows],
        arrays["identity"][fit_rows],
        1.0 - arrays["baseline_probability"][fit_rows, 0],
    )
    baseline_event, _ = corrected._baseline_evidence(dataset_run, verifier_run)
    baseline_delta, _, _ = _frozen_delta_baseline(
        verifier_run, arrays, delta_target, delta_valid
    )
    np.savez_compressed(
        output_dir / "sampling_contract.npz",
        fit_global_sequence_index=arrays["global_sequence_index"][fit_rows],
        fit_event_weight=weights,
        fit_regression_global_sequence_index=arrays["global_sequence_index"][regression_rows],
        selection_global_sequence_index=arrays["global_sequence_index"][selection_rows],
        delta_scale=np.asarray(GEOMETRY_DELTA_SCALE, dtype=np.float64),
    )

    device = torch.device(args.device)
    seed_summaries: dict[str, Any] = {}
    seed_probability = []
    seed_delta = []
    for seed in (0, 1, 2):
        _seed(seed)
        seed_dir = output_dir / f"seed{seed}"
        seed_dir.mkdir()
        checkpoint_path = training_run / f"artifacts/models/seed{seed}/best.pt"
        source = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        backbone = GeometrySemanticEventNet().to(device)
        backbone.load_state_dict(source["model"], strict=True)
        backbone.eval()
        for parameter in backbone.parameters():
            parameter.requires_grad_(False)
        head = CausalGeometryDeltaEventHead().to(device)
        optimizer = torch.optim.AdamW(head.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        generator = np.random.default_rng(seed)
        best_rank = None
        best_metrics = None
        best_probability = None
        best_delta = None
        optimizer_steps = 0
        epoch_file = seed_dir / "epoch_metrics.jsonl"
        with epoch_file.open("w", encoding="utf-8") as stream:
            for epoch in range(1, args.epochs + 1):
                dataset.augment_azimuth = True
                dataset.augmentation_seed = seed
                dataset.set_epoch(epoch)
                regression_order = generator.permutation(regression_rows)
                event_local = generator.choice(
                    len(fit_rows), size=args.event_draws_per_epoch, replace=True, p=weights
                )
                event_rows = fit_rows[event_local]
                regression_chunks = np.array_split(
                    regression_order,
                    math.ceil(len(regression_order) / args.regression_batch_size),
                )
                event_chunks = np.array_split(event_rows, len(regression_chunks))
                head.train()
                sums = {"structural": 0.0, "conditional_event": 0.0, "geometry_delta": 0.0, "total": 0.0}
                event_seen = 0
                regression_seen = 0
                for regression_batch, event_batch in zip(regression_chunks, event_chunks, strict=True):
                    combined = np.concatenate((regression_batch, event_batch))
                    student = _batch_student(dataset, combined, arrays["dataset_row"], device)
                    event_target = torch.from_numpy(arrays["event"][event_batch]).long().to(device)
                    delta_value = torch.from_numpy(delta_target[regression_batch]).to(device)
                    optimizer.zero_grad(set_to_none=True)
                    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                        output = _forward(backbone, head, student)
                        event_output = _slice_event_outputs(output, slice(len(regression_batch), None))
                        event_losses = directional_structural_event_loss(event_output, event_target)
                        delta_loss = causal_geometry_delta_loss(
                            output["geometry_delta_normalized"][: len(regression_batch)],
                            delta_value,
                            torch.ones(len(regression_batch), dtype=torch.bool, device=device),
                        )
                        total = event_losses["total"] + delta_loss
                    total.backward()
                    norm = torch.nn.utils.clip_grad_norm_(head.parameters(), 5.0)
                    if not torch.isfinite(norm):
                        raise RuntimeError("nonfinite geometry-delta multitask gradient")
                    optimizer.step()
                    if any(parameter.grad is not None for parameter in backbone.parameters()):
                        raise RuntimeError("frozen GSE backbone received a gradient")
                    optimizer_steps += 1
                    event_size = len(event_batch)
                    regression_size = len(regression_batch)
                    event_seen += event_size
                    regression_seen += regression_size
                    sums["structural"] += float(event_losses["structural"].detach().cpu()) * event_size
                    sums["conditional_event"] += float(event_losses["conditional_event"].detach().cpu()) * event_size
                    sums["geometry_delta"] += float(delta_loss.detach().cpu()) * regression_size
                    sums["total"] += float(total.detach().cpu())
                if event_seen != args.event_draws_per_epoch or regression_seen != EXPECTED_VALID[0]:
                    raise RuntimeError("multitask epoch population drift")
                record: dict[str, Any] = {
                    "epoch": epoch,
                    "event_draws": event_seen,
                    "regression_unique_rows": regression_seen,
                    "optimizer_steps": optimizer_steps,
                    "train_loss": {
                        "structural": sums["structural"] / event_seen,
                        "conditional_event": sums["conditional_event"] / event_seen,
                        "geometry_delta": sums["geometry_delta"] / regression_seen,
                        "total_step_mean": sums["total"] / len(regression_chunks),
                    },
                    "selection": None,
                }
                if epoch % args.evaluate_every == 0 or epoch == args.epochs:
                    metrics, probability, delta_prediction = _evaluate(
                        dataset, arrays, selection_rows, backbone, head, device, args.evaluation_batch_size
                    )
                    rank = _ranking(metrics, epoch)
                    record["selection"] = metrics
                    record["event_gate"] = corrected_causal_event_gate(metrics["event"])
                    if best_rank is None or rank > best_rank:
                        best_rank = rank
                        best_metrics = metrics
                        best_probability = probability.copy()
                        best_delta = delta_prediction.copy()
                        torch.save(
                            {
                                "schema_version": "gse_causal_geometry_delta_multitask_checkpoint_v1",
                                "seed": seed,
                                "epoch": epoch,
                                "head": {name: value.detach().cpu() for name, value in head.state_dict().items()},
                                "source_backbone_checkpoint": str(checkpoint_path.relative_to(PROJECT_ROOT)),
                                "source_backbone_epoch": int(source["epoch"]),
                                "optimizer_steps": optimizer_steps,
                                "architecture": "frozen_gse_shared_directional_change_explicit_delta_plus_event_v1",
                                "delta_scale": GEOMETRY_DELTA_SCALE,
                            },
                            seed_dir / "best.pt",
                        )
                stream.write(json.dumps(record, separators=(",", ":"), sort_keys=True) + "\n")
                print(json.dumps({"seed": seed, **record}, sort_keys=True), flush=True)
        if best_metrics is None or best_probability is None or best_delta is None:
            raise RuntimeError(f"seed{seed} produced no selected checkpoint")
        np.savez_compressed(
            seed_dir / "selection_outputs.npz",
            global_sequence_index=arrays["global_sequence_index"][selection_rows],
            probability=best_probability,
            geometry_delta=best_delta,
            geometry_delta_valid=delta_valid[selection_rows],
            geometry_delta_target=delta_target[selection_rows],
            event=arrays["event"][selection_rows],
            identity=arrays["identity"][selection_rows],
            family=arrays["family"][selection_rows],
        )
        best_epoch = int(torch.load(seed_dir / "best.pt", map_location="cpu", weights_only=False)["epoch"])
        seed_summary = {
            "seed": seed,
            "best_epoch": best_epoch,
            "optimizer_steps": optimizer_steps,
            "backbone_optimizer_steps": 0,
            "metrics": best_metrics,
            "event_gate": corrected_causal_event_gate(best_metrics["event"]),
        }
        (seed_dir / "summary.json").write_text(json.dumps(seed_summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        seed_summaries[str(seed)] = seed_summary
        seed_probability.append(best_probability)
        seed_delta.append(best_delta)
        del backbone, head, optimizer
        torch.cuda.empty_cache()

    ensemble_probability = np.mean(np.stack(seed_probability).astype(np.float64), axis=0)
    ensemble_delta = np.mean(np.stack(seed_delta).astype(np.float64), axis=0)
    ensemble_event = evaluate_rare_event_corrective(
        ensemble_probability,
        arrays["event"][selection_rows],
        arrays["identity"][selection_rows],
        arrays["family"][selection_rows],
    )
    selection_valid = delta_valid[selection_rows]
    ensemble_delta_metrics = _delta_metrics(
        ensemble_delta[selection_valid], delta_target[selection_rows][selection_valid]
    )
    event_gate = corrected_causal_event_gate(ensemble_event)
    old_event = baseline_event["old_directional_head"]
    seed_event_gain = {
        str(seed): float(seed_summaries[str(seed)]["metrics"]["event"]["event"]["macro_f1"])
        - float(old_event["seeds"][str(seed)]["event"]["macro_f1"])
        for seed in (0, 1, 2)
    }
    seed_delta_improvement = {
        str(seed): 1.0
        - float(seed_summaries[str(seed)]["metrics"]["geometry_delta"]["normalized_mae"])
        / float(baseline_delta[str(seed)]["normalized_mae"])
        for seed in (0, 1, 2)
    }
    ensemble_event_gain = float(ensemble_event["event"]["macro_f1"]) - OLD_DIRECTIONAL_ENSEMBLE_MACRO_F1
    ensemble_delta_improvement = 1.0 - float(ensemble_delta_metrics["normalized_mae"]) / float(
        baseline_delta["ensemble"]["normalized_mae"]
    )
    requirements = {
        "corrected_causal_event_gate": bool(event_gate["passed"]),
        "ensemble_event_macro_f1_gain_at_least_0p05": ensemble_event_gain >= 0.05,
        "at_least_two_seed_event_gains_at_least_0p05": sum(value >= 0.05 for value in seed_event_gain.values()) >= 2,
        "no_seed_event_regression": all(value >= 0.0 for value in seed_event_gain.values()),
        "ensemble_delta_normalized_mae_improvement_at_least_10_percent": ensemble_delta_improvement >= 0.10,
        "at_least_two_seed_delta_improvements_at_least_10_percent": sum(value >= 0.10 for value in seed_delta_improvement.values()) >= 2,
        "no_seed_delta_regression_over_5_percent": all(value >= -0.05 for value in seed_delta_improvement.values()),
    }
    scientific_pass = all(requirements.values())
    np.savez_compressed(
        output_dir / "ensemble_selection_outputs.npz",
        global_sequence_index=arrays["global_sequence_index"][selection_rows],
        probability=ensemble_probability.astype(np.float32),
        geometry_delta=ensemble_delta.astype(np.float32),
        geometry_delta_valid=selection_valid,
        geometry_delta_target=delta_target[selection_rows],
        event=arrays["event"][selection_rows],
        identity=arrays["identity"][selection_rows],
        family=arrays["family"][selection_rows],
    )
    summary = {
        "schema_version": "gse_causal_geometry_delta_multitask_training_v1",
        "overall_status": PASS_STATUS if scientific_pass else FAIL_STATUS,
        "scientific_pass": scientific_pass,
        "method": "three_seed_frozen_gse_shared_directional_change_explicit_delta_plus_event",
        "fit_worlds": 60,
        "fit_observations": len(fit_rows),
        "fit_geometry_valid_lag_pairs": len(regression_rows),
        "selection_worlds": 20,
        "selection_observations": len(selection_rows),
        "selection_geometry_valid_lag_pairs": int(np.sum(selection_valid)),
        "epochs": args.epochs,
        "event_draws_per_epoch": args.event_draws_per_epoch,
        "regression_unique_rows_per_epoch": len(regression_rows),
        "optimizer_steps": int(sum(value["optimizer_steps"] for value in seed_summaries.values())),
        "backbone_optimizer_steps": 0,
        "seeds": seed_summaries,
        "ensemble_event_metrics": ensemble_event,
        "ensemble_event_gate": event_gate,
        "ensemble_geometry_delta_metrics": ensemble_delta_metrics,
        "baseline_event_evidence": baseline_event,
        "baseline_geometry_delta_metrics": baseline_delta,
        "seed_event_macro_f1_gain": seed_event_gain,
        "ensemble_event_macro_f1_gain": ensemble_event_gain,
        "seed_delta_normalized_mae_improvement": seed_delta_improvement,
        "ensemble_delta_normalized_mae_improvement": ensemble_delta_improvement,
        "aggregate_requirements": requirements,
        "strict_test_worlds_read": 0,
        "c09_worlds_read": 0,
        "mtare_worlds_read": 0,
        "duration_seconds": time.monotonic() - started,
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-run", required=True)
    parser.add_argument("--training-run", required=True)
    parser.add_argument("--verifier-run", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--event-draws-per-epoch", type=int, default=24000)
    parser.add_argument("--regression-batch-size", type=int, default=32)
    parser.add_argument("--evaluation-batch-size", type=int, default=48)
    parser.add_argument("--evaluate-every", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    args = parser.parse_args()
    if (
        args.epochs != 6
        or args.event_draws_per_epoch != 24000
        or args.regression_batch_size != 32
        or args.evaluation_batch_size != 48
        or args.evaluate_every != 2
        or not math.isclose(args.learning_rate, 1e-3)
        or not math.isclose(args.weight_decay, 1e-4)
    ):
        raise RuntimeError("geometry-delta multitask V1 hyperparameters are frozen")
    result = train(args)
    return 0 if result["scientific_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
