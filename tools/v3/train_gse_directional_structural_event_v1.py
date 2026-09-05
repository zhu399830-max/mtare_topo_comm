#!/usr/bin/env python3
"""Train three frozen-backbone directional structural-event heads."""

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

from _bootstrap import PROJECT_ROOT
from mtare_topo.data.gse_training_dataset import GSESequenceDataset
from mtare_topo.representation.gse_directional_structural_event import (
    DirectionalStructuralEventHead,
    directional_identity_balanced_weights,
    directional_structural_event_loss,
)
from mtare_topo.representation.gse_graph import GeometrySemanticEventNet
from mtare_topo.representation.gse_rare_event_corrective import (
    corrective_gate,
    evaluate_rare_event_corrective,
)


PASS_STATUS = "PASS_GSE_DIRECTIONAL_STRUCTURAL_EVENT_TRAINING_V1"
FAIL_STATUS = "FAIL_GSE_DIRECTIONAL_STRUCTURAL_EVENT_TRAINING_V1"


def _seed(value: int) -> None:
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    torch.cuda.manual_seed_all(value)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False


def _load_population(
    dataset_run: Path,
    verifier_run: Path,
) -> tuple[GSESequenceDataset, dict[str, np.ndarray]]:
    dataset = GSESequenceDataset(dataset_run, "train", augment_azimuth=False)
    cache = verifier_run / "artifacts/pair_cache/pairs.npz"
    with np.load(cache, allow_pickle=False) as archive:
        compact_global = archive["compact_to_global_sequence_index"].astype(np.int64)
        partition = archive["partition_code"].astype(np.uint8)
        parent = archive["parent_id"].astype(str)
    if (
        compact_global.shape != (188126,)
        or partition.shape != (188126,)
        or int(np.sum(partition == 0)) != 142184
        or int(np.sum(partition == 1)) != 45942
    ):
        raise RuntimeError("directional event population drift")
    row_by_global = {
        int(record["global_sequence_index"]): row for row, record in enumerate(dataset.records)
    }
    try:
        dataset_row = np.asarray([row_by_global[int(value)] for value in compact_global], dtype=np.int64)
    except KeyError as exc:
        raise RuntimeError("directional event population is absent from the dataset") from exc
    event = dataset.event_labels()[dataset_row].astype(np.int64)
    identity = dataset.association_labels()[dataset_row].astype(np.int64)
    probability = []
    for seed in (0, 1, 2):
        features = np.load(
            verifier_run / f"artifacts/models/seed{seed}/frozen_observation_features.npy",
            mmap_mode="r",
        )
        if features.shape != (188126, 146):
            raise RuntimeError(f"seed{seed} frozen observation feature drift")
        value = np.asarray(features[:, :5], dtype=np.float64)
        value /= value.sum(axis=1, keepdims=True)
        probability.append(value)
    baseline_probability = np.mean(np.stack(probability), axis=0)
    expected = {
        0: (102874, 32249),
        1: (19743, 6865),
        2: (5551, 1974),
        3: (1482, 521),
        4: (12534, 4333),
    }
    if any(
        (
            int(np.sum((partition == 0) & (event == key))),
            int(np.sum((partition == 1) & (event == key))),
        )
        != count
        for key, count in expected.items()
    ) or np.any((event == 0) != (identity < 0)):
        raise RuntimeError("directional event labels or identity contract drift")
    return dataset, {
        "global_sequence_index": compact_global,
        "dataset_row": dataset_row,
        "partition": partition,
        "parent": parent,
        "family": np.asarray([value[:3] for value in parent]),
        "event": event,
        "identity": identity,
        "baseline_probability": baseline_probability.astype(np.float32),
    }


def _batch(
    dataset: GSESequenceDataset,
    population_rows: np.ndarray,
    dataset_rows: np.ndarray,
    event: np.ndarray,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    examples = [dataset[int(dataset_rows[int(row)])] for row in population_rows]
    student = torch.from_numpy(np.stack([item["student"] for item in examples])).to(
        device, non_blocking=True
    )
    target = torch.from_numpy(event[population_rows]).long().to(device, non_blocking=True)
    return student, target


def _forward_head(
    backbone: GeometrySemanticEventNet,
    head: DirectionalStructuralEventHead,
    student: torch.Tensor,
) -> dict[str, torch.Tensor]:
    with torch.no_grad():
        causal = backbone.encode_causal_features(student)
        baseline_logits = backbone.event_head(causal["context"])
    return head(causal["azimuth_sequence"], causal["context"], baseline_logits)


def _evaluate(
    dataset: GSESequenceDataset,
    arrays: dict[str, np.ndarray],
    rows: np.ndarray,
    backbone: GeometrySemanticEventNet,
    head: DirectionalStructuralEventHead,
    device: torch.device,
    batch_size: int,
) -> tuple[dict[str, Any], np.ndarray]:
    dataset.set_epoch(0)
    head.eval()
    values = []
    with torch.inference_mode():
        for start in range(0, len(rows), batch_size):
            batch_rows = rows[start : start + batch_size]
            student, _ = _batch(
                dataset,
                batch_rows,
                arrays["dataset_row"],
                arrays["event"],
                device,
            )
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                causal = backbone.encode_causal_features(student)
                baseline_logits = backbone.event_head(causal["context"])
                output = head(causal["azimuth_sequence"], causal["context"], baseline_logits)
            values.append(output["event_probability"].float().cpu().numpy())
    probability = np.concatenate(values).astype(np.float32)
    metrics = evaluate_rare_event_corrective(
        probability,
        arrays["event"][rows],
        arrays["identity"][rows],
        arrays["family"][rows],
    )
    return metrics, probability


def _ranking(metrics: dict[str, Any], epoch: int) -> tuple[Any, ...]:
    gate = corrective_gate(metrics)
    coverage = metrics.get("identity_coverage") or {}
    selection = metrics.get("structural_selection") or {}
    return (
        bool(gate["passed"]),
        min(
            float(coverage.get("turn", {}).get("correct_class_identity_coverage", 0.0)),
            float(coverage.get("geometry_transition", {}).get("correct_class_identity_coverage", 0.0)),
        ),
        float(metrics.get("event", {}).get("macro_f1", 0.0)),
        float(selection.get("recall", 0.0)),
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
        raise RuntimeError("directional event output directory already exists")
    output_dir.mkdir(parents=True)
    dataset, arrays = _load_population(dataset_run, verifier_run)
    fit_rows = np.where(arrays["partition"] == 0)[0]
    selection_rows = np.where(arrays["partition"] == 1)[0]
    weights = directional_identity_balanced_weights(
        arrays["event"][fit_rows],
        arrays["identity"][fit_rows],
        1.0 - arrays["baseline_probability"][fit_rows, 0],
    )
    np.savez_compressed(
        output_dir / "sampling_contract.npz",
        fit_global_sequence_index=arrays["global_sequence_index"][fit_rows],
        fit_weight=weights,
        selection_global_sequence_index=arrays["global_sequence_index"][selection_rows],
    )
    device = torch.device(args.device)
    seed_summaries = {}
    seed_probabilities = []
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
        head = DirectionalStructuralEventHead().to(device)
        optimizer = torch.optim.AdamW(
            head.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
        )
        generator = np.random.default_rng(seed)
        best_rank = None
        best_metrics = None
        best_probability = None
        optimizer_steps = 0
        epoch_file = seed_dir / "epoch_metrics.jsonl"
        with epoch_file.open("w", encoding="utf-8") as stream:
            for epoch in range(1, args.epochs + 1):
                dataset.augment_azimuth = True
                dataset.augmentation_seed = seed
                dataset.set_epoch(epoch)
                sampled_local = generator.choice(
                    len(fit_rows), size=args.draws_per_epoch, replace=True, p=weights
                )
                sampled = fit_rows[sampled_local]
                head.train()
                sums = {"structural": 0.0, "conditional_event": 0.0, "total": 0.0}
                seen = 0
                for start in range(0, len(sampled), args.batch_size):
                    rows = sampled[start : start + args.batch_size]
                    student, target = _batch(
                        dataset,
                        rows,
                        arrays["dataset_row"],
                        arrays["event"],
                        device,
                    )
                    optimizer.zero_grad(set_to_none=True)
                    with torch.autocast(
                        device_type=device.type,
                        dtype=torch.bfloat16,
                        enabled=device.type == "cuda",
                    ):
                        output = _forward_head(backbone, head, student)
                        losses = directional_structural_event_loss(output, target)
                    losses["total"].backward()
                    norm = torch.nn.utils.clip_grad_norm_(head.parameters(), 5.0)
                    if not torch.isfinite(norm):
                        raise RuntimeError("nonfinite directional event gradient")
                    optimizer.step()
                    if any(parameter.grad is not None for parameter in backbone.parameters()):
                        raise RuntimeError("frozen GSE backbone received a gradient")
                    optimizer_steps += 1
                    size = len(rows)
                    seen += size
                    for key in sums:
                        sums[key] += float(losses[key].detach().cpu()) * size
                record: dict[str, Any] = {
                    "epoch": epoch,
                    "draws": seen,
                    "optimizer_steps": optimizer_steps,
                    "train_loss": {key: value / seen for key, value in sums.items()},
                    "selection": None,
                }
                if epoch % args.evaluate_every == 0 or epoch == args.epochs:
                    dataset.augment_azimuth = False
                    metrics, probability = _evaluate(
                        dataset,
                        arrays,
                        selection_rows,
                        backbone,
                        head,
                        device,
                        args.evaluation_batch_size,
                    )
                    rank = _ranking(metrics, epoch)
                    record["selection"] = metrics
                    record["gate"] = corrective_gate(metrics)
                    if best_rank is None or rank > best_rank:
                        best_rank = rank
                        best_metrics = metrics
                        best_probability = probability.copy()
                        torch.save(
                            {
                                "schema_version": "gse_directional_structural_event_checkpoint_v1",
                                "seed": seed,
                                "epoch": epoch,
                                "head": {name: value.detach().cpu() for name, value in head.state_dict().items()},
                                "source_backbone_checkpoint": str(checkpoint_path.relative_to(PROJECT_ROOT)),
                                "source_backbone_epoch": int(source["epoch"]),
                                "optimizer_steps": optimizer_steps,
                                "architecture": "frozen_gse_azimuth_sequence_directional_binary_plus_conditional_event_v1",
                            },
                            seed_dir / "best.pt",
                        )
                stream.write(json.dumps(record, separators=(",", ":"), sort_keys=True) + "\n")
                print(json.dumps({"seed": seed, **record}, sort_keys=True), flush=True)
        if best_metrics is None or best_probability is None or best_rank is None:
            raise RuntimeError(f"seed{seed} produced no evaluated checkpoint")
        np.savez_compressed(
            seed_dir / "selection_outputs.npz",
            global_sequence_index=arrays["global_sequence_index"][selection_rows],
            probability=best_probability,
            event=arrays["event"][selection_rows],
            identity=arrays["identity"][selection_rows],
            family=arrays["family"][selection_rows],
        )
        seed_summary = {
            "seed": seed,
            "best_epoch": int(torch.load(seed_dir / "best.pt", map_location="cpu", weights_only=False)["epoch"]),
            "optimizer_steps": optimizer_steps,
            "backbone_optimizer_steps": 0,
            "metrics": best_metrics,
            "gate": corrective_gate(best_metrics),
        }
        (seed_dir / "summary.json").write_text(
            json.dumps(seed_summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        seed_summaries[str(seed)] = seed_summary
        seed_probabilities.append(best_probability)
        del backbone, head, optimizer
        torch.cuda.empty_cache()
    ensemble_probability = np.mean(np.stack(seed_probabilities).astype(np.float64), axis=0)
    ensemble_metrics = evaluate_rare_event_corrective(
        ensemble_probability,
        arrays["event"][selection_rows],
        arrays["identity"][selection_rows],
        arrays["family"][selection_rows],
    )
    ensemble_gate = corrective_gate(ensemble_metrics)
    np.savez_compressed(
        output_dir / "ensemble_selection_outputs.npz",
        global_sequence_index=arrays["global_sequence_index"][selection_rows],
        probability=ensemble_probability.astype(np.float32),
        event=arrays["event"][selection_rows],
        identity=arrays["identity"][selection_rows],
        family=arrays["family"][selection_rows],
    )
    summary = {
        "schema_version": "gse_directional_structural_event_training_v1",
        "overall_status": PASS_STATUS if ensemble_gate["passed"] else FAIL_STATUS,
        "scientific_pass": bool(ensemble_gate["passed"]),
        "method": "three_seed_frozen_backbone_directional_binary_plus_conditional_event",
        "fit_worlds": 60,
        "fit_observations": len(fit_rows),
        "selection_worlds": 20,
        "selection_observations": len(selection_rows),
        "epochs": args.epochs,
        "draws_per_epoch": args.draws_per_epoch,
        "optimizer_steps": int(sum(value["optimizer_steps"] for value in seed_summaries.values())),
        "backbone_optimizer_steps": 0,
        "seeds": seed_summaries,
        "ensemble_metrics": ensemble_metrics,
        "ensemble_gate": ensemble_gate,
        "strict_test_worlds_read": 0,
        "c09_worlds_read": 0,
        "mtare_worlds_read": 0,
        "duration_seconds": time.monotonic() - started,
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-run", required=True)
    parser.add_argument("--training-run", required=True)
    parser.add_argument("--verifier-run", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--draws-per-epoch", type=int, default=24000)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--evaluation-batch-size", type=int, default=48)
    parser.add_argument("--evaluate-every", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    args = parser.parse_args()
    if (
        args.epochs != 12
        or args.draws_per_epoch != 24000
        or args.batch_size != 24
        or args.evaluation_batch_size != 48
        or args.evaluate_every != 3
        or not math.isclose(args.learning_rate, 1e-3)
        or not math.isclose(args.weight_decay, 1e-4)
    ):
        raise RuntimeError("directional event V1 hyperparameters are frozen")
    summary = train(args)
    print(json.dumps(summary, sort_keys=True))
    return 0 if summary["scientific_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

