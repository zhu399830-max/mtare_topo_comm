#!/usr/bin/env python3
"""Train three deterministic rare-event residual heads on frozen GSE features."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import random
from typing import Any

import numpy as np
import torch
from torch import nn

from _bootstrap import PROJECT_ROOT
from mtare_topo.data.gse_training_dataset import GSESequenceDataset
from mtare_topo.governance import write_json
from mtare_topo.representation.gse_rare_event_corrective import (
    GSERareEventCorrective,
    corrective_gate,
    evaluate_rare_event_corrective,
    identity_class_balanced_weights,
)


PASS_STATUS = "PASS_GSE_RARE_EVENT_CORRECTIVE_TRAINING_V1"
FAIL_STATUS = "FAIL_GSE_RARE_EVENT_CORRECTIVE_TRAINING_V1"


def _seed(value: int) -> None:
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    torch.cuda.manual_seed_all(value)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False


def _load_arrays(verifier_run: Path, dataset_run: Path) -> dict[str, np.ndarray]:
    cache_path = verifier_run / "artifacts/pair_cache/pairs.npz"
    with np.load(cache_path, allow_pickle=False) as archive:
        compact_global = archive["compact_to_global_sequence_index"].astype(np.int64)
        partition = archive["partition_code"].astype(np.uint8)
        parent = archive["parent_id"].astype(str)
    if (
        compact_global.shape != (188126,)
        or partition.shape != (188126,)
        or np.sum(partition == 0) != 142184
        or np.sum(partition == 1) != 45942
    ):
        raise RuntimeError("rare-event pair-cache population drift")
    seed_features = []
    for seed in (0, 1, 2):
        directory = verifier_run / f"artifacts/models/seed{seed}"
        features = np.load(directory / "frozen_observation_features.npy", mmap_mode="r")
        with np.load(directory / "frozen_exit_token_outputs.npz", allow_pickle=False) as archive:
            identities = archive["global_sequence_index"].astype(np.int64)
        if features.shape != (188126, 146) or not np.array_equal(identities, compact_global):
            raise RuntimeError(f"seed{seed} frozen feature identity drift")
        seed_features.append(np.asarray(features, dtype=np.float32))
    stacked = np.stack(seed_features)
    mean_feature = stacked.mean(axis=0, dtype=np.float64).astype(np.float32)
    std_feature = stacked.std(axis=0, dtype=np.float64).astype(np.float32)
    baseline_probability = mean_feature[:, :5].astype(np.float64)
    baseline_probability /= baseline_probability.sum(axis=1, keepdims=True)
    feature = np.concatenate((mean_feature, std_feature), axis=1).astype(np.float32)

    dataset = GSESequenceDataset(dataset_run, "train", augment_azimuth=False)
    event_by_dataset = dataset.event_labels()
    identity_by_dataset = dataset.association_labels()
    row_by_global = {
        int(record["global_sequence_index"]): row for row, record in enumerate(dataset.records)
    }
    try:
        dataset_rows = np.asarray([row_by_global[int(value)] for value in compact_global], dtype=np.int64)
    except KeyError as exc:
        raise RuntimeError("frozen feature identity is absent from the training dataset") from exc
    event = event_by_dataset[dataset_rows].astype(np.int64)
    identity = identity_by_dataset[dataset_rows].astype(np.int64)
    family = np.asarray([value[:3] for value in parent])
    expected_counts = {
        0: (102874, 32249),
        1: (19743, 6865),
        2: (5551, 1974),
        3: (1482, 521),
        4: (12534, 4333),
    }
    if any(
        (int(np.sum((partition == 0) & (event == key))), int(np.sum((partition == 1) & (event == key)))) != value
        for key, value in expected_counts.items()
    ) or np.any((event == 0) != (identity < 0)):
        raise RuntimeError("rare-event labels or partition counts drifted")
    return {
        "feature": feature,
        "baseline_probability": baseline_probability.astype(np.float32),
        "event": event,
        "identity": identity,
        "family": family,
        "partition": partition,
        "global_sequence_index": compact_global,
    }


def _probabilities(
    model: GSERareEventCorrective,
    feature: torch.Tensor,
    baseline: torch.Tensor,
    *,
    batch_size: int = 8192,
) -> np.ndarray:
    values = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(feature), batch_size):
            logits = model(feature[start : start + batch_size], baseline[start : start + batch_size])
            values.append(torch.softmax(logits, dim=1).cpu().numpy())
    return np.concatenate(values).astype(np.float32)


def _ranking(metrics: dict[str, Any], validation_loss: float, epoch: int) -> tuple[Any, ...]:
    gate = corrective_gate(metrics)
    coverage = metrics.get("identity_coverage") or {}
    return (
        bool(gate["passed"]),
        min(
            float(coverage.get("turn", {}).get("correct_class_identity_coverage", 0.0)),
            float(coverage.get("geometry_transition", {}).get("correct_class_identity_coverage", 0.0)),
        ),
        float(metrics["event"]["macro_f1"]),
        float((metrics.get("structural_selection") or {}).get("recall", 0.0)),
        -float(validation_loss),
        -int(epoch),
    )


def train(
    verifier_run: Path,
    dataset_run: Path,
    output_dir: Path,
    *,
    device: str = "cuda:0",
    epochs: int = 40,
    batch_size: int = 512,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
) -> dict[str, Any]:
    verifier_run = verifier_run.resolve()
    dataset_run = dataset_run.resolve()
    output_dir = output_dir.resolve()
    for path in (verifier_run, dataset_run, output_dir.parent):
        path.relative_to(PROJECT_ROOT)
    if output_dir.exists():
        raise RuntimeError("rare-event corrective output already exists")
    output_dir.mkdir(parents=True)
    arrays = _load_arrays(verifier_run, dataset_run)
    fit = arrays["partition"] == 0
    selection = arrays["partition"] == 1
    mean = arrays["feature"][fit].mean(axis=0, dtype=np.float64).astype(np.float32)
    scale = arrays["feature"][fit].std(axis=0, dtype=np.float64).astype(np.float32)
    scale = np.maximum(scale, 1e-6)
    normalized = ((arrays["feature"] - mean) / scale).astype(np.float32)
    np.savez_compressed(output_dir / "normalization.npz", mean=mean, scale=scale)
    fit_rows = np.where(fit)[0]
    selection_rows = np.where(selection)[0]
    sample_weight = identity_class_balanced_weights(
        arrays["event"][fit_rows], arrays["identity"][fit_rows]
    )
    torch_device = torch.device(device)
    fit_feature = torch.from_numpy(normalized[fit_rows]).to(torch_device)
    fit_baseline = torch.from_numpy(arrays["baseline_probability"][fit_rows]).to(torch_device)
    fit_target = torch.from_numpy(arrays["event"][fit_rows]).long().to(torch_device)
    selection_feature = torch.from_numpy(normalized[selection_rows]).to(torch_device)
    selection_baseline = torch.from_numpy(arrays["baseline_probability"][selection_rows]).to(torch_device)
    selection_target = torch.from_numpy(arrays["event"][selection_rows]).long().to(torch_device)

    best_probabilities = []
    seed_summaries = {}
    for seed in (0, 1, 2):
        _seed(seed)
        seed_dir = output_dir / f"seed{seed}"
        seed_dir.mkdir()
        model = GSERareEventCorrective().to(torch_device)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
        generator = np.random.default_rng(seed)
        epoch_path = seed_dir / "epoch_metrics.jsonl"
        best = None
        best_state = None
        best_probability = None
        best_metrics = None
        optimizer_steps = 0
        with epoch_path.open("w", encoding="utf-8") as stream:
            for epoch in range(1, epochs + 1):
                model.train()
                sampled = generator.choice(
                    len(fit_rows), size=len(fit_rows), replace=True, p=sample_weight
                )
                losses = []
                for start in range(0, len(sampled), batch_size):
                    index = torch.from_numpy(sampled[start : start + batch_size]).to(torch_device)
                    optimizer.zero_grad(set_to_none=True)
                    logits = model(fit_feature[index], fit_baseline[index])
                    loss = nn.functional.cross_entropy(logits, fit_target[index])
                    loss.backward()
                    optimizer.step()
                    optimizer_steps += 1
                    losses.append(float(loss.detach().cpu()))
                probability = _probabilities(model, selection_feature, selection_baseline)
                validation_loss = float(
                    nn.functional.nll_loss(
                        torch.from_numpy(np.log(np.maximum(probability, 1e-12))),
                        selection_target.cpu(),
                    )
                )
                metrics = evaluate_rare_event_corrective(
                    probability,
                    arrays["event"][selection_rows],
                    arrays["identity"][selection_rows],
                    arrays["family"][selection_rows],
                )
                gate = corrective_gate(metrics)
                record = {
                    "epoch": epoch,
                    "train_loss": float(np.mean(losses)),
                    "selection_nll": validation_loss,
                    "metrics": metrics,
                    "gate": gate,
                    "optimizer_steps": optimizer_steps,
                }
                stream.write(json.dumps(record, separators=(",", ":"), sort_keys=True) + "\n")
                rank = _ranking(metrics, validation_loss, epoch)
                if best is None or rank > best:
                    best = rank
                    best_state = {name: value.detach().cpu() for name, value in model.state_dict().items()}
                    best_probability = probability.copy()
                    best_metrics = metrics
                    torch.save({
                        "schema_version": "gse_rare_event_corrective_checkpoint_v1",
                        "seed": seed,
                        "epoch": epoch,
                        "model": best_state,
                        "optimizer_steps": optimizer_steps,
                        "architecture": "292-128-64-5_residual_over_log_mean_event_probability",
                    }, seed_dir / "best.pt")
        if best_probability is None or best_metrics is None or best_state is None:
            raise RuntimeError(f"seed{seed} did not produce a corrective checkpoint")
        np.savez_compressed(
            seed_dir / "selection_outputs.npz",
            global_sequence_index=arrays["global_sequence_index"][selection_rows],
            probability=best_probability,
            event=arrays["event"][selection_rows],
            identity=arrays["identity"][selection_rows],
            family=arrays["family"][selection_rows],
        )
        checkpoint = torch.load(seed_dir / "best.pt", map_location="cpu", weights_only=False)
        summary = {
            "seed": seed,
            "best_epoch": int(checkpoint["epoch"]),
            "optimizer_steps": optimizer_steps,
            "best_metrics": best_metrics,
            "best_gate": corrective_gate(best_metrics),
        }
        write_json(seed_dir / "summary.json", summary)
        seed_summaries[str(seed)] = summary
        best_probabilities.append(best_probability)

    ensemble_probability = np.mean(best_probabilities, axis=0, dtype=np.float64)
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
        seed_probability=np.stack(best_probabilities),
        ensemble_probability=ensemble_probability,
        event=arrays["event"][selection_rows],
        identity=arrays["identity"][selection_rows],
        family=arrays["family"][selection_rows],
    )
    total_steps = sum(int(value["optimizer_steps"]) for value in seed_summaries.values())
    summary = {
        "schema_version": "gse_rare_event_corrective_training_v1",
        "overall_status": PASS_STATUS if ensemble_gate["passed"] else FAIL_STATUS,
        "scientific_pass": bool(ensemble_gate["passed"]),
        "method": "identity_class_balanced_three_seed_residual_event_corrective",
        "fit_worlds": 60,
        "fit_observations": 142184,
        "selection_worlds": 20,
        "selection_observations": 45942,
        "seeds": seed_summaries,
        "ensemble_metrics": ensemble_metrics,
        "ensemble_gate": ensemble_gate,
        "optimizer_steps": total_steps,
        "backbone_optimizer_steps": 0,
        "c09_worlds_read": 0,
        "strict_test_worlds_read": 0,
        "mtare_worlds_read": 0,
        "hyperparameters": {
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
        },
    }
    write_json(output_dir / "summary.json", summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verifier-run", required=True, type=Path)
    parser.add_argument("--dataset-run", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    args = parser.parse_args()
    result = train(
        args.verifier_run,
        args.dataset_run,
        args.output_dir,
        device=args.device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    print(json.dumps({"overall_status": result["overall_status"]}, indent=2))
    return 0 if result["scientific_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
