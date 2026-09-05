#!/usr/bin/env python3
"""Train one open-set verifier seed on frozen GSE outputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import time
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from _bootstrap import PROJECT_ROOT
from mtare_topo.data.gse_training_dataset import GSESequenceDataset
from mtare_topo.representation.gse_graph import GeometrySemanticEventNet
from mtare_topo.representation.gse_open_set_association import (
    GSEOpenSetAssociationVerifier,
    OpenSetAssociationContract,
    observation_features,
    select_nonvacuous_threshold,
    symmetric_pair_features,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False


def _collate(samples: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "student": torch.from_numpy(np.stack([sample["student"] for sample in samples])),
        "global_sequence_index": np.asarray(
            [sample["global_sequence_index"] for sample in samples], dtype=np.int64
        ),
    }


@torch.no_grad()
def _extract_features(
    dataset_run: Path, checkpoint_path: Path, device: torch.device
) -> tuple[np.ndarray, np.ndarray, str]:
    dataset = GSESequenceDataset(dataset_run, "train", augment_azimuth=False)
    if len(dataset) != 188126:
        raise RuntimeError("frozen C01--C08 dataset population drift")
    loader = DataLoader(dataset, batch_size=256, shuffle=False, num_workers=0, collate_fn=_collate)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if checkpoint.get("schema_version") != "gse_graph_checkpoint_v1":
        raise RuntimeError("unexpected frozen GSE checkpoint schema")
    model = GeometrySemanticEventNet().to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    feature_batches = []
    identity_batches = []
    for batch in loader:
        student = batch["student"].to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            output = model(student)
        numpy_output = {
            key: value.float().cpu().numpy()
            for key, value in output.items()
            if key in {
                "event_logits", "local_axis", "width_m", "height_m", "slope_deg",
                "curvature_per_m", "place_descriptor", "uncertainty", "exit_confidence",
                "exit_heading_unit", "exit_opening_width_m", "exit_vertical_profile",
            }
        }
        batch_features = observation_features(numpy_output)
        feature_batches.append(batch_features)
        identity_batches.append(batch["global_sequence_index"])
    if not feature_batches:
        raise RuntimeError("feature extraction produced no batches")
    features = np.concatenate(feature_batches)
    identities = np.concatenate(identity_batches)
    order = np.argsort(identities, kind="stable")
    features = features[order]
    identities = identities[order]
    if (
        features.shape[0] != len(dataset)
        or identities.shape != (len(dataset),)
        or np.any(np.diff(identities) <= 0)
        or not np.all(np.isfinite(features))
    ):
        raise RuntimeError("feature extraction did not cover C01--C08 exactly once")
    return features, identities, _sha256(checkpoint_path)


def _balanced_bce_logits(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    labels = labels.float()
    positive = labels.sum()
    negative = labels.numel() - positive
    if positive <= 0 or negative <= 0:
        raise RuntimeError("balanced BCE batch requires both classes")
    weights = torch.where(labels > 0.5, 0.5 / positive, 0.5 / negative) * labels.numel()
    return torch.nn.functional.binary_cross_entropy_with_logits(logits, labels, weight=weights)


def _balanced_bce_probability(probability: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    labels = labels.float()
    positive = labels.sum()
    negative = labels.numel() - positive
    weights = torch.where(labels > 0.5, 0.5 / positive, 0.5 / negative) * labels.numel()
    probability = probability.clamp(1e-6, 1.0 - 1e-6)
    return -(weights * (labels * probability.log() + (1.0 - labels) * (1.0 - probability).log())).mean()


def _evaluate(
    model: GSEOpenSetAssociationVerifier,
    observation: torch.Tensor,
    pair: torch.Tensor,
    left: np.ndarray,
    right: np.ndarray,
    labels: np.ndarray,
    families: np.ndarray,
    device: torch.device,
    contract: OpenSetAssociationContract,
) -> tuple[dict[str, Any], np.ndarray]:
    model.eval()
    scores = []
    losses = []
    with torch.no_grad():
        for start in range(0, len(labels), 4096):
            stop = min(start + 4096, len(labels))
            li = torch.from_numpy(left[start:stop]).to(device)
            ri = torch.from_numpy(right[start:stop]).to(device)
            target = torch.from_numpy(labels[start:stop].astype(np.float32)).to(device)
            output = model(observation[li], observation[ri], pair[start:stop])
            loss = _balanced_bce_probability(output["association_score"], target)
            losses.append((float(loss.cpu()), stop - start))
            scores.append(output["association_score"].cpu().numpy())
    score = np.concatenate(scores).astype(np.float64)
    selection = None
    try:
        selection = select_nonvacuous_threshold(score, labels, families, contract)
    except RuntimeError:
        pass
    return {
        "weighted_bce": float(sum(value * count for value, count in losses) / len(labels)),
        "threshold_selection": selection,
        "positive_pairs": int(np.sum(labels)),
        "negative_pairs": int(np.sum(~labels.astype(bool))),
    }, score


def train(
    dataset_run: Path,
    checkpoint_path: Path,
    pair_cache: Path,
    output_dir: Path,
    seed: int,
) -> dict[str, Any]:
    if seed not in (0, 1, 2):
        raise ValueError("formal verifier seed must be 0, 1 or 2")
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise RuntimeError("verifier seed output already exists")
    output_dir.mkdir(parents=True)
    _seed(seed)
    device = torch.device("cuda:0")
    started = time.monotonic()
    features, feature_global_indices, checkpoint_sha = _extract_features(
        dataset_run, checkpoint_path, device
    )
    np.save(output_dir / "frozen_observation_features.npy", features, allow_pickle=False)
    with np.load(pair_cache / "pairs.npz", allow_pickle=False) as source:
        arrays = {name: source[name] for name in source.files}
    if not np.array_equal(arrays["compact_to_global_sequence_index"], feature_global_indices):
        raise RuntimeError("pair cache sequence identity drift")
    partition_code = arrays["partition_code"].astype(np.uint8)
    if np.sum(partition_code == 0) != 142184 or np.sum(partition_code == 1) != 45942:
        raise RuntimeError("fit/selection observation partition drift")
    fit_observation = features[partition_code == 0]
    observation_mean = fit_observation.mean(axis=0, dtype=np.float64).astype(np.float32)
    observation_std = fit_observation.std(axis=0, dtype=np.float64).astype(np.float32)
    observation_std = np.maximum(observation_std, 1e-5)
    normalized = ((features - observation_mean) / observation_std).astype(np.float32)
    pair_by_partition = {}
    for partition in ("fit", "selection"):
        left = arrays[f"{partition}_left"].astype(np.int64)
        right = arrays[f"{partition}_right"].astype(np.int64)
        pair_by_partition[partition] = symmetric_pair_features(
            normalized[left], normalized[right], arrays[f"{partition}_distance_m"]
        )
    pair_mean = pair_by_partition["fit"].mean(axis=0, dtype=np.float64).astype(np.float32)
    pair_std = pair_by_partition["fit"].std(axis=0, dtype=np.float64).astype(np.float32)
    pair_std = np.maximum(pair_std, 1e-5)
    for partition in pair_by_partition:
        pair_by_partition[partition] = (
            (pair_by_partition[partition] - pair_mean) / pair_std
        ).astype(np.float32)
    np.savez_compressed(
        output_dir / "normalization.npz",
        observation_mean=observation_mean,
        observation_std=observation_std,
        pair_mean=pair_mean,
        pair_std=pair_std,
    )
    observation_tensor = torch.from_numpy(normalized).to(device)
    fit_pair = torch.from_numpy(pair_by_partition["fit"]).to(device)
    selection_pair = torch.from_numpy(pair_by_partition["selection"]).to(device)
    fit_left = arrays["fit_left"].astype(np.int64)
    fit_right = arrays["fit_right"].astype(np.int64)
    fit_label = arrays["fit_label"].astype(np.float32)
    selection_left = arrays["selection_left"].astype(np.int64)
    selection_right = arrays["selection_right"].astype(np.int64)
    selection_label = arrays["selection_label"].astype(np.uint8)
    selection_family = arrays["selection_family"].astype(str)
    valid = arrays["association_valid"].astype(np.float32)
    model = GSEOpenSetAssociationVerifier().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    generator = np.random.default_rng(seed)
    best_loss = math.inf
    best_epoch = 0
    stale = 0
    optimizer_steps = 0
    history = []
    contract = OpenSetAssociationContract()
    for epoch in range(1, 31):
        model.train()
        order = generator.permutation(len(fit_label))
        epoch_sum = 0.0
        epoch_count = 0
        for start in range(0, len(order), 2048):
            indices = order[start : start + 2048]
            # The final small batch is deterministically joined with the first
            # samples if it lacks either class, preserving full pair coverage.
            if len(np.unique(fit_label[indices])) < 2:
                indices = np.concatenate((indices, order[:2048]))
            li = torch.from_numpy(fit_left[indices]).to(device)
            ri = torch.from_numpy(fit_right[indices]).to(device)
            target = torch.from_numpy(fit_label[indices]).to(device)
            left_match = torch.from_numpy(valid[fit_left[indices]]).to(device)
            right_match = torch.from_numpy(valid[fit_right[indices]]).to(device)
            pair_index = torch.from_numpy(indices).to(device)
            optimizer.zero_grad(set_to_none=True)
            output = model(observation_tensor[li], observation_tensor[ri], fit_pair[pair_index])
            final_loss = _balanced_bce_probability(output["association_score"], target)
            pair_loss = _balanced_bce_logits(output["pair_logit"], target)
            match_loss = 0.5 * (
                _balanced_bce_logits(output["left_matchability_logit"], left_match)
                + torch.nn.functional.binary_cross_entropy_with_logits(
                    output["right_matchability_logit"], right_match
                )
            )
            loss = final_loss + 0.25 * pair_loss + 0.5 * match_loss
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            if not torch.isfinite(norm):
                raise RuntimeError("nonfinite verifier gradient")
            optimizer.step()
            optimizer_steps += 1
            epoch_sum += float(loss.detach().cpu()) * len(indices)
            epoch_count += len(indices)
        selection_metrics, _ = _evaluate(
            model, observation_tensor, selection_pair, selection_left, selection_right,
            selection_label, selection_family, device, contract,
        )
        record = {
            "epoch": epoch,
            "train_loss": epoch_sum / epoch_count,
            "selection": selection_metrics,
            "optimizer_steps": optimizer_steps,
        }
        history.append(record)
        with (output_dir / "epoch_metrics.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, separators=(",", ":"), sort_keys=True) + "\n")
        score = float(selection_metrics["weighted_bce"])
        if score < best_loss - 1e-8:
            best_loss = score
            best_epoch = epoch
            stale = 0
            torch.save(
                {
                    "schema_version": "gse_open_set_association_checkpoint_v1",
                    "model": model.state_dict(),
                    "seed": seed,
                    "epoch": epoch,
                    "selection_weighted_bce": score,
                    "source_gse_checkpoint_sha256": checkpoint_sha,
                },
                output_dir / "best.pt",
            )
        else:
            stale += 1
        if stale >= 6:
            break
    checkpoint = torch.load(output_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    final_metrics, scores = _evaluate(
        model, observation_tensor, selection_pair, selection_left, selection_right,
        selection_label, selection_family, device, contract,
    )
    np.savez_compressed(
        output_dir / "selection_outputs.npz",
        score=scores,
        label=selection_label,
        family=selection_family,
        left=selection_left,
        right=selection_right,
    )
    threshold = final_metrics["threshold_selection"]
    passed = threshold is not None
    summary = {
        "schema_version": "gse_open_set_association_training_seed_v1",
        "overall_status": (
            "PASS_GSE_OPEN_SET_ASSOCIATION_TRAINING_SEED_V1"
            if passed else "FAIL_GSE_OPEN_SET_ASSOCIATION_TRAINING_SEED_V1"
        ),
        "seed": seed,
        "best_epoch": best_epoch,
        "epochs_completed": len(history),
        "optimizer_steps": optimizer_steps,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "source_gse_checkpoint_sha256": checkpoint_sha,
        "fit_pairs": len(fit_label),
        "selection_pairs": len(selection_label),
        "best_selection": final_metrics,
        "selection_contract": contract.to_dict(),
        "duration_seconds": time.monotonic() - started,
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()),
        "c09_worlds_read": 0,
        "strict_test_worlds_read": 0,
        "mtare_worlds_read": 0,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-run", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--pair-cache", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--seed", required=True, type=int)
    args = parser.parse_args()
    result = train(args.dataset_run, args.checkpoint, args.pair_cache, args.output_dir, args.seed)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["overall_status"].startswith("PASS_") else 2


if __name__ == "__main__":
    raise SystemExit(main())
