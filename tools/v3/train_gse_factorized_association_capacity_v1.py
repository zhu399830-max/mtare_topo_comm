#!/usr/bin/env python3
"""Train one frozen-perception Factorized GSE association capacity seed."""

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

from mtare_topo.representation.gse_exit_token_association import hard_negative_tail_loss
from mtare_topo.representation.gse_factorized_association import (
    FactorizedAssociationVerifier,
    factorized_pair_features,
    learned_geometry_profiles,
    parameter_count,
)
from mtare_topo.representation.gse_open_set_association import (
    OpenSetAssociationContract,
    select_nonvacuous_threshold,
)
from mtare_topo.teacher.gse_factorized_association_teacher import causal_history_row_references


TOKEN_KEYS = (
    "exit_confidence", "exit_heading_unit", "exit_opening_width_m",
    "exit_vertical_profile", "exit_descriptor",
)


def _read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def _pairs(manifest: list[dict], partition: str) -> dict[str, np.ndarray]:
    rows = [row for row in manifest if row["partition"] == partition]
    return {
        "left": np.asarray([row["query_observation_row"] for row in rows for _ in (0, 1)], dtype=np.int64),
        "right": np.asarray([
            value for row in rows
            for value in (row["positive_observation_row"], row["hard_negative_observation_row"])
        ], dtype=np.int64),
        "label": np.asarray([value for _ in rows for value in (1, 0)], dtype=np.uint8),
        "family": np.asarray([row["family"] for row in rows for _ in (0, 1)]),
        "physical_positive": np.asarray([
            value for row in rows for value in (
                row["positive_view_kind"] != "singleton_circular_shift_augmentation", True
            )
        ], dtype=np.bool_),
    }


def _threshold_metrics(
    scores: np.ndarray,
    labels: np.ndarray,
    families: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    accepted = np.asarray(scores) >= float(threshold)
    labels = np.asarray(labels, dtype=np.bool_)
    families = np.asarray(families).astype(str)
    tp = int(np.sum(accepted & labels))
    fp = int(np.sum(accepted & ~labels))
    positive = int(np.sum(labels))
    total = tp + fp
    per_family = {}
    for family in sorted(set(families.tolist())):
        rows = families == family
        current_tp = int(np.sum(accepted & labels & rows))
        current_fp = int(np.sum(accepted & ~labels & rows))
        current_positive = int(np.sum(labels & rows))
        current_total = current_tp + current_fp
        per_family[family] = {
            "accepted": current_total,
            "precision": current_tp / current_total if current_total else 0.0,
            "recall": current_tp / current_positive if current_positive else 0.0,
        }
    return {
        "accepted": total, "true_positive": tp, "false_positive": fp,
        "precision": tp / total if total else 0.0,
        "false_accept_rate": fp / total if total else 0.0,
        "recall": tp / positive if positive else 0.0, "per_family": per_family,
    }


def _safe_threshold(
    scores: np.ndarray, labels: np.ndarray, families: np.ndarray,
) -> dict[str, Any] | None:
    try:
        return select_nonvacuous_threshold(
            np.asarray(scores, dtype=np.float64), labels, families, OpenSetAssociationContract()
        )
    except RuntimeError:
        return None


def _curve(scores: np.ndarray, labels: np.ndarray) -> list[dict[str, float | int]]:
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.bool_)
    order = np.argsort(-scores, kind="stable")
    rows = []
    tp = fp = 0
    positives = int(np.sum(labels))
    start = 0
    while start < len(order):
        stop = start + 1
        threshold = float(scores[order[start]])
        while stop < len(order) and scores[order[stop]] == threshold:
            stop += 1
        group = labels[order[start:stop]]
        tp += int(np.sum(group)); fp += int(np.sum(~group))
        rows.append({
            "threshold": threshold, "accepted": tp + fp,
            "precision": tp / (tp + fp), "recall": tp / positives,
            "false_accept_rate": fp / (tp + fp),
        })
        start = stop
    return rows


def _evaluate(
    model: FactorizedAssociationVerifier,
    features: np.ndarray,
    labels: np.ndarray,
    families: np.ndarray,
) -> tuple[dict[str, Any], np.ndarray]:
    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(features)).float()
        scores = torch.sigmoid(logits).cpu().numpy().astype(np.float64)
        target = torch.from_numpy(labels.astype(np.float32))
        bce = float(torch.nn.functional.binary_cross_entropy_with_logits(logits, target))
    return {"balanced_bce": bce, "threshold_selection": _safe_threshold(scores, labels, families)}, scores


def _train_variant(
    fit_features: np.ndarray,
    selection_features: np.ndarray,
    fit: dict[str, np.ndarray],
    selection: dict[str, np.ndarray],
    *,
    seed: int,
    include_route_geometry: bool,
    output_dir: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=False)
    mean = fit_features.mean(axis=0, dtype=np.float64).astype(np.float32)
    std = np.maximum(fit_features.std(axis=0, dtype=np.float64).astype(np.float32), 1e-5)
    fit_normalized = ((fit_features - mean) / std).astype(np.float32)
    selection_normalized = ((selection_features - mean) / std).astype(np.float32)
    np.savez_compressed(output_dir / "normalization.npz", mean=mean, std=std)
    _seed(seed + (0 if include_route_geometry else 1000))
    model = FactorizedAssociationVerifier(include_route_geometry=include_route_geometry)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    generator = np.random.default_rng(seed + (0 if include_route_geometry else 1000))
    fit_tensor = torch.from_numpy(fit_normalized)
    target_tensor = torch.from_numpy(fit["label"].astype(np.float32))
    best_bce = math.inf
    best_epoch = 0
    stale = 0
    optimizer_steps = 0
    history = []
    for epoch in range(1, 51):
        model.train()
        order = generator.permutation(len(fit_tensor))
        total_loss = 0.0
        for start in range(0, len(order), 256):
            indices = order[start:start + 256]
            if len(np.unique(fit["label"][indices])) < 2:
                indices = np.concatenate((indices, order[:256]))
            index = torch.from_numpy(indices)
            optimizer.zero_grad(set_to_none=True)
            logits = model(fit_tensor[index])
            target = target_tensor[index]
            probability = torch.sigmoid(logits)
            loss = (
                torch.nn.functional.binary_cross_entropy_with_logits(logits, target)
                + 0.5 * hard_negative_tail_loss(probability, target)
            )
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            if not torch.isfinite(norm):
                raise RuntimeError("factorized association gradient is nonfinite")
            optimizer.step()
            optimizer_steps += 1
            total_loss += float(loss.detach()) * len(indices)
        validation, _ = _evaluate(
            model, selection_normalized, selection["label"], selection["family"]
        )
        record = {
            "epoch": epoch, "fit_loss": total_loss / len(order),
            "selection_balanced_bce": validation["balanced_bce"],
            "selection_threshold_available": validation["threshold_selection"] is not None,
            "optimizer_steps": optimizer_steps,
        }
        history.append(record)
        with (output_dir / "epoch_metrics.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, separators=(",", ":"), sort_keys=True) + "\n")
        if validation["balanced_bce"] < best_bce - 1e-8:
            best_bce = float(validation["balanced_bce"]); best_epoch = epoch; stale = 0
            torch.save({
                "schema_version": "gse_factorized_association_checkpoint_v1",
                "model": model.state_dict(), "seed": seed, "epoch": epoch,
                "include_route_geometry": include_route_geometry,
                "selection_balanced_bce": best_bce,
            }, output_dir / "best.pt")
        else:
            stale += 1
        if stale >= 8:
            break
    checkpoint = torch.load(output_dir / "best.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"])
    metrics, scores = _evaluate(
        model, selection_normalized, selection["label"], selection["family"]
    )
    threshold = metrics["threshold_selection"]
    physical = None
    if threshold is not None:
        keep = selection["physical_positive"]
        physical = _threshold_metrics(
            scores[keep], selection["label"][keep], selection["family"][keep], threshold["threshold"]
        )
    np.savez_compressed(
        output_dir / "selection_outputs.npz", score=scores,
        label=selection["label"], family=selection["family"],
        physical_positive=selection["physical_positive"],
    )
    with (output_dir / "selection_curve.jsonl").open("w", encoding="utf-8") as stream:
        for row in _curve(scores, selection["label"]):
            stream.write(json.dumps(row, separators=(",", ":"), sort_keys=True) + "\n")
    summary = {
        "schema_version": "gse_factorized_association_capacity_variant_v1",
        "seed": seed, "variant": "full_route_conditioned" if include_route_geometry else "no_route_geometry",
        "parameters": parameter_count(include_route_geometry), "best_epoch": best_epoch,
        "epochs_completed": len(history), "optimizer_steps": optimizer_steps,
        "selection": metrics, "physical_only_at_selected_threshold": physical,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def train_seed(
    teacher_path: Path,
    manifest_path: Path,
    observation_path: Path,
    token_path: Path,
    output_dir: Path,
    seed: int,
) -> dict[str, Any]:
    started = time.monotonic()
    if output_dir.exists():
        raise RuntimeError("factorized association seed output already exists")
    output_dir.mkdir(parents=True)
    rows = _read_jsonl(teacher_path)
    manifest = _read_jsonl(manifest_path)
    if len(rows) != 188126 or len(manifest) != 1066:
        raise RuntimeError("factorized association population drift")
    observation = np.load(observation_path, allow_pickle=False)
    with np.load(token_path, allow_pickle=False) as source:
        if not np.array_equal(source["global_sequence_index"], np.asarray([row["global_sequence_index"] for row in rows])):
            raise RuntimeError("factorized token global identity drift")
        token = {name: source[name] for name in TOKEN_KEYS}
    references, mask = causal_history_row_references(
        np.asarray([row["traversal_id"] for row in rows]),
        np.asarray([row["sequence_index"] for row in rows], dtype=np.int64),
    )
    profiles = learned_geometry_profiles(observation, references, mask)
    fit = _pairs(manifest, "fit"); selection = _pairs(manifest, "selection")
    if len(fit["label"]) != 1584 or len(selection["label"]) != 548:
        raise RuntimeError("factorized fit/selection pair count drift")
    feature = {}
    for partition, pair in (("fit", fit), ("selection", selection)):
        feature[(partition, True)] = factorized_pair_features(
            observation, token, profiles, pair["left"], pair["right"], include_route_geometry=True
        )
        feature[(partition, False)] = factorized_pair_features(
            observation, token, profiles, pair["left"], pair["right"], include_route_geometry=False
        )
    descriptor = observation[:, 12:140]
    descriptor /= np.maximum(np.linalg.norm(descriptor, axis=1, keepdims=True), 1e-8)
    descriptor_score = np.clip((np.sum(
        descriptor[selection["left"]] * descriptor[selection["right"]], axis=1
    ) + 1.0) / 2.0, 0.0, 1.0)
    descriptor_threshold = _safe_threshold(
        descriptor_score, selection["label"], selection["family"]
    )
    np.savez_compressed(
        output_dir / "descriptor_only_selection_outputs.npz", score=descriptor_score,
        label=selection["label"], family=selection["family"],
    )
    with (output_dir / "descriptor_only_selection_curve.jsonl").open("w", encoding="utf-8") as stream:
        for row in _curve(descriptor_score, selection["label"]):
            stream.write(json.dumps(row, separators=(",", ":"), sort_keys=True) + "\n")
    full = _train_variant(
        feature[("fit", True)], feature[("selection", True)], fit, selection,
        seed=seed, include_route_geometry=True, output_dir=output_dir / "full_route_conditioned",
    )
    no_route = _train_variant(
        feature[("fit", False)], feature[("selection", False)], fit, selection,
        seed=seed, include_route_geometry=False, output_dir=output_dir / "no_route_geometry",
    )
    summary = {
        "schema_version": "gse_factorized_association_capacity_seed_v1",
        "seed": seed, "fit_pairs": len(fit["label"]), "selection_pairs": len(selection["label"]),
        "descriptor_only_threshold": descriptor_threshold,
        "full_route_conditioned": full, "no_route_geometry": no_route,
        "duration_seconds": time.monotonic() - started,
        "backbone_optimizer_steps": 0, "model_inference_frames": 0,
        "c09_worlds_read": 0, "strict_test_worlds_read": 0, "mtare_worlds_read": 0,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--observation", required=True, type=Path)
    parser.add_argument("--tokens", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--seed", required=True, type=int)
    args = parser.parse_args()
    if args.seed not in (0, 1, 2) or args.output_dir.exists():
        raise RuntimeError("formal factorized association seed/output contract drift")
    result = train_seed(
        args.teacher, args.manifest, args.observation, args.tokens, args.output_dir, args.seed
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
