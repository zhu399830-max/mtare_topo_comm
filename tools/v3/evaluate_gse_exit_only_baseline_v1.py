#!/usr/bin/env python3
"""Evaluate frozen M1D checkpoints on the GSE validation sequences.

Only the current (fifth) LiDAR frame is passed to M1D.  Sequence metadata and
Teacher targets are used after inference solely for metrics and provenance.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from _bootstrap import PROJECT_ROOT
from mtare_topo.data.gse_training_dataset import GSESequenceDataset
from mtare_topo.evaluation.gse_exit_only_baseline import (
    five_event_scores,
    m1d_role_logits_to_gse,
)
from mtare_topo.evaluation.phase3_semantic_metrics import decode_direction_components, match_headings
from mtare_topo.representation.phase3_structural_semantics import StructuralSemanticNet


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_checkpoint(value: str) -> tuple[int, Path, str]:
    pieces = value.split(":", 2)
    if len(pieces) != 3:
        raise argparse.ArgumentTypeError("checkpoint must be SEED:PATH:SHA256")
    seed = int(pieces[0])
    path = Path(pieces[1]).resolve()
    expected = pieces[2].lower()
    if seed not in (0, 1, 2) or len(expected) != 64:
        raise argparse.ArgumentTypeError("checkpoint seed/hash contract invalid")
    return seed, path, expected


def collate(samples: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "student": torch.from_numpy(np.stack([sample["student"][-1] for sample in samples])),
        "event_index": np.asarray([sample["targets"]["event_index"] for sample in samples], dtype=np.int64),
        "exit_mask": np.stack([sample["targets"]["exit_mask"] for sample in samples]),
        "exit_heading_unit": np.stack([sample["targets"]["exit_heading_unit"] for sample in samples]),
        "global_sequence_index": np.asarray([sample["global_sequence_index"] for sample in samples], dtype=np.int64),
        "observation_id": [sample["observation_id"] for sample in samples],
        "parent_id": [sample["parent_id"] for sample in samples],
    }


def _target_headings(mask: np.ndarray, unit: np.ndarray) -> list[float]:
    indices = np.flatnonzero(np.asarray(mask, dtype=np.uint8))
    return sorted(
        float(np.degrees(np.arctan2(float(unit[index, 0]), float(unit[index, 1]))) % 360.0)
        for index in indices
    )


@torch.no_grad()
def evaluate_seed(
    dataset: GSESequenceDataset,
    checkpoint: tuple[int, Path, str],
    output_dir: Path,
    *,
    device: torch.device,
) -> dict[str, Any]:
    seed, path, expected_hash = checkpoint
    actual_hash = sha256(path)
    if actual_hash != expected_hash:
        raise RuntimeError(f"M1D seed {seed} checkpoint hash drift: {actual_hash}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("mode") != "M1D" or int(payload.get("seed", -1)) != seed:
        raise RuntimeError(f"M1D seed {seed} checkpoint identity mismatch")
    model = StructuralSemanticNet().to(device)
    model.load_state_dict(payload["model"], strict=True)
    model.eval()
    loader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=0, collate_fn=collate)

    arrays: dict[str, list[np.ndarray]] = {
        key: []
        for key in (
            "role_logits",
            "count_logits",
            "direction_logits",
            "z_role",
            "target_event_index",
            "target_exit_mask",
            "target_exit_heading_unit",
            "global_sequence_index",
        )
    }
    observation_ids: list[str] = []
    parent_ids: list[str] = []
    matched = predicted_count = target_count = 0
    angular_errors: list[float] = []
    count_absolute = count_exact = 0
    for batch in loader:
        student = batch["student"].to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            outputs = model(student)
        role_logits = outputs["role_logits"].float().cpu().numpy()
        count_logits = outputs["count_logits"].float().cpu().numpy()
        direction_logits = outputs["direction_logits"].float().cpu().numpy()
        predicted_counts = count_logits.argmax(axis=1) + 1
        true_counts = batch["exit_mask"].sum(axis=1)
        count_absolute += int(np.abs(predicted_counts - true_counts).sum())
        count_exact += int(np.sum(predicted_counts == true_counts))
        for row in range(len(direction_logits)):
            prediction = decode_direction_components(direction_logits[row], 0.5)
            truth = _target_headings(batch["exit_mask"][row], batch["exit_heading_unit"][row])
            current_matched, current_predicted, current_truth, errors = match_headings(prediction, truth)
            matched += current_matched
            predicted_count += current_predicted
            target_count += current_truth
            angular_errors.extend(errors)
        arrays["role_logits"].append(role_logits.astype(np.float16))
        arrays["count_logits"].append(count_logits.astype(np.float16))
        arrays["direction_logits"].append(direction_logits.astype(np.float16))
        arrays["z_role"].append(outputs["z_role"].float().cpu().numpy().astype(np.float16))
        arrays["target_event_index"].append(batch["event_index"])
        arrays["target_exit_mask"].append(batch["exit_mask"])
        arrays["target_exit_heading_unit"].append(batch["exit_heading_unit"])
        arrays["global_sequence_index"].append(batch["global_sequence_index"])
        observation_ids.extend(batch["observation_id"])
        parent_ids.extend(batch["parent_id"])

    packed = {key: np.concatenate(values) for key, values in arrays.items()}
    if (
        len(packed["target_event_index"]) != len(dataset)
        or len(dataset) != 24462
        or len(np.unique(np.asarray(parent_ids))) != 10
    ):
        raise RuntimeError("M1D baseline evaluation lost validation observations")
    event_probabilities = m1d_role_logits_to_gse(packed["role_logits"])
    precision = matched / predicted_count if predicted_count else 0.0
    recall = matched / target_count if target_count else 0.0
    direction_f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    metrics = {
        "seed": seed,
        "checkpoint": str(path.relative_to(PROJECT_ROOT)),
        "checkpoint_sha256": actual_hash,
        "checkpoint_epoch": int(payload["epoch"]),
        "validation_sequences": len(dataset),
        "student_input": "current_fifth_frame_range_valid_only",
        "event_mapping": {
            "interior": "corridor",
            "junction": "junction",
            "terminal": "terminal",
            "turn": "unavailable",
            "geometry_transition": "unavailable",
        },
        "event": five_event_scores(packed["target_event_index"], event_probabilities),
        "per_parent_event": [
            {
                "parent_id": parent_id,
                "frames": int(np.sum(np.asarray(parent_ids) == parent_id)),
                "event": five_event_scores(
                    packed["target_event_index"][np.asarray(parent_ids) == parent_id],
                    event_probabilities[np.asarray(parent_ids) == parent_id],
                ),
            }
            for parent_id in sorted(np.unique(np.asarray(parent_ids)).tolist())
        ],
        "direction": {
            "matched": matched,
            "predicted": predicted_count,
            "target": target_count,
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(direction_f1),
            "mean_matched_angular_error_deg": float(np.mean(angular_errors)) if angular_errors else None,
        },
        "count": {
            "exact_accuracy": float(count_exact / len(dataset)),
            "mean_absolute_error": float(count_absolute / len(dataset)),
        },
    }
    packed["event_probabilities"] = event_probabilities.astype(np.float16)
    packed["observation_id"] = np.asarray(observation_ids, dtype="U128")
    packed["parent_id"] = np.asarray(parent_ids, dtype="U64")
    if (
        len(metrics["per_parent_event"]) != 10
        or sum(row["frames"] for row in metrics["per_parent_event"]) != len(dataset)
    ):
        raise RuntimeError("M1D per-parent event diagnostics do not cover C09")
    np.savez_compressed(output_dir / f"m1d_seed{seed}_validation_outputs.npz", **packed)
    (output_dir / f"m1d_seed{seed}_metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-run", required=True, type=Path)
    parser.add_argument("--checkpoint", action="append", required=True, type=parse_checkpoint)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    checkpoints = sorted(args.checkpoint, key=lambda item: item[0])
    if [item[0] for item in checkpoints] != [0, 1, 2]:
        raise ValueError("exactly M1D checkpoint seeds 0/1/2 are required")
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    dataset = GSESequenceDataset(args.dataset_run.resolve(), "validation", augment_azimuth=False)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    per_seed = [evaluate_seed(dataset, checkpoint, output_dir, device=device) for checkpoint in checkpoints]
    summary = {
        "status": "PASS_GSE_EXIT_ONLY_BASELINE_VALIDATION_V1",
        "validation_sequences": len(dataset),
        "checkpoint_seeds": [0, 1, 2],
        "model_inputs": ["range", "valid_mask"],
        "forbidden_model_inputs": [
            "teacher_event",
            "teacher_geometry",
            "pose",
            "world_id",
            "trajectory_id",
            "future_frame",
            "C10",
            "M-TARE",
        ],
        "per_seed": per_seed,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
