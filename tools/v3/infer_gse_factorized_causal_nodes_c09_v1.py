#!/usr/bin/env python3
"""Run frozen past-only junction/terminal node inference on all C09 worlds."""

from __future__ import annotations

import argparse
from collections import Counter
import gc
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from mtare_topo.data.cano_sensor_smoke import MAX_RANGE_M, NEAR_RANGE_M
from mtare_topo.data.gse_causal_episode_cache import compact_global_references
from mtare_topo.representation.gse_causal_episode_detector import (
    CausalEpisodeDetector,
    DIRECTIONAL_BINS,
    ENCODER_DIM,
    encode_spatial_scan_features,
    materialize_past_only_references,
)
from mtare_topo.representation.gse_causal_episode_runtime import (
    align_baseline_event_logits,
)
from mtare_topo.representation.gse_graph import GeometrySemanticEventNet
from mtare_topo.topology.factorized_gse_graph import FactorizedCausalDecisionBackend


STRUCTURAL_THRESHOLD = 0.986
EXPECTED_WORLDS = 10
EXPECTED_FRAMES = 32_678
EXPECTED_OBSERVATIONS = 24_462
EXPECTED_TRAVERSALS = 2_054


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _deployment_population(dataset_run: Path):
    import zarr

    sequence_rows = [
        row for row in _read_jsonl(dataset_run / "artifacts/sequence_manifest.jsonl")
        if str(row["parent_id"]).endswith("_C09")
    ]
    sequence_rows.sort(key=lambda row: int(row["global_sequence_index"]))
    parents = sorted({str(row["parent_id"]) for row in sequence_rows})
    if len(sequence_rows) != EXPECTED_OBSERVATIONS or len(parents) != EXPECTED_WORLDS:
        raise RuntimeError("C09 deployment sequence population drift")
    groups = {
        parent: zarr.open_group(
            str(dataset_run / "artifacts/dataset/validation" / f"{parent}.zarr"), mode="r"
        )
        for parent in parents
    }
    frame_chunks = []
    frame_shards = []
    previous_global = -1
    for parent in parents:
        group = groups[parent]
        frames = np.asarray(group["global_frame_index"][:], dtype=np.int64)
        if (
            group.attrs.get("schema_version") != "gse_deduplicated_world_v1"
            or group.attrs.get("split") != "validation"
            or group.attrs.get("parent_id") != parent
            or len(frames) == 0
            or np.any(np.diff(frames) <= 0)
            or int(frames[0]) <= previous_global
        ):
            raise RuntimeError(f"C09 frame shard identity drift: {parent}")
        previous_global = int(frames[-1])
        frame_chunks.append(frames)
        frame_shards.append({
            "parent_id": parent,
            "frames": len(frames),
            "first_global_frame_index": int(frames[0]),
            "last_global_frame_index": int(frames[-1]),
        })
    global_frames = np.concatenate(frame_chunks)
    if len(global_frames) != EXPECTED_FRAMES or len(np.unique(global_frames)) != EXPECTED_FRAMES:
        raise RuntimeError("C09 unique frame population drift")

    sequence_counter: Counter[str] = Counter()
    deployment_rows = []
    keys = []
    parent_id = []
    traversal_id = []
    sequence_index = []
    observation_id = []
    for record in sequence_rows:
        parent = str(record["parent_id"])
        traversal = str(record["traversal_id"])
        group = groups[parent]
        world_row = int(record["world_sequence_row"])
        anchor = int(record["world_anchor_frame_row"])
        key = int(record["global_sequence_index"])
        references = np.asarray(group["global_frame_references"][world_row], dtype=np.int64)
        local_references = np.asarray(group["local_frame_references"][world_row], dtype=np.int64)
        frame = int(group["local_frame_index"][anchor])
        current_sequence = int(sequence_counter[traversal])
        sequence_counter[traversal] += 1
        if (
            int(group["global_sequence_index"][world_row]) != key
            or references.shape != (5,)
            or local_references.shape != (5,)
            or int(local_references[-1]) != anchor
            or np.any(np.diff(references) != 1)
            or np.any(np.diff(local_references) != 1)
            or frame < 4
        ):
            raise RuntimeError("C09 deployment observation identity drift")
        deployment_rows.append({
            "traversal_id": traversal,
            "sequence_index": current_sequence,
            "frame_index": frame,
            "global_frame_references": references.tolist(),
        })
        keys.append(key)
        parent_id.append(parent)
        traversal_id.append(traversal)
        sequence_index.append(current_sequence)
        observation_id.append(str(record["observation_id"]))
    if len(sequence_counter) != EXPECTED_TRAVERSALS:
        raise RuntimeError("C09 directed traversal count drift")
    bank = materialize_past_only_references(deployment_rows)
    compact = compact_global_references(
        bank.global_frame_references, bank.valid_history_mask, global_frames
    )
    return {
        "sequence_rows": sequence_rows,
        "groups": groups,
        "frame_shards": frame_shards,
        "global_frames": global_frames,
        "compact_references": compact,
        "global_references": bank.global_frame_references,
        "history_mask": bank.valid_history_mask,
        "global_sequence_index": np.asarray(keys, dtype=np.int64),
        "parent_id": np.asarray(parent_id),
        "traversal_id": np.asarray(traversal_id),
        "sequence_index": np.asarray(sequence_index, dtype=np.int64),
        "observation_id": np.asarray(observation_id),
    }


def _encode_all_frames(base_model, population, *, device, batch_size: int):
    import torch

    pooled = np.empty((EXPECTED_FRAMES, ENCODER_DIM), dtype=np.float16)
    directional = np.empty(
        (EXPECTED_FRAMES, ENCODER_DIM, DIRECTIONAL_BINS), dtype=np.float16
    )
    offset = 0
    with torch.inference_mode():
        for shard in population["frame_shards"]:
            parent = shard["parent_id"]
            group = population["groups"][parent]
            count = int(shard["frames"])
            for start in range(0, count, batch_size):
                stop = min(start + batch_size, count)
                range_m = np.asarray(group["range_m"][start:stop], dtype=np.float32)
                valid = np.asarray(group["valid_mask"][start:stop], dtype=np.float32)
                if (
                    range_m.shape != (stop - start, 16, 720)
                    or valid.shape != range_m.shape
                    or not np.all(np.isfinite(range_m))
                    or np.any(range_m < NEAR_RANGE_M)
                    or np.any(range_m > MAX_RANGE_M)
                    or not np.all((valid == 0.0) | (valid == 1.0))
                ):
                    raise RuntimeError(f"C09 sensor payload drift: {parent}")
                scans = np.stack((range_m / MAX_RANGE_M, valid), axis=1)[:, None]
                features = encode_spatial_scan_features(
                    base_model.encoder,
                    torch.from_numpy(scans).to(device=device, dtype=torch.float32),
                )
                destination = slice(offset + start, offset + stop)
                pooled[destination] = features["pooled"][:, 0].cpu().numpy().astype(np.float16)
                directional[destination] = features["directional"][:, 0].cpu().numpy().astype(np.float16)
            offset += count
    if offset != EXPECTED_FRAMES:
        raise RuntimeError("C09 encoded frame count drift")
    return pooled, directional


def _infer_episode_model(
    model, pooled, directional, compact, history_mask, baseline_logits,
    *, device, batch_size: int,
):
    import torch

    probability = np.empty((EXPECTED_OBSERVATIONS, 5), dtype=np.float32)
    boundary = np.empty(EXPECTED_OBSERVATIONS, dtype=np.float32)
    uncertainty = np.empty(EXPECTED_OBSERVATIONS, dtype=np.float32)
    with torch.inference_mode():
        for start in range(0, EXPECTED_OBSERVATIONS, batch_size):
            stop = min(start + batch_size, EXPECTED_OBSERVATIONS)
            references = compact[start:stop]
            mask = history_mask[start:stop]
            local_pooled = np.zeros((stop - start, 12, ENCODER_DIM), dtype=np.float16)
            local_directional = np.zeros(
                (stop - start, 12, ENCODER_DIM, DIRECTIONAL_BINS), dtype=np.float16
            )
            local_pooled[mask] = pooled[references[mask]]
            local_directional[mask] = directional[references[mask]]
            outputs = model(
                torch.from_numpy(local_pooled).to(device=device, dtype=torch.float32),
                torch.from_numpy(local_directional).to(device=device, dtype=torch.float32),
                torch.from_numpy(mask).to(device=device),
                torch.from_numpy(baseline_logits[start:stop]).to(device=device, dtype=torch.float32),
            )
            probability[start:stop] = outputs["event_probability"].cpu().numpy()
            boundary[start:stop] = outputs["boundary_offset_m"].cpu().numpy()
            uncertainty[start:stop] = outputs["uncertainty"].cpu().numpy()
    return probability, boundary, uncertainty


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-run", required=True, type=Path)
    parser.add_argument("--training-run", required=True, type=Path)
    parser.add_argument("--episode-run", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--frame-batch-size", type=int, default=128)
    parser.add_argument("--episode-batch-size", type=int, default=128)
    args = parser.parse_args()
    started = time.monotonic()
    output = args.output_dir.resolve()
    if output.exists():
        raise RuntimeError("C09 causal node inference output already exists")
    output.mkdir(parents=True)

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("formal C09 causal node inference requires CUDA")
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda")
    dataset = args.dataset_run.resolve()
    training = args.training_run.resolve()
    episode = args.episode_run.resolve()
    population = _deployment_population(dataset)
    np.savez_compressed(
        output / "past_only_references.npz",
        global_sequence_index=population["global_sequence_index"],
        global_frame_references=population["global_references"],
        valid_history_mask=population["history_mask"],
    )

    seed_probability = []
    seed_boundary = []
    seed_uncertainty = []
    checkpoints = {}
    for seed in (0, 1, 2):
        base_path = training / f"artifacts/models/seed{seed}/best.pt"
        episode_path = episode / f"artifacts/models/seed{seed}/best.pt"
        base_checkpoint = torch.load(base_path, map_location="cpu", weights_only=False)
        if base_checkpoint.get("schema_version") != "gse_graph_checkpoint_v1":
            raise RuntimeError(f"seed{seed} base checkpoint drift")
        base_model = GeometrySemanticEventNet()
        base_model.load_state_dict(base_checkpoint["model"], strict=True)
        base_model.eval().to(device)
        for parameter in base_model.parameters():
            parameter.requires_grad_(False)
        pooled, directional = _encode_all_frames(
            base_model, population, device=device, batch_size=args.frame_batch_size
        )
        del base_model, base_checkpoint
        torch.cuda.empty_cache()

        with np.load(
            training / f"artifacts/models/seed{seed}/validation_outputs.npz",
            allow_pickle=False,
        ) as archive:
            validation_global = archive["global_sequence_index"].astype(np.int64)
            validation_parent = archive["parent_id"].astype(str)
            baseline_logits = archive["event_logits"].astype(np.float32)
        try:
            baseline_logits = align_baseline_event_logits(
                validation_global, validation_parent, baseline_logits,
                population["global_sequence_index"], population["parent_id"],
            )
        except (ValueError, RuntimeError) as exc:
            raise RuntimeError(f"seed{seed} C09 baseline population drift") from exc
        episode_checkpoint = torch.load(episode_path, map_location="cpu", weights_only=False)
        if (
            episode_checkpoint.get("schema_version")
            != "gse_causal_episode_detector_checkpoint_v1"
            or episode_checkpoint.get("seed") != seed
        ):
            raise RuntimeError(f"seed{seed} causal episode checkpoint drift")
        episode_model = CausalEpisodeDetector()
        episode_model.load_state_dict(episode_checkpoint["model"], strict=True)
        episode_model.eval().to(device)
        for parameter in episode_model.parameters():
            parameter.requires_grad_(False)
        probability, boundary, uncertainty = _infer_episode_model(
            episode_model, pooled, directional,
            population["compact_references"], population["history_mask"], baseline_logits,
            device=device, batch_size=args.episode_batch_size,
        )
        np.savez_compressed(
            output / f"seed{seed}_outputs.npz",
            global_sequence_index=population["global_sequence_index"],
            probability=probability,
            boundary_offset_m=boundary,
            uncertainty=uncertainty,
        )
        seed_probability.append(probability)
        seed_boundary.append(boundary)
        seed_uncertainty.append(uncertainty)
        checkpoints[str(seed)] = {
            "base_checkpoint_sha256": _sha256(base_path),
            "episode_checkpoint_sha256": _sha256(episode_path),
        }
        del pooled, directional, episode_model, episode_checkpoint, baseline_logits
        gc.collect()
        torch.cuda.empty_cache()

    ensemble_probability = np.mean(np.stack(seed_probability), axis=0).astype(np.float32)
    ensemble_boundary = np.mean(np.stack(seed_boundary), axis=0).astype(np.float32)
    ensemble_uncertainty = np.mean(np.stack(seed_uncertainty), axis=0).astype(np.float32)
    backend = FactorizedCausalDecisionBackend(
        global_sequence_index=population["global_sequence_index"],
        event_probability=ensemble_probability,
        traversal_id=population["traversal_id"],
        sequence_index=population["sequence_index"],
        boundary_offset_m=ensemble_boundary,
        uncertainty=ensemble_uncertainty,
        structural_threshold=STRUCTURAL_THRESHOLD,
    )
    np.savez_compressed(
        output / "ensemble_deployment_outputs.npz",
        global_sequence_index=population["global_sequence_index"],
        parent_id=population["parent_id"],
        observation_id=population["observation_id"],
        traversal_id=population["traversal_id"],
        sequence_index=population["sequence_index"],
        probability=ensemble_probability,
        boundary_offset_m=ensemble_boundary,
        uncertainty=ensemble_uncertainty,
    )
    with (output / "decision_triggers.jsonl").open("w", encoding="utf-8") as stream:
        for key in population["global_sequence_index"]:
            decision = backend.evaluate_node(int(key))
            if decision["accepted"]:
                stream.write(json.dumps({"global_sequence_index": int(key), **decision}, sort_keys=True) + "\n")
    valid_counts = population["history_mask"].sum(axis=1)
    manifest = {
        "schema_version": "gse_factorized_causal_node_c09_inference_v1",
        "worlds": EXPECTED_WORLDS,
        "unique_lidar_frames": EXPECTED_FRAMES,
        "causal_observations": EXPECTED_OBSERVATIONS,
        "directed_traversals": EXPECTED_TRAVERSALS,
        "valid_past_reference_cells": int(population["history_mask"].sum()),
        "minimum_history_frames": int(valid_counts.min()),
        "maximum_history_frames": int(valid_counts.max()),
        "future_reference_cells": 0,
        "teacher_inputs_read": 0,
        "teacher_identity_inputs_read": 0,
        "structural_threshold": STRUCTURAL_THRESHOLD,
        "raw_structural_triggers": backend.raw_structural_trigger_count,
        "decision_triggers": backend.decision_trigger_count,
        "spatial_encoder_inference_frames": 3 * EXPECTED_FRAMES,
        "episode_detector_inference_observations": 3 * EXPECTED_OBSERVATIONS,
        "optimizer_steps": 0,
        "model_updates": 0,
        "checkpoint_selection_steps": 0,
        "c10_worlds_read": 0,
        "strict_test_worlds_read": 0,
        "mtare_worlds_read": 0,
        "checkpoints": checkpoints,
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()),
        "duration_seconds": time.monotonic() - started,
        "frame_shards": population["frame_shards"],
    }
    (output / "inference_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
