#!/usr/bin/env python3
"""Run the three already-frozen M1D checkpoints on the approved C08 stream."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
import zarr

from _bootstrap import PROJECT_ROOT
from mtare_topo.data.c08_causal_replay import validate_m1d_checkpoint_identity
from mtare_topo.governance import write_json
from mtare_topo.representation.phase3_structural_semantics import StructuralSemanticNet


WORLDS = ("S01_flat_tree_small_C08", "S06_3d_branch_medium_C08", "S10_3d_complex_C08")
CHECKPOINTS = {
    0: PROJECT_ROOT / "results/gate2_representation/gate2_20260812_cano_phase3_masking_corrective_ray_dropout_v1r3_seed0/artifacts/models/m1d_seed0/best.pt",
    1: PROJECT_ROOT / "results/gate2_representation/gate2_20260812_cano_phase3_masking_corrective_ray_dropout_v1r3_seed0/artifacts/models/m1d_seed1/best.pt",
    2: PROJECT_ROOT / "results/gate2_representation/gate2_20260813_cano_phase3_masking_corrective_recovery_seed2_v1_seed2/artifacts/models/m1d_seed2/best.pt",
}
EXPECTED_SHA256 = {
    0: "55f6602fb7fd74e3a44c3a4697a7c4be4d31469f46348f631dc606612125f9fe",
    1: "3822d27af6d5fb0ef920da3e310927d42c5a112c5d671b4cbdc81d9db251f8df",
    2: "20b1e9c8198b06473b68e31f4d8a3eb50eb5f2fc1b7aa577aa1aac92ba35346e",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    output_root = run_dir / "artifacts/inference"
    output_root.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    records = []
    total = 0
    for seed, checkpoint_path in CHECKPOINTS.items():
        actual = sha256(checkpoint_path)
        if actual != EXPECTED_SHA256[seed]:
            raise RuntimeError(f"seed {seed} checkpoint hash drift: {actual}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        validate_m1d_checkpoint_identity(checkpoint, seed)
        model = StructuralSemanticNet().to(device)
        model.load_state_dict(checkpoint["model"])
        model.eval()
        for world in WORLDS:
            shard = zarr.open_group(str(run_dir / f"artifacts/sensor/{world}.zarr"), mode="r")
            frames = int(shard["range_m"].shape[0])
            direction_logits = np.empty((frames, 720), dtype=np.float32)
            count_probabilities = np.empty((frames, 6), dtype=np.float32)
            role_probabilities = np.empty((frames, 3), dtype=np.float32)
            z_role = np.empty((frames, 128), dtype=np.float32)
            with torch.inference_mode():
                for start in range(0, frames, args.batch_size):
                    end = min(frames, start + args.batch_size)
                    ranges = np.asarray(shard["range_m"][start:end], dtype=np.float32) / 50.0
                    valid = np.asarray(shard["valid_mask"][start:end], dtype=np.float32)
                    student = torch.from_numpy(np.stack((ranges, valid), axis=1)).to(device)
                    outputs = model(student)
                    direction_logits[start:end] = outputs["direction_logits"].cpu().numpy()
                    count_probabilities[start:end] = torch.softmax(outputs["count_logits"], dim=1).cpu().numpy()
                    role_probabilities[start:end] = torch.softmax(outputs["role_logits"], dim=1).cpu().numpy()
                    z_role[start:end] = outputs["z_role"].cpu().numpy()
            output = output_root / f"m1d_seed{seed}_{world}.npz"
            np.savez_compressed(
                output, direction_logits=direction_logits,
                count_probabilities=count_probabilities,
                role_probabilities=role_probabilities, z_role=z_role,
            )
            record = {
                "seed": seed, "world": world, "frames": frames,
                "checkpoint": str(checkpoint_path.relative_to(PROJECT_ROOT)),
                "checkpoint_sha256": actual,
                "checkpoint_epoch": int(checkpoint["epoch"]),
                "device": str(device), "output": str(output.relative_to(run_dir)),
            }
            records.append(record)
            total += frames
            print(json.dumps(record), flush=True)
    expected = sum(item["frames"] for item in records[:3]) * 3
    if len(records) != 9 or total != 4773 * 3 or expected != total:
        raise RuntimeError(f"inference count mismatch: records={len(records)}, frames={total}")
    summary = {
        "schema_version": "cano_c08_frozen_inference_stage_v1",
        "overall_status": "PASS_C08_FROZEN_INFERENCE_STAGE_V1",
        "checkpoint_count": 3, "world_passes": 9,
        "unique_sensor_frames": 4773, "model_inference_frames": total,
        "training_samples_consumed": 0, "optimizer_steps": 0,
        "device": str(device), "records": records,
        "duration_seconds": time.monotonic() - started,
    }
    write_json(run_dir / "metrics/inference_summary.json", summary)
    print(json.dumps(summary), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
