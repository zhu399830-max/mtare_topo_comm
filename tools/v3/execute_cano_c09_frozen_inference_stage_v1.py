#!/usr/bin/env python3
"""Run the three frozen M1D checkpoints once over all C09 sensor frames."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import zarr

import execute_cano_c08_frozen_inference_stage_v1 as core
from mtare_topo.data.c08_causal_replay import validate_m1d_checkpoint_identity
from mtare_topo.governance import write_json
from mtare_topo.representation.phase3_structural_semantics import StructuralSemanticNet


WORLDS = tuple(f"S{index:02d}_{name}_C09" for index, name in enumerate((
    "flat_tree_small", "3d_tree_small", "flat_unicyclic_small", "3d_unicyclic_small",
    "flat_branch_medium", "3d_branch_medium", "flat_loop_rich", "3d_loop_rich",
    "flat_complex", "3d_complex",
), 1))
EXPECTED_FRAMES = 15833


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    output_root = run_dir / "artifacts/inference"
    output_root.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    records, total = [], 0
    for seed, checkpoint_path in core.CHECKPOINTS.items():
        actual = core.sha256(checkpoint_path)
        if actual != core.EXPECTED_SHA256[seed]:
            raise RuntimeError(f"seed {seed} checkpoint hash drift: {actual}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        validate_m1d_checkpoint_identity(checkpoint, seed)
        model = StructuralSemanticNet().to(device)
        model.load_state_dict(checkpoint["model"])
        model.eval()
        for world in WORLDS:
            shard = zarr.open_group(str(run_dir / f"artifacts/sensor/{world}.zarr"), mode="r")
            frames = int(shard["range_m"].shape[0])
            arrays = {
                "direction_logits": np.empty((frames, 720), dtype=np.float32),
                "count_probabilities": np.empty((frames, 6), dtype=np.float32),
                "role_probabilities": np.empty((frames, 3), dtype=np.float32),
                "z_role": np.empty((frames, 128), dtype=np.float32),
            }
            with torch.inference_mode():
                for start in range(0, frames, args.batch_size):
                    end = min(frames, start + args.batch_size)
                    ranges = np.asarray(shard["range_m"][start:end], dtype=np.float32) / 50.0
                    valid = np.asarray(shard["valid_mask"][start:end], dtype=np.float32)
                    outputs = model(torch.from_numpy(np.stack((ranges, valid), axis=1)).to(device))
                    arrays["direction_logits"][start:end] = outputs["direction_logits"].cpu().numpy()
                    arrays["count_probabilities"][start:end] = torch.softmax(outputs["count_logits"], 1).cpu().numpy()
                    arrays["role_probabilities"][start:end] = torch.softmax(outputs["role_logits"], 1).cpu().numpy()
                    arrays["z_role"][start:end] = outputs["z_role"].cpu().numpy()
            output = output_root / f"m1d_seed{seed}_{world}.npz"
            np.savez_compressed(output, **arrays)
            record = {"seed": seed, "world": world, "frames": frames,
                      "checkpoint": str(checkpoint_path.relative_to(core.PROJECT_ROOT)),
                      "checkpoint_sha256": actual, "checkpoint_epoch": int(checkpoint["epoch"]),
                      "device": str(device), "output": str(output.relative_to(run_dir))}
            records.append(record); total += frames
            print(json.dumps(record), flush=True)
    if len(records) != 30 or total != EXPECTED_FRAMES * 3:
        raise RuntimeError(f"inference count mismatch: records={len(records)}, frames={total}")
    summary = {"schema_version": "cano_c09_frozen_inference_stage_v1",
               "overall_status": "PASS_C09_FROZEN_INFERENCE_STAGE_V1",
               "checkpoint_count": 3, "world_passes": 30,
               "unique_sensor_frames": EXPECTED_FRAMES, "model_inference_frames": total,
               "training_samples_consumed": 0, "optimizer_steps": 0, "device": str(device),
               "records": records, "c10_worlds_read": 0, "mtare_worlds_read": 0,
               "duration_seconds": time.monotonic() - started}
    write_json(run_dir / "metrics/inference_summary.json", summary)
    print(json.dumps(summary), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
