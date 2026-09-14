from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from learning.structural_learning.dataset import StructuralMapDataset, collate_numpy, collate_torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", default="results/lamp_structural_dataset_v1")
    parser.add_argument("--split", default="train")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--require-torch", action="store_true")
    args = parser.parse_args()

    ds = StructuralMapDataset(args.dataset_dir, args.split)
    samples = [ds[i] for i in range(min(args.batch_size, len(ds)))]
    batch = collate_numpy(samples)
    expected = {
        "partial_map": (len(samples), 8, 100, 100),
        "teacher_map": (len(samples), 8, 100, 100),
        "direction_reachability": (len(samples), 16),
        "direction_distance": (len(samples), 16),
        "direction_clearance": (len(samples), 16),
    }
    for key, shape in expected.items():
        if batch[key].shape != shape:
            raise RuntimeError(f"{key} shape {batch[key].shape} != {shape}")
        if not np.isfinite(batch[key]).all():
            raise RuntimeError(f"{key} contains NaN/Inf")

    torch_status = "not_requested"
    torch_shapes = {}
    if args.require_torch:
        tbatch = collate_torch(samples)
        torch_status = "ok"
        torch_shapes = {k: list(v.shape) for k, v in tbatch.items()}

    print(json.dumps({
        "dataset_dir": args.dataset_dir,
        "split": args.split,
        "batch_size": len(samples),
        "numpy_shapes": {k: list(v.shape) for k, v in batch.items()},
        "torch_status": torch_status,
        "torch_shapes": torch_shapes,
    }, indent=2))


if __name__ == "__main__":
    main()
