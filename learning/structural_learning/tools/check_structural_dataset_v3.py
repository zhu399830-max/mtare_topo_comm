from __future__ import annotations

import argparse
import json

import numpy as np

from learning.structural_learning.dataset import StructuralSurfaceDataset, collate_surface_numpy, collate_surface_torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", default="results/structural_dataset_v3")
    parser.add_argument("--split", default="train")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--all-samples", action="store_true")
    parser.add_argument("--require-torch", action="store_true")
    args = parser.parse_args()
    dataset = StructuralSurfaceDataset(args.dataset_dir, args.split)
    samples = [dataset[i] for i in range(min(args.batch_size, len(dataset)))]
    batch = collate_surface_numpy(samples)
    expected = {"input_surface": (len(samples), 4, 100, 100), "teacher_surface": (len(samples), 4, 100, 100), "center_pose": (len(samples), 7)}
    for key, shape in expected.items():
        if batch[key].shape != shape:
            raise RuntimeError(f"{key} shape {batch[key].shape} != {shape}")
        if not np.isfinite(batch[key]).all():
            raise RuntimeError(f"{key} contains NaN/Inf")
    checked = len(samples)
    if args.all_samples:
        for index in range(len(dataset)):
            sample = dataset[index]
            for key, shape in (("input_surface", (4, 100, 100)), ("teacher_surface", (4, 100, 100)), ("center_pose", (7,))):
                if sample[key].shape != shape or not np.isfinite(sample[key]).all():
                    raise RuntimeError(f"invalid {key} in {sample['path']}")
            if not sample["robot"] or not sample["split"] or not sample["region_id"]:
                raise RuntimeError(f"incomplete metadata in {sample['path']}")
        checked = len(dataset)
    torch_info = {"status": "not_requested", "shapes": {}}
    if args.require_torch:
        try:
            tbatch = collate_surface_torch(samples)
            torch_info = {"status": "ok", "shapes": {key: list(value.shape) for key, value in tbatch.items()}}
        except ModuleNotFoundError:
            torch_info = {"status": "unavailable", "shapes": {}}
    print(json.dumps({"dataset_dir": args.dataset_dir, "split": args.split, "batch_size": len(samples), "checked_samples": checked, "numpy_shapes": {key: list(value.shape) for key, value in batch.items()}, "torch": torch_info}, indent=2))


if __name__ == "__main__":
    main()
