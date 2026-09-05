#!/usr/bin/env python3
"""Train one causal circular exit-set process seed on C01-C06."""

from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
import sys

import numpy as np
import torch

import train_gse_circular_peak_geometry_v1 as core
from mtare_topo.representation.gse_circular_exit_set_process import (
    CausalCircularExitSetProcessNet,
    circular_exit_set_process_loss,
)


EXPECTED_PARAMETERS = 769784


def _infer_world(model, teacher_path: Path, source_root: Path, output: Path, *, device: torch.device, batch_size: int) -> int:
    world = core._load_world(teacher_path, source_root)
    values: dict[str, list[np.ndarray]] = defaultdict(list)
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(world["presence"]), batch_size):
            indices = np.arange(start, min(start + batch_size, len(world["presence"])))
            scans, _ = core._batch(world, indices, device=device)
            predicted = model(scans)
            values["exit_mass"].append(predicted["exit_mass"].cpu().numpy().astype(np.float16))
            values["exit_count_probability"].append(predicted["exit_count_probability"].cpu().numpy().astype(np.float32))
            values["heading_residual_deg"].append(predicted["peak_heading_residual_deg"].cpu().numpy().astype(np.float16))
            values["opening_width_m"].append(predicted["peak_opening_width_m"].cpu().numpy().astype(np.float16))
            values["vertical_profile_m"].append(predicted["peak_vertical_profile_m"].cpu().numpy().astype(np.float16))
            values["local_axis"].append(predicted["local_axis"].cpu().numpy().astype(np.float32))
            values["geometry"].append(torch.stack((predicted["width_m"], predicted["height_m"], predicted["slope_deg"], predicted["curvature_per_m"]), dim=-1).cpu().numpy().astype(np.float32))
            values["observation_uncertainty"].append(predicted["observation_uncertainty"].cpu().numpy().astype(np.float32))
    arrays = {name: np.concatenate(parts) for name, parts in values.items()}
    arrays["global_sequence_index"] = world["global_sequence_index"]
    np.savez_compressed(output / f"{world['parent']}.npz", **arrays)
    return len(world["presence"])


def _argument_value(name: str) -> Path:
    index = sys.argv.index(name)
    return Path(sys.argv[index + 1]).resolve()


def main() -> int:
    output_dir = _argument_value("--output-dir")
    core.CircularPeakGeometrySemanticNet = CausalCircularExitSetProcessNet
    core.circular_peak_geometry_loss = circular_exit_set_process_loss
    core.EXPECTED_PARAMETERS = EXPECTED_PARAMETERS
    core._infer_world = _infer_world
    return_code = core.main()
    checkpoint_path = output_dir / "best.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    checkpoint["schema_version"] = "gse_causal_circular_exit_set_process_checkpoint_v1"
    checkpoint["parameters"] = EXPECTED_PARAMETERS
    checkpoint["exit_set_process"] = {
        "cardinality": [1, 4],
        "set_likelihood": "mean true-exit log mass from one normalized circular probability budget",
        "decode": "predicted count plus radius-one diverse modes and bounded continuous residual",
        "existence_threshold": None,
    }
    torch.save(checkpoint, checkpoint_path)
    summary_path = output_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["schema_version"] = "gse_causal_circular_exit_set_process_training_seed_v1"
    summary["parameters"] = EXPECTED_PARAMETERS
    summary["exit_set_process"] = checkpoint["exit_set_process"]
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
