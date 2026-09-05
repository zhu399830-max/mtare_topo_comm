#!/usr/bin/env python3
"""Train one Joint Cyclic Gap Simplex seed."""
from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
import sys

import numpy as np
import torch

import train_gse_circular_peak_geometry_v1 as core
from mtare_topo.representation.gse_joint_cyclic_gap_simplex import CONTRACT, JointCyclicGapSimplexNet, joint_cyclic_gap_simplex_loss


EXPECTED_PARAMETERS = 789650


def _infer_world(model, teacher_path: Path, source_root: Path, output: Path, *, device: torch.device, batch_size: int) -> int:
    world = core._load_world(teacher_path, source_root); values = defaultdict(list); model.eval()
    with torch.inference_mode():
        for start in range(0, len(world["presence"]), batch_size):
            indices = np.arange(start, min(start + batch_size, len(world["presence"])))
            scans, _ = core._batch(world, indices, device=device); predicted = model(scans)
            arrays = {
                "joint_bearing_deg": predicted["joint_bearing_deg"], "joint_bearing_scale_deg": predicted["joint_bearing_scale_deg"],
                "phase_concentration": predicted["phase_concentration"], "gap_fraction": predicted["gap_fraction"],
                "exit_opening_width_m": predicted["exit_opening_width_m"], "exit_vertical_profile_m": predicted["exit_vertical_profile_m"],
                "exit_descriptor": predicted["exit_descriptor"], "exit_uncertainty": predicted["exit_uncertainty"],
                "exit_count_probability": predicted["exit_count_probability"], "local_axis": predicted["local_axis"],
                "observation_uncertainty": predicted["observation_uncertainty"],
            }
            for name, value in arrays.items():
                dtype = np.float16 if name in ("gap_fraction", "exit_opening_width_m", "exit_vertical_profile_m", "exit_descriptor", "exit_uncertainty") else np.float32
                values[name].append(value.cpu().numpy().astype(dtype))
            geometry = torch.stack((predicted["width_m"], predicted["height_m"], predicted["slope_deg"], predicted["curvature_per_m"]), dim=-1)
            values["geometry"].append(geometry.cpu().numpy().astype(np.float32))
    arrays = {name: np.concatenate(parts) for name, parts in values.items()}; arrays["global_sequence_index"] = world["global_sequence_index"]
    np.savez_compressed(output / f"{world['parent']}.npz", **arrays); return len(world["presence"])


def _argument(name: str) -> Path:
    return Path(sys.argv[sys.argv.index(name) + 1]).resolve()


def main() -> int:
    output = _argument("--output-dir")
    core.CircularPeakGeometrySemanticNet = JointCyclicGapSimplexNet; core.circular_peak_geometry_loss = joint_cyclic_gap_simplex_loss; core.EXPECTED_PARAMETERS = EXPECTED_PARAMETERS; core._infer_world = _infer_world
    code = core.main(); checkpoint_path = output / "best.pt"; checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    checkpoint.update({"schema_version": "gse_joint_cyclic_gap_simplex_checkpoint_v1", "parameters": EXPECTED_PARAMETERS, "decoder": {"representation": CONTRACT.representation, "closure": CONTRACT.closure, "assignment": CONTRACT.assignment, "confidence": CONTRACT.confidence, "independent_slot_localizers": False}})
    torch.save(checkpoint, checkpoint_path); summary_path = output / "summary.json"; summary = json.loads(summary_path.read_text(encoding="utf-8")); summary.update({"schema_version": "gse_joint_cyclic_gap_simplex_training_seed_v1", "parameters": EXPECTED_PARAMETERS, "decoder": checkpoint["decoder"]}); summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"); return code


if __name__ == "__main__": raise SystemExit(main())
