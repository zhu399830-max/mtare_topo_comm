#!/usr/bin/env python3
"""Replay frozen final slope heads and integrate them into C01--C08 GSE features."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from _bootstrap import PROJECT_ROOT
from mtare_topo.data.gse_slope_corrective_dataset import load_slope_corrective_partition
from mtare_topo.governance import load_json, write_json
from mtare_topo.representation.gse_slope_corrective import PhysicsGuidedSlopeResidualNet
from mtare_topo.representation.gse_slope_risk_calibration import apply_residual_scale
from mtare_topo.representation.gse_unified_observation import replace_normalized_slope


EXPECTED_OBSERVATIONS = 188_126
EXPECTED_FIT = 142_184
EXPECTED_SELECTION = 45_942


def _teacher_indices(path: Path) -> np.ndarray:
    values: list[int] = []
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                values.append(int(json.loads(line)["global_sequence_index"]))
    result = np.asarray(values, dtype=np.int64)
    if result.shape != (EXPECTED_OBSERVATIONS,) or len(np.unique(result)) != len(result):
        raise RuntimeError("unified observation Teacher identity drift")
    return result


@torch.no_grad()
def _predict(
    model: PhysicsGuidedSlopeResidualNet,
    features: np.ndarray,
    prior: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    blocks = []
    for start in range(0, len(features), 4096):
        stop = min(start + 4096, len(features))
        output = model(
            torch.from_numpy(features[start:stop]).to(device),
            torch.from_numpy(prior[start:stop]).to(device),
        )
        blocks.append(output["slope_deg"].float().cpu().numpy())
    result = np.concatenate(blocks).astype(np.float32, copy=False)
    if result.shape != prior.shape or not np.all(np.isfinite(result)):
        raise RuntimeError("frozen slope replay produced invalid output")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", required=True, type=Path)
    parser.add_argument("--corrective-run", required=True, type=Path)
    parser.add_argument("--risk-calibration", required=True, type=Path)
    parser.add_argument("--teacher", required=True, type=Path)
    for seed in range(3):
        parser.add_argument(f"--observation{seed}", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("exact frozen slope replay requires the original CUDA execution path")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    corrective = args.corrective_run.resolve()
    fit = load_slope_corrective_partition(args.cache_dir.resolve(), "fit")
    selection = load_slope_corrective_partition(args.cache_dir.resolve(), "selection")
    if len(fit["features"]) != EXPECTED_FIT or len(selection["features"]) != EXPECTED_SELECTION:
        raise RuntimeError("unified observation slope cache population drift")
    global_index = np.concatenate((fit["global_sequence_index"], selection["global_sequence_index"]))
    prior = np.concatenate((fit["prior_slope_deg"], selection["prior_slope_deg"]))
    teacher_index = _teacher_indices(args.teacher.resolve())
    if len(np.unique(global_index)) != EXPECTED_OBSERVATIONS or not np.array_equal(
        np.sort(global_index), np.sort(teacher_index)
    ):
        raise RuntimeError("unified observation cache/Teacher identity mismatch")
    calibration = load_json(args.risk_calibration.resolve())
    if (
        calibration.get("schema_version") != "gse_slope_residual_risk_calibration_v2"
        or calibration.get("selection_sequences") != EXPECTED_SELECTION
        or calibration.get("selection_worlds") != 20
        or calibration.get("c09_worlds_read_during_selection") != 0
        or float(calibration.get("residual_scale", -1)) != 0.89
    ):
        raise RuntimeError("unified observation risk calibration drift")
    device = torch.device("cuda:0")
    seed_rows = []
    for seed in range(3):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.use_deterministic_algorithms(True)
        checkpoint_path = corrective / f"artifacts/models/seed{seed}/best.pt"
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if (
            checkpoint.get("schema_version") != "gse_slope_corrective_checkpoint_v1"
            or checkpoint.get("seed") != seed
        ):
            raise RuntimeError(f"unified observation checkpoint drift for seed {seed}")
        model = PhysicsGuidedSlopeResidualNet().to(device)
        model.load_state_dict(checkpoint["model"])
        fit_raw = _predict(model, fit["features"], fit["prior_slope_deg"], device)
        selection_raw = _predict(
            model, selection["features"], selection["prior_slope_deg"], device
        )
        with np.load(
            corrective / f"artifacts/models/seed{seed}/selection_outputs.npz",
            allow_pickle=False,
        ) as archive:
            if (
                not np.array_equal(selection["global_sequence_index"], archive["global_sequence_index"])
                or not np.array_equal(selection_raw, archive["corrected_slope_deg"])
            ):
                raise RuntimeError(f"seed {seed} exact GPU selection replay drift")
        raw = np.concatenate((fit_raw, selection_raw))
        corrected = apply_residual_scale(prior, raw, float(calibration["residual_scale"]))
        original = np.load(getattr(args, f"observation{seed}").resolve(), allow_pickle=False)
        unified, audit = replace_normalized_slope(
            original, teacher_index, global_index, corrected
        )
        if audit.changed_columns != (10,) or not audit.unchanged_columns_byte_exact:
            raise RuntimeError(f"seed {seed} unified slope-only replacement drift")
        np.save(output / f"seed{seed}_unified_observation_features.npy", unified, allow_pickle=False)
        np.savez_compressed(
            output / f"seed{seed}_slope_outputs.npz",
            global_sequence_index=global_index,
            five_frame_prior_slope_deg=prior,
            raw_full_residual_slope_deg=raw,
            corrected_slope_deg=corrected,
            residual_scale=np.asarray(0.89, dtype=np.float32),
        )
        seed_rows.append(
            {
                "seed": seed,
                "checkpoint_epoch": int(checkpoint["epoch"]),
                "selection_gpu_replay_byte_exact": True,
                "audit": audit.to_dict(),
            }
        )
        del model, original, unified
        torch.cuda.empty_cache()
    summary = {
        "schema_version": "gse_unified_observation_features_v1",
        "overall_status": "PASS_GSE_UNIFIED_OBSERVATION_FEATURES_V1",
        "observations_per_seed": EXPECTED_OBSERVATIONS,
        "fit_sequences": EXPECTED_FIT,
        "selection_sequences": EXPECTED_SELECTION,
        "seeds": seed_rows,
        "residual_scale": 0.89,
        "slope_corrective_inference_sequences": 3 * EXPECTED_OBSERVATIONS,
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()),
        "gse_backbone_inference_frames": 0,
        "optimizer_steps": 0,
        "c09_worlds_read": 0,
        "strict_test_worlds_read": 0,
        "mtare_worlds_read": 0,
        "teacher": str(args.teacher.resolve().relative_to(PROJECT_ROOT)),
    }
    write_json(output / "summary.json", summary)
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
