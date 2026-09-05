#!/usr/bin/env python3
"""Calibrate a frozen three-seed open-set structural-node gate."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from _bootstrap import PROJECT_ROOT
from mtare_topo.representation.gse_exit_token_association import GSEExitTokenAssociationVerifier
from mtare_topo.representation.gse_open_set_association import (
    OpenSetAssociationContract,
    select_nonvacuous_threshold,
)


PASS_STATUS = "PASS_GSE_NODE_MATCHABILITY_ENSEMBLE_CALIBRATION_V1"
FAIL_STATUS = "FAIL_GSE_NODE_MATCHABILITY_ENSEMBLE_CALIBRATION_V1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def evaluate(source_run: Path, output_dir: Path) -> dict[str, Any]:
    source_run = source_run.resolve()
    output_dir = output_dir.resolve()
    source_run.relative_to(PROJECT_ROOT)
    output_dir.relative_to(PROJECT_ROOT)
    if output_dir.exists():
        raise RuntimeError("node-matchability calibration output already exists")
    output_dir.mkdir(parents=True)
    cache_path = source_run / "artifacts/pair_cache/pairs.npz"
    with np.load(cache_path, allow_pickle=False) as archive:
        cache = {name: archive[name] for name in archive.files}
    partition = cache["partition_code"].astype(np.uint8)
    label = cache["association_valid"].astype(np.uint8)
    parent = cache["parent_id"].astype(str)
    global_index = cache["compact_to_global_sequence_index"].astype(np.int64)
    if (
        len(partition) != 188126
        or np.sum(partition == 0) != 142184
        or np.sum(partition == 1) != 45942
        or int(np.sum(label[partition == 0])) != 39310
        or int(np.sum(label[partition == 1])) != 13693
        or len(np.unique(global_index)) != len(global_index)
    ):
        raise RuntimeError("node-matchability observation population drift")
    seed_scores = []
    source_sha = {}
    for seed in (0, 1, 2):
        directory = source_run / f"artifacts/models/seed{seed}"
        features_path = directory / "frozen_observation_features.npy"
        normalization_path = directory / "normalization.npz"
        checkpoint_path = directory / "best.pt"
        features = np.load(features_path, mmap_mode="r")
        with np.load(normalization_path, allow_pickle=False) as archive:
            mean = archive["observation_mean"].astype(np.float32)
            std = archive["observation_std"].astype(np.float32)
        if features.shape != (188126, 146):
            raise RuntimeError(f"seed{seed} frozen observation feature shape drift")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if checkpoint.get("schema_version") != "gse_exit_token_association_checkpoint_v2":
            raise RuntimeError(f"seed{seed} verifier checkpoint schema drift")
        model = GSEExitTokenAssociationVerifier()
        model.load_state_dict(checkpoint["model"])
        model.eval()
        scores = []
        with torch.inference_mode():
            for start in range(0, len(features), 8192):
                normalized = (
                    (np.asarray(features[start : start + 8192]) - mean) / std
                ).astype(np.float32)
                scores.append(
                    torch.sigmoid(model.matchability(torch.from_numpy(normalized)).squeeze(1)).numpy()
                )
        seed_scores.append(np.concatenate(scores).astype(np.float32))
        source_sha[f"seed{seed}"] = {
            "features": _sha256(features_path),
            "normalization": _sha256(normalization_path),
            "checkpoint": _sha256(checkpoint_path),
        }
    stacked = np.stack(seed_scores)
    ensemble = stacked.mean(axis=0, dtype=np.float64)
    selection_mask = partition == 1
    selection_label = label[selection_mask].astype(bool)
    selection_family = np.asarray([value[:3] for value in parent[selection_mask]])
    contract = OpenSetAssociationContract()
    selection = None
    selection_error = None
    try:
        selection = select_nonvacuous_threshold(
            ensemble[selection_mask], selection_label, selection_family, contract
        )
    except RuntimeError as exc:
        selection_error = str(exc)
    np.savez_compressed(
        output_dir / "node_matchability_selection_outputs.npz",
        global_sequence_index=global_index[selection_mask],
        family=selection_family,
        label=selection_label.astype(np.uint8),
        seed_score=stacked[:, selection_mask],
        ensemble_score=ensemble[selection_mask],
    )
    passed = selection is not None
    summary = {
        "schema_version": "gse_node_matchability_ensemble_calibration_v1",
        "overall_status": PASS_STATUS if passed else FAIL_STATUS,
        "scientific_pass": passed,
        "method": "arithmetic_mean_of_three_frozen_v2_matchability_scores",
        "fit_observations": 142184,
        "fit_positive": 39310,
        "fit_negative": 102874,
        "selection_observations": 45942,
        "selection_positive": 13693,
        "selection_negative": 32249,
        "selection": selection,
        "selection_error": selection_error,
        "contract": contract.to_dict(),
        "source_sha256": source_sha,
        "pair_cache_sha256": _sha256(cache_path),
        "optimizer_steps": 0,
        "backbone_optimizer_steps": 0,
        "c09_worlds_read": 0,
        "strict_test_worlds_read": 0,
        "mtare_worlds_read": 0,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-run", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    result = evaluate(args.source_run, args.output_dir)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["scientific_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
