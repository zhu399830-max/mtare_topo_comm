#!/usr/bin/env python3
"""Calibrate a C01-C08-only three-seed structural-event node score."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Any

import numpy as np

from _bootstrap import PROJECT_ROOT
from mtare_topo.governance import write_json
from mtare_topo.representation.gse_open_set_association import (
    OpenSetAssociationContract,
    select_nonvacuous_threshold,
)


PASS_STATUS = "PASS_GSE_EVENT_NODE_ENSEMBLE_CALIBRATION_V1"
FAIL_STATUS = "FAIL_GSE_EVENT_NODE_ENSEMBLE_CALIBRATION_V1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def evaluate(verifier_run: Path, node_run: Path, output_dir: Path) -> dict[str, Any]:
    verifier_run = verifier_run.resolve()
    node_run = node_run.resolve()
    output_dir = output_dir.resolve()
    for path in (verifier_run, node_run, output_dir.parent):
        path.relative_to(PROJECT_ROOT)
    if output_dir.exists():
        raise RuntimeError("event-node calibration output already exists")
    output_dir.mkdir(parents=True)

    node_path = node_run / "artifacts/calibration/node_matchability_selection_outputs.npz"
    with np.load(node_path, allow_pickle=False) as archive:
        global_index = archive["global_sequence_index"].astype(np.int64)
        family = archive["family"].astype(str)
        label = archive["label"].astype(np.uint8)
        node_score = archive["ensemble_score"].astype(np.float64)
    if (
        global_index.shape != (45942,)
        or len(np.unique(global_index)) != 45942
        or int(label.sum()) != 13693
        or set(family.tolist()) != {f"S{index:02d}" for index in range(1, 11)}
        or not np.all(np.isfinite(node_score))
    ):
        raise RuntimeError("frozen node selection population drift")

    probabilities = []
    source_sha256 = {"node_selection_outputs": _sha256(node_path)}
    for seed in (0, 1, 2):
        directory = verifier_run / f"artifacts/models/seed{seed}"
        features_path = directory / "frozen_observation_features.npy"
        identities_path = directory / "frozen_exit_token_outputs.npz"
        features = np.load(features_path, mmap_mode="r")
        with np.load(identities_path, allow_pickle=False) as archive:
            identities = archive["global_sequence_index"].astype(np.int64)
        if features.shape != (188126, 146) or identities.shape != (188126,):
            raise RuntimeError(f"seed{seed} frozen feature population drift")
        positions = np.searchsorted(identities, global_index)
        if np.any(positions >= len(identities)) or not np.array_equal(identities[positions], global_index):
            raise RuntimeError(f"seed{seed} selection identities do not align")
        values = np.asarray(features[positions, :5], dtype=np.float64)
        values /= values.sum(axis=1, keepdims=True)
        if not np.all(np.isfinite(values)) or np.any(values < 0.0):
            raise RuntimeError(f"seed{seed} event probabilities are invalid")
        probabilities.append(values)
        source_sha256[f"seed{seed}"] = {
            "features": _sha256(features_path),
            "identities": _sha256(identities_path),
        }

    seed_probability = np.stack(probabilities)
    mean_probability = seed_probability.mean(axis=0, dtype=np.float64)
    structural_probability = 1.0 - mean_probability[:, 0]
    combined_score = node_score * structural_probability
    contract = OpenSetAssociationContract()
    selection = None
    selection_error = None
    try:
        selection = select_nonvacuous_threshold(
            combined_score, label.astype(bool), family, contract
        )
    except RuntimeError as exc:
        selection_error = str(exc)

    np.savez_compressed(
        output_dir / "event_node_ensemble_selection_outputs.npz",
        global_sequence_index=global_index,
        family=family,
        label=label,
        seed_event_probability=seed_probability,
        mean_event_probability=mean_probability,
        node_ensemble_score=node_score,
        structural_probability=structural_probability,
        combined_score=combined_score,
    )
    passed = selection is not None
    summary = {
        "schema_version": "gse_event_node_ensemble_calibration_v1",
        "overall_status": PASS_STATUS if passed else FAIL_STATUS,
        "scientific_pass": passed,
        "method": "frozen_node_ensemble_times_equal_weight_three_seed_structural_probability",
        "selection_observations": 45942,
        "selection_positive": 13693,
        "selection_negative": 32249,
        "selection": selection,
        "selection_error": selection_error,
        "contract": contract.to_dict(),
        "source_sha256": source_sha256,
        "optimizer_steps": 0,
        "backbone_optimizer_steps": 0,
        "model_inference_frames": 0,
        "c09_worlds_read": 0,
        "strict_test_worlds_read": 0,
        "mtare_worlds_read": 0,
    }
    write_json(output_dir / "summary.json", summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verifier-run", required=True, type=Path)
    parser.add_argument("--node-run", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    result = evaluate(args.verifier_run, args.node_run, args.output_dir)
    print(result["overall_status"])
    return 0 if result["scientific_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
