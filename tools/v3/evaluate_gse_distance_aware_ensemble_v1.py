#!/usr/bin/env python3
"""Calibrate the frozen three-seed V2 ensemble on the online candidate domain."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from _bootstrap import PROJECT_ROOT
from mtare_topo.representation.gse_open_set_association import (
    OpenSetAssociationContract,
    select_online_candidate_threshold,
)


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
        raise RuntimeError("distance-aware ensemble output already exists")
    output_dir.mkdir(parents=True)
    pair_path = source_run / "artifacts/pair_cache/pairs.npz"
    with np.load(pair_path, allow_pickle=False) as archive:
        pair = {name: archive[name] for name in archive.files}
    required_pair = (
        "selection_label",
        "selection_family",
        "selection_distance_m",
        "selection_left",
        "selection_right",
    )
    if any(name not in pair for name in required_pair):
        raise RuntimeError("V2 pair cache lacks selection evidence")
    label = pair["selection_label"].astype(np.uint8)
    family = pair["selection_family"].astype(str)
    distance = pair["selection_distance_m"].astype(np.float64)
    left = pair["selection_left"].astype(np.int64)
    right = pair["selection_right"].astype(np.int64)
    if len(label) != 45372:
        raise RuntimeError("V2 selection pair population drift")
    seed_scores = []
    source_outputs = {}
    for seed in (0, 1, 2):
        path = source_run / f"artifacts/models/seed{seed}/selection_outputs.npz"
        with np.load(path, allow_pickle=False) as archive:
            observed = {name: archive[name] for name in archive.files}
        if (
            not np.array_equal(observed["label"].astype(np.uint8), label)
            or not np.array_equal(observed["family"].astype(str), family)
            or not np.array_equal(observed["left"].astype(np.int64), left)
            or not np.array_equal(observed["right"].astype(np.int64), right)
        ):
            raise RuntimeError(f"seed{seed} selection identity drift")
        score = observed["score"].astype(np.float64)
        if score.shape != label.shape or not np.all(np.isfinite(score)):
            raise RuntimeError(f"seed{seed} score evidence invalid")
        seed_scores.append(score)
        source_outputs[f"seed{seed}"] = _sha256(path)
    stacked = np.stack(seed_scores, axis=0)
    ensemble_score = stacked.mean(axis=0, dtype=np.float64)
    contract = OpenSetAssociationContract()
    threshold = None
    error = None
    try:
        threshold = select_online_candidate_threshold(
            ensemble_score, label.astype(bool), family, distance, contract
        )
    except RuntimeError as exc:
        error = str(exc)
    eligible = distance <= contract.maximum_candidate_distance_m + 1e-12
    domain_counts = {
        "all_pairs": int(len(label)),
        "eligible_pairs": int(np.sum(eligible)),
        "excluded_pairs": int(np.sum(~eligible)),
        "eligible_positive_pairs": int(np.sum(label[eligible])),
        "eligible_negative_pairs": int(np.sum(eligible) - np.sum(label[eligible])),
        "excluded_positive_pairs": int(np.sum(label[~eligible])),
        "excluded_negative_pairs": int(np.sum(~eligible) - np.sum(label[~eligible])),
    }
    expected_domain = {
        "all_pairs": 45372,
        "eligible_pairs": 31469,
        "excluded_pairs": 13903,
        "eligible_positive_pairs": 12813,
        "eligible_negative_pairs": 18656,
        "excluded_positive_pairs": 755,
        "excluded_negative_pairs": 13148,
    }
    if domain_counts != expected_domain:
        raise RuntimeError(f"online candidate-domain count drift: {domain_counts}")
    np.savez_compressed(
        output_dir / "ensemble_selection_outputs.npz",
        seed_score=stacked,
        ensemble_score=ensemble_score,
        label=label,
        family=family,
        distance_m=distance,
        eligible=eligible.astype(np.uint8),
        left=left,
        right=right,
    )
    passed = threshold is not None
    summary = {
        "schema_version": "gse_distance_aware_ensemble_calibration_v1",
        "overall_status": (
            "PASS_GSE_DISTANCE_AWARE_ENSEMBLE_CALIBRATION_V1"
            if passed else "FAIL_GSE_DISTANCE_AWARE_ENSEMBLE_CALIBRATION_V1"
        ),
        "scientific_pass": passed,
        "method": "arithmetic_mean_of_three_frozen_v2_seed_scores",
        "candidate_domain": "selection_distance_m_le_16",
        "domain_counts": domain_counts,
        "selection": threshold,
        "selection_error": error,
        "contract": contract.to_dict(),
        "source_selection_output_sha256": source_outputs,
        "pair_cache_sha256": _sha256(pair_path),
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
