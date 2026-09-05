#!/usr/bin/env python3
"""Calibrate all three frozen GSE checkpoints on validation outputs only."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from _bootstrap import PROJECT_ROOT
from mtare_topo.evaluation.gse_validation_evaluator import calibrate_validation_outputs
from mtare_topo.governance import load_json, write_json


EXPECTED_RUN_ID = "gate2_20260824_gse_graph_three_seed_training_v1r_seed0"
EXPECTED_STATUS = "PASS_GSE_GRAPH_THREE_SEED_TRAINING_V1R"
PASS_STATUS = "PASS_GSE_VALIDATION_CALIBRATION_V1"
FAIL_STATUS = "FAIL_GSE_VALIDATION_CALIBRATION_V1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, separators=(",", ":"), sort_keys=True) + "\n")


def _extract_records(result: dict[str, Any], output_dir: Path, seed: int) -> dict[str, Any]:
    compact = {
        **result,
        "event": {**result["event"]},
        "place_association": {**result["place_association"]},
        "exit_tokens": {
            **result["exit_tokens"],
            "descriptor_association": {**result["exit_tokens"]["descriptor_association"]},
        },
    }
    record_fields = (
        (compact["event"], "rejection_curve", f"seed{seed}_event_rejection_curve.jsonl"),
        (
            compact["uncertainty_diagnostic"],
            "curve",
            f"seed{seed}_uncertainty_diagnostic_curve.jsonl",
        ),
        (compact["place_association"], "curve", f"seed{seed}_place_association_curve.jsonl"),
        (compact["place_association"], "causal_decisions", f"seed{seed}_place_association_decisions.jsonl"),
        (compact["exit_tokens"], "presence_curve", f"seed{seed}_exit_presence_curve.jsonl"),
        (
            compact["exit_tokens"]["descriptor_association"],
            "curve",
            f"seed{seed}_exit_descriptor_curve.jsonl",
        ),
        (
            compact["exit_tokens"]["descriptor_association"],
            "causal_decisions",
            f"seed{seed}_exit_descriptor_decisions.jsonl",
        ),
    )
    for container, field, filename in record_fields:
        records = container.pop(field)
        _write_jsonl(output_dir / filename, records)
        container[f"{field}_file"] = filename
        container[f"{field}_records"] = len(records)
    return compact


def evaluate(training_run: Path, output_dir: Path) -> dict[str, Any]:
    root = PROJECT_ROOT.resolve()
    training_run = training_run.resolve()
    output_dir = output_dir.resolve()
    training_run.relative_to(root)
    output_dir.relative_to(root)
    if training_run.name != EXPECTED_RUN_ID:
        raise RuntimeError("unexpected three-seed GSE source run")
    state = load_json(training_run / "RUN_STATE.json")
    source_summary = load_json(training_run / "metrics/summary.json")
    if (
        state.get("state") != "COMPLETED"
        or state.get("overall_status") != EXPECTED_STATUS
        or source_summary.get("overall_status") != EXPECTED_STATUS
        or source_summary.get("strict_test_worlds_read") != 0
        or source_summary.get("mtare_worlds_read") != 0
    ):
        raise RuntimeError("three-seed GSE source is not a completed sealed PASS")
    if output_dir.exists():
        raise RuntimeError("validation calibration output directory already exists")
    output_dir.mkdir(parents=True)

    seal_path = training_run / "artifacts/evidence_sha256.txt"
    sealed = {}
    for line in seal_path.read_text(encoding="utf-8").splitlines():
        expected, relative = line.split("  ", 1)
        sealed[relative] = expected
    seed_results = []
    overall = PASS_STATUS
    for seed in (0, 1, 2):
        archive = training_run / f"artifacts/models/seed{seed}/validation_outputs.npz"
        relative = str(archive.relative_to(root))
        if sealed.get(relative) != _sha256(archive):
            raise RuntimeError(f"seed {seed} validation archive is unsealed or drifted")
        with np.load(archive, allow_pickle=False) as source:
            arrays = {name: source[name] for name in source.files}
        result = calibrate_validation_outputs(arrays)
        if result["validation_frames"] != 24462 or result["validation_parents"] != 10:
            raise RuntimeError(f"seed {seed} validation population drift")
        compact = _extract_records(result, output_dir, seed)
        place = compact["place_association"]["selection"]
        exit_association = compact["exit_tokens"]["descriptor_association"]["selection"]
        seed_pass = (
            place["precision"] >= 0.98
            and place["false_accept_rate"] <= 0.01
            and exit_association["precision"] >= 0.98
            and exit_association["false_accept_rate"] <= 0.01
        )
        compact["overall_status"] = "PASS_GSE_VALIDATION_CALIBRATION_SEED_V1" if seed_pass else "FAIL_GSE_VALIDATION_CALIBRATION_SEED_V1"
        if not seed_pass:
            overall = FAIL_STATUS
        write_json(output_dir / f"seed{seed}_summary.json", compact)
        seed_results.append(
            {
                "seed": seed,
                "overall_status": compact["overall_status"],
                "event_temperature": compact["event"]["temperature"]["temperature"],
                "event_probability_threshold": compact["event"]["rejection_selection"]["threshold"],
                "maximum_uncertainty": compact["event"]["rejection_selection"]["maximum_uncertainty"],
                "event_macro_f1": compact["event"]["rejection_selection"]["macro_f1"],
                "place_descriptor_minimum_similarity": place["threshold"],
                "place_association_precision": place["precision"],
                "place_association_recall": place["recall"],
                "place_false_loop_merge_rate": place["false_accept_rate"],
                "place_association_accepted": place["accepted"],
                "exit_presence_threshold": compact["exit_tokens"]["presence_threshold"]["threshold"],
                "exit_descriptor_minimum_similarity": exit_association["threshold"],
                "exit_descriptor_association_precision": exit_association["precision"],
                "exit_descriptor_association_recall": exit_association["recall"],
                "exit_descriptor_false_accept_rate": exit_association["false_accept_rate"],
                "exit_descriptor_association_accepted": exit_association["accepted"],
            }
        )
    summary = {
        "schema_version": "gse_validation_calibration_v1",
        "overall_status": overall,
        "source_training_run": str(training_run.relative_to(root)),
        "source_training_seal_sha256": _sha256(seal_path),
        "selection_policy": {
            "event_temperature": "minimum full-validation NLL",
            "event_rejection": "maximum full-validation five-class macro-F1 using min(calibrated confidence, 1-uncertainty)",
            "exit_presence": "maximum full-validation matched-token F1",
            "place_and_exit_association": "maximum recovered correct causal matches subject to precision>=0.98 and false-accept-rate<=0.01",
        },
        "seeds": seed_results,
        "strict_test_worlds_read": 0,
        "mtare_worlds_read": 0,
        "optimizer_steps": 0,
        "model_updates": 0,
    }
    write_json(output_dir / "summary.json", summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-run", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    result = evaluate(args.training_run, args.output_dir)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["overall_status"] == PASS_STATUS else 2


if __name__ == "__main__":
    raise SystemExit(main())
