#!/usr/bin/env python3
"""Finalize the recorded validation-only serialization recovery for one Cano run."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

from _bootstrap import PROJECT_ROOT
from mtare_topo.governance import load_json, write_json


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _duration(log_path: Path) -> float:
    text = log_path.read_text(encoding="utf-8")
    matches = re.findall(r"duration_seconds=([0-9.]+)", text)
    if len(matches) != 1:
        raise ValueError(f"expected one duration in {log_path}, got {matches}")
    return float(matches[0])


def _manifest(run_dir: Path) -> int:
    manifest = run_dir / "artifacts/evidence_sha256.txt"
    files = sorted(path for path in run_dir.rglob("*") if path.is_file() and path != manifest)
    manifest.write_text(
        "\n".join(
            f"{_sha256(path)}  {path.relative_to(PROJECT_ROOT)}" for path in files
        )
        + "\n",
        encoding="utf-8",
    )
    return len(files)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    args = parser.parse_args()
    run_dir = args.run_dir if args.run_dir.is_absolute() else PROJECT_ROOT / args.run_dir
    summary_path = run_dir / "metrics/summary.json"
    validation_path = run_dir / "metrics/adapter_validation.json"
    deviation_path = run_dir / "artifacts/validation_deviation.json"
    initial_log = run_dir / "logs/03_strict_validation_v1_serialization_failure.log"
    corrected_log = run_dir / "logs/03_strict_validation_v2_serialization_corrected.log"
    summary = load_json(summary_path)
    validation = load_json(validation_path)
    deviation = load_json(deviation_path)
    if summary.get("overall_status") != "FAIL_ADAPTER_VALIDATOR_NO_RESULT":
        raise RuntimeError("initial validator failure status was not preserved")
    if summary.get("execution", {}).get("generation_exit_code") != 0:
        raise RuntimeError("generation did not complete exactly once")
    if validation.get("overall_status") not in {
        "PASS_ADAPTER_SINGLE_WORLD_SMOKE",
        "FAIL_ADAPTER_MESH_GATE",
        "FAIL_ADAPTER_CONTRACT",
    }:
        raise RuntimeError("unexpected corrected validation status")
    if deviation.get("recovery_scope", "").find("No second topology/mesh generation") < 0:
        raise RuntimeError("deviation does not preserve no-regeneration boundary")

    execution = dict(summary["execution"])
    execution.update(
        generation_attempts=1,
        validator_attempts=2,
        validator_initial_exit_code=execution["validator_exit_code"],
        validator_initial_duration_seconds=execution["validator_duration_seconds"],
        validator_exit_code=2 if validation["overall_status"] != "PASS_ADAPTER_SINGLE_WORLD_SMOKE" else 0,
        validator_duration_seconds=_duration(corrected_log),
        validation_only_recovery=True,
        whole_world_retries=0,
    )
    summary["execution"] = execution
    summary["initial_outcome_before_serialization_recovery"] = "FAIL_ADAPTER_VALIDATOR_NO_RESULT"
    summary["overall_status"] = validation["overall_status"]
    summary["validation"] = validation
    summary["recorded_deviation"] = str(deviation_path.relative_to(PROJECT_ROOT))
    summary["validation_logs"] = {
        "initial_serialization_failure": str(initial_log.relative_to(PROJECT_ROOT)),
        "corrected_validation": str(corrected_log.relative_to(PROJECT_ROOT)),
    }
    write_json(summary_path, summary)
    write_json(
        run_dir / "RUN_STATE.json",
        {
            "schema_version": "v3_run_state_v1",
            "run_id": run_dir.name,
            "state": validation["overall_status"],
            "note": summary["claim_boundary"],
            "recorded_deviation": str(deviation_path.relative_to(PROJECT_ROOT)),
        },
    )
    tool_hashes_path = run_dir / "config/tool_hashes.json"
    tool_hashes = load_json(tool_hashes_path)
    tool_hashes["validation_recovery_finalizer"] = {
        "path": str(Path(__file__).resolve().relative_to(PROJECT_ROOT)),
        "sha256": _sha256(Path(__file__).resolve()),
    }
    write_json(tool_hashes_path, tool_hashes)
    entries = _manifest(run_dir)
    print(
        json.dumps(
            {
                "overall_status": summary["overall_status"],
                "generation_attempts": 1,
                "whole_world_retries": 0,
                "manifest_entries": entries,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
