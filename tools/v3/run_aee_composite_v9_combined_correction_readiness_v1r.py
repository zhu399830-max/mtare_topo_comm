#!/home/zeng-workstation/anaconda3/bin/python
"""Retry V5 six-case readiness after the startup-only import-path fix."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import run_aee_composite_v9_combined_correction_readiness_v1 as base
from mtare_topo.governance import load_json, write_json


RUN_ID = "gate6_20260823_aee_composite_v9_combined_correction_readiness_v1r_seed20260823"
STATUS_PASS = "PASS_AEE_COMPOSITE_V9_COMBINED_CORRECTION_READINESS_V1R"
STATUS_FAIL = "FAIL_AEE_COMPOSITE_V9_COMBINED_CORRECTION_READINESS_V1R"
FAILED_STATUS = "FAIL_AEE_COMPOSITE_V9_COMBINED_CORRECTION_READINESS_V1_STARTUP_IMPORT"


def verify_failed_source(spec: dict, project_root: Path = base.PROJECT_ROOT) -> dict:
    root = project_root.resolve()
    source = (root / spec["failed_readiness_run"]).resolve()
    source.relative_to(root)
    state = load_json(source / "RUN_STATE.json")
    summary = load_json(source / "metrics/summary.json")
    seal_path = source / "artifacts/evidence_sha256.txt"
    if (
        state.get("state") != "FAILED"
        or state.get("overall_status") != FAILED_STATUS
        or summary.get("overall_status") != FAILED_STATUS
        or summary.get("completed_cases") != 0
        or summary.get("gazebo_started") is not False
    ):
        raise RuntimeError("V1R source is not the exact startup-only failure")
    if base.sha256(seal_path) != spec["failed_readiness_seal_sha256"]:
        raise RuntimeError("V1R failed-source seal identity drift")
    entries = 0
    for line in seal_path.read_text(encoding="utf-8").splitlines():
        digest, relative = line.split("  ", 1)
        if base.sha256(root / relative) != digest:
            raise RuntimeError(f"V1R failed-source evidence drift: {relative}")
        entries += 1
    return {
        "schema_version": "aee_combined_readiness_failed_source_v1",
        "run": spec["failed_readiness_run"],
        "status": FAILED_STATUS,
        "seal_sha256": spec["failed_readiness_seal_sha256"],
        "sealed_files": entries,
        "completed_cases": 0,
        "gazebo_started": False,
        "partial_reuse": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    known, _ = parser.parse_known_args(sys.argv[1:])
    spec = load_json(known.spec.resolve())
    failed_source = verify_failed_source(spec)
    write_json(known.run_dir.resolve() / "config/failed_readiness_source.json", failed_source)
    base.RUN_ID = RUN_ID
    base.STATUS_PASS = STATUS_PASS
    base.STATUS_FAIL = STATUS_FAIL
    return base.main()


if __name__ == "__main__":
    raise SystemExit(main())
