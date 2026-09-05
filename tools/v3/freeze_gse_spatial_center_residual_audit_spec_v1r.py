#!/usr/bin/env python3
"""Freeze the identity-count-semantics corrective residual audit."""

from __future__ import annotations

import hashlib

from _bootstrap import PROJECT_ROOT
from mtare_topo.governance import load_json, write_json


RUN_ID = "gate3_20260828_gse_spatial_center_residual_audit_v1r_seed0"
SOURCE_SPEC = PROJECT_ROOT / "configs/v3/gate3/gse_spatial_center_residual_audit_v1.json"
SPEC_PATH = PROJECT_ROOT / "configs/v3/gate3/gse_spatial_center_residual_audit_v1r.json"
PREDECESSOR = "results/gate3_semantics/gate3_20260828_gse_spatial_center_residual_audit_v1_seed0"
PYTHON = "/home/zeng-workstation/.local/share/mtare_topo_comm/envs/phase3_torch290_cu129_zarr2187_v1/bin/python"


def _sha(path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    spec = load_json(SOURCE_SPEC)
    spec.update({
        "slug": "gse_spatial_center_residual_audit_v1r",
        "question": "After explicitly separating 274 total identities from 272 pair-capable identities, which component causes the remaining spatial-center within-4m deficit?",
        "method": spec["method"] + " The corrective explicitly excludes the two valid single-traversal identities from cross-traversal pair metrics while retaining them in inventory provenance.",
        "user_authorization": {
            "status": "APPROVED", "approved_by": "user-standing-authorization",
            "approved_at": "2026-08-28T11:30:00+08:00",
            "scope": "One immutable read-only residual-audit corrective for 274 total/272 pair-capable identity semantics; zero training and zero C09/C10/M-TARE.",
            "confirmation_reference": "User instructed automatic best-choice execution without routine approval prompts.",
        },
    })
    predecessor_files = [f"{PREDECESSOR}/RUN_STATE.json", f"{PREDECESSOR}/metrics/summary.json", f"{PREDECESSOR}/artifacts/evidence_sha256.txt"]
    spec["frozen_inputs"].update({relative: _sha(PROJECT_ROOT / relative) for relative in predecessor_files})
    tools = {
        "residual_metrics": "src/mtare_topo/evaluation/gse_spatial_center_residual.py",
        "center_teacher": "src/mtare_topo/teacher/gse_event_center_teacher.py",
        "executor": "tools/v3/execute_gse_spatial_center_residual_audit_v1.py",
        "runner_wrapper": "tools/v3/run_gse_spatial_center_residual_audit_v1r.py",
        "runner_base": "tools/v3/run_gse_spatial_center_residual_audit_v1.py",
        "freezer": "tools/v3/freeze_gse_spatial_center_residual_audit_spec_v1r.py",
        "preflight": "tools/v3/preflight.py", "create_run": "tools/v3/create_run.py",
    }
    spec["frozen_tools"] = {name: {"path": relative, "sha256": _sha(PROJECT_ROOT / relative)} for name, relative in tools.items()}
    spec["command"] = [
        "/usr/bin/timeout", "--signal=INT", "--kill-after=10s", "300s", PYTHON,
        "tools/v3/run_gse_spatial_center_residual_audit_v1r.py",
        "--spec", str(SPEC_PATH),
        "--run-dir", str(PROJECT_ROOT / "results/gate3_semantics" / RUN_ID),
    ]
    write_json(SPEC_PATH, spec)
    print(SPEC_PATH.relative_to(PROJECT_ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
