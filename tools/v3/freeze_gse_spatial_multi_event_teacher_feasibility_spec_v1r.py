#!/usr/bin/env python3
"""Freeze the one corrective spec after the v1 JSON-root system failure."""

from __future__ import annotations

import hashlib
from pathlib import Path

from _bootstrap import PROJECT_ROOT
from mtare_topo.governance import load_json, write_json


RUN_ID = "gate3_20260828_gse_spatial_multi_event_teacher_feasibility_v1r_seed0"
CARD = PROJECT_ROOT / "configs/v3/gate3/data_cards/gse_spatial_multi_event_teacher_feasibility_v1r.json"
SPEC = PROJECT_ROOT / "configs/v3/gate3/gse_spatial_multi_event_teacher_feasibility_v1r.json"
ORIGINAL_CARD = PROJECT_ROOT / "configs/v3/gate3/data_cards/gse_spatial_multi_event_teacher_feasibility_v1.json"
ORIGINAL_SPEC = PROJECT_ROOT / "configs/v3/gate3/gse_spatial_multi_event_teacher_feasibility_v1.json"
ORIGINAL_RUN = "results/gate3_semantics/gate3_20260828_gse_spatial_multi_event_teacher_feasibility_v1_seed0"
SIDECAR = "/tmp/mtare_gate4_meshing_sidecar_v1/bin/python"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    if CARD.exists() or SPEC.exists():
        raise RuntimeError("spatial multi-event corrective card/spec already exists")
    card = load_json(ORIGINAL_CARD)
    approval = {
        "status": "APPROVED",
        "approved_by": "user-standing-authorization",
        "approved_at": "2026-08-28T00:00:00+08:00",
        "authorized_gates": [3],
        "authorized_operations": ["audit"],
        "scope": "One corrective immutable execution after v1 stopped before all world/raycast processing because a four-row JSON array was passed to an object-only loader. Scientific data, method and thresholds are unchanged.",
        "confirmation_reference": "User instructed automatic best-choice execution without routine approval prompts.",
    }
    card.update(
        {
            "card_id": "gse_spatial_multi_event_teacher_feasibility_v1r",
            "title": "GSE spatial multi-event Teacher native-mesh LOS feasibility corrective",
            "status": "APPROVED_FOR_ONE_IMMUTABLE_GSE_SPATIAL_MULTI_EVENT_TEACHER_FEASIBILITY_V1",
            "approval": approval,
            "corrective_provenance": {
                "supersedes_execution_only": ORIGINAL_RUN,
                "original_failure": "ValueError before world processing: governance.load_json requires an object root while low_selection_rows.json is a sealed four-object JSON array.",
                "single_change": "Read the already frozen rare evidence with json.loads and validate list[object].",
                "scientific_contract_changed": False,
            },
        }
    )
    write_json(CARD, card)

    old_spec = load_json(ORIGINAL_SPEC)
    inputs = dict(old_spec["frozen_inputs"])
    for relative in (
        str(ORIGINAL_CARD.relative_to(PROJECT_ROOT)),
        str(ORIGINAL_SPEC.relative_to(PROJECT_ROOT)),
        f"{ORIGINAL_RUN}/RUN_STATE.json",
        f"{ORIGINAL_RUN}/metrics/summary.json",
        f"{ORIGINAL_RUN}/artifacts/evidence_sha256.txt",
    ):
        inputs[relative] = _sha256(PROJECT_ROOT / relative)
    tools = {
        "data_card": str(CARD.relative_to(PROJECT_ROOT)),
        "teacher_interface": "src/mtare_topo/teacher/gse_spatial_multi_event_teacher.py",
        "executor": "tools/v3/execute_gse_spatial_multi_event_teacher_feasibility_v1.py",
        "runner": "tools/v3/run_gse_spatial_multi_event_teacher_feasibility_v1.py",
        "freezer": "tools/v3/freeze_gse_spatial_multi_event_teacher_feasibility_spec_v1r.py",
        "teacher_tests": "tests/v3/unit/test_gse_spatial_multi_event_teacher.py",
        "executor_regression_tests": "tests/v3/unit/test_gse_spatial_multi_event_teacher_executor.py",
        "governance": "src/mtare_topo/governance.py",
        "preflight": "tools/v3/preflight.py",
        "create_run": "tools/v3/create_run.py",
    }
    spec = dict(old_spec)
    spec.update(
        {
            "slug": "gse_spatial_multi_event_teacher_feasibility_v1r",
            "data_card": str(CARD.relative_to(PROJECT_ROOT)),
            "config_path": str(CARD.relative_to(PROJECT_ROOT)),
            "user_authorization": approval,
            "corrective_provenance": card["corrective_provenance"],
            "frozen_inputs": dict(sorted(inputs.items())),
            "frozen_tools": {
                name: {"path": path, "sha256": _sha256(PROJECT_ROOT / path)}
                for name, path in tools.items()
            },
            "command": [
                "/usr/bin/timeout",
                "3600s",
                SIDECAR,
                "tools/v3/run_gse_spatial_multi_event_teacher_feasibility_v1.py",
                "--spec",
                str(SPEC),
                "--run-dir",
                str(PROJECT_ROOT / f"results/gate3_semantics/{RUN_ID}"),
            ],
        }
    )
    write_json(SPEC, spec)
    print(CARD.relative_to(PROJECT_ROOT))
    print(SPEC.relative_to(PROJECT_ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
