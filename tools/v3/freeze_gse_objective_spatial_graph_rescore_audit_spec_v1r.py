#!/usr/bin/env python3
"""Freeze governance-field-only V1R of the objective spatial rescore audit."""

from __future__ import annotations

import hashlib

from _bootstrap import PROJECT_ROOT
from mtare_topo.governance import load_json, write_json


RUN_ID = "gate3_20260828_gse_objective_spatial_graph_rescore_audit_v1r_seed0"
OLD_SPEC = PROJECT_ROOT / "configs/v3/gate3/gse_objective_spatial_graph_rescore_audit_v1.json"
OLD_CARD = PROJECT_ROOT / "configs/v3/gate3/data_cards/gse_objective_spatial_graph_rescore_audit_v1.json"
SPEC = PROJECT_ROOT / "configs/v3/gate3/gse_objective_spatial_graph_rescore_audit_v1r.json"
CARD = PROJECT_ROOT / "configs/v3/gate3/data_cards/gse_objective_spatial_graph_rescore_audit_v1r.json"
PYTHON = "/home/zeng-workstation/.local/share/mtare_topo_comm/envs/phase3_torch290_cu129_zarr2187_v1/bin/python"


def _sha(path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    if SPEC.exists() or CARD.exists():
        raise RuntimeError("objective spatial graph V1R card/spec exists; overwrite is forbidden")
    old_spec = load_json(OLD_SPEC)
    card = load_json(OLD_CARD)
    approval = dict(card["approval"])
    approval["scope"] = "One immutable V1R adding only preflight-required top-level method, baseline and estimated_cost fields; scientific contract unchanged."
    card.update({
        "card_id": "gse_objective_spatial_graph_rescore_audit_v1r",
        "title": "Objective 3D one-to-one graph rescore audit V1R",
        "status": "APPROVED_FOR_ONE_IMMUTABLE_GSE_OBJECTIVE_SPATIAL_GRAPH_RESCORE_AUDIT_V1R",
        "approval": approval,
        "predecessor_preflight_failure": {
            "spec": str(OLD_SPEC.relative_to(PROJECT_ROOT)),
            "run_created": False, "data_read": False, "scientific_conclusion": None,
            "errors": ["method must be a non-empty string", "baseline must be a non-empty string", "estimated_cost must be an object"],
        },
    })
    write_json(CARD, card)
    tools = {
        "data_card": str(CARD.relative_to(PROJECT_ROOT)),
        "scorer": "src/mtare_topo/evaluation/gse_objective_spatial_graph_score.py",
        "relation_helper": "src/mtare_topo/evaluation/gse_trace_commit_failure_funnel.py",
        "executor": "tools/v3/execute_gse_objective_spatial_graph_rescore_audit_v1.py",
        "runner": "tools/v3/run_gse_objective_spatial_graph_rescore_audit_v1r.py",
        "runner_implementation": "tools/v3/run_gse_objective_spatial_graph_rescore_audit_v1.py",
        "freezer": "tools/v3/freeze_gse_objective_spatial_graph_rescore_audit_spec_v1r.py",
        "unit_test": "tests/v3/unit/test_gse_objective_spatial_graph_score.py",
    }
    inputs = dict(old_spec["frozen_inputs"])
    inputs[str(OLD_SPEC.relative_to(PROJECT_ROOT))] = _sha(OLD_SPEC)
    inputs[str(OLD_CARD.relative_to(PROJECT_ROOT))] = _sha(OLD_CARD)
    spec = dict(old_spec)
    spec.update({
        "slug": "gse_objective_spatial_graph_rescore_audit_v1r",
        "question": "Under the unchanged scientific contract, does objective 3D one-to-one node matching change the sealed graph conclusion?",
        "method": card["methods"]["main"],
        "baseline": card["methods"]["baseline"],
        "estimated_cost": card["estimated_cost"],
        "config_path": str(CARD.relative_to(PROJECT_ROOT)),
        "data_card": str(CARD.relative_to(PROJECT_ROOT)),
        "user_authorization": approval,
        "frozen_inputs": inputs,
        "frozen_tools": {
            name: {"path": relative, "sha256": _sha(PROJECT_ROOT / relative)}
            for name, relative in tools.items()
        },
        "command": [
            "/usr/bin/timeout", "--signal=INT", "--kill-after=30s", "900s", PYTHON,
            "tools/v3/run_gse_objective_spatial_graph_rescore_audit_v1r.py",
            "--spec", str(SPEC),
            "--run-dir", str(PROJECT_ROOT / "results/gate3_semantics" / RUN_ID),
        ],
    })
    write_json(SPEC, spec)
    print(SPEC.relative_to(PROJECT_ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
