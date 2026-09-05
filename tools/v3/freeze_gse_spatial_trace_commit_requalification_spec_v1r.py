#!/usr/bin/env python3
"""Freeze V1R after the float32 linear-identity checker correction."""

from __future__ import annotations

import hashlib
from pathlib import Path

from _bootstrap import PROJECT_ROOT
from mtare_topo.governance import load_json, write_json


RUN_ID = "gate3_20260828_gse_spatial_trace_commit_requalification_v1r_seed0"
SPEC = PROJECT_ROOT / "configs/v3/gate3/gse_spatial_trace_commit_requalification_v1r.json"
CARD = PROJECT_ROOT / "configs/v3/gate3/data_cards/gse_spatial_trace_commit_requalification_v1r.json"
OLD_SPEC = PROJECT_ROOT / "configs/v3/gate3/gse_spatial_trace_commit_requalification_v1.json"
OLD_CARD = PROJECT_ROOT / "configs/v3/gate3/data_cards/gse_spatial_trace_commit_requalification_v1.json"
OLD_RUN = PROJECT_ROOT / "results/gate3_semantics/gate3_20260828_gse_spatial_trace_commit_requalification_v1_seed0"
PYTHON = "/home/zeng-workstation/.local/share/mtare_topo_comm/envs/phase3_torch290_cu129_zarr2187_v1/bin/python"


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    if SPEC.exists() or CARD.exists():
        raise RuntimeError("spatial trace-commit V1R card/spec exists; overwrite is forbidden")
    old_spec = load_json(OLD_SPEC)
    card = load_json(OLD_CARD)
    approval = {
        "status": "APPROVED", "approved_by": "user-standing-authorization",
        "approved_at": "2026-08-28T12:20:00+08:00", "authorized_gates": [3],
        "authorized_operations": ["audit"],
        "scope": "One immutable V1R replay with only the float32-aware sub-millimetre linear-identity checker correction; zero training and zero C09/C10/M-TARE.",
        "confirmation_reference": "User instructed automatic best-choice execution without routine approval prompts.",
    }
    card.update({
        "card_id": "gse_spatial_trace_commit_requalification_v1r",
        "title": "Qualified spatial-center trace-commit graph requalification V1R",
        "status": "APPROVED_FOR_ONE_IMMUTABLE_GSE_SPATIAL_TRACE_COMMIT_REQUALIFICATION_V1R",
        "approval": approval,
        "purpose": "Repeat V1 unchanged after replacing its invalid 2e-6 m float32 linear-identity assertion with a documented 1e-4 m sub-millimetre bound; V1 stopped before graph replay.",
    })
    card["methods"]["main"] += " The canonical ensemble remains the arithmetic mean of the three archived seed centers, exactly matching qualification."
    card["metrics_and_pre_registered_gates"]["numeric_reproduction"] = (
        "Mean-of-projected-seed centers versus projection-of-mean-vector must differ by <=1e-4 m; V1 observed 4.0691e-5 m at world coordinates near 513 m."
    )
    card["failure_evidence"] = {
        "run": str(OLD_RUN.relative_to(PROJECT_ROOT)),
        "status": "FAIL_SYSTEM_BEFORE_GRAPH_REPLAY_FLOAT32_LINEAR_IDENTITY_CHECK_4P0691E_5M_VS_2E_6M",
        "seal_sha256": _sha(OLD_RUN / "artifacts/evidence_sha256.txt"),
        "scientific_conclusion": "NONE_GRAPH_NOT_EXECUTED",
    }
    write_json(CARD, card)

    input_paths = list(old_spec["frozen_inputs"])
    input_paths.extend([
        str((OLD_RUN / "RUN_STATE.json").relative_to(PROJECT_ROOT)),
        str((OLD_RUN / "metrics/summary.json").relative_to(PROJECT_ROOT)),
        str((OLD_RUN / "artifacts/evidence_sha256.txt").relative_to(PROJECT_ROOT)),
    ])
    tool_paths = {name: value["path"] for name, value in old_spec["frozen_tools"].items()}
    tool_paths.update({
        "data_card": str(CARD.relative_to(PROJECT_ROOT)),
        "center_projection": "src/mtare_topo/evaluation/gse_spatial_center_projection.py",
        "center_combiner": "tools/v3/combine_gse_spatial_longitudinal_center_v1.py",
        "runner": "tools/v3/run_gse_spatial_trace_commit_requalification_v1r.py",
        "runner_implementation": "tools/v3/run_gse_spatial_trace_commit_requalification_v1.py",
        "freezer": "tools/v3/freeze_gse_spatial_trace_commit_requalification_spec_v1r.py",
    })
    spec = dict(old_spec)
    spec.update({
        "slug": "gse_spatial_trace_commit_requalification_v1r",
        "question": "Under the unchanged graph contract, does the qualified 3D event center pass after a float32-resolution-correct V1R reproduction check?",
        "method": card["methods"]["main"],
        "baseline": card["methods"]["baselines"],
        "fallback": card["methods"]["fallback"],
        "data_card": str(CARD.relative_to(PROJECT_ROOT)),
        "config_path": str(CARD.relative_to(PROJECT_ROOT)),
        "user_authorization": approval,
        "acceptance_criteria": list(card["metrics_and_pre_registered_gates"].values()),
        "frozen_inputs": {relative: _sha(PROJECT_ROOT / relative) for relative in input_paths},
        "frozen_tools": {
            name: {"path": relative, "sha256": _sha(PROJECT_ROOT / relative)}
            for name, relative in tool_paths.items()
        },
        "command": [
            "/usr/bin/timeout", "--signal=INT", "--kill-after=60s", "7200s", PYTHON,
            "tools/v3/run_gse_spatial_trace_commit_requalification_v1r.py",
            "--spec", str(SPEC),
            "--run-dir", str(PROJECT_ROOT / "results/gate3_semantics" / RUN_ID),
        ],
    })
    spec["expected_evidence"] = list(spec["expected_evidence"]) + [
        "V1 failure provenance and measured float32 linear-identity error bounded below 0.1 mm."
    ]
    write_json(SPEC, spec)
    print(SPEC.relative_to(PROJECT_ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
