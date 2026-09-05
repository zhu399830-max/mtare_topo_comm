#!/usr/bin/env python3
"""Freeze V1R after correcting non-decision identity attribution."""

from __future__ import annotations

import hashlib

from _bootstrap import PROJECT_ROOT
from mtare_topo.governance import load_json, write_json


RUN_ID = "gate3_20260828_gse_trace_commit_failure_funnel_audit_v1r_seed0"
SPEC = PROJECT_ROOT / "configs/v3/gate3/gse_trace_commit_failure_funnel_audit_v1r.json"
CARD = PROJECT_ROOT / "configs/v3/gate3/data_cards/gse_trace_commit_failure_funnel_audit_v1r.json"
OLD_SPEC = PROJECT_ROOT / "configs/v3/gate3/gse_trace_commit_failure_funnel_audit_v1.json"
OLD_CARD = PROJECT_ROOT / "configs/v3/gate3/data_cards/gse_trace_commit_failure_funnel_audit_v1.json"
OLD_RUN = PROJECT_ROOT / "results/gate3_semantics/gate3_20260828_gse_trace_commit_failure_funnel_audit_v1_seed0"
PYTHON = "/home/zeng-workstation/.local/share/mtare_topo_comm/envs/phase3_torch290_cu129_zarr2187_v1/bin/python"


def _sha(path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    if SPEC.exists() or CARD.exists():
        raise RuntimeError("failure-funnel V1R card/spec exists; overwrite is forbidden")
    old_spec = load_json(OLD_SPEC)
    card = load_json(OLD_CARD)
    approval = {
        "status": "APPROVED", "approved_by": "user-standing-authorization",
        "approved_at": "2026-08-28T13:10:00+08:00", "authorized_gates": [3],
        "authorized_operations": ["audit"],
        "scope": "One immutable V1R with only corrected scorer-equivalent non-decision identity attribution; zero model/graph changes and zero C09/C10/M-TARE.",
        "confirmation_reference": "User instructed automatic best-choice execution without routine approval prompts.",
    }
    card.update({
        "card_id": "gse_trace_commit_failure_funnel_audit_v1r",
        "title": "GSE trace-commit node/edge causal failure funnel V1R",
        "status": "APPROVED_FOR_ONE_IMMUTABLE_GSE_TRACE_COMMIT_FAILURE_FUNNEL_AUDIT_V1R",
        "approval": approval,
        "purpose": "Repeat the unchanged read-only audit after distinguishing single non-decision identities from correct junction/terminal identities in scorer reproduction.",
        "failure_evidence": {
            "run": str(OLD_RUN.relative_to(PROJECT_ROOT)),
            "status": "FAIL_SYSTEM_SCORER_REPRODUCTION_NON_DECISION_IDENTITY_MISCLASSIFIED",
            "seal_sha256": _sha(OLD_RUN / "artifacts/evidence_sha256.txt"),
            "observed_partition": "198 correct unique + 6 duplicate excess + 2 empty identity + 1 non-decision identity = 207 committed",
            "scientific_conclusion": "NONE_ATTRIBUTION_NOT_WRITTEN",
        },
    })
    card["acceptance"]["scorer_reproduction"] = (
        "Exactly reproduce 198 correct unique nodes, 6 duplicate committed excess, 2 empty-identity hypotheses, 1 non-decision identity hypothesis, 0 mixed loops and 2 correct edges."
    )
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
        "executor": "tools/v3/execute_gse_trace_commit_failure_funnel_audit_v1.py",
        "runner": "tools/v3/run_gse_trace_commit_failure_funnel_audit_v1r.py",
        "runner_implementation": "tools/v3/run_gse_trace_commit_failure_funnel_audit_v1.py",
        "freezer": "tools/v3/freeze_gse_trace_commit_failure_funnel_audit_spec_v1r.py",
    })
    spec = dict(old_spec)
    spec.update({
        "slug": "gse_trace_commit_failure_funnel_audit_v1r",
        "question": "After scorer-equivalent identity partitioning, which causal stage limits node safety and edge recall?",
        "data_card": str(CARD.relative_to(PROJECT_ROOT)),
        "config_path": str(CARD.relative_to(PROJECT_ROOT)),
        "user_authorization": approval,
        "acceptance_criteria": list(card["acceptance"].values()),
        "frozen_inputs": {relative: _sha(PROJECT_ROOT / relative) for relative in input_paths},
        "frozen_tools": {
            name: {"path": relative, "sha256": _sha(PROJECT_ROOT / relative)}
            for name, relative in tool_paths.items()
        },
        "command": [
            "/usr/bin/timeout", "--signal=INT", "--kill-after=30s", "900s", PYTHON,
            "tools/v3/run_gse_trace_commit_failure_funnel_audit_v1r.py",
            "--spec", str(SPEC),
            "--run-dir", str(PROJECT_ROOT / "results/gate3_semantics" / RUN_ID),
        ],
    })
    spec["expected_evidence"] = list(spec["expected_evidence"]) + [
        "V1 failure provenance and explicit 198+6+2+1 scorer partition."
    ]
    write_json(SPEC, spec)
    print(SPEC.relative_to(PROJECT_ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
