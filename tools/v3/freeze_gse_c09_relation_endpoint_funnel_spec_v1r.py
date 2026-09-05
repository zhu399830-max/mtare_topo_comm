#!/usr/bin/env python3
"""Freeze V1R after correcting canonical Teacher row alignment."""

from __future__ import annotations

import hashlib

from _bootstrap import PROJECT_ROOT
from mtare_topo.governance import load_json, write_json


RUN_ID = "gate3_20260828_gse_c09_relation_endpoint_funnel_v1r_seed0"
SPEC = PROJECT_ROOT / "configs/v3/gate3/gse_c09_relation_endpoint_funnel_v1r.json"
CARD = PROJECT_ROOT / "configs/v3/gate3/data_cards/gse_c09_relation_endpoint_funnel_v1r.json"
OLD_SPEC = PROJECT_ROOT / "configs/v3/gate3/gse_c09_relation_endpoint_funnel_v1.json"
OLD_CARD = PROJECT_ROOT / "configs/v3/gate3/data_cards/gse_c09_relation_endpoint_funnel_v1.json"
OLD_RUN = PROJECT_ROOT / "results/gate3_semantics/gate3_20260828_gse_c09_relation_endpoint_funnel_v1_seed0"
PYTHON = "/home/zeng-workstation/.local/share/mtare_topo_comm/envs/phase3_torch290_cu129_zarr2187_v1/bin/python"


def _sha(path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    if SPEC.exists() or CARD.exists():
        raise RuntimeError("C09 endpoint-funnel V1R card/spec exists; overwrite is forbidden")
    old_spec = load_json(OLD_SPEC)
    card = load_json(OLD_CARD)
    approval = {
        "status": "APPROVED", "approved_by": "user-standing-authorization",
        "approved_at": "2026-08-28T00:00:00+08:00", "authorized_gates": [3],
        "authorized_operations": ["audit"],
        "scope": "One immutable V1R changing only Teacher alignment from file order to one-to-one stable global_sequence_index mapping; zero science changes, inference, training, C10 or M-TARE.",
        "confirmation_reference": "User instructed automatic best-choice execution without routine approval prompts.",
    }
    card.update({
        "card_id": "gse_c09_relation_endpoint_funnel_v1r",
        "title": "C09 frozen graph relation-endpoint causal failure funnel V1R",
        "status": "APPROVED_FOR_ONE_IMMUTABLE_GSE_C09_RELATION_ENDPOINT_FUNNEL_V1R",
        "approval": approval,
        "failure_evidence": {
            "run": str(OLD_RUN.relative_to(PROJECT_ROOT)),
            "status": "FAIL_SYSTEM_ACTION_TEACHER_FILE_ORDER_MISMATCH",
            "seal_sha256": _sha(OLD_RUN / "artifacts/evidence_sha256.txt"),
            "observed": "24462 unique IDs in each source; identical sets but different row order",
            "scientific_conclusion": "NONE",
        },
    })
    card["split"]["leakage_audit"] = (
        "Teacher is joined one-to-one to the frozen canonical action order by stable global_sequence_index; the identical 24462-ID sets and uniqueness are mandatory. Teacher cannot alter the graph or any parameter."
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
        "executor": "tools/v3/execute_gse_c09_relation_endpoint_funnel_v1.py",
        "runner": "tools/v3/run_gse_c09_relation_endpoint_funnel_v1r.py",
        "runner_implementation": "tools/v3/run_gse_c09_relation_endpoint_funnel_v1.py",
        "freezer": "tools/v3/freeze_gse_c09_relation_endpoint_funnel_spec_v1r.py",
    })
    spec = dict(old_spec)
    spec.update({
        "slug": "gse_c09_relation_endpoint_funnel_v1r",
        "question": "After stable-ID alignment, at which causal stage are the seven missing frozen C09 relations lost?",
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
            "/usr/bin/timeout", "--signal=INT", "--kill-after=30s", "360s", PYTHON,
            "tools/v3/run_gse_c09_relation_endpoint_funnel_v1r.py", "--spec", str(SPEC),
            "--run-dir", str(PROJECT_ROOT / "results/gate3_semantics" / RUN_ID),
        ],
    })
    spec["expected_evidence"] = list(spec["expected_evidence"]) + [
        "V1 system-failure provenance and exact stable-global-ID alignment proof."
    ]
    write_json(SPEC, spec)
    print(CARD.relative_to(PROJECT_ROOT)); print(SPEC.relative_to(PROJECT_ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
