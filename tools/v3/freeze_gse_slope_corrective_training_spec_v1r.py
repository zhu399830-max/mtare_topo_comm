#!/usr/bin/env python3
"""Freeze V1R after the V1 traversal-local-index checker system failure."""

from __future__ import annotations

import hashlib
from pathlib import Path

from _bootstrap import PROJECT_ROOT
from mtare_topo.governance import load_json, validate_data_card, write_json


OLD_CARD = PROJECT_ROOT / "configs/v3/gate2/data_cards/gse_slope_corrective_three_seed_training_v1.json"
OLD_SPEC = PROJECT_ROOT / "configs/v3/gate2/gse_slope_corrective_three_seed_training_v1.json"
CARD = PROJECT_ROOT / "configs/v3/gate2/data_cards/gse_slope_corrective_three_seed_training_v1r.json"
SPEC = PROJECT_ROOT / "configs/v3/gate2/gse_slope_corrective_three_seed_training_v1r.json"
PREDECESSOR = PROJECT_ROOT / "results/gate2_representation/gate2_20260824_gse_slope_corrective_three_seed_training_v1_seed0"
RUN_ID = "gate2_20260824_gse_slope_corrective_three_seed_training_v1r_seed0"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _record(relative: str) -> dict[str, str]:
    return {"path": relative, "sha256": _sha256(PROJECT_ROOT / relative)}


def freeze() -> dict:
    if CARD.exists() or SPEC.exists():
        raise RuntimeError("slope corrective V1R card/spec already exists; refusing overwrite")
    state = load_json(PREDECESSOR / "RUN_STATE.json")
    summary = load_json(PREDECESSOR / "metrics/summary.json")
    seal = PREDECESSOR / "artifacts/evidence_sha256.txt"
    if (
        state.get("state") != "FAILED"
        or state.get("overall_status") != "FAIL_GSE_SLOPE_CORRECTIVE_THREE_SEED_TRAINING_V1"
        or summary.get("completed_seeds") != []
        or summary.get("c09_worlds_read") != 0
        or summary.get("strict_test_worlds_read") != 0
        or summary.get("mtare_worlds_read") != 0
        or not seal.is_file()
        or len(seal.read_text(encoding="utf-8").splitlines()) != 11
        or _sha256(seal) != "eb75870c060b9c0f744b1a255bd136a3392b11f8c4a3cdd372fecaaf7b286c62"
    ):
        raise RuntimeError("V1 system-failure predecessor evidence drift")
    card = load_json(OLD_CARD)
    card["card_id"] = "gse_slope_corrective_three_seed_training_v1r"
    card["status"] = "APPROVED_FOR_ONE_IMMUTABLE_SLOPE_CORRECTIVE_TRAINING_V1R"
    card["purpose"] = (
        "Repeat the unchanged train-only slope corrective contract after fixing only the V1 checker assumption that traversal-local frame indices must be world-global arange values."
    )
    card["approval"]["approved_at"] = "2026-08-24T16:25:00+08:00"
    card["approval"]["scope"] = (
        "One immutable V1R recovery run with exactly the V1 data, model, seeds, optimization and gates; the only change accepts documented local_frame_index resets between directed traversals while requiring +1 within every five-frame sequence."
    )
    card["approval"]["confirmation_reference"] = (
        "The user granted continuous execution for the fixed GSE paper scope. V1 failed before cache completion with zero seeds/optimizer/C09 reads; this V1R is the minimal system-only recovery."
    )
    card["split"]["historical_pollution_audit"] += (
        " V1 stopped before training because its checker incorrectly required world-global local_frame_index. V1R changes no data or learning rule and additionally verifies all 188126 sequences advance local frame indices by exactly one."
    )
    card["evidence"]["failure_policy"] += (
        " The V1 FAIL and 11-file seal remain immutable predecessor evidence."
    )
    report = validate_data_card(card)
    if not report.passed:
        raise RuntimeError(f"V1R Data Card invalid: {report.errors}")
    write_json(CARD, card)
    old_spec = load_json(OLD_SPEC)
    spec = dict(old_spec)
    spec["slug"] = "gse_slope_corrective_three_seed_training_v1r"
    spec["question"] = (
        "Under the unchanged V1 scientific contract, can the slope residual pass after correcting only the false world-global local-frame-index assertion while preserving strict within-sequence causality?"
    )
    spec["method"] = old_spec["method"] + (
        " V1R recognizes that local_frame_index resets at each directed traversal and now requires exact +1 steps inside every five-frame reference; no scientific input or hyperparameter changes."
    )
    spec["data_card"] = str(CARD.relative_to(PROJECT_ROOT))
    spec["config_path"] = str(CARD.relative_to(PROJECT_ROOT))
    spec["user_authorization"] = card["approval"]
    command = list(old_spec["command"])
    command[command.index(str(OLD_SPEC))] = str(SPEC)
    old_run = str(PROJECT_ROOT / "results/gate2_representation/gate2_20260824_gse_slope_corrective_three_seed_training_v1_seed0")
    command[command.index(old_run)] = str(PROJECT_ROOT / "results/gate2_representation" / RUN_ID)
    spec["command"] = command
    spec["acceptance_criteria"] = list(old_spec["acceptance_criteria"]) + [
        "All 188126 five-frame sequences have consecutive Zarr rows, local traversal indices advancing exactly +1, and global references matching the frozen global frame index; resets between traversals are valid and cannot be crossed by a sequence."
    ]
    spec["expected_evidence"] = list(old_spec["expected_evidence"]) + [
        "Immutable V1 system-failure predecessor identity, zero-update/read counts and exact 11-file seal."
    ]
    tool_paths = {name: item["path"] for name, item in old_spec["frozen_tools"].items()}
    tool_paths["data_card"] = str(CARD.relative_to(PROJECT_ROOT))
    spec["frozen_tools"] = {name: _record(relative) for name, relative in tool_paths.items()}
    spec["frozen_inputs"] = dict(old_spec["frozen_inputs"])
    for path in (
        PREDECESSOR / "RUN_STATE.json",
        PREDECESSOR / "metrics/summary.json",
        PREDECESSOR / "artifacts/evidence_sha256.txt",
    ):
        spec["frozen_inputs"][str(path.relative_to(PROJECT_ROOT))] = _sha256(path)
    spec["predecessor"] = {
        "run": str(PREDECESSOR.relative_to(PROJECT_ROOT)),
        "status": "FAIL_GSE_SLOPE_CORRECTIVE_THREE_SEED_TRAINING_V1",
        "failure_class": "SYSTEM_CHECKER_FALSE_WORLD_GLOBAL_LOCAL_FRAME_INDEX_ASSUMPTION",
        "completed_seeds": 0,
        "optimizer_steps": 0,
        "c09_worlds_read": 0,
        "seal_sha256": _sha256(seal),
    }
    write_json(SPEC, spec)
    return {
        "data_card": str(CARD.relative_to(PROJECT_ROOT)),
        "data_card_sha256": _sha256(CARD),
        "spec": str(SPEC.relative_to(PROJECT_ROOT)),
        "spec_sha256": _sha256(SPEC),
    }


if __name__ == "__main__":
    print(freeze())
