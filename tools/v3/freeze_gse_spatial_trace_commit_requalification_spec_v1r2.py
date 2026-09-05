#!/usr/bin/env python3
"""Freeze V1R2 with canonical all-row batch inference tolerance."""

from __future__ import annotations

import hashlib
from pathlib import Path

from _bootstrap import PROJECT_ROOT
from mtare_topo.governance import load_json, write_json


RUN_ID = "gate3_20260828_gse_spatial_trace_commit_requalification_v1r2_seed0"
SPEC = PROJECT_ROOT / "configs/v3/gate3/gse_spatial_trace_commit_requalification_v1r2.json"
CARD = PROJECT_ROOT / "configs/v3/gate3/data_cards/gse_spatial_trace_commit_requalification_v1r2.json"
OLD_SPEC = PROJECT_ROOT / "configs/v3/gate3/gse_spatial_trace_commit_requalification_v1r.json"
OLD_CARD = PROJECT_ROOT / "configs/v3/gate3/data_cards/gse_spatial_trace_commit_requalification_v1r.json"
OLD_RUN = PROJECT_ROOT / "results/gate3_semantics/gate3_20260828_gse_spatial_trace_commit_requalification_v1r_seed0"
PYTHON = "/home/zeng-workstation/.local/share/mtare_topo_comm/envs/phase3_torch290_cu129_zarr2187_v1/bin/python"


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    if SPEC.exists() or CARD.exists():
        raise RuntimeError("spatial trace-commit V1R2 card/spec exists; overwrite is forbidden")
    old_spec = load_json(OLD_SPEC)
    card = load_json(OLD_CARD)
    approval = {
        "status": "APPROVED", "approved_by": "user-standing-authorization",
        "approved_at": "2026-08-28T12:30:00+08:00", "authorized_gates": [3],
        "authorized_operations": ["audit"],
        "scope": "One immutable V1R2 with canonical global-row batch128 deployment inference and a 0.25 mm qualification reproduction bound; zero training and zero C09/C10/M-TARE.",
        "confirmation_reference": "User instructed automatic best-choice execution without routine approval prompts.",
    }
    card.update({
        "card_id": "gse_spatial_trace_commit_requalification_v1r2",
        "title": "Qualified spatial-center trace-commit graph requalification V1R2",
        "status": "APPROVED_FOR_ONE_IMMUTABLE_GSE_SPATIAL_TRACE_COMMIT_REQUALIFICATION_V1R2",
        "approval": approval,
        "purpose": "Run the unchanged graph audit with canonical all-188126-row batch128 inference after V1R showed that different GPU batch packing is deterministic but not bit-exact.",
    })
    card["metrics_and_pre_registered_gates"]["center_reproduction"] = (
        "All 8,839 C07-C08 event rows must reproduce each qualified seed and ensemble within 0.00025 m maximum absolute local/world coordinate error; identities remain exact."
    )
    card["metrics_and_pre_registered_gates"]["numeric_reproduction"] = (
        "Canonical all-row batching is fixed before graph replay; V1R observed seed/ensemble maxima 0.000179/0.000061 m, below the 0.00025 m bound."
    )
    card["failure_evidence_v1r"] = {
        "run": str(OLD_RUN.relative_to(PROJECT_ROOT)),
        "status": "FAIL_SYSTEM_BEFORE_GRAPH_REPLAY_DIFFERENT_GPU_BATCH_PACKING_NOT_BIT_EXACT",
        "seal_sha256": _sha(OLD_RUN / "artifacts/evidence_sha256.txt"),
        "observed_seed_max_abs_m": 0.00017881393432617188,
        "observed_ensemble_max_abs_m": 0.00006103515625,
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
        "runner": "tools/v3/run_gse_spatial_trace_commit_requalification_v1r2.py",
        "runner_implementation": "tools/v3/run_gse_spatial_trace_commit_requalification_v1.py",
        "freezer": "tools/v3/freeze_gse_spatial_trace_commit_requalification_spec_v1r2.py",
    })
    spec = dict(old_spec)
    spec.update({
        "slug": "gse_spatial_trace_commit_requalification_v1r2",
        "question": "With canonical deployment batching, does the qualified 3D event center pass the unchanged trace-commit graph gates?",
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
            "tools/v3/run_gse_spatial_trace_commit_requalification_v1r2.py",
            "--spec", str(SPEC),
            "--run-dir", str(PROJECT_ROOT / "results/gate3_semantics" / RUN_ID),
        ],
    })
    spec["hyperparameters"] = dict(spec["hyperparameters"])
    spec["hyperparameters"].update({
        "canonical_inference_batching": "global_rows_0_to_188125_batch128",
        "qualification_reproduction_tolerance_m": 0.00025,
    })
    spec["expected_evidence"] = list(spec["expected_evidence"]) + [
        "V1R failure provenance and measured per-seed/ensemble batch-layout reproduction errors."
    ]
    write_json(SPEC, spec)
    print(SPEC.relative_to(PROJECT_ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
