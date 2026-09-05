#!/usr/bin/env python3
"""Freeze V1R with explicit total/observed/zero-observation traversal counts."""

from __future__ import annotations

import hashlib
from pathlib import Path

from _bootstrap import PROJECT_ROOT
from mtare_topo.governance import load_json, write_json


RUN_ID = "gate3_20260827_gse_causal_episode_reference_proof_v1r_seed0"
CARD_PATH = PROJECT_ROOT / "configs/v3/gate3/data_cards/gse_causal_episode_reference_proof_v1r.json"
SPEC_PATH = PROJECT_ROOT / "configs/v3/gate3/gse_causal_episode_reference_proof_v1r.json"
SOURCE_CARD = PROJECT_ROOT / "configs/v3/gate3/data_cards/gse_causal_episode_reference_proof_v1.json"
DATASET = "results/gate2_representation/gate2_20260824_gse_deduplicated_dataset_export_v1_seed0"
TEACHER = "results/gate2_representation/gate2_20260827_gse_corrected_causal_teacher_manifest_v1r_seed0"
AUDIT = "results/gate3_semantics/gate3_20260827_gse_causal_event_supervision_audit_v1_seed0"
CHANGE_PROOF = "results/gate3_semantics/gate3_20260826_gse_causal_change_point_proof_v1_seed0"
FAILED_V1 = "results/gate3_semantics/gate3_20260827_gse_causal_episode_reference_proof_v1_seed0"
PYTHON = "/home/zeng-workstation/.local/share/mtare_topo_comm/envs/phase3_torch290_cu129_zarr2187_v1/bin/python"


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    old = load_json(SOURCE_CARD)
    failed = load_json(PROJECT_ROOT / FAILED_V1 / "metrics/summary.json")
    inner = failed.get("reference_proof", {})
    if (
        failed.get("overall_status") != "FAIL_GSE_CAUSAL_EPISODE_REFERENCE_PROOF_V1"
        or failed.get("error") != "RuntimeError: causal episode reference proof violated its contract"
        or inner.get("overall_status") != "PASS_GSE_CAUSAL_EPISODE_REFERENCE_PROOF_V1"
        or inner.get("directed_traversals") != 16076
    ):
        raise RuntimeError("V1R requires the exact sealed traversal-count system failure")
    card = {
        **old,
        "card_id": "gse_causal_episode_reference_proof_v1r",
        "status": "APPROVED_FOR_ONE_IMMUTABLE_GSE_CAUSAL_EPISODE_REFERENCE_PROOF_V1R",
        "purpose": "Repeat the unchanged twelve-scan reference proof while reporting the complete traversal inventory as 16078 total = 16076 with observations + 2 objective zero-observation short traversals.",
        "approval": {
            "status": "APPROVED", "approved_by": "user", "approved_at": "2026-08-27T23:59:58+08:00",
            "authorized_gates": [3], "authorized_operations": ["audit"],
            "scope": "One immutable V1R read-only proof; only traversal count semantics change, all references/data/method/split remain exact.",
            "confirmation_reference": "Standing autonomous authorization; V1 system defect reported immediately and recommended count-semantic correction applied.",
        },
        "method": {
            **old["method"],
            "main": old["method"]["main"] + " Report total, observed and zero-observation traversal populations separately using the sealed causal change-point inventory.",
        },
        "acceptance": {
            "audit_completion": "Exact 16078 inventory traversals decomposed into 16076 observed Teacher traversals and 2 zero-observation short traversals, plus the unchanged 80/252430/188126/5306 reference contract.",
            "decision_rule": old["acceptance"]["decision_rule"],
        },
    }
    write_json(CARD_PATH, card)
    inputs = []
    for run in (DATASET, TEACHER, AUDIT, CHANGE_PROOF, FAILED_V1):
        inputs.extend([f"{run}/RUN_STATE.json", f"{run}/metrics/summary.json", f"{run}/artifacts/evidence_sha256.txt"])
    inputs.extend([f"{TEACHER}/artifacts/teacher_observations.jsonl", f"{DATASET}/artifacts/shard_manifest.json", f"{AUDIT}/metrics/supervision_audit.json", f"{CHANGE_PROOF}/metrics/summary.json"])
    tools = {
        "audit_card": str(CARD_PATH.relative_to(PROJECT_ROOT)),
        "reference_contract": "src/mtare_topo/representation/gse_causal_episode_detector.py",
        "supervision_contract": "src/mtare_topo/evaluation/gse_causal_event_supervision.py",
        "executor": "tools/v3/execute_gse_causal_episode_reference_proof_v1.py",
        "runner": "tools/v3/run_gse_causal_episode_reference_proof_v1r.py",
        "shared_runner": "tools/v3/run_gse_causal_episode_reference_proof_v1.py",
        "freezer": "tools/v3/freeze_gse_causal_episode_reference_proof_spec_v1r.py",
        "evidence_integrity": "src/mtare_topo/evaluation/gse_evidence_integrity.py",
        "governance": "src/mtare_topo/governance.py", "preflight": "tools/v3/preflight.py", "create_run": "tools/v3/create_run.py",
    }
    spec = {
        "schema_version": "v3_run_spec_v1", "gate": 3, "execution_phase": 3, "date": "20260827",
        "slug": "gse_causal_episode_reference_proof_v1r", "seed": 0, "operation": "audit",
        "question": "Do all observed C01-C08 Teacher rows have exact 12-scan past-only references when the two legitimate zero-observation traversals are represented separately?",
        "method": card["method"]["main"], "baseline": card["method"]["baseline"], "fallback": card["method"]["fallback"],
        "audit_card": str(CARD_PATH.relative_to(PROJECT_ROOT)), "config_path": str(CARD_PATH.relative_to(PROJECT_ROOT)),
        "user_authorization": card["approval"],
        "acceptance_criteria": [
            "Exact 16078 total traversals = 16076 observed + 2 zero-observation, with unchanged 80 worlds, 252430 unique frames, 188126 observations and 5306 episodes.",
            "Every valid reference resolves within its parent shard and consecutive traversal-local indices; array digests must equal the sealed V1 inner results.",
            "Zero payload export, rays, inference, training, C09/C10/M-TARE access; source unchanged and complete seal.",
        ],
        "expected_counts": {"inventory_directed_traversals": 16078, "observed_directed_traversals": 16076, "zero_observation_traversals": 2, "worlds": 80, "unique_frames": 252430, "causal_observations": 188126, "structural_episodes": 5306, "new_rays": 0, "sensor_payload_bytes_written": 0, "optimizer_steps": 0, "model_inference_frames": 0, "c09_worlds_read": 0, "strict_test_worlds_read": 0, "mtare_worlds_read": 0},
        "expected_evidence": ["Corrected traversal decomposition, unchanged reference histograms/digests, per-world summary, source integrity, environment, log, RUN_STATE and seal."],
        "estimated_cost": card["estimated_cost"],
        "frozen_inputs": {path: _sha(PROJECT_ROOT / path) for path in inputs},
        "frozen_tools": {name: {"path": path, "sha256": _sha(PROJECT_ROOT / path)} for name, path in tools.items()},
        "working_directory": str(PROJECT_ROOT),
        "command": ["/usr/bin/systemd-inhibit", "--what=sleep:shutdown", "--why=GSE causal episode reference proof V1R", "--mode=block", "/usr/bin/timeout", "--signal=INT", "--kill-after=30s", "360s", PYTHON, "tools/v3/run_gse_causal_episode_reference_proof_v1r.py", "--spec", str(SPEC_PATH), "--run-dir", str(PROJECT_ROOT / "results/gate3_semantics" / RUN_ID)],
    }
    write_json(SPEC_PATH, spec)
    print(CARD_PATH.relative_to(PROJECT_ROOT))
    print(SPEC_PATH.relative_to(PROJECT_ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
