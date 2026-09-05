#!/usr/bin/env python3
"""Freeze one C01-C08 twelve-scan reference proof."""

from __future__ import annotations

import hashlib
from pathlib import Path

from _bootstrap import PROJECT_ROOT
from mtare_topo.governance import load_json, write_json


RUN_ID = "gate3_20260827_gse_causal_episode_reference_proof_v1_seed0"
CARD_PATH = PROJECT_ROOT / "configs/v3/gate3/data_cards/gse_causal_episode_reference_proof_v1.json"
SPEC_PATH = PROJECT_ROOT / "configs/v3/gate3/gse_causal_episode_reference_proof_v1.json"
SOURCE_CARD = PROJECT_ROOT / "configs/v3/gate3/data_cards/gse_causal_event_supervision_audit_v1.json"
DATASET = "results/gate2_representation/gate2_20260824_gse_deduplicated_dataset_export_v1_seed0"
TEACHER = "results/gate2_representation/gate2_20260827_gse_corrected_causal_teacher_manifest_v1r_seed0"
AUDIT = "results/gate3_semantics/gate3_20260827_gse_causal_event_supervision_audit_v1_seed0"
PYTHON = "/home/zeng-workstation/.local/share/mtare_topo_comm/envs/phase3_torch290_cu129_zarr2187_v1/bin/python"


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    source = load_json(SOURCE_CARD)
    card = {
        **source,
        "card_id": "gse_causal_episode_reference_proof_v1",
        "status": "APPROVED_FOR_ONE_IMMUTABLE_GSE_CAUSAL_EPISODE_REFERENCE_PROOF_V1",
        "purpose": "Mechanically prove that every corrected C01-C08 observation can reference up to twelve exact past LiDAR scans inside one traversal and resolve those references in the sealed deduplicated shards without copying sensor payload.",
        "approval": {
            "status": "APPROVED", "approved_by": "user", "approved_at": "2026-08-27T23:59:55+08:00",
            "authorized_gates": [3], "authorized_operations": ["audit"],
            "scope": "One immutable read-only C01-C08 reference proof; no sensor payload export, ray generation, inference, training, C09/C10 or M-TARE access.",
            "confirmation_reference": "Standing user authorization for autonomous optimal execution within the GSE-Graph paper scope.",
        },
        "method": {
            "main": "For each corrected Teacher row, derive a left-padded suffix of at most twelve global frames from the sealed current frame and traversal-local frame index; resolve every valid reference inside the same parent shard and prove local indices remain consecutive across no reset.",
            "baseline": "The sealed five-frame global/local reference arrays already used by GSE V1R.",
            "fallback": "Any unresolved or cross-reset reference stops the twelve-frame method and requires a new data representation; no new raycast or payload duplication is allowed.",
        },
        "acceptance": {
            "audit_completion": "Exact 80 worlds, 16078 traversals, 252430 unique frames, 188126 observations, 5306 episodes and all reference cells resolved within parent/traversal.",
            "decision_rule": "PASS authorizes a training Data Card for a 12-frame detector; it does not itself export data, train a model or qualify semantics/topology.",
        },
        "estimated_cost": {"compute": "CPU-only index and Zarr metadata audit", "wall_time_hours": 0.1, "host_ram_gb": 4, "disk_gb": 0.0625},
        "retention": "Keep per-world reference summary, array digests, exact histograms, config, log, RUN_STATE and seal; write zero sensor payload bytes.",
    }
    write_json(CARD_PATH, card)
    inputs = []
    for run in (DATASET, TEACHER, AUDIT):
        inputs.extend([f"{run}/RUN_STATE.json", f"{run}/metrics/summary.json", f"{run}/artifacts/evidence_sha256.txt"])
    inputs.extend([f"{TEACHER}/artifacts/teacher_observations.jsonl", f"{DATASET}/artifacts/shard_manifest.json", f"{AUDIT}/metrics/supervision_audit.json"])
    tools = {
        "audit_card": str(CARD_PATH.relative_to(PROJECT_ROOT)),
        "reference_contract": "src/mtare_topo/representation/gse_causal_episode_detector.py",
        "supervision_contract": "src/mtare_topo/evaluation/gse_causal_event_supervision.py",
        "executor": "tools/v3/execute_gse_causal_episode_reference_proof_v1.py",
        "runner": "tools/v3/run_gse_causal_episode_reference_proof_v1.py",
        "freezer": "tools/v3/freeze_gse_causal_episode_reference_proof_spec_v1.py",
        "evidence_integrity": "src/mtare_topo/evaluation/gse_evidence_integrity.py",
        "governance": "src/mtare_topo/governance.py", "preflight": "tools/v3/preflight.py", "create_run": "tools/v3/create_run.py",
    }
    spec = {
        "schema_version": "v3_run_spec_v1", "gate": 3, "execution_phase": 3, "date": "20260827",
        "slug": "gse_causal_episode_reference_proof_v1", "seed": 0, "operation": "audit",
        "question": "Can every corrected C01-C08 observation consume an exact past-only 12-scan history from existing shards without crossing a traversal or duplicating sensor data?",
        "method": card["method"]["main"], "baseline": card["method"]["baseline"], "fallback": card["method"]["fallback"],
        "audit_card": str(CARD_PATH.relative_to(PROJECT_ROOT)), "config_path": str(CARD_PATH.relative_to(PROJECT_ROOT)),
        "user_authorization": card["approval"],
        "acceptance_criteria": [
            "Exact 80/16078/252430/188126 population and 5306 contiguous structural episodes.",
            "Every unmasked 12-frame global reference resolves in the matching C01-C08 parent shard and local frame indices are consecutive without crossing a traversal reset.",
            "Preserve the sealed five-frame suffix exactly; report history histogram and deterministic array digests.",
            "Write zero sensor payload bytes, cast zero rays, run zero inference/optimizer steps and read zero C09/C10/M-TARE worlds.",
        ],
        "expected_counts": {"worlds": 80, "directed_traversals": 16078, "unique_frames": 252430, "causal_observations": 188126, "structural_episodes": 5306, "new_rays": 0, "sensor_payload_bytes_written": 0, "optimizer_steps": 0, "model_inference_frames": 0, "c09_worlds_read": 0, "strict_test_worlds_read": 0, "mtare_worlds_read": 0},
        "expected_evidence": ["Per-world reference summary, length histogram, array digests, source integrity, environment, raw log, RUN_STATE and complete SHA-256 seal."],
        "estimated_cost": card["estimated_cost"],
        "frozen_inputs": {path: _sha(PROJECT_ROOT / path) for path in inputs},
        "frozen_tools": {name: {"path": path, "sha256": _sha(PROJECT_ROOT / path)} for name, path in tools.items()},
        "working_directory": str(PROJECT_ROOT),
        "command": ["/usr/bin/systemd-inhibit", "--what=sleep:shutdown", "--why=GSE causal episode reference proof", "--mode=block", "/usr/bin/timeout", "--signal=INT", "--kill-after=30s", "360s", PYTHON, "tools/v3/run_gse_causal_episode_reference_proof_v1.py", "--spec", str(SPEC_PATH), "--run-dir", str(PROJECT_ROOT / "results/gate3_semantics" / RUN_ID)],
    }
    write_json(SPEC_PATH, spec)
    print(CARD_PATH.relative_to(PROJECT_ROOT))
    print(SPEC_PATH.relative_to(PROJECT_ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
