#!/usr/bin/env python3
"""Freeze schema-corrected V1R token-validity audit; science is unchanged."""

from __future__ import annotations

import hashlib

from _bootstrap import PROJECT_ROOT
from mtare_topo.governance import write_json


RUN_ID = "gate3_20260829_gse_token_validity_action_track_feasibility_v1r_seed0"
SPEC = PROJECT_ROOT / "configs/v3/gate3/gse_token_validity_action_track_feasibility_v1r.json"
DATA_CARD = "configs/v3/gate3/data_cards/gse_token_validity_action_track_feasibility_v1r.json"
PYTHON = "/home/zeng-workstation/.local/share/mtare_topo_comm/envs/phase3_torch290_cu129_zarr2187_v1/bin/python"


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    if SPEC.exists():
        raise RuntimeError("token-validity V1R spec already exists")
    dataset = "results/gate2_representation/gate2_20260824_gse_deduplicated_dataset_export_v1_seed0"
    action = "results/gate3_semantics/gate3_20260828_gse_action_set_node_training_v1_seed0"
    cache = f"{action}/scratch/action_set_cache"
    transport = "results/gate3_semantics/gate3_20260828_gse_exit_action_transport_feasibility_v1_seed0"
    v1 = "results/gate3_semantics/gate3_20260829_gse_relational_commit_policy_feasibility_v1_seed0"
    v2 = "results/gate3_semantics/gate3_20260829_gse_structured_exact_one_event_training_v1_seed0"
    inputs = [
        DATA_CARD,
        f"{dataset}/RUN_STATE.json", f"{dataset}/metrics/summary.json", f"{dataset}/artifacts/shard_manifest.json", f"{dataset}/artifacts/evidence_sha256.txt",
        f"{action}/RUN_STATE.json", f"{action}/metrics/summary.json", f"{action}/artifacts/evidence_sha256.txt",
        f"{transport}/RUN_STATE.json", f"{transport}/metrics/summary.json", f"{transport}/artifacts/evidence_sha256.txt",
        f"{v1}/RUN_STATE.json", f"{v1}/metrics/summary.json", f"{v1}/artifacts/evidence_sha256.txt",
        f"{v2}/RUN_STATE.json", f"{v2}/metrics/summary.json", f"{v2}/metrics/selection/summary.json", f"{v2}/artifacts/evidence_sha256.txt",
        f"{cache}/manifest.json",
    ]
    inputs.extend(f"{cache}/{name}.npy" for name in ("raw_tokens", "partition_code", "parent_id", "traversal_id", "sequence_index", "global_sequence_index"))
    tools = {
        "validity_metrics": "src/mtare_topo/evaluation/gse_token_validity_feasibility.py",
        "transport_assignment": "src/mtare_topo/evaluation/gse_exit_action_transport.py",
        "executor": "tools/v3/execute_gse_token_validity_action_track_feasibility_v1.py",
        "runner": "tools/v3/run_gse_token_validity_action_track_feasibility_v1.py",
        "freezer": "tools/v3/freeze_gse_token_validity_action_track_feasibility_spec_v1r.py",
        "tests": "tests/v3/unit/test_gse_token_validity_feasibility.py",
        "governance": "src/mtare_topo/governance.py", "preflight": "tools/v3/preflight.py", "create_run": "tools/v3/create_run.py",
    }
    spec = {
        "schema_version": "v3_run_spec_v1", "gate": 3, "execution_phase": 3, "date": "20260829", "slug": "gse_token_validity_action_track_feasibility_v1r", "seed": 0,
        "operation": "audit", "data_card": DATA_CARD,
        "question": "Can frozen free-query exit tokens be safely validated using confidence, five-frame descriptor tracks or three-seed geometry consensus?",
        "method": "Schema-only V1R: verify C01-C08; attach Teacher validity for scoring only; evaluate tied-threshold current confidence, past-only five-frame descriptor-track mean and Teacher-free three-seed heading/width/profile consensus.",
        "baseline": "Frozen single-frame query confidence, V1 stateful direct event head and V2 exact-one direct event head.",
        "fallback": "On scientific failure, stop the frozen free-query validity route and prepare a dense circular traversability field without query objectness.",
        "supersedes_preflight_only_spec": "configs/v3/gate3/gse_token_validity_action_track_feasibility_v1.json",
        "user_authorization": {"status": "APPROVED", "approved_by": "user-standing-authorization", "approved_at": "2026-08-29T01:35:00+08:00", "authorized_gates": [3], "authorized_operations": ["audit"], "scope": "One immutable schema-corrected V1R zero-training C01-C08 audit.", "confirmation_reference": "User explicitly delegated best-method choices and continuous execution."},
        "acceptance_criteria": ["Exact 80 worlds, 252430 frames, 188126 observations, C07/C08 21548/24394 and per-seed 396913 positive plus 731843 negative slots.", "Retained Teacher action-set and all-seed descriptor transport metrics remain at least 0.98.", "Every seed reaches token recall at least 0.50 at tied-threshold precision at least 0.995.", "Every seed's best causal/consensus score improves AP at least 0.05 and retains high-precision recall at least 0.50.", "Zero optimizer/new inference/threshold fitting/graph/C09/C10/M-TARE; source unchanged and complete evidence."],
        "expected_counts": {"worlds": 80, "raw_frames": 252430, "observations": 188126, "fit_observations": 142184, "selection_observations": 45942, "c07_observations": 21548, "c08_observations": 24394, "slots_per_seed": 1128756, "positive_tokens_per_seed": 396913, "negative_tokens_per_seed": 731843, "seed_archives": 3, "optimizer_steps": 0, "model_inference_frames": 0, "threshold_selection_steps": 0, "graph_replays": 0, "c09_worlds_read": 0, "c10_worlds_read": 0, "mtare_worlds_read": 0},
        "expected_evidence": ["Per-seed/partition validity ranking metrics with complete score ties.", "Five-frame causal and three-seed geometry attribution against current confidence.", "Retained action-set/descriptor capacity and sealed direct-event comparisons.", "PNG/PDF/SVG/source, raw log, environment, RUN_STATE and seal."],
        "estimated_cost": {"compute": "CPU read-only full shard, causal-track and cross-seed consensus audit", "wall_time_hours": 0.1, "host_ram_gb": 6, "gpu_memory_gb": 0, "disk_gb": 0.1, "gpu": "none"},
        "frozen_inputs": {path: sha256(PROJECT_ROOT / path) for path in sorted(inputs)},
        "frozen_tools": {name: {"path": path, "sha256": sha256(PROJECT_ROOT / path)} for name, path in tools.items()},
        "working_directory": str(PROJECT_ROOT),
        "command": ["/usr/bin/timeout", "1200s", PYTHON, "tools/v3/run_gse_token_validity_action_track_feasibility_v1.py", "--spec", str(SPEC), "--run-dir", str(PROJECT_ROOT / f"results/gate3_semantics/{RUN_ID}")],
    }
    write_json(SPEC, spec)
    print(SPEC.relative_to(PROJECT_ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
