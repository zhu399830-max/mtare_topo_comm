#!/usr/bin/env python3
"""Freeze the C01--C08 structured learned-exit capacity audit."""

from __future__ import annotations

import hashlib

from _bootstrap import PROJECT_ROOT
from mtare_topo.governance import write_json


RUN_ID = "gate3_20260828_gse_route_conditioned_structured_node_capacity_v1_seed0"
SPEC = PROJECT_ROOT / "configs/v3/gate3/gse_route_conditioned_structured_node_capacity_v1.json"
PYTHON = "/home/zeng-workstation/.local/share/mtare_topo_comm/envs/phase3_torch290_cu129_zarr2187_v1/bin/python"
TEACHER = "results/gate2_representation/gate2_20260827_gse_corrected_causal_teacher_manifest_v1r_seed0"
SOURCE = "results/gate3_semantics/gate3_20260826_gse_exit_token_association_corrective_v2_seed0"


def _sha(path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    if SPEC.exists():
        raise RuntimeError("structured-node capacity spec exists; overwrite is forbidden")
    approval = {"status": "APPROVED", "approved_by": "user", "approved_at": "2026-08-28T01:15:00+08:00", "authorized_operations": ["audit"], "authorized_gates": [3], "scope": "One zero-training C01-C08 route-conditioned structured learned-exit capacity audit; fixed 900 fit configurations, one C07-C08 validation, zero C09/C10/M-TARE.", "confirmation_reference": "User explicitly authorized automatic best in-scope execution without routine approval prompts."}
    inputs = [f"{TEACHER}/RUN_STATE.json", f"{TEACHER}/metrics/summary.json", f"{TEACHER}/artifacts/evidence_sha256.txt", f"{TEACHER}/artifacts/teacher_observations.jsonl", f"{SOURCE}/RUN_STATE.json", f"{SOURCE}/metrics/summary.json", f"{SOURCE}/artifacts/evidence_sha256.txt", f"{SOURCE}/artifacts/pair_cache/pairs.npz"] + [f"{SOURCE}/artifacts/models/seed{seed}/frozen_exit_token_outputs.npz" for seed in (0, 1, 2)]
    tools = {"decoder": "src/mtare_topo/representation/gse_route_conditioned_node.py", "episode_reference": "src/mtare_topo/representation/gse_causal_episode_detector.py", "executor": "tools/v3/execute_gse_route_conditioned_structured_node_capacity_v1.py", "runner": "tools/v3/run_gse_route_conditioned_structured_node_capacity_v1.py", "freezer": "tools/v3/freeze_gse_route_conditioned_structured_node_capacity_spec_v1.py", "governance": "src/mtare_topo/governance.py", "preflight": "tools/v3/preflight.py", "create_run": "tools/v3/create_run.py"}
    spec = {
        "schema_version": "v3_run_spec_v1", "gate": 3, "execution_phase": 3, "date": "20260828", "slug": "gse_route_conditioned_structured_node_capacity_v1", "seed": 0, "operation": "audit",
        "question": "Can online incoming-action removal turn frozen learned exit tokens into safe junction/terminal proposals without another black-box classifier?",
        "method": "For each frozen perception seed, remove the learned exit cluster aligned with the causally executed incoming heading, 20-degree NMS the remaining learned exits, classify 0/1/2+ outgoing actions as terminal/no-node/junction, then require seed consensus and temporal persistence. Select one of 900 fixed configurations on C01-C06 only and apply it once to C07-C08.",
        "baseline": "Completed action-set black-box detector: at all minimum recalls, best C07-C08 aggregate P/false/R=0.965870/0.034130/0.498239 and junction precision=0.951977.",
        "fallback": "None inside this audit. If no fit-safe configuration or the frozen selection result fails, stop this node-proposal representation; do not change gates or graph logic.",
        "config_path": "configs/v3/gate3/data_cards/gse_action_set_node_training_v1.json", "data_card": "configs/v3/gate3/data_cards/gse_action_set_node_training_v1.json", "user_authorization": approval,
        "parameter_grid": {"confidence_threshold": "0.02..1.00 step 0.02 (50)", "incoming_half_angle_deg": [20, 35, 50], "persistence_observations": [1, 2, 3], "seed_consensus": [2, 3], "configurations": 900, "token_nms_deg": 20},
        "acceptance_criteria": ["C01-C06 selection requires aggregate P>=0.997/false<=0.003/R>=0.25 and junction/terminal each P>=0.995/R>=0.25.", "The single frozen C07-C08 result requires aggregate P>=0.995/false<=0.005/R>=0.25, each event P>=0.99/R>=0.25 and all ten families with a correct trigger.", "Zero optimizer/inference and zero C09/C10/M-TARE reads; source unchanged and complete seal."],
        "expected_evidence": ["All 900 fit records, selected configuration, one C07-C08 result, per-event/family/identity diagnostics, logs, source integrity, RUN_STATE and SHA-256 seal."],
        "estimated_cost": {"compute": "CPU-only structured token audit", "wall_time_hours": 0.5, "disk_gb": 0.1},
        "expected_counts": {"fit_observations": 142184, "selection_observations": 45942, "grid_configurations": 900, "optimizer_steps": 0, "model_inference_frames": 0, "c09_worlds_read": 0, "c10_worlds_read": 0, "mtare_worlds_read": 0},
        "frozen_inputs": {relative: _sha(PROJECT_ROOT / relative) for relative in inputs},
        "frozen_tools": {name: {"path": relative, "sha256": _sha(PROJECT_ROOT / relative)} for name, relative in tools.items()},
        "command": ["/usr/bin/timeout", "--signal=INT", "--kill-after=30s", "3600s", PYTHON, "tools/v3/run_gse_route_conditioned_structured_node_capacity_v1.py", "--spec", str(SPEC), "--run-dir", str(PROJECT_ROOT / f"results/gate3_semantics/{RUN_ID}")],
        "working_directory": str(PROJECT_ROOT),
    }
    write_json(SPEC, spec)
    print({"spec": str(SPEC), "frozen_inputs": len(inputs), "frozen_tools": len(tools)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
