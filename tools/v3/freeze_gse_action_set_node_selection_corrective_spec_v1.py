#!/usr/bin/env python3
"""Freeze the zero-training action-set selection corrective audit."""

from __future__ import annotations

import hashlib

from _bootstrap import PROJECT_ROOT
from mtare_topo.governance import write_json


RUN_ID = "gate3_20260828_gse_action_set_node_selection_corrective_v1_seed0"
SPEC = PROJECT_ROOT / "configs/v3/gate3/gse_action_set_node_selection_corrective_v1.json"
PYTHON = "/home/zeng-workstation/.local/share/mtare_topo_comm/envs/phase3_torch290_cu129_zarr2187_v1/bin/python"
SOURCE = "results/gate3_semantics/gate3_20260828_gse_action_set_node_training_v1_seed0"


def _sha(path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    if SPEC.exists():
        raise RuntimeError("action-set corrective spec exists; overwrite is forbidden")
    approval = {"status": "APPROVED", "approved_by": "user", "approved_at": "2026-08-28T00:55:00+08:00", "authorized_operations": ["audit"], "authorized_gates": [3], "scope": "One zero-training C01-C08 corrective evaluation of the fully completed sealed V1 models; same fixed 1001 thresholds and gates, zero C09/C10/M-TARE.", "confirmation_reference": "User explicitly authorized automatic best in-scope execution without routine approval prompts."}
    inputs = [f"{SOURCE}/RUN_STATE.json", f"{SOURCE}/metrics/summary.json", f"{SOURCE}/artifacts/evidence_sha256.txt", f"{SOURCE}/scratch/action_set_cache/partition_code.npy", f"{SOURCE}/scratch/action_set_cache/traversal_id.npy", f"{SOURCE}/scratch/action_set_cache/sequence_index.npy", f"{SOURCE}/scratch/action_set_cache/identity.npy", f"{SOURCE}/scratch/action_set_cache/parent_id.npy", f"{SOURCE}/scratch/action_set_cache/decision_target.npy", f"{SOURCE}/scratch/action_set_cache/decision_episode_id.npy"] + [f"{SOURCE}/artifacts/models/seed{seed}/selection_outputs.npz" for seed in (0, 1, 2)]
    tools = {"evaluator": "tools/v3/evaluate_gse_action_set_node_selection_v1.py", "metrics": "src/mtare_topo/evaluation/gse_causal_episode_metrics.py", "runner": "tools/v3/run_gse_action_set_node_selection_corrective_v1.py", "freezer": "tools/v3/freeze_gse_action_set_node_selection_corrective_spec_v1.py", "governance": "src/mtare_topo/governance.py", "preflight": "tools/v3/preflight.py", "create_run": "tools/v3/create_run.py"}
    spec = {
        "schema_version": "v3_run_spec_v1", "gate": 3, "execution_phase": 3, "date": "20260828", "slug": "gse_action_set_node_selection_corrective_v1", "seed": 0, "operation": "audit",
        "question": "What exact C07-C08 precision/recall frontier did the completed action-set models achieve when no threshold met every safety gate?",
        "method": "Read only the three sealed V1 selection outputs; scan the unchanged 0.000--1.000/0.001 ensemble threshold grid and emit a scientific PASS/FAIL plus the highest-precision point satisfying all minimum recall requirements.",
        "baseline": "V1 evaluator process error after select_decision_mass_threshold raised on an empty safe set; all three trainings completed 39996 steps and are reused read-only.",
        "fallback": "None. This run performs zero training and cannot alter the models, threshold grid, dataset or gates.",
        "config_path": "configs/v3/gate3/data_cards/gse_action_set_node_training_v1.json", "data_card": "configs/v3/gate3/data_cards/gse_action_set_node_training_v1.json", "user_authorization": approval,
        "acceptance_criteria": ["Same aggregate P>=0.995/false<=0.005/R>=0.25 and per-event P>=0.99/R>=0.25 gates; all ten families must have a correct trigger.", "On failure, report the highest-precision nonvacuous threshold satisfying all three recall minima.", "Zero optimizer/inference and zero C09/C10/M-TARE reads; exact source seal remains unchanged."],
        "expected_evidence": ["Corrected selection summary, ensemble outputs, source integrity, log, RUN_STATE and SHA-256 seal."],
        "estimated_cost": {"compute": "CPU-only read-only threshold audit", "wall_time_hours": 0.05, "disk_gb": 0.05},
        "expected_counts": {"selection_observations": 45942, "thresholds": 1001, "optimizer_steps": 0, "model_inference_frames": 0, "c09_worlds_read": 0, "c10_worlds_read": 0, "mtare_worlds_read": 0},
        "frozen_inputs": {relative: _sha(PROJECT_ROOT / relative) for relative in inputs},
        "frozen_tools": {name: {"path": relative, "sha256": _sha(PROJECT_ROOT / relative)} for name, relative in tools.items()},
        "command": ["/usr/bin/timeout", "--signal=INT", "--kill-after=30s", "360s", PYTHON, "tools/v3/run_gse_action_set_node_selection_corrective_v1.py", "--spec", str(SPEC), "--run-dir", str(PROJECT_ROOT / f"results/gate3_semantics/{RUN_ID}")],
        "working_directory": str(PROJECT_ROOT),
    }
    write_json(SPEC, spec)
    print({"spec": str(SPEC), "frozen_inputs": len(inputs), "frozen_tools": len(tools)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
