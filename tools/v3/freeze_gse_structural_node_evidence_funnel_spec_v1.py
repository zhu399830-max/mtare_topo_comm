#!/usr/bin/env python3
"""Freeze one C01-C07 zero-training structural-node evidence audit."""

from __future__ import annotations

import hashlib

from _bootstrap import PROJECT_ROOT
from mtare_topo.governance import write_json


RUN_ID = "gate3_20260904_gse_structural_node_evidence_funnel_v1_seed0"
SPEC = PROJECT_ROOT / "configs/v3/gate3/gse_structural_node_evidence_funnel_v1.json"
PYTHON = "/home/zeng-workstation/.local/share/mtare_topo_comm/envs/phase3_torch290_cu129_zarr2187_v1/bin/python"
P1A = "results/gate3_semantics/gate3_20260830_primitive_relation_p1a_lossless_storage_corrective_v1_seed0"
P1B = "results/gate3_semantics/gate3_20260830_primitive_relation_p1b_teacher_materialization_v1r_seed0"


def _sha(path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    if SPEC.exists():
        raise RuntimeError("structural-node evidence spec exists; overwrite is forbidden")
    approval = {
        "status": "APPROVED", "approved_by": "user-standing-authorization",
        "approved_at": "2026-09-04T22:00:00+08:00", "authorized_operations": ["audit"],
        "authorized_gates": [3],
        "scope": "One immutable zero-training C01-C07 structural-node evidence funnel; zero C08-C10, graph, planner or M-TARE.",
        "confirmation_reference": "User instructed continuous autonomous execution, optimal in-scope choices and no routine approval prompts."
    }
    inputs = [
        f"{P1A}/RUN_STATE.json", f"{P1A}/metrics/summary.json",
        f"{P1A}/artifacts/evidence_sha256.txt", f"{P1A}/artifacts/task_manifest.json",
        f"{P1A}/artifacts/traversal_manifest.jsonl",
        f"{P1B}/RUN_STATE.json", f"{P1B}/metrics/summary.json",
        f"{P1B}/artifacts/evidence_sha256.txt", f"{P1B}/artifacts/task_manifest.json",
    ]
    tools = {
        "descriptor": "src/mtare_topo/representation/gse_structural_node_evidence.py",
        "descriptor_tests": "tests/v3/unit/test_gse_structural_node_evidence.py",
        "executor": "tools/v3/execute_gse_structural_node_evidence_funnel_v1.py",
        "runner": "tools/v3/run_gse_structural_node_evidence_funnel_v1.py",
        "freezer": "tools/v3/freeze_gse_structural_node_evidence_funnel_spec_v1.py",
        "governance": "src/mtare_topo/governance.py",
        "preflight": "tools/v3/preflight.py",
        "create_run": "tools/v3/create_run.py"
    }
    spec = {
        "schema_version": "v3_run_spec_v1", "gate": 3, "execution_phase": 3,
        "date": "20260904", "slug": "gse_structural_node_evidence_funnel_v1", "seed": 0,
        "operation": "audit",
        "question": "Does causal five-frame primitive composition contain enough invariant information for high-precision structural-node association before any further training?",
        "method": "At the final causal observation of every directed traversal, aggregate the incident swept-superellipse ports into a slot-, pose- and rotation-invariant node descriptor. Compare full-composition oracle, five-frame observed and single-frame observed modes. Select one tie-safe 0.98-precision distance threshold on C01-C06 and apply it once to C07.",
        "baseline": "The completed endpoint relation metric improved ranking but failed 0/3 seeds: best precision 0.118684-0.198213 and no nonempty precision>=0.98 operating point.",
        "fallback": "If the oracle itself fails, stop this representation. If oracle passes but five-frame fails, keep the geometry backbone and replace only node evidence aggregation before any training.",
        "config_path": "configs/v3/gate3/data_cards/gse_structural_node_evidence_funnel_v1.json",
        "data_card": "configs/v3/gate3/data_cards/gse_structural_node_evidence_funnel_v1.json",
        "user_authorization": approval,
        "acceptance_criteria": [
            "Full-composition oracle has a nonempty tie-safe precision>=0.98 fit region.",
            "Causal five-frame fit and C07 both have precision>=0.98, recall>=0.25 and accepted-pair false fraction<=0.01.",
            "Every C07 family S01-S10 has at least one true-positive node association.",
            "Exactly 180 fit and 30 C07 tasks, 36,318 fit and 5,874 C07 node observations; zero optimizer/model inference/C08-C10/graph/M-TARE."
        ],
        "expected_evidence": ["Mode-wise threshold/precision/recall, per-family C07 metrics, deterministic pair digest, source integrity, logs, RUN_STATE and SHA-256 seal."],
        "estimated_cost": {"compute": "CPU-only descriptor and pair audit", "wall_time_hours": 0.5, "host_ram_gb": 4, "disk_gb": 0.1, "gpu": 0},
        "expected_counts": {"fit_tasks": 180, "c07_tasks": 30, "fit_node_observations": 36318, "c07_node_observations": 5874, "optimizer_steps": 0, "model_inference_frames": 0, "c08_c09_c10_worlds_read": 0, "graph_replays": 0, "mtare_runs": 0},
        "frozen_inputs": {relative: _sha(PROJECT_ROOT / relative) for relative in inputs},
        "frozen_tools": {name: {"path": relative, "sha256": _sha(PROJECT_ROOT / relative)} for name, relative in tools.items()},
        "command": ["/usr/bin/timeout", "--signal=INT", "--kill-after=30s", "3600s", PYTHON, "tools/v3/run_gse_structural_node_evidence_funnel_v1.py", "--spec", str(SPEC), "--run-dir", str(PROJECT_ROOT / f"results/gate3_semantics/{RUN_ID}")],
        "working_directory": str(PROJECT_ROOT)
    }
    write_json(SPEC, spec)
    print({"spec": str(SPEC), "frozen_inputs": len(inputs), "frozen_tools": len(tools)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
