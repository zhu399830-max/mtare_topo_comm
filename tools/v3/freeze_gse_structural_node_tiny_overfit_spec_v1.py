#!/usr/bin/env python3
"""Freeze one family-balanced structural-node tiny-overfit gate."""

from __future__ import annotations

import hashlib

from _bootstrap import PROJECT_ROOT
from mtare_topo.governance import write_json


SPEC = PROJECT_ROOT / "configs/v3/gate3/gse_structural_node_tiny_overfit_v1.json"
RUN_ID = "gate3_20260904_gse_structural_node_tiny_overfit_v1_seed0"
PYTHON = "/home/zeng-workstation/.local/share/mtare_topo_comm/envs/phase3_torch290_cu129_zarr2187_v1/bin/python"


def _sha(path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    if SPEC.exists(): raise RuntimeError("tiny-overfit spec exists; overwrite is forbidden")
    p1a = "results/gate3_semantics/gate3_20260830_primitive_relation_p1a_lossless_storage_corrective_v1_seed0"
    p1b = "results/gate3_semantics/gate3_20260830_primitive_relation_p1b_teacher_materialization_v1r_seed0"
    obs = "results/gate3_semantics/gate3_20260902_primitive_attachment_observability_sidecar_v1_seed0"
    source_run = "results/gate3_semantics/gate3_20260902_primitive_relation_observable_three_seed_training_v1_seed0"
    capacity = "results/gate3_semantics/gate3_20260904_gse_structural_node_graph_consistency_attribution_v1_seed0"
    inputs = [f"{p1a}/RUN_STATE.json", f"{p1a}/metrics/summary.json", f"{p1a}/artifacts/evidence_sha256.txt", f"{p1a}/artifacts/task_manifest.json", f"{p1a}/artifacts/traversal_manifest.jsonl", f"{p1b}/RUN_STATE.json", f"{p1b}/metrics/summary.json", f"{p1b}/artifacts/evidence_sha256.txt", f"{p1b}/artifacts/task_manifest.json", f"{obs}/RUN_STATE.json", f"{obs}/metrics/summary.json", f"{obs}/artifacts/evidence_sha256.txt", f"{source_run}/artifacts/models/seed0/selected.pt", f"{capacity}/RUN_STATE.json", f"{capacity}/metrics/attribution/summary.json", f"{capacity}/artifacts/evidence_sha256.txt"]
    tools = {"evidence": "src/mtare_topo/representation/gse_structural_node_evidence.py", "student": "src/mtare_topo/representation/gse_structural_node_student.py", "evidence_executor": "tools/v3/execute_gse_structural_node_evidence_funnel_v1.py", "trainer": "tools/v3/train_gse_structural_node_tiny_overfit_v1.py", "runner": "tools/v3/run_gse_structural_node_tiny_overfit_v1.py", "freezer": "tools/v3/freeze_gse_structural_node_tiny_overfit_spec_v1.py", "tests_evidence": "tests/v3/unit/test_gse_structural_node_evidence.py", "tests_student": "tests/v3/unit/test_gse_structural_node_student.py", "governance": "src/mtare_topo/governance.py", "preflight": "tools/v3/preflight.py", "create_run": "tools/v3/create_run.py"}
    approval = {"status": "APPROVED", "approved_by": "user-standing-authorization", "approved_at": "2026-09-04T23:00:00+08:00", "authorized_operations": ["training"], "authorized_gates": [3], "scope": "One immutable 180-observation/500-step seed0 tiny-overfit gate; zero C07-C10, graph or M-TARE.", "confirmation_reference": "User instructed continuous autonomous execution, automatic optimal in-scope decisions and no routine approval prompts."}
    spec = {"schema_version": "v3_run_spec_v1", "gate": 3, "execution_phase": 3, "date": "20260904", "slug": "gse_structural_node_tiny_overfit_v1", "seed": 0, "operation": "training", "question": "Can a small permutation-invariant node head memorize the 44-D structural-node evidence and same-node relation from frozen causal-LiDAR primitive predictions within 500 updates?", "method": "Select exactly 18 C01 c1_mixed final-traversal observations from each S01-S10 family (180 observations/100 nodes). Run the frozen seed0 observable primitive backbone once, cache 64x44 endpoint tokens, and train only a small permutation-invariant node aggregation head for 500 full-batch AdamW steps.", "baseline": "Untrained head on the identical cached features; old endpoint metric C07 best precision 0.198213; zero-training Teacher node capacity plus 16 m graph gate already passed.", "fallback": "If tiny overfit fails, stop before one-seed training and redesign only the node aggregation/readout. Do not add steps, unfreeze the backbone, read C07 or lower gates.", "config_path": "configs/v3/gate3/data_cards/gse_structural_node_tiny_overfit_v1.json", "data_card": "configs/v3/gate3/data_cards/gse_structural_node_tiny_overfit_v1.json", "user_authorization": approval, "acceptance_criteria": ["Exactly 180 observations, 100 nodes, 10 families, 10 tasks and 80 same-node positive pairs.", "Exactly 500 head-only updates; total loss reduction>=95%, geometry RMSE<=0.01 and degree accuracy>=0.99.", "Tie-safe tiny relation precision>=0.99 and recall>=0.95 with nonempty accepted pairs.", "All head gradients finite, cached/final repeat exact and frozen backbone/source unchanged.", "Zero C07-C10, graph replay and M-TARE."], "expected_evidence": ["180-row manifest, compact frozen features, initial/final losses and pair metrics, loss curve PNG/PDF/SVG, tiny checkpoint, resource monitor, environment, RUN_STATE and seal."], "estimated_cost": {"compute": "180 frozen-backbone forwards plus 500 small-head full-batch updates", "wall_time_hours": 0.25, "host_ram_gb": 4, "disk_gb": 0.1, "gpu": 1}, "expected_counts": {"families": 10, "tasks": 10, "observations": 180, "nodes": 100, "positive_pairs": 80, "optimizer_steps": 500, "c07_rows_read": 0, "c08_c09_c10_worlds_read": 0, "graph_replays": 0, "mtare_runs": 0}, "frozen_inputs": {relative: _sha(PROJECT_ROOT / relative) for relative in inputs}, "frozen_tools": {name: {"path": relative, "sha256": _sha(PROJECT_ROOT / relative)} for name, relative in tools.items()}, "command": ["/usr/bin/timeout", "--signal=INT", "--kill-after=30s", "1800s", PYTHON, "tools/v3/run_gse_structural_node_tiny_overfit_v1.py", "--spec", str(SPEC), "--run-dir", str(PROJECT_ROOT / f"results/gate3_semantics/{RUN_ID}")], "working_directory": str(PROJECT_ROOT)}
    write_json(SPEC, spec); print({"spec": str(SPEC), "frozen_inputs": len(inputs), "frozen_tools": len(tools)}); return 0


if __name__ == "__main__": raise SystemExit(main())
