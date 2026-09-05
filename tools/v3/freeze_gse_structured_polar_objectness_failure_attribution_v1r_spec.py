#!/usr/bin/env python3
from __future__ import annotations

import hashlib

from _bootstrap import PROJECT_ROOT
from mtare_topo.governance import write_json


RUN_ID = "gate3_20260828_gse_structured_polar_objectness_failure_attribution_v1r_seed0"
SPEC = PROJECT_ROOT / "configs/v3/gate3/gse_structured_polar_objectness_failure_attribution_v1r.json"
DATA_CARD = "configs/v3/gate3/data_cards/gse_structured_polar_objectness_failure_attribution_v1r.json"
PYTHON = "/home/zeng-workstation/.local/share/mtare_topo_comm/envs/phase3_torch290_cu129_zarr2187_v1/bin/python"


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""): digest.update(block)
    return digest.hexdigest()


def main() -> int:
    if SPEC.exists(): raise RuntimeError("V1R spec already exists")
    teacher = "results/gate2_representation/gate2_20260828_gse_observable_spatial_event_teacher_v2_seed0"
    capacity = "results/gate3_semantics/gate3_20260828_gse_structured_polar_multidepth_capacity_v1_seed0"
    v1 = "results/gate3_semantics/gate3_20260828_gse_structured_polar_objectness_failure_attribution_v1_seed0"
    inputs = [DATA_CARD, f"{teacher}/RUN_STATE.json", f"{teacher}/metrics/summary.json", f"{teacher}/artifacts/evidence_sha256.txt", f"{teacher}/artifacts/export/summary.json", f"{teacher}/artifacts/export/teacher_shard_manifest.jsonl", f"{capacity}/RUN_STATE.json", f"{capacity}/metrics/summary.json", f"{capacity}/metrics/capacity/summary.json", f"{capacity}/artifacts/evidence_sha256.txt", f"{v1}/RUN_STATE.json", f"{v1}/metrics/summary.json", f"{v1}/artifacts/audit/summary.json", f"{v1}/artifacts/evidence_sha256.txt"]
    for seed in (0, 1, 2): inputs.extend((f"{capacity}/artifacts/models/seed{seed}/summary.json", f"{capacity}/artifacts/models/seed{seed}/selection_outputs.npz"))
    tools = {"executor": "tools/v3/execute_gse_structured_polar_objectness_failure_attribution_v1r.py", "v1_helpers": "tools/v3/execute_gse_structured_polar_objectness_failure_attribution_v1.py", "capacity_helpers": "tools/v3/evaluate_gse_structured_polar_multidepth_capacity_v1.py", "runner": "tools/v3/run_gse_structured_polar_objectness_failure_attribution_v1r.py", "freezer": "tools/v3/freeze_gse_structured_polar_objectness_failure_attribution_v1r_spec.py", "teacher_loader": "src/mtare_topo/data/gse_observable_spatial_event_dataset.py", "metrics": "src/mtare_topo/evaluation/gse_spatial_event_set_metrics.py", "tests": "tests/v3/unit/test_gse_structured_polar_objectness_failure_attribution_v1r.py", "governance": "src/mtare_topo/governance.py", "preflight": "tools/v3/preflight.py", "create_run": "tools/v3/create_run.py"}
    spec = {
        "schema_version": "v3_run_spec_v1", "gate": 3, "execution_phase": 3, "date": "20260828", "slug": "gse_structured_polar_objectness_failure_attribution_v1r", "seed": 0, "operation": "audit", "data_card": DATA_CARD,
        "question": "Does the V1 top16 oracle gain actually use the second radial-depth slot, or did all seeds collapse to slot0 so that objectness-only refit is invalid?",
        "method": "Replay all deterministic typed 4m matches while retaining predicted depth-slot provenance; count top16/selected slot use, second-depth target source slots and simultaneous same-bin near/far coverage with distinct slots.",
        "baseline": "V1 PASS decision FROZEN_GEOMETRY_OBJECTNESS_REFIT_REQUIRED based on typed oracle recall but without semantic slot-utilization evidence.",
        "fallback": "Keep V1 only if every seed uses slot1 for a second-depth target and one distinct-slot pair; otherwise supersede V1 and stop structured multi-depth in favor of executable full exit/action tokens.",
        "user_authorization": {"status": "APPROVED", "approved_by": "user-standing-authorization", "approved_at": "2026-08-28T23:50:00+08:00", "authorized_gates": [3], "authorized_operations": ["audit"], "scope": "One immutable V1R slot-provenance correction.", "confirmation_reference": "User delegated best-method decisions and continuous execution."},
        "acceptance_criteria": ["Exact 45942 observations, 33145 targets, 79 second-depth targets and three archives.", "Replay all top16 typed matches with exact predicted slot provenance and report same-bin pair simultaneous coverage.", "Emit KEEP_FROZEN_GEOMETRY_OBJECTNESS_REFIT only if every seed has nonzero slot1 second-depth and distinct-slot pair use; otherwise stop the route.", "Zero optimizer/inference/threshold selection/C09/C10/M-TARE/graph/planner; source unchanged and complete seal."],
        "expected_counts": {"worlds": 20, "observations": 45942, "target_tokens": 33145, "second_depth_targets": 79, "seed_archives": 3, "optimizer_steps": 0, "model_inference_frames": 0, "threshold_selection_steps": 0, "c09_worlds_read": 0, "c10_worlds_read": 0, "mtare_worlds_read": 0},
        "expected_evidence": ["Per-seed top16/selected/matched depth-slot provenance and same-bin pair coverage.", "CSV, PNG/PDF/SVG and exact JSON source.", "Commands, raw log, hashes, RUN_STATE and seal."],
        "estimated_cost": {"compute": "CPU read-only provenance audit", "wall_time_hours": 0.05, "host_ram_gb": 4, "gpu_memory_gb": 0, "disk_gb": 0.1, "gpu": "none"},
        "frozen_inputs": {path: sha256(PROJECT_ROOT / path) for path in sorted(inputs)}, "frozen_tools": {name: {"path": path, "sha256": sha256(PROJECT_ROOT / path)} for name, path in tools.items()}, "working_directory": str(PROJECT_ROOT),
        "command": ["/usr/bin/timeout", "900s", PYTHON, "tools/v3/run_gse_structured_polar_objectness_failure_attribution_v1r.py", "--spec", str(SPEC), "--run-dir", str(PROJECT_ROOT / f"results/gate3_semantics/{RUN_ID}")]
    }
    write_json(SPEC, spec); print(SPEC.relative_to(PROJECT_ROOT)); return 0


if __name__ == "__main__": raise SystemExit(main())
