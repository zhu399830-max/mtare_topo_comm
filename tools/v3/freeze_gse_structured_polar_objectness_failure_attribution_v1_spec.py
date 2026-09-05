#!/usr/bin/env python3
from __future__ import annotations

import hashlib

from _bootstrap import PROJECT_ROOT
from mtare_topo.governance import write_json


RUN_ID = "gate3_20260828_gse_structured_polar_objectness_failure_attribution_v1_seed0"
SPEC = PROJECT_ROOT / "configs/v3/gate3/gse_structured_polar_objectness_failure_attribution_v1.json"
DATA_CARD = "configs/v3/gate3/data_cards/gse_structured_polar_objectness_failure_attribution_v1.json"
PYTHON = "/home/zeng-workstation/.local/share/mtare_topo_comm/envs/phase3_torch290_cu129_zarr2187_v1/bin/python"


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    if SPEC.exists():
        raise RuntimeError("structured-polar attribution V1 spec already exists")
    teacher = "results/gate2_representation/gate2_20260828_gse_observable_spatial_event_teacher_v2_seed0"
    capacity = "results/gate3_semantics/gate3_20260828_gse_structured_polar_multidepth_capacity_v1_seed0"
    inputs = [
        DATA_CARD,
        f"{teacher}/RUN_STATE.json",
        f"{teacher}/metrics/summary.json",
        f"{teacher}/artifacts/evidence_sha256.txt",
        f"{teacher}/artifacts/export/summary.json",
        f"{teacher}/artifacts/export/teacher_shard_manifest.jsonl",
        f"{capacity}/RUN_STATE.json",
        f"{capacity}/metrics/summary.json",
        f"{capacity}/metrics/capacity/summary.json",
        f"{capacity}/metrics/capacity/figure_source.json",
        f"{capacity}/artifacts/evidence_sha256.txt",
    ]
    for seed in (0, 1, 2):
        inputs.extend((f"{capacity}/artifacts/models/seed{seed}/summary.json", f"{capacity}/artifacts/models/seed{seed}/selection_outputs.npz"))
    tools = {
        "executor": "tools/v3/execute_gse_structured_polar_objectness_failure_attribution_v1.py",
        "capacity_helpers": "tools/v3/evaluate_gse_structured_polar_multidepth_capacity_v1.py",
        "runner": "tools/v3/run_gse_structured_polar_objectness_failure_attribution_v1.py",
        "freezer": "tools/v3/freeze_gse_structured_polar_objectness_failure_attribution_v1_spec.py",
        "teacher_loader": "src/mtare_topo/data/gse_observable_spatial_event_dataset.py",
        "metrics": "src/mtare_topo/evaluation/gse_spatial_event_set_metrics.py",
        "tests": "tests/v3/unit/test_gse_structured_polar_objectness_failure_attribution.py",
        "governance": "src/mtare_topo/governance.py",
        "preflight": "tools/v3/preflight.py",
        "create_run": "tools/v3/create_run.py",
    }
    authorization = {"status": "APPROVED", "approved_by": "user-standing-authorization", "approved_at": "2026-08-28T23:35:00+08:00", "authorized_gates": [3], "authorized_operations": ["audit"], "scope": "One immutable read-only C07-C08 structured-polar objectness failure attribution.", "confirmation_reference": "User instructed automatic best-choice execution without routine approval prompts."}
    spec = {
        "schema_version": "v3_run_spec_v1",
        "gate": 3,
        "execution_phase": 3,
        "date": "20260828",
        "slug": "gse_structured_polar_objectness_failure_attribution_v1",
        "seed": 0,
        "operation": "audit",
        "data_card": DATA_CARD,
        "question": "Do the sealed 180x2 top16 candidates contain enough typed 4m proposal capacity for a minimal objectness corrective, or must the dense event route stop despite partial second-depth gains?",
        "method": "Replay every sealed C07-C08 top16 archive with no inference. Decompose typed/position oracle proposal recall, confidence ranking, depth-slot and same-bin false positives, empty rows, one-per-bin suppression, Teacher-cardinality upper bound and distance strata.",
        "baseline": "Same-population exclusive-center recall=0.262905 and F1=0.390133, plus each seed's sealed 0.95 result.",
        "fallback": "Emit one frozen decision. No training or parameter change occurs; C09/C10/M-TARE, graph and planner remain forbidden.",
        "user_authorization": authorization,
        "acceptance_criteria": [
            "Exactly 20 worlds, 45942 observations, 33145 targets, 6073 multi-event rows, 79 second-depth targets, 275 identities and three sealed top16 archives.",
            "All three sealed metrics replay exactly at threshold 0.95; no prediction, target, checkpoint or source changes.",
            "Report typed and position oracle proposal recall, candidate confidence ranking, slot0/slot1 and same-bin false positives, empty-row/cardinality diagnostics and distance strata.",
            "Emit exactly one decision from the frozen proposal, one-per-bin and Teacher-cardinality gates.",
            "Zero optimizer, model inference, threshold selection, C09/C10/M-TARE, graph or planner; complete CSV, paper figure and seal."
        ],
        "expected_counts": {"worlds": 20, "observations": 45942, "target_tokens": 33145, "target_types": [7578, 25567], "event_identities": 275, "multi_event_rows": 6073, "second_depth_targets": 79, "cardinality": [19159, 20710, 5786, 285, 2, 0], "seed_archives": 3, "optimizer_steps": 0, "model_inference_frames": 0, "threshold_selection_steps": 0, "c09_worlds_read": 0, "c10_worlds_read": 0, "mtare_worlds_read": 0},
        "hyperparameters": {"match_radius_m": 4.0, "sealed_thresholds": [0.95, 0.95, 0.95], "proposal_recall_gain_gate": 0.10, "deterministic_f1_gain_gate": 0.05},
        "expected_evidence": ["Three complete proposal/ranking/slot/cardinality records and an explicit decision.", "CSV plus PNG/PDF/SVG diagnostic figure and exact JSON source.", "Commands, raw log, source hashes, summary, RUN_STATE and SHA-256 seal."],
        "estimated_cost": {"compute": "CPU read-only deterministic matching over three sealed C07-C08 archives; no model inference.", "wall_time_hours": 0.15, "host_ram_gb": 4, "gpu_memory_gb": 0, "disk_gb": 0.1, "gpu": "none"},
        "frozen_inputs": {path: sha256(PROJECT_ROOT / path) for path in sorted(inputs)},
        "frozen_tools": {name: {"path": path, "sha256": sha256(PROJECT_ROOT / path)} for name, path in tools.items()},
        "working_directory": str(PROJECT_ROOT),
        "command": ["/usr/bin/timeout", "1500s", PYTHON, "tools/v3/run_gse_structured_polar_objectness_failure_attribution_v1.py", "--spec", str(SPEC), "--run-dir", str(PROJECT_ROOT / f"results/gate3_semantics/{RUN_ID}")]
    }
    write_json(SPEC, spec)
    print(SPEC.relative_to(PROJECT_ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
