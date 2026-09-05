#!/usr/bin/env python3
"""Freeze the one formal JCGS frozen-output failure-attribution spec."""
from __future__ import annotations

import hashlib

from _bootstrap import PROJECT_ROOT
from mtare_topo.governance import load_json, validate_data_card, write_json


RUN_ID = "gate3_20260829_gse_joint_cyclic_gap_simplex_failure_attribution_v1_seed0"
SPEC = PROJECT_ROOT / "configs/v3/gate3/gse_joint_cyclic_gap_simplex_failure_attribution_v1.json"
DATA_CARD = "configs/v3/gate3/data_cards/gse_joint_cyclic_gap_simplex_failure_attribution_v1.json"
PYTHON = "/home/zeng-workstation/.local/share/mtare_topo_comm/envs/phase3_torch290_cu129_zarr2187_v1/bin/python"


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    if SPEC.exists():
        raise RuntimeError("JCGS attribution spec already exists")
    card = load_json(PROJECT_ROOT / DATA_CARD)
    validation = validate_data_card(card)
    if not validation.passed:
        raise RuntimeError(f"JCGS attribution card invalid: {validation.errors}")
    teacher = "results/gate2_representation/gate2_20260829_gse_circular_exit_geometry_field_teacher_export_v1_seed0"
    training = "results/gate3_semantics/gate3_20260829_gse_joint_cyclic_gap_simplex_three_seed_training_v1r2_seed0"
    inputs = [
        DATA_CARD,
        f"{teacher}/RUN_STATE.json", f"{teacher}/metrics/summary.json",
        f"{teacher}/artifacts/export/teacher_shard_manifest.jsonl", f"{teacher}/artifacts/evidence_sha256.txt",
        f"{training}/RUN_STATE.json", f"{training}/metrics/summary.json",
        f"{training}/metrics/selection/summary.json", f"{training}/artifacts/evidence_sha256.txt",
        "configs/v3/gate2/environments/phase3_torch290_cu129_zarr2187_v1r2.json",
    ]
    for seed in range(3):
        inputs.append(f"{training}/artifacts/models/seed{seed}/summary.json")
        inputs.extend(str(path.relative_to(PROJECT_ROOT)) for path in sorted((PROJECT_ROOT / training / f"artifacts/models/seed{seed}/development_predictions").glob("*.npz")))
    tools = {
        "executor": "tools/v3/execute_gse_joint_cyclic_gap_simplex_failure_attribution_v1.py",
        "runner": "tools/v3/run_gse_joint_cyclic_gap_simplex_failure_attribution_v1.py",
        "freezer": "tools/v3/freeze_gse_joint_cyclic_gap_simplex_failure_attribution_spec_v1.py",
        "tests": "tests/v3/unit/test_gse_joint_cyclic_gap_simplex_failure_attribution.py",
        "formal_evaluator": "tools/v3/evaluate_gse_joint_cyclic_gap_simplex_selection_v1.py",
        "base_evaluator": "tools/v3/evaluate_gse_cardinality_conditioned_circular_slot_transport_selection_v1.py",
        "cyclic_alignment": "tools/v3/evaluate_gse_cyclic_ordered_unimodal_slot_transport_selection_v1.py",
        "branch_contract": "src/mtare_topo/representation/gse_cardinality_conditioned_circular_slot_transport.py",
        "governance": "src/mtare_topo/governance.py",
        "preflight": "tools/v3/preflight.py",
        "create_run": "tools/v3/create_run.py",
    }
    spec = {
        "schema_version": "v3_run_spec_v1", "gate": 3, "execution_phase": 3, "date": "20260829",
        "slug": "gse_joint_cyclic_gap_simplex_failure_attribution_v1", "seed": 0,
        "operation": "audit", "data_card": DATA_CARD,
        "question": "Is JCGS failure primarily cardinality, cyclic seed alignment, phase, gap shape, or confidence, and which representation components are scientifically defensible to retain?",
        "method": "Read the sealed C07/C08 Teacher and three prediction trees once; exactly reproduce formal metrics, then measure per-seed/ensemble count confusion, oracle count, target-aligned seed ceiling, true-count phase/gap interventions, gap closure/collapse and five predeclared confidence scores.",
        "baseline": "Sealed formal JCGS C07/C08 exact-set@2deg 0.2008074995/0.1572107895, including K3/K4 exact-set@2/4/10deg equal to zero.",
        "fallback": "If formal reproduction or population integrity fails, seal system FAIL. If both phase and gap fail consistently for K>=3, replace complete-set regression with event-plus-relation factorization; no post-processing, retraining or planner compensation in this run.",
        "user_authorization": card["approval"],
        "acceptance_criteria": [
            "Exactly 10 C07 and 10 C08 worlds, 21548/24394 observations and 45504/51537 exits, with exact joins.",
            "Reproduce sealed formal ensemble exact-set@2deg on each split to absolute error <=1e-12.",
            "Report per-seed and ensemble cardinality confusion, oracle count and target-aligned ensemble ceilings at 2/4/10 degrees.",
            "For true-count K1-K4, report phase and gap errors and oracle-phase/oracle-gaps exact sets at 2/4/10 degrees.",
            "Select only predeclared confidence thresholds on C07 at exit precision >=0.995 and transfer unchanged once to C08.",
            "Five unit tests pass; RAM <=8 GiB; zero optimizer, new inference, checkpoint selection, C09/C10/M-TARE/graph/planner; sources unchanged."
        ],
        "expected_counts": {"c07_worlds": 10, "c08_worlds": 10, "c07_observations": 21548, "c08_observations": 24394, "c07_exits": 45504, "c08_exits": 51537, "seeds": 3, "confidence_features": 5, "tolerances": 3, "unit_tests": 5, "optimizer_steps": 0, "new_model_inference_observations": 0, "checkpoint_selection_observations": 0, "c09_worlds_read": 0, "c10_worlds_read": 0, "mtare_worlds_read": 0, "graph_replays": 0},
        "expected_evidence": [
            "Count, seed-alignment, phase, gap and confidence attribution summary with explicit cross-split diagnosis.",
            "Decoder and component CSV tables plus paper-ready PNG/PDF/SVG and figure source.",
            "Unit-test and raw attribution logs, environment, RUN_STATE and SHA-256 seal."
        ],
        "estimated_cost": {"compute": "CPU-only deterministic frozen-output attribution", "wall_time_hours": 0.1, "host_ram_gb": 8, "gpu_memory_gb": 0, "disk_gb": 0.1},
        "frozen_inputs": {path: sha256(PROJECT_ROOT / path) for path in sorted(inputs)},
        "frozen_tools": {name: {"path": path, "sha256": sha256(PROJECT_ROOT / path)} for name, path in tools.items()},
        "working_directory": str(PROJECT_ROOT),
        "command": ["/usr/bin/timeout", "7200s", PYTHON, "tools/v3/run_gse_joint_cyclic_gap_simplex_failure_attribution_v1.py", "--spec", str(SPEC), "--run-dir", str(PROJECT_ROOT / f"results/gate3_semantics/{RUN_ID}")]
    }
    write_json(SPEC, spec)
    print(SPEC.relative_to(PROJECT_ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
