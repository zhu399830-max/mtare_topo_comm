#!/usr/bin/env python3
from __future__ import annotations

import hashlib
from _bootstrap import PROJECT_ROOT
from mtare_topo.governance import load_json, validate_data_card, write_json

RUN_ID = "gate3_20260829_gse_sparse_relation_objective_recall_corrective_v1r_seed0"
SPEC = PROJECT_ROOT / "configs/v3/gate3/gse_sparse_relation_objective_recall_corrective_v1r.json"
CARD = "configs/v3/gate3/data_cards/gse_sparse_relation_objective_recall_corrective_v1r.json"
PYTHON = "/home/zeng-workstation/.local/share/mtare_topo_comm/envs/phase3_torch290_cu129_zarr2187_v1/bin/python"


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    if SPEC.exists():
        raise RuntimeError("objective-recall corrective V1R spec exists")
    card = load_json(PROJECT_ROOT / CARD)
    validation = validate_data_card(card)
    if not validation.passed:
        raise RuntimeError(str(validation.errors))
    dataset = "results/gate2_representation/gate2_20260824_gse_deduplicated_dataset_export_v1_seed0"
    failed_v2 = "results/gate3_semantics/gate3_20260829_gse_sparse_circular_relation_transport_three_seed_training_v2_seed0"
    failed_v2r = "results/gate3_semantics/gate3_20260829_gse_sparse_circular_relation_transport_three_seed_training_v2r_seed0"
    failed_v2r2 = "results/gate3_semantics/gate3_20260829_gse_sparse_circular_relation_transport_three_seed_training_v2r2_seed0"
    corrective_v1 = "results/gate3_semantics/gate3_20260829_gse_sparse_relation_objective_recall_corrective_v1_seed0"
    inputs = [
        CARD,
        f"{dataset}/RUN_STATE.json",
        f"{dataset}/metrics/summary.json",
        f"{dataset}/artifacts/evidence_sha256.txt",
        f"{dataset}/artifacts/sequence_manifest.jsonl",
        *[item for run in (failed_v2, failed_v2r, failed_v2r2, corrective_v1) for item in (f"{run}/RUN_STATE.json", f"{run}/metrics/summary.json", f"{run}/artifacts/evidence_sha256.txt")],
        "configs/v3/gate2/environments/phase3_torch290_cu129_zarr2187_v1r2.json",
    ]
    tools = {
        "evaluator": "tools/v3/evaluate_gse_sparse_circular_relation_transport_v2.py",
        "evaluation_tests": "tests/v3/unit/test_evaluate_gse_sparse_circular_relation_transport_v2.py",
        "training_tests": "tests/v3/unit/test_train_gse_sparse_circular_relation_transport_v2.py",
        "model_tests": "tests/v3/unit/test_gse_sparse_circular_relation_transport.py",
        "count_tests": "tests/v3/unit/test_gse_sparse_relation_cardinality_corrective.py",
        "sparse_loader": "tools/v3/train_gse_sparse_circular_relation_transport_v2.py",
        "base_loader": "tools/v3/train_gse_axis_anchored_event_relation_v1.py",
        "executor": "tools/v3/execute_gse_sparse_relation_objective_recall_corrective_v1r.py",
        "runner": "tools/v3/run_gse_sparse_relation_objective_recall_corrective_v1r.py",
        "freezer": "tools/v3/freeze_gse_sparse_relation_objective_recall_corrective_spec_v1r.py",
        "governance": "src/mtare_topo/governance.py",
        "preflight": "tools/v3/preflight.py",
        "create_run": "tools/v3/create_run.py",
    }
    spec = {
        "schema_version": "v3_run_spec_v1",
        "gate": 3,
        "execution_phase": 3,
        "date": "20260829",
        "slug": "gse_sparse_relation_objective_recall_corrective_v1r",
        "seed": 0,
        "operation": "audit",
        "data_card": CARD,
        "question": "Does the evaluator apply the complete objective Teacher denominator consistently to tokens, causal relations and uncertainty-refused structural events?",
        "method": "Read all C01-C08 causal Teacher histories once, preserve the V1 objective set-relation proof, and prove with missing-endpoint constructions that token, relation and structural-refusal positives/FN/recall/F1 all use the full Teacher denominator while precision remains candidate-based.",
        "baseline": "The sealed V2 evaluator used candidate-only relation positives and could inflate recall; its system-failed runs contain zero scientific conclusion.",
        "fallback": "Any population/join/accounting/source defect stops. No training may restart until this corrective passes.",
        "user_authorization": card["approval"],
        "acceptance_criteria": [
            "Exactly 60 fit, 10 C07 and 10 C08 worlds; 142184/21548/24394 observations and 448152/66752/77490 valid causal pairs.",
            "Persistent, reveal and withdraw have independently counted positive populations on every split.",
            "A missing-endpoint proof records three FN out of five objective positives while retaining candidate precision 1.0.",
            "Token, relation and structural-refusal FN, positives, recall and F1 use the objective denominator; all unit tests pass.",
            "Zero optimizer, checkpoint, model forward, threshold selection, C09/C10/M-TARE/graph/planner and unchanged inputs/tools.",
        ],
        "expected_counts": {"fit_worlds": 60, "c07_worlds": 10, "c08_worlds": 10, "fit_observations": 142184, "c07_observations": 21548, "c08_observations": 24394, "fit_valid_pairs": 448152, "c07_valid_pairs": 66752, "c08_valid_pairs": 77490, "optimizer_steps": 0, "model_forward_observations": 0, "checkpoints_written": 0, "c09_worlds_read": 0, "c10_worlds_read": 0, "mtare_worlds_read": 0, "graph_replays": 0},
        "expected_evidence": ["Exact per-world and per-split objective relation populations.", "Synthetic missing-endpoint comparison of invalid and corrected recall.", "Unit tests, PNG/PDF/SVG/source, logs, environment, RUN_STATE and SHA-256 seal."],
        "estimated_cost": {"compute": "CPU-only Teacher metadata read; zero model forward", "wall_time_hours": 0.2, "host_ram_gb": 8, "gpu_memory_gb": 0, "disk_gb": 0.1},
        "frozen_inputs": {path: sha(PROJECT_ROOT / path) for path in sorted(inputs)},
        "frozen_tools": {name: {"path": path, "sha256": sha(PROJECT_ROOT / path)} for name, path in tools.items()},
        "working_directory": str(PROJECT_ROOT),
        "command": ["/usr/bin/timeout", "1800s", PYTHON, "tools/v3/run_gse_sparse_relation_objective_recall_corrective_v1r.py", "--spec", str(SPEC), "--run-dir", str(PROJECT_ROOT / f"results/gate3_semantics/{RUN_ID}")],
    }
    write_json(SPEC, spec)
    print(SPEC.relative_to(PROJECT_ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
