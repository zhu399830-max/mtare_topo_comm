#!/usr/bin/env python3
"""Freeze the axis-anchored event-relation three-seed training spec."""
from __future__ import annotations

import hashlib

from _bootstrap import PROJECT_ROOT
from mtare_topo.governance import load_json, validate_data_card, write_json


RUN_ID = "gate3_20260829_gse_axis_anchored_event_relation_three_seed_training_v1_seed0"
SPEC = PROJECT_ROOT / "configs/v3/gate3/gse_axis_anchored_event_relation_three_seed_training_v1.json"
DATA_CARD = "configs/v3/gate3/data_cards/gse_axis_anchored_event_relation_three_seed_training_v1.json"
PYTHON = "/home/zeng-workstation/.local/share/mtare_topo_comm/envs/phase3_torch290_cu129_zarr2187_v1/bin/python"


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""): digest.update(block)
    return digest.hexdigest()


def main() -> int:
    if SPEC.exists(): raise RuntimeError("axis-anchored training spec already exists")
    card = load_json(PROJECT_ROOT / DATA_CARD); validation = validate_data_card(card)
    if not validation.passed: raise RuntimeError(f"axis-anchored training card invalid: {validation.errors}")
    dataset = "results/gate2_representation/gate2_20260824_gse_deduplicated_dataset_export_v1_seed0"
    readiness = "results/gate3_semantics/gate3_20260829_gse_axis_anchored_event_relation_readiness_v2_seed0"
    inputs = [
        DATA_CARD, f"{dataset}/RUN_STATE.json", f"{dataset}/metrics/summary.json",
        f"{dataset}/artifacts/sequence_manifest.jsonl", f"{dataset}/artifacts/association_pairs_numeric.jsonl",
        f"{dataset}/artifacts/shard_manifest.json", f"{dataset}/artifacts/evidence_sha256.txt",
        f"{readiness}/RUN_STATE.json", f"{readiness}/metrics/summary.json", f"{readiness}/metrics/readiness/summary.json", f"{readiness}/artifacts/evidence_sha256.txt",
        "configs/v3/gate2/environments/phase3_torch290_cu129_zarr2187_v1r2.json",
    ]
    tools = {
        "model":"src/mtare_topo/representation/gse_axis_anchored_event_relation.py",
        "encoder":"src/mtare_topo/representation/phase3_structural_semantics.py",
        "teacher_and_sampler":"src/mtare_topo/data/gse_axis_anchored_event_relation_training.py",
        "trainer":"tools/v3/train_gse_axis_anchored_event_relation_v1.py",
        "evaluator":"tools/v3/evaluate_gse_axis_anchored_event_relation_selection_v1.py",
        "runner":"tools/v3/run_gse_axis_anchored_event_relation_three_seed_training_v1.py",
        "freezer":"tools/v3/freeze_gse_axis_anchored_event_relation_three_seed_training_spec_v1.py",
        "geometry_baseline":"src/mtare_topo/semantics/range_geometry_baseline.py",
        "metrics":"src/mtare_topo/evaluation/gse_metrics.py",
        "model_tests":"tests/v3/unit/test_gse_axis_anchored_event_relation.py",
        "teacher_tests":"tests/v3/unit/test_gse_axis_anchored_event_relation_training.py",
        "governance":"src/mtare_topo/governance.py", "preflight":"tools/v3/preflight.py", "create_run":"tools/v3/create_run.py",
    }
    spec = {
        "schema_version":"v3_run_spec_v1", "gate":3, "execution_phase":3, "date":"20260829", "slug":"gse_axis_anchored_event_relation_three_seed_training_v1", "seed":0, "operation":"training", "data_card":DATA_CARD,
        "question":"Does the frozen axis-anchored multi-label method improve five-event semantics and continuous geometry while learning safe temporal relations, current branches, association and uncertainty refusal on C07 and zero-adaptation C08?",
        "method":"Three from-scratch 819067-parameter seeds. Every C01-C06 row trains event/relation/branch/global geometry; deterministic per-world identity-balanced batches separately train place/branch descriptors. Complete checkpoint selection and all thresholds use C07 only; C08 transfers once.",
        "baseline":"Frozen five-event macro-F1 0.6879041032, deterministic same-scan RangeGeometryBaseline, categorical relation V1/V1R failure and sealed rule/exit-token association evidence.",
        "fallback":"Any scientific gate failure seals checkpoints and predictions then stops before graph; one read-only attribution may follow, but no threshold relaxation, C08 calibration, seed selection, extra epoch or retry.",
        "user_authorization":card["approval"],
        "acceptance_criteria":[
            "Exactly three seeds, ten epochs each, 11370 core plus 860 descriptor optimizer steps per seed; 819067 parameters.",
            "Ensemble five-event macro-F1 >=0.7379041032 on C07 and C08.",
            "All three temporal relation channels transfer with precision>=0.98 and recall>=0.25; current branch field precision>=0.995 and recall>=0.50.",
            "Four-field geometry improves the deterministic baseline by >=10% macro with no field regression >5%; axis error <=10 degrees.",
            "Combined place/branch association transfers with precision>=0.98, false merge<=1% and recall>=0.25.",
            "Uncertainty-refused structural events transfer with precision>=0.98, false acceptance<=1% and recall>=0.25.",
            "C08 checkpoint observations zero; full 33083-entry source seal; C09/C10/M-TARE/graph/planner zero; sources unchanged."
        ],
        "expected_counts":{"fit_worlds":60,"c07_worlds":10,"c08_worlds":10,"fit_observations":142184,"c07_observations":21548,"c08_observations":24394,"fit_descriptor_rows_per_epoch":6347,"c07_descriptor_rows":928,"c08_descriptor_rows":1173,"parameters":819067,"seeds":3,"epochs_per_seed":10,"core_optimizer_steps_per_seed":11370,"descriptor_optimizer_steps_per_seed":860,"optimizer_steps_per_seed":12230,"optimizer_steps":36690,"c08_checkpoint_observations":0,"c09_worlds_read":0,"c10_worlds_read":0,"mtare_worlds_read":0,"graph_replays":0},
        "expected_evidence":["Three checkpoints, histories, compact C07/C08 prediction archives and exact optimizer populations.","C07 threshold selection and unchanged C08 transfer for events, relations, branch field, geometry, association and refusal.","Per-world metrics, PNG/PDF/SVG/source, environment, commands, raw logs, RUN_STATE and SHA-256 seal."],
        "estimated_cost":card["estimated_cost"],
        "frozen_inputs":{path:sha256(PROJECT_ROOT/path) for path in sorted(inputs)},
        "frozen_tools":{name:{"path":path,"sha256":sha256(PROJECT_ROOT/path)} for name,path in tools.items()},
        "working_directory":str(PROJECT_ROOT),
        "command":["/usr/bin/timeout","21600s",PYTHON,"tools/v3/run_gse_axis_anchored_event_relation_three_seed_training_v1.py","--spec",str(SPEC),"--run-dir",str(PROJECT_ROOT/f"results/gate3_semantics/{RUN_ID}")],
    }
    write_json(SPEC, spec); print(SPEC.relative_to(PROJECT_ROOT)); return 0


if __name__ == "__main__": raise SystemExit(main())
