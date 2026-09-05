#!/usr/bin/env python3
"""Freeze the Data Card and run spec for the consensus/metric corrective."""

from __future__ import annotations

from copy import deepcopy
import hashlib

from _bootstrap import PROJECT_ROOT
from mtare_topo.governance import load_json, write_json


RUN_ID = "gate3_20260828_gse_factorized_consensus_metric_corrective_v1_seed0"
CARD = PROJECT_ROOT / "configs/v3/gate3/data_cards/gse_factorized_consensus_metric_corrective_v1.json"
SPEC = PROJECT_ROOT / "configs/v3/gate3/gse_factorized_consensus_metric_corrective_v1.json"
SOURCE_CARD = PROJECT_ROOT / "configs/v3/gate3/data_cards/gse_factorized_association_c09_qualification_v1.json"
CAPACITY_CARD = PROJECT_ROOT / "configs/v3/gate3/data_cards/gse_factorized_association_capacity_v1r.json"
DATASET = "results/gate2_representation/gate2_20260824_gse_deduplicated_dataset_export_v1_seed0"
TEACHER = "results/gate2_representation/gate2_20260827_gse_corrected_causal_teacher_manifest_v1r_seed0"
TOKEN = "results/gate3_semantics/gate3_20260826_gse_exit_token_association_corrective_v2_seed0"
CAPACITY = "results/gate3_semantics/gate3_20260827_gse_factorized_association_capacity_v1r_seed0"
C09_FAIL = "results/gate3_semantics/gate3_20260827_gse_factorized_association_c09_qualification_v1_seed0"
PYTHON = "/home/zeng-workstation/.local/share/mtare_topo_comm/envs/phase3_torch290_cu129_zarr2187_v1/bin/python"


def _sha(path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    card = deepcopy(load_json(SOURCE_CARD))
    capacity_card = load_json(CAPACITY_CARD)
    card.update({
        "card_id": "gse_factorized_consensus_metric_corrective_v1",
        "status": "APPROVED_FOR_ONE_IMMUTABLE_GSE_FACTORIZED_CONSENSUS_METRIC_CORRECTIVE_V1",
        "purpose": "Select a robust multi-seed consensus and runtime metric-locality contract on C07-C08 only, then apply that sealed contract once to archived C09 scores.",
    })
    card["worlds"]["train"] = deepcopy(capacity_card["worlds"]["train"])
    card["worlds"]["validation"] = sorted(set(
        capacity_card["worlds"]["validation"] + card["worlds"]["validation"]
    ))
    card["approval"] = {
        "status": "APPROVED", "approved_by": "user-standing-authorization",
        "approved_at": "2026-08-28T00:30:00+08:00", "authorized_gates": [3],
        "authorized_operations": ["threshold_calibration"],
        "scope": "One immutable two-process C01-C08 selector followed by one archived-C09 applicator; no model update, C10, M-TARE, graph or planner access.",
        "confirmation_reference": "User explicitly instructed automatic best-choice execution without routine approval prompts.",
    }
    card["sampling"] = {
        "raw_frame_count": 188_126 + 24_462,
        "effective_sample_count": 45_942 + 24_462,
        "effective_structure_event_count": 12_924,
        "spatial_interval_m": 1.0,
        "rule": "Five-frame causal sequences are sampled at 1 m arc spacing; selection uses all C07-C08 balanced aliases and same-world strictly-past runtime candidates, then validation applies the frozen rule to all archived C09 pairs.",
        "independent_units": "Twenty C07-C08 worlds select the deployment rule; ten disjoint C09 worlds evaluate it once.",
        "selection_balanced_domain": "274 decision identities and 548 identity-balanced pairs already sealed upstream.",
        "selection_runtime_domain": "8839 decision queries; 8604 with a prior candidate; 8565 with a same-identity prior; 9380 pairs = 8565 positive + 815 negative.",
        "validation_balanced_domain": "130 decision identities and 260 balanced pairs.",
        "validation_runtime_domain": "4085 decision queries and 4201 pairs = 3955 positive + 246 negative.",
        "temporal_context": "Current plus at most four causal traversal rows; runtime candidates are same-world and strictly past.",
        "structure_event_counts": {"junction": 10_081, "terminal": 2_843},
    }
    card["source"] = {
        "dataset_run": DATASET, "corrected_teacher_run": TEACHER,
        "frozen_token_feature_run": TOKEN, "unified_capacity_run": CAPACITY,
        "archived_c09_failure_run": C09_FAIL,
        "raw_sources": [
            "Sealed C01-C08 unified observations, exit tokens, balanced selection outputs and runtime metadata.",
            "Three frozen Factorized verifier checkpoints, normalizations and thresholds.",
            "Archived C09 balanced/runtime pairs and per-seed scores from the immutable failed qualification.",
        ],
        "license_or_allowed_use": "Local research use with sealed provenance.",
    }
    card["trajectories"] = [
        row for row in capacity_card["trajectories"]
        if str(row["world"]).endswith(("_C07", "_C08"))
    ] + deepcopy(load_json(SOURCE_CARD)["trajectories"])
    card["split"] = {
        "model_fit": "Frozen upstream on C01-C06; no fit occurs here.",
        "selection": "Exactly C07-C08. Process 1 has no C09 argument or input path.",
        "validation": "Exactly archived C09, opened only by process 2 after calibration.json exists and process 1 exits.",
        "strict_test": "C10 and all M-TARE/closed-loop worlds remain unread.",
        "world_disjoint": True, "trajectory_disjoint": True,
        "historical_pollution_audit": "C09 exposed the single-seed safety failure and motivated the method-family revision, but cannot select votes or distance. Exact values are selected solely by the C07-C08 process. This is development validation, not the strict test claim.",
        "leakage_audit": "The runner enforces process separation; selector summary must report c09_worlds_read=0 before launching the applicator.",
    }
    card["teacher"] = deepcopy(capacity_card["teacher"])
    card["teacher"].update({
        "source": "Sealed corrected TNG/spline/mesh Teacher, sealed C07-C08 balanced outputs/runtime metadata, and archived C09 pair labels.",
        "association": "Same structural identity is positive; a different identity is negative.",
        "runtime_candidates": "Same-world, strictly-past decision observations within 16 m, nearest view retained per candidate identity.",
        "valid_mask": "All association-valid junction/terminal queries are retained: 8839 on C07-C08 and 4085 on C09; no world, query, identity or pair is silently removed.",
        "planner_consistency_plan": "The frozen rule will later be evaluated offline before any planner use; rejected or ambiguous matches remain provisional, and an edge may be committed only after physical traversal.",
        "student_forbidden_fields": ["identity", "world ID", "absolute position as classifier input", "future frames", "C09 during selection"],
    })
    card["methods"] = {
        "main": "Count accept votes from the three frozen Factorized models. On runtime candidates additionally reject pairs farther than a selected metric cap.",
        "selection": "Grid votes={1,2,3}, distance={0.5,...,16.0} m. Require C07-C08 precision>=0.995, false-accept<=0.005, recall>=0.25 and balanced family precision>=0.98/recall>=0.10; maximize runtime recall, then balanced recall, then smaller cap, then larger vote count.",
        "baseline": "The previously sealed single-seed C09 qualification is the direct comparison and documented failure mode.",
        "fallback": "If the once-frozen C09 corrective misses any original safety/recall gate, stop the Factorized association claim; no second calibration or planner compensation.",
    }
    card["metrics_and_pre_registered_gates"] = {
        "selection": "The deterministic C07-C08 rule must select 2-of-3 votes and a 4.0 m runtime cap from at least one margin-qualified configuration.",
        "balanced": "C09 aggregate precision>=0.98, false accept<=0.01, recall>=0.25; every family precision>=0.95 and recall>=0.10.",
        "physical_only": "Same balanced gate after excluding the one singleton augmentation positive.",
        "runtime": "C09 aggregate precision>=0.98, false accept<=0.01 and recall>=0.25; S02/S03 zero-negative families explicitly unidentifiable.",
        "population": "Exact C07-C08 45942 sequences/8839 queries/9380 runtime pairs and C09 24462 sequences/4085 queries/260 balanced/4201 runtime pairs.",
        "forbidden": "Optimizer/model/checkpoint selection, C10, strict test, M-TARE, graph and planner operations all zero.",
    }
    card["estimated_cost"] = {
        "compute": "CPU-only 28140 small-head selection inferences plus archived C09 boolean application",
        "disk_gb": .1, "host_ram_gb": 4, "gpu_memory_gb": 0, "wall_time_hours": .5,
    }
    card["retention"] = "Retain the selection grid/calibration, C07-C08 scores, C09 decisions/metrics, PNG/PDF/SVG/source, environment, commands, logs, RUN_STATE and exact SHA-256 seal."
    card["failure_policy"] = "Any input/process separation/population/selection/resource drift or C09 gate failure seals FAIL; no retry, grid change, masking, C10 read or planner tuning."
    card["evidence"] = {
        "machine_metrics": "Complete selection and validation metrics with per-family support.",
        "complete_visual_review": "Selection-versus-C09 precision and recall in PNG/PDF/SVG with machine source.",
        "failure_policy": card["failure_policy"],
    }
    write_json(CARD, card)

    inputs = [
        f"{DATASET}/RUN_STATE.json", f"{DATASET}/metrics/summary.json", f"{DATASET}/artifacts/evidence_sha256.txt", f"{DATASET}/artifacts/sequence_manifest.jsonl",
        f"{TEACHER}/RUN_STATE.json", f"{TEACHER}/metrics/summary.json", f"{TEACHER}/artifacts/evidence_sha256.txt", f"{TEACHER}/artifacts/teacher_observations.jsonl",
        f"{TOKEN}/RUN_STATE.json", f"{TOKEN}/metrics/summary.json", f"{TOKEN}/artifacts/evidence_sha256.txt", f"{TOKEN}/artifacts/pair_cache/pairs.npz",
        f"{CAPACITY}/RUN_STATE.json", f"{CAPACITY}/metrics/summary.json", f"{CAPACITY}/artifacts/evidence_sha256.txt",
        f"{C09_FAIL}/RUN_STATE.json", f"{C09_FAIL}/metrics/summary.json", f"{C09_FAIL}/metrics/factorized_association_c09_qualification.json", f"{C09_FAIL}/artifacts/evidence_sha256.txt",
        f"{C09_FAIL}/artifacts/c09_balanced_alias_pairs.npz", f"{C09_FAIL}/artifacts/c09_runtime_candidate_pairs.npz",
    ]
    for seed in range(3):
        inputs.extend([
            f"{TOKEN}/artifacts/models/seed{seed}/frozen_exit_token_outputs.npz",
            f"{CAPACITY}/artifacts/unified_observation/seed{seed}_unified_observation_features.npy",
            f"{CAPACITY}/artifacts/models/seed{seed}/summary.json",
            f"{CAPACITY}/artifacts/models/seed{seed}/full_route_conditioned/selection_outputs.npz",
            f"{CAPACITY}/artifacts/models/seed{seed}/full_route_conditioned/best.pt",
            f"{CAPACITY}/artifacts/models/seed{seed}/full_route_conditioned/normalization.npz",
            f"{C09_FAIL}/artifacts/seed{seed}_c09_qualification_scores.npz",
        ])
    tools = {
        "data_card": str(CARD.relative_to(PROJECT_ROOT)),
        "method_plan": "docs/GSE_GRAPH_FACTORIZED_RESEARCH_PLAN_V2.md",
        "qualification_helpers": "src/mtare_topo/evaluation/gse_factorized_qualification.py",
        "factorized_model": "src/mtare_topo/representation/gse_factorized_association.py",
        "teacher_contract": "src/mtare_topo/teacher/gse_factorized_association_teacher.py",
        "selector": "tools/v3/select_gse_factorized_consensus_metric_v1.py",
        "applicator": "tools/v3/apply_gse_factorized_consensus_metric_c09_v1.py",
        "runner": "tools/v3/run_gse_factorized_consensus_metric_corrective_v1.py",
        "freezer": "tools/v3/freeze_gse_factorized_consensus_metric_corrective_spec_v1.py",
        "tests": "tests/v3/unit/test_gse_factorized_qualification.py",
        "governance": "src/mtare_topo/governance.py",
        "preflight": "tools/v3/preflight.py", "create_run": "tools/v3/create_run.py",
    }
    criteria = card["metrics_and_pre_registered_gates"]
    spec = {
        "schema_version": "v3_run_spec_v1", "gate": 3, "execution_phase": 3,
        "date": "20260828", "slug": "gse_factorized_consensus_metric_corrective_v1", "seed": 0,
        "operation": "threshold_calibration",
        "question": "Can a C07-C08-selected multi-seed consensus and metric-locality contract recover safe Factorized association on C09?",
        "method": card["methods"]["main"], "baseline": card["methods"]["baseline"],
        "fallback": card["methods"]["fallback"],
        "data_card": str(CARD.relative_to(PROJECT_ROOT)), "config_path": str(CARD.relative_to(PROJECT_ROOT)),
        "user_authorization": deepcopy(card["approval"]),
        "acceptance_criteria": [criteria[name] for name in ("selection", "balanced", "physical_only", "runtime", "population", "forbidden")],
        "expected_counts": {
            "selection_worlds": 20, "selection_sequences": 45_942,
            "selection_runtime_queries": 8_839, "selection_runtime_pairs": 9_380,
            "validation_worlds": 10, "validation_sequences": 24_462,
            "validation_balanced_pairs": 260, "validation_runtime_queries": 4_085,
            "validation_runtime_pairs": 4_201, "association_inference_pairs": 28_140,
            "optimizer_steps": 0, "model_updates": 0, "c10_worlds_read": 0, "mtare_worlds_read": 0,
        },
        "expected_evidence": ["Selection grid/calibration, C07-C08 scores, C09 decisions/metrics, PNG/PDF/SVG/source JSON, environment, commands, logs, RUN_STATE and exact SHA-256 seal."],
        "estimated_cost": card["estimated_cost"],
        "frozen_inputs": {path: _sha(PROJECT_ROOT / path) for path in inputs},
        "frozen_tools": {name: {"path": path, "sha256": _sha(PROJECT_ROOT / path)} for name, path in tools.items()},
        "working_directory": str(PROJECT_ROOT),
        "command": [
            "/usr/bin/timeout", "--signal=INT", "--kill-after=30s", "1900s", PYTHON,
            "tools/v3/run_gse_factorized_consensus_metric_corrective_v1.py",
            "--spec", str(SPEC), "--run-dir", str(PROJECT_ROOT / "results/gate3_semantics" / RUN_ID),
        ],
    }
    write_json(SPEC, spec)
    print(CARD.relative_to(PROJECT_ROOT))
    print(SPEC.relative_to(PROJECT_ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
