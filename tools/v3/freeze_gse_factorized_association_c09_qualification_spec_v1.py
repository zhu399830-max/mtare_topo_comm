#!/usr/bin/env python3
"""Freeze the Data Card and spec for one C09 Factorized qualification."""

from __future__ import annotations

from copy import deepcopy
import hashlib

from _bootstrap import PROJECT_ROOT
from mtare_topo.governance import load_json, write_json


RUN_ID = "gate3_20260827_gse_factorized_association_c09_qualification_v1_seed0"
CARD = PROJECT_ROOT / "configs/v3/gate3/data_cards/gse_factorized_association_c09_qualification_v1.json"
SPEC = PROJECT_ROOT / "configs/v3/gate3/gse_factorized_association_c09_qualification_v1.json"
SOURCE_CARD = PROJECT_ROOT / "configs/v3/gate3/data_cards/gse_risk_calibrated_perception_validation_v2.json"
DATASET = "results/gate2_representation/gate2_20260824_gse_deduplicated_dataset_export_v1_seed0"
TEACHER = "results/gate2_representation/gate2_20260824_gse_teacher_manifest_v1_seed0"
TRAINING = "results/gate2_representation/gate2_20260824_gse_graph_three_seed_training_v1r_seed0"
RISK = "results/gate3_semantics/gate3_20260825_gse_risk_calibrated_perception_validation_v2_seed0"
CAPACITY = "results/gate3_semantics/gate3_20260827_gse_factorized_association_capacity_v1r_seed0"
PYTHON = "/home/zeng-workstation/.local/share/mtare_topo_comm/envs/phase3_torch290_cu129_zarr2187_v1/bin/python"


def _sha(path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    card = deepcopy(load_json(SOURCE_CARD))
    card.update({
        "card_id": "gse_factorized_association_c09_qualification_v1",
        "status": "APPROVED_FOR_ONE_IMMUTABLE_GSE_FACTORIZED_ASSOCIATION_C09_QUALIFICATION_V1",
        "purpose": "Evaluate the frozen unified-slope route-conditioned association models and C01-C08 thresholds on unseen C09 balanced structural aliases and actual strictly-past runtime candidates without selection.",
    })
    card["approval"] = {
        "status": "APPROVED", "approved_by": "user-standing-authorization",
        "approved_at": "2026-08-27T23:59:59+08:00", "authorized_gates": [3],
        "authorized_operations": ["threshold_calibration"],
        "scope": "One immutable read-only C09 audit over 10 worlds/24462 sequences using already frozen V1R models, normalizations and thresholds; no threshold calibration, optimization, C10, M-TARE, graph or planner access.",
        "confirmation_reference": "User explicitly instructed automatic evidence-supported decisions without routine approval prompts.",
    }
    card["sampling"] = {
        "raw_frame_count": 32678, "effective_sample_count": 24462,
        "effective_structure_event_count": 4085, "spatial_interval_m": 1.0,
        "independent_units": "Ten unseen C09 procedural topology worlds; decision identity is the balanced-domain reporting unit and world is the generalization unit.",
        "balanced_domain": "130 junction/terminal identities, one positive and one same-event/degree objective-geometry-nearest hard negative per identity: 260 pairs.",
        "runtime_domain": "4085 association-valid junction/terminal queries; same-world strictly-past decision observations within 16 m; nearest observation per candidate identity: 4201 pairs = 3955 positive + 246 negative.",
        "temporal_context": "Current plus at most four causal rows from the same directed traversal; runtime candidates are strictly earlier in world_sequence_row.",
        "structure_event_counts": {"junction": 3216, "terminal": 869},
        "singleton_policy": "One full-history C09 terminal singleton is retained and excluded in a separately required physical-only balanced audit.",
    }
    card["source"] = {
        "dataset_run": DATASET, "teacher_run": TEACHER, "frozen_gse_training_run": TRAINING,
        "risk_calibrated_slope_run": RISK, "unified_capacity_run": CAPACITY,
        "raw_sources": [
            "Three sealed C09 validation output archives, each with 24462 complete GSE observations and six exit tokens.",
            "Three sealed C09 risk-calibrated slope archives using the fixed C07-C08 residual scale 0.89.",
            "Sealed C09 Teacher/sequence metadata and validation sensor positions for pair labels and runtime candidate construction only.",
            "Three sealed V1R full-route checkpoints, normalizations and thresholds selected only on C01-C08.",
        ],
        "license_or_allowed_use": "Local research use of project procedural worlds and locally trained frozen outputs with complete provenance.",
    }
    card["trajectories"] = [row for row in card["trajectories"] if row["world"].endswith("_C09")]
    card["split"] = {
        "model_fit_and_selection": "Already sealed upstream: C01-C06 fit and C07-C08 checkpoint/threshold selection. This operation does not reread their sensor payloads.",
        "validation": "Exactly ten C09 worlds and all 24462 sequences; no world, family, query or pair selection.",
        "strict_test": "C10 and all M-TARE/closed-loop worlds remain unread.",
        "world_disjoint": True, "trajectory_disjoint": True,
        "historical_pollution_audit": "C09 was previously used only for perception qualification, not association training or threshold selection. Association V1R fit/checkpoint/thresholds were frozen on C01-C08 before this audit.",
        "leakage_audit": "Identity/event/objective geometry construct labels and balanced negatives only; model inputs contain frozen learned observation/token fields and causal learned geometry. Absolute position only gates runtime candidates at <=16 m and is never a classifier feature.",
    }
    card["teacher"] = {
        "balanced_positive": "Same C09 decision identity using deterministic cross-view priority; 71 different edge, 55 reverse view, 3 distinct observation and 1 explicit singleton unit.",
        "balanced_negative": "Different C09 identity with same event and incident degree, nearest in masked objective geometry profile.",
        "runtime": "Same identity is positive; different identity is negative after same-world strict-past <=16 m candidate gating and per-identity nearest-view reduction.",
        "student_forbidden_fields": ["identity", "parent_id", "world_id", "Teacher event", "objective geometry", "absolute pose", "future frame"],
    }
    card["methods"] = {
        "main": "For each frozen seed, build the final 146D observation by replacing only slope column 10 with the sealed risk-calibrated value; compute the unchanged route-conditioned pair relation; apply the sealed V1R normalization, checkpoint and threshold.",
        "baseline": "The frozen threshold itself is evaluated on two independently shaped domains: identity-balanced structural aliases and actual runtime candidates; physical-only balanced results exclude the singleton augmentation.",
        "fallback": "If any seed fails balanced, physical-only or runtime aggregate safety, stop the Factorized association claim and do not enter offline graph or planner tuning.",
    }
    card["metrics_and_pre_registered_gates"] = {
        "balanced": "All three seeds: aggregate precision>=0.98, false accept<=0.01, recall>=0.25; every family nonempty precision>=0.95 and recall>=0.10.",
        "physical_only": "Same complete balanced gate after removing the one singleton augmentation positive.",
        "runtime": "All three seeds: aggregate precision>=0.98, false accept<=0.01 and recall>=0.25 at the frozen threshold.",
        "runtime_family_reporting": "Report positive/negative support, accepted, precision, recall and rejection for every family. S02/S03 have zero runtime negatives and must be marked precision-unidentifiable, not reported as evidence of perfect safety.",
        "population": "Exact 10 worlds/24462 sequences/130 balanced identities/260 balanced pairs/4085 runtime queries/4201 runtime pairs.",
        "forbidden": "Checkpoint and threshold selection, optimizer/model updates, C10, strict-test, M-TARE, graph and planner operations all zero.",
    }
    card["estimated_cost"] = {"compute": "CPU-only frozen small-head scoring of 13383 pairs plus read-only feature assembly", "disk_gb": .1, "host_ram_gb": 4, "gpu_memory_gb": 0, "wall_time_hours": .5}
    card["retention"] = "Retain balanced/runtime pair manifests, three unified C09 arrays and score archives, per-seed/family metrics, PNG/PDF/SVG/source JSON, environment, command, raw log, RUN_STATE and SHA-256 seal."
    card["failure_policy"] = "Any source/identity/count/causality/interface/threshold/resource drift or seed safety failure seals FAIL; no retry, threshold selection, world deletion or family masking."
    card["evidence"] = {"machine_metrics": "Frozen-threshold balanced, physical-only and runtime metrics for all seeds/families with explicit zero-negative identifiability.", "complete_visual_review": "C09 recall and precision by seed/domain in PNG/PDF/SVG with machine source.", "failure_policy": card["failure_policy"]}
    write_json(CARD, card)

    inputs = [
        f"{DATASET}/RUN_STATE.json", f"{DATASET}/metrics/summary.json", f"{DATASET}/artifacts/evidence_sha256.txt", f"{DATASET}/artifacts/sequence_manifest.jsonl",
        f"{TEACHER}/RUN_STATE.json", f"{TEACHER}/metrics/summary.json", f"{TEACHER}/artifacts/evidence_sha256.txt", f"{TEACHER}/artifacts/teacher_observations.jsonl",
        f"{TRAINING}/RUN_STATE.json", f"{TRAINING}/metrics/summary.json", f"{TRAINING}/artifacts/evidence_sha256.txt",
        f"{RISK}/RUN_STATE.json", f"{RISK}/metrics/summary.json", f"{RISK}/artifacts/evidence_sha256.txt", f"{RISK}/artifacts/risk_calibrated_slope_c09/risk_calibration.json",
        f"{CAPACITY}/RUN_STATE.json", f"{CAPACITY}/metrics/summary.json", f"{CAPACITY}/artifacts/evidence_sha256.txt",
    ]
    for seed in range(3):
        inputs.extend([
            f"{TRAINING}/artifacts/models/seed{seed}/validation_outputs.npz",
            f"{RISK}/artifacts/risk_calibrated_slope_c09/seed{seed}_outputs.npz",
            f"{CAPACITY}/artifacts/models/seed{seed}/summary.json",
            f"{CAPACITY}/artifacts/models/seed{seed}/full_route_conditioned/best.pt",
            f"{CAPACITY}/artifacts/models/seed{seed}/full_route_conditioned/normalization.npz",
        ])
    tools = {
        "data_card": str(CARD.relative_to(PROJECT_ROOT)),
        "method_plan": "docs/GSE_GRAPH_FACTORIZED_RESEARCH_PLAN_V2.md",
        "qualification_helpers": "src/mtare_topo/evaluation/gse_factorized_qualification.py",
        "factorized_model": "src/mtare_topo/representation/gse_factorized_association.py",
        "unified_interface": "src/mtare_topo/representation/gse_unified_observation.py",
        "teacher_contract": "src/mtare_topo/teacher/gse_factorized_association_teacher.py",
        "executor": "tools/v3/execute_gse_factorized_association_c09_qualification_v1.py",
        "runner": "tools/v3/run_gse_factorized_association_c09_qualification_v1.py",
        "freezer": "tools/v3/freeze_gse_factorized_association_c09_qualification_spec_v1.py",
        "tests_qualification": "tests/v3/unit/test_gse_factorized_qualification.py",
        "tests_teacher": "tests/v3/unit/test_gse_factorized_association_teacher.py",
        "tests_unified": "tests/v3/unit/test_gse_unified_observation.py",
        "governance": "src/mtare_topo/governance.py", "preflight": "tools/v3/preflight.py", "create_run": "tools/v3/create_run.py",
    }
    criteria = card["metrics_and_pre_registered_gates"]
    spec = {
        "schema_version": "v3_run_spec_v1", "gate": 3, "execution_phase": 3,
        "date": "20260827", "slug": "gse_factorized_association_c09_qualification_v1", "seed": 0, "operation": "audit",
        "question": "Do frozen unified-slope Factorized association models retain safety on unseen C09 balanced aliases and actual strictly-past runtime candidates?",
        "method": card["methods"]["main"], "baseline": card["methods"]["baseline"], "fallback": card["methods"]["fallback"],
        "data_card": str(CARD.relative_to(PROJECT_ROOT)), "config_path": str(CARD.relative_to(PROJECT_ROOT)), "user_authorization": deepcopy(card["approval"]),
        "acceptance_criteria": [criteria[name] for name in ("balanced", "physical_only", "runtime", "runtime_family_reporting", "population", "forbidden")],
        "expected_counts": {"validation_worlds": 10, "validation_sequences": 24462, "balanced_identities": 130, "balanced_pairs": 260, "runtime_queries": 4085, "runtime_pairs": 4201, "runtime_positive_pairs": 3955, "runtime_negative_pairs": 246, "association_inference_pairs": 13383, "checkpoint_selection_steps": 0, "threshold_selection_steps": 0, "optimizer_steps": 0, "model_updates": 0, "c10_worlds_read": 0, "mtare_worlds_read": 0},
        "expected_evidence": ["Balanced/runtime manifests and scores, all seed/domain/family metrics, three unified C09 arrays, PNG/PDF/SVG/source JSON, environment, command, raw log, RUN_STATE and exact SHA-256 seal."],
        "estimated_cost": card["estimated_cost"],
        "frozen_inputs": {path: _sha(PROJECT_ROOT / path) for path in inputs},
        "frozen_tools": {name: {"path": path, "sha256": _sha(PROJECT_ROOT / path)} for name, path in tools.items()},
        "working_directory": str(PROJECT_ROOT),
        "command": ["/usr/bin/timeout", "--signal=INT", "--kill-after=30s", "1900s", PYTHON, "tools/v3/run_gse_factorized_association_c09_qualification_v1.py", "--spec", str(SPEC), "--run-dir", str(PROJECT_ROOT / "results/gate3_semantics" / RUN_ID)],
    }
    write_json(SPEC, spec)
    print(CARD.relative_to(PROJECT_ROOT)); print(SPEC.relative_to(PROJECT_ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
