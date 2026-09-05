#!/usr/bin/env python3
"""Freeze the final-slope unified Factorized GSE association capacity run."""

from __future__ import annotations

from copy import deepcopy
import hashlib

from _bootstrap import PROJECT_ROOT
from mtare_topo.governance import load_json, write_json


RUN_ID = "gate3_20260827_gse_factorized_association_capacity_v1r_seed0"
CARD = PROJECT_ROOT / "configs/v3/gate3/data_cards/gse_factorized_association_capacity_v1r.json"
SPEC = PROJECT_ROOT / "configs/v3/gate3/gse_factorized_association_capacity_v1r.json"
SOURCE_CARD = PROJECT_ROOT / "configs/v3/gate3/data_cards/gse_factorized_association_capacity_v1.json"
TEACHER = "results/gate2_representation/gate2_20260827_gse_corrected_causal_teacher_manifest_v1r_seed0"
MANIFEST = "results/gate3_semantics/gate3_20260827_gse_factorized_association_teacher_manifest_v1_seed0"
FEATURES = "results/gate3_semantics/gate3_20260826_gse_exit_token_association_corrective_v2_seed0"
CORRECTIVE = "results/gate2_representation/gate2_20260824_gse_slope_corrective_three_seed_training_v1r_seed0"
RISK = "results/gate3_semantics/gate3_20260825_gse_risk_calibrated_perception_validation_v2_seed0"
OLD_CAPACITY = "results/gate3_semantics/gate3_20260827_gse_factorized_association_capacity_v1_seed0"
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
        "card_id": "gse_factorized_association_capacity_v1r",
        "status": "APPROVED_FOR_ONE_IMMUTABLE_GSE_FACTORIZED_ASSOCIATION_CAPACITY_V1R",
        "purpose": "Verify the route-conditioned association claim after replacing its legacy slope feature with the final frozen risk-calibrated slope used by the unified GSE perception interface.",
    })
    card["approval"] = {
        "status": "APPROVED", "approved_by": "user-standing-authorization",
        "approved_at": "2026-08-27T23:59:45+08:00", "authorized_gates": [3],
        "authorized_operations": ["training", "normalization", "checkpoint_selection", "threshold_calibration"],
        "scope": "One immutable C01-C08 unified-interface corrective: exact CUDA replay of three frozen slope correctives, slope-column-only integration, then the unchanged six small association verifier trainings; zero C09/C10/M-TARE/graph/planner access.",
        "confirmation_reference": "User explicitly instructed automatic evidence-supported decisions without routine approval prompts.",
    }
    card["source"].update({
        "slope_corrective_run": CORRECTIVE,
        "risk_calibration_run": RISK,
        "old_capacity_component_run": OLD_CAPACITY,
    })
    card["source"]["raw_sources"].extend([
        "Sealed C01-C08 slope-corrective cache with 142184 fit and 45942 selection causal sequences and three frozen best checkpoints.",
        "Sealed validation-only risk calibration residual scale 0.89 selected on C07-C08 without C09 access.",
    ])
    card["methods"]["main"] = "Exact CUDA replay of the frozen physics-guided slope corrective, fixed 0.89 residual scaling, global-sequence-ID alignment and replacement of only normalized observation column 10; then the unchanged 8017-parameter route-conditioned verifier."
    card["methods"]["fallback"] = "If exact replay, slope-only integration, ablation identity, safety or route gain fails, stop before C09; do not mix legacy and final slope semantics or alter thresholds."
    card["metrics_and_pre_registered_gates"]["interface"] = "For all three seeds, 188126 identities align exactly, archived C07-C08 GPU slope outputs reproduce byte-exactly, only feature column 10 changes, and all other 145 columns remain byte-exact."
    card["metrics_and_pre_registered_gates"]["ablation_identity"] = "Descriptor-only and no-route-geometry selection archives remain byte-exact to V1; only the route-conditioned model may change."
    card["metrics_and_pre_registered_gates"]["forbidden"] = "GSE backbone inference/updates, C09/C10/strict/M-TARE/graph/planner reads all zero; the only frozen inference is 188126 sequences x 3 slope correctives."
    card["optimization"]["slope_corrective"] = "Frozen; exact original CUDA path; 564378 sequence inferences; zero optimizer steps; fixed residual scale 0.89."
    card["estimated_cost"] = {"compute": "One exact CUDA replay of three frozen 5-frame slope correctives plus six CPU-only small verifier trainings", "disk_gb": 0.5, "host_ram_gb": 4, "gpu_memory_gb": 8, "wall_time_hours": 1}
    card["retention"] = "Retain three unified 146D arrays and slope provenance archives, six verifier checkpoints/normalizations/logs/scores, PNG/PDF/SVG/source JSON, environment, commands, RUN_STATE and SHA-256 seal."
    card["failure_policy"] = "Any exact CUDA replay mismatch, identity drift, non-slope feature change, V1 ablation drift, safety/gain failure, forbidden read or resource/evidence drift seals FAIL; no retry or threshold change."
    card["evidence"]["failure_policy"] = card["failure_policy"]
    card["leakage_audit"]["unified_interface"] = "Final corrected slope uses only five causal LiDAR frames. Fit normalization is C01-C06; fixed residual scale and association selection are C07-C08; C09/C10/M-TARE are absent."
    write_json(CARD, card)

    inputs = [
        f"{TEACHER}/RUN_STATE.json", f"{TEACHER}/metrics/summary.json", f"{TEACHER}/artifacts/evidence_sha256.txt", f"{TEACHER}/artifacts/teacher_observations.jsonl",
        f"{MANIFEST}/RUN_STATE.json", f"{MANIFEST}/metrics/summary.json", f"{MANIFEST}/artifacts/evidence_sha256.txt", f"{MANIFEST}/artifacts/identity_balanced_association_manifest.jsonl",
        f"{FEATURES}/RUN_STATE.json", f"{FEATURES}/metrics/summary.json", f"{FEATURES}/artifacts/evidence_sha256.txt",
        f"{CORRECTIVE}/RUN_STATE.json", f"{CORRECTIVE}/metrics/summary.json", f"{CORRECTIVE}/artifacts/evidence_sha256.txt", f"{CORRECTIVE}/artifacts/slope_corrective_cache/manifest.json", f"{CORRECTIVE}/artifacts/slope_corrective_cache/normalization.json",
        f"{RISK}/RUN_STATE.json", f"{RISK}/metrics/summary.json", f"{RISK}/artifacts/evidence_sha256.txt", f"{RISK}/artifacts/risk_calibrated_slope_c09/risk_calibration.json",
        f"{OLD_CAPACITY}/RUN_STATE.json", f"{OLD_CAPACITY}/metrics/summary.json", f"{OLD_CAPACITY}/artifacts/evidence_sha256.txt",
    ]
    for seed in range(3):
        inputs.extend([
            f"{FEATURES}/artifacts/models/seed{seed}/frozen_observation_features.npy",
            f"{FEATURES}/artifacts/models/seed{seed}/frozen_exit_token_outputs.npz",
            f"{CORRECTIVE}/artifacts/models/seed{seed}/best.pt",
            f"{CORRECTIVE}/artifacts/models/seed{seed}/selection_outputs.npz",
            f"{OLD_CAPACITY}/artifacts/models/seed{seed}/descriptor_only_selection_outputs.npz",
            f"{OLD_CAPACITY}/artifacts/models/seed{seed}/no_route_geometry/selection_outputs.npz",
        ])
    tools = {
        "data_card": str(CARD.relative_to(PROJECT_ROOT)),
        "method_plan": "docs/GSE_GRAPH_FACTORIZED_RESEARCH_PLAN_V2.md",
        "unified_interface": "src/mtare_topo/representation/gse_unified_observation.py",
        "factorized_verifier": "src/mtare_topo/representation/gse_factorized_association.py",
        "slope_corrective": "src/mtare_topo/representation/gse_slope_corrective.py",
        "slope_risk_calibration": "src/mtare_topo/representation/gse_slope_risk_calibration.py",
        "cache_contract": "src/mtare_topo/data/gse_slope_corrective_dataset.py",
        "unified_builder": "tools/v3/build_gse_unified_observation_features_v1.py",
        "trainer": "tools/v3/train_gse_factorized_association_capacity_v1.py",
        "executor": "tools/v3/execute_gse_factorized_association_capacity_v1.py",
        "runner": "tools/v3/run_gse_factorized_association_capacity_v1r.py",
        "freezer": "tools/v3/freeze_gse_factorized_association_capacity_spec_v1r.py",
        "unit_tests_unified": "tests/v3/unit/test_gse_unified_observation.py",
        "unit_tests_factorized": "tests/v3/unit/test_gse_factorized_association.py",
        "governance": "src/mtare_topo/governance.py", "preflight": "tools/v3/preflight.py", "create_run": "tools/v3/create_run.py",
    }
    criteria = card["metrics_and_pre_registered_gates"]
    spec = {
        "schema_version": "v3_run_spec_v1", "gate": 3, "execution_phase": 3,
        "date": "20260827", "slug": "gse_factorized_association_capacity_v1r", "seed": 0, "operation": "training",
        "question": "Does route-conditioned association remain safe and independently useful under the final unified risk-calibrated slope interface?",
        "method": card["methods"]["main"], "baseline": card["methods"]["baseline"], "fallback": card["methods"]["fallback"],
        "optimization": card["optimization"], "data_card": str(CARD.relative_to(PROJECT_ROOT)), "config_path": str(CARD.relative_to(PROJECT_ROOT)), "user_authorization": deepcopy(card["approval"]),
        "acceptance_criteria": [criteria[name] for name in ("interface", "ablation_identity", "safety", "descriptor_gain", "route_gain", "population", "forbidden")],
        "expected_counts": {"fit_worlds": 60, "fit_observations": 142184, "fit_identities": 792, "fit_pairs": 1584, "selection_worlds": 20, "selection_observations": 45942, "selection_identities": 274, "selection_pairs": 548, "seeds": 3, "variants": 2, "slope_corrective_inference_sequences": 564378, "gse_backbone_inference_frames": 0, "c09_worlds_read": 0, "strict_test_worlds_read": 0, "mtare_worlds_read": 0},
        "expected_evidence": ["Three unified feature arrays and slope provenance archives; exact replay/column/ablation audits; six checkpoints and score curves; PNG/PDF/SVG/source JSON; environment, commands, raw logs, RUN_STATE and SHA-256 seal."],
        "estimated_cost": card["estimated_cost"],
        "frozen_inputs": {path: _sha(PROJECT_ROOT / path) for path in inputs},
        "frozen_tools": {name: {"path": path, "sha256": _sha(PROJECT_ROOT / path)} for name, path in tools.items()},
        "working_directory": str(PROJECT_ROOT),
        "command": ["/usr/bin/timeout", "--signal=INT", "--kill-after=30s", "3700s", PYTHON, "tools/v3/run_gse_factorized_association_capacity_v1r.py", "--spec", str(SPEC), "--run-dir", str(PROJECT_ROOT / "results/gate3_semantics" / RUN_ID)],
    }
    write_json(SPEC, spec)
    print(CARD.relative_to(PROJECT_ROOT)); print(SPEC.relative_to(PROJECT_ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
