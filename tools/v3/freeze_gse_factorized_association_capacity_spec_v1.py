#!/usr/bin/env python3
"""Freeze Data Card and run spec for factorized association capacity V1."""

from __future__ import annotations

from copy import deepcopy
import hashlib

from _bootstrap import PROJECT_ROOT
from mtare_topo.governance import load_json, write_json


RUN_ID = "gate3_20260827_gse_factorized_association_capacity_v1_seed0"
CARD = PROJECT_ROOT / "configs/v3/gate3/data_cards/gse_factorized_association_capacity_v1.json"
SPEC = PROJECT_ROOT / "configs/v3/gate3/gse_factorized_association_capacity_v1.json"
SOURCE_CARD = PROJECT_ROOT / "configs/v3/gate3/data_cards/gse_factorized_association_teacher_manifest_v1.json"
TEACHER = "results/gate2_representation/gate2_20260827_gse_corrected_causal_teacher_manifest_v1r_seed0"
MANIFEST = "results/gate3_semantics/gate3_20260827_gse_factorized_association_teacher_manifest_v1_seed0"
FEATURES = "results/gate3_semantics/gate3_20260826_gse_exit_token_association_corrective_v2_seed0"
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
        "card_id": "gse_factorized_association_capacity_v1",
        "status": "APPROVED_FOR_ONE_IMMUTABLE_GSE_FACTORIZED_ASSOCIATION_CAPACITY_V1",
        "purpose": "Test whether deployment-only learned action tokens and executed incoming-edge geometry safely improve decision-node association over descriptor-only and no-route-geometry baselines.",
    })
    card["approval"] = {
        "status": "APPROVED", "approved_by": "user-standing-authorization",
        "approved_at": "2026-08-27T23:59:45+08:00",
        "authorized_operations": ["training", "normalization", "checkpoint_selection", "threshold_calibration"],
        "authorized_gates": [3],
        "scope": "One immutable CPU-only C01-C08 capacity proof: three frozen perception seeds, full/no-route small verifiers, C01-C06 fit, C07-C08 checkpoint/threshold selection, zero C09/C10/M-TARE/graph/planner access.",
        "confirmation_reference": "User explicitly instructed automatic evidence-supported decisions without routine approval prompts.",
    }
    card["source"] = {
        "raw_sources": [
            "Sealed corrected C01-C08 Teacher with 188126 causal observations.",
            "Sealed PASS identity-balanced association manifest with 1066 identities and 2132 pairs.",
            "Three sealed frozen 146D observation arrays and complete six-token outputs from the failed V2 association run; only scientifically valid frozen perception components are reused.",
        ],
        "corrected_teacher_run": TEACHER, "association_teacher_run": MANIFEST,
        "frozen_perception_component_run": FEATURES,
        "license_or_allowed_use": "Local research use of project procedural worlds and locally trained frozen outputs with complete provenance.",
    }
    card["sampling"] = {
        "raw_frame_count": 252430, "effective_sample_count": 188126,
        "effective_structure_event_count": 1066,
        "structure_event_counts": {"junction": 563, "terminal": 503},
        "spatial_interval_m": 1.0,
        "fit": "C01-C06: 60 worlds, 142184 observations, 792 decision identities, 1584 balanced pairs.",
        "selection": "C07-C08: 20 worlds, 45942 observations, 274 decision identities, 548 balanced pairs.",
        "independent_units": "Decision identity is the reporting and balancing unit; observations from one identity never cross fit/selection.",
        "temporal_context": "Each observation and incoming-edge profile uses current plus at most four strictly earlier rows on the same directed traversal.",
        "rule": "Five-frame causal sequences are sampled at 1 m arc spacing within each directed traversal; identity-balanced association units are one positive and one same-event/degree hard negative per junction/terminal identity, split before pairing.",
        "singleton_policy": "Five singleton short terminals remain explicit circular-shift augmentation units; selection safety is also required after removing the one selection augmentation positive.",
    }
    card["split"] = {
        "fit": "C01-C06 only; normalization and optimizer updates.",
        "selection": "C07-C08 only; complete-epoch checkpoint selection and one non-vacuous refusal threshold per seed/variant.",
        "strict_test": "C09, C10 and M-TARE worlds are forbidden and read counts must remain zero.",
        "leakage_audit": "Student API excludes identity/world/parent/TNG event/objective geometry/absolute pose. Teacher objective geometry selected hard negatives only. Spatial distance defines the <=16m runtime candidate domain but is not a cross-world structural-alias classifier input.",
        "world_disjoint": True,
        "trajectory_disjoint": True,
        "historical_pollution_audit": "C01-C06 and C07-C08 are world- and traversal-disjoint. C09/C10/M-TARE paths are absent from all frozen feature, Teacher and manifest sources; old failed association models/checkpoints are not reused, only their immutable frozen perception arrays are inputs.",
    }
    card["teacher"] = {
        "source": "Sealed corrected TNG/spline/mesh Teacher plus the sealed identity-balanced structural-alias manifest.",
        "positive": "Same decision identity; 1061 observed cross-view/distinct positives and five explicit circular-shift singleton terminals.",
        "negative": "Different identity, same split/event/incident degree, nearest masked objective geometry profile.",
        "valid_mask": "All 1066 decision identities are retained. Twenty identities with fewer than five valid profile observations carry explicit causal-length masks; five singleton terminals carry explicit augmentation kind. No row is silently dropped.",
        "planner_consistency_plan": "This proof trains association only. Runtime graph integration will apply the frozen score only inside the unchanged <=16 m strictly-past candidate domain, create provisional nodes for rejected/ambiguous matches, and commit edges only after physical traversal.",
        "student_forbidden_fields": ["identity", "world_id", "parent_id", "teacher_event", "objective_geometry_profile", "absolute_pose"],
    }
    card["methods"] = {
        "main": "Per frozen perception seed, an 8017-parameter symmetric verifier combines learned event/axis/place/uncertainty, complete permutation/yaw-invariant exit-token relations, causal incoming-edge geometry and incoming-width/slope-to-candidate-exit width/vertical compatibility.",
        "baseline": "Descriptor cosine with the same safety selector, plus a 6673-parameter otherwise identical no-route-geometry ablation.",
        "fallback": "If safety or independent route-geometry gain fails, stop the learned route-conditioned association claim and do not enter C09/graph/planner tuning.",
    }
    card["optimization"] = {
        "seeds": [0, 1, 2], "variants_per_seed": ["full_route_conditioned", "no_route_geometry"],
        "optimizer": "AdamW", "learning_rate": .001, "weight_decay": .0001,
        "batch_size": 256, "maximum_epochs": 50, "patience": 8,
        "loss": "balanced BCE plus 0.5 equal-count lowest-positive/highest-negative tail loss",
        "checkpoint_selection": "minimum complete C07-C08 balanced BCE independently per seed and variant",
        "threshold_selection": "maximum recall subject to aggregate precision>=0.98, false accept<=0.01, recall>=0.25 and every-family nonempty precision>=0.95/recall>=0.10",
        "frozen_backbone": True, "backbone_optimizer_steps": 0,
    }
    card["metrics_and_pre_registered_gates"] = {
        "safety": "All three full seeds and their physical-only subsets pass precision>=0.98, false accept<=0.01, recall>=0.25 and per-family nonempty precision>=0.95/recall>=0.10.",
        "descriptor_gain": "Three-seed mean full safe recall exceeds descriptor-only mean by at least 0.05.",
        "route_gain": "Mean full safe recall exceeds no-route geometry by at least 0.02; no seed regresses by more than 0.02.",
        "population": "Exact fit/selection identity and pair counts 792/274 and 1584/548.",
        "forbidden": "Backbone updates, model inference, C09/C10/strict/M-TARE/graph/planner reads all zero.",
    }
    card["estimated_cost"] = {
        "compute": "CPU-only six small verifier trainings over frozen features", "disk_gb": .5,
        "host_ram_gb": 4, "gpu_memory_gb": 0, "wall_time_hours": 1,
    }
    card["retention"] = "Retain six checkpoints/normalizations/epoch logs/selection scores and curves, three-seed summaries, PNG/PDF/SVG/source JSON, environment, raw log, RUN_STATE and SHA-256 seal."
    card["evidence"] = {
        "machine_metrics": "Per-seed/variant safety, physical-only safety, descriptor/no-route gains, parameter/update/resource counts and zero forbidden reads.",
        "complete_visual_review": "Three-seed safe-recall comparison and full-model precision-recall curves in PNG/PDF/SVG with source JSON.",
        "failure_policy": "Any source/split/feature/symmetry/population/resource/evidence drift or scientific-gate miss seals FAIL; no retry, threshold relaxation or graph tuning.",
    }
    card["failure_policy"] = card["evidence"]["failure_policy"]
    write_json(CARD, card)

    inputs = [
        f"{TEACHER}/RUN_STATE.json", f"{TEACHER}/metrics/summary.json",
        f"{TEACHER}/artifacts/evidence_sha256.txt", f"{TEACHER}/artifacts/teacher_observations.jsonl",
        f"{MANIFEST}/RUN_STATE.json", f"{MANIFEST}/metrics/summary.json",
        f"{MANIFEST}/artifacts/evidence_sha256.txt", f"{MANIFEST}/artifacts/identity_balanced_association_manifest.jsonl",
        f"{FEATURES}/RUN_STATE.json", f"{FEATURES}/metrics/summary.json", f"{FEATURES}/artifacts/evidence_sha256.txt",
    ]
    for seed in range(3):
        inputs.extend([
            f"{FEATURES}/artifacts/models/seed{seed}/frozen_observation_features.npy",
            f"{FEATURES}/artifacts/models/seed{seed}/frozen_exit_token_outputs.npz",
        ])
    tools = {
        "data_card": str(CARD.relative_to(PROJECT_ROOT)),
        "method_plan": "docs/GSE_GRAPH_FACTORIZED_RESEARCH_PLAN_V2.md",
        "teacher_contract": "src/mtare_topo/teacher/gse_factorized_association_teacher.py",
        "factorized_verifier": "src/mtare_topo/representation/gse_factorized_association.py",
        "base_threshold": "src/mtare_topo/representation/gse_open_set_association.py",
        "token_relation": "src/mtare_topo/representation/gse_exit_token_association.py",
        "unit_tests": "tests/v3/unit/test_gse_factorized_association.py",
        "trainer": "tools/v3/train_gse_factorized_association_capacity_v1.py",
        "executor": "tools/v3/execute_gse_factorized_association_capacity_v1.py",
        "runner": "tools/v3/run_gse_factorized_association_capacity_v1.py",
        "freezer": "tools/v3/freeze_gse_factorized_association_capacity_spec_v1.py",
        "governance": "src/mtare_topo/governance.py", "preflight": "tools/v3/preflight.py",
        "create_run": "tools/v3/create_run.py",
    }
    spec = {
        "schema_version": "v3_run_spec_v1", "gate": 3, "execution_phase": 3,
        "date": "20260827", "slug": "gse_factorized_association_capacity_v1",
        "seed": 0, "operation": "training",
        "question": "Can deployment-only learned action tokens and executed-edge geometry safely improve decision-node association over descriptor and no-route baselines?",
        "method": card["methods"]["main"], "baseline": card["methods"]["baseline"],
        "fallback": card["methods"]["fallback"], "optimization": card["optimization"],
        "data_card": str(CARD.relative_to(PROJECT_ROOT)), "config_path": str(CARD.relative_to(PROJECT_ROOT)),
        "user_authorization": deepcopy(card["approval"]),
        "acceptance_criteria": [
            card["metrics_and_pre_registered_gates"]["safety"],
            card["metrics_and_pre_registered_gates"]["descriptor_gain"],
            card["metrics_and_pre_registered_gates"]["route_gain"],
            card["metrics_and_pre_registered_gates"]["population"],
            card["metrics_and_pre_registered_gates"]["forbidden"],
        ],
        "expected_counts": {
            "fit_worlds": 60, "fit_observations": 142184, "fit_identities": 792, "fit_pairs": 1584,
            "selection_worlds": 20, "selection_observations": 45942,
            "selection_identities": 274, "selection_pairs": 548, "seeds": 3, "variants": 2,
            "backbone_optimizer_steps": 0, "model_inference_frames": 0,
            "c09_worlds_read": 0, "strict_test_worlds_read": 0, "mtare_worlds_read": 0,
        },
        "expected_evidence": [
            "Six checkpoints/normalizations/epoch logs/score archives and curves, exact per-seed/variant metrics, PNG/PDF/SVG/source JSON, environment, command, raw log, RUN_STATE and SHA-256 seal."
        ],
        "estimated_cost": card["estimated_cost"],
        "frozen_inputs": {path: _sha(PROJECT_ROOT / path) for path in inputs},
        "frozen_tools": {name: {"path": path, "sha256": _sha(PROJECT_ROOT / path)} for name, path in tools.items()},
        "working_directory": str(PROJECT_ROOT),
        "command": [
            "/usr/bin/timeout", "--signal=INT", "--kill-after=30s", "3700s", PYTHON,
            "tools/v3/run_gse_factorized_association_capacity_v1.py", "--spec", str(SPEC),
            "--run-dir", str(PROJECT_ROOT / "results/gate3_semantics" / RUN_ID),
        ],
    }
    write_json(SPEC, spec)
    print(CARD.relative_to(PROJECT_ROOT)); print(SPEC.relative_to(PROJECT_ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
