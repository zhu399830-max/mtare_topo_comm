#!/usr/bin/env python3
"""Freeze the zero-training P2 primitive-relation model readiness audit."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from _bootstrap import PROJECT_ROOT


CARD = PROJECT_ROOT / "configs/v3/gate3/data_cards/primitive_relation_model_readiness_v1.json"
SPEC = PROJECT_ROOT / "configs/v3/gate3/primitive_relation_model_readiness_v1.json"
RUN_ID = "gate3_20260830_primitive_relation_model_readiness_v1_seed0"
PYTHON = "/home/zeng-workstation/.local/share/mtare_topo_comm/envs/phase3_torch290_cu129_zarr2187_v1/bin/python"
P1A = PROJECT_ROOT / "results/gate3_semantics/gate3_20260830_primitive_relation_p1a_lossless_storage_corrective_v1_seed0"
P1B = PROJECT_ROOT / "results/gate3_semantics/gate3_20260830_primitive_relation_p1b_teacher_materialization_v1r_seed0"
CARD_ID = "primitive_relation_model_readiness_v1"
SLUG = "primitive_relation_model_readiness_v1"
RUNNER = "tools/v3/run_primitive_relation_model_readiness_v1.py"
EXPECTED_TESTS = 29
CARD_STATUS = "APPROVED_FOR_ONE_IMMUTABLE_PRIMITIVE_RELATION_MODEL_READINESS_V1"
CORRECTIVE_OF: Path | None = None


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    p1a_summary = json.loads((P1A / "metrics/summary.json").read_text())
    p1a_state = json.loads((P1A / "RUN_STATE.json").read_text())
    p1b_summary = json.loads((P1B / "metrics/summary.json").read_text())
    p1b_state = json.loads((P1B / "RUN_STATE.json").read_text())
    if not (
        p1a_summary.get("scientific_pass")
        and p1b_summary.get("scientific_pass")
        and p1a_state.get("state") == "COMPLETED"
        and p1b_state.get("state") == "COMPLETED"
        and p1a_state.get("error") is None
        and p1b_state.get("error") is None
        and p1b_summary.get("sequences") == 564378
    ):
        raise RuntimeError("P2 readiness requires the exact sealed P1a/P1b PASS")

    approval = {
        "status": "APPROVED",
        "approved_by": "user-standing-authorization",
        "approved_at": "2026-08-30T00:00:00+08:00",
        "authorized_gates": [3],
        "authorized_operations": ["audit"],
        "confirmation_reference": "User authorized continuous autonomous execution of the frozen primitive-relation paper plan without repeated routine approvals.",
        "scope": "One immutable zero-training P2 model-readiness audit on one fit source sequence rendered with the three paired geometry realizations. No optimizer step, checkpoint, selection, graph, C07/C08/C09/C10 or M-TARE read.",
    }
    strict_worlds = [
        f"S{family:02d}_{name}_C{split:02d}"
        for split in (9, 10)
        for family, name in (
            (1, "flat_tree_small"), (2, "3d_tree_small"),
            (3, "flat_unicyclic_small"), (4, "3d_unicyclic_small"),
            (5, "flat_branch_medium"), (6, "3d_branch_medium"),
            (7, "flat_loop_rich"), (8, "3d_loop_rich"),
            (9, "flat_complex"), (10, "3d_complex"),
        )
    ]
    card = {
        "schema_version": "v3_data_card_v1",
        "card_id": CARD_ID,
        "status": CARD_STATUS,
        "approval": approval,
        "purpose": "Prove that the frozen bottom-up model/loss/reader contract can consume real five-frame LiDAR and exact construction supervision deterministically before any training is authorized.",
        "source": {
            "raw_sources": [
                "sealed P1a corrected causal LiDAR/provenance shards",
                "sealed P1b 32-slot primitive geometry/relation Teacher shards",
            ],
            "license_or_allowed_use": "Local research use of project-generated Cano assets; redistribution remains subject to upstream licensing.",
        },
        "worlds": {
            "train": ["S10_3d_complex_C04"],
            "validation": ["S10_3d_complex_C07"],
            "ssl": [], "normalization": [], "teacher_calibration": [],
            "threshold_calibration": [], "augmentation_tuning": [],
            "checkpoint_selection": [], "strict_test": strict_worlds,
        },
        "trajectories": [{
            "id": "S10_3d_complex_C04_source_sequence_188724",
            "world": "S10_3d_complex_C04", "split": "train", "independent": True,
            "duration_s": 4.0, "distance_m": 4.0, "spatial_coverage_m": 4.0,
        }],
        "sampling": {
            "independent_sampling_units": "One source topology sequence. Ellipse, rounded-rectangle and C1-mixed geometry rows are paired repeated measures, not three independent samples.",
            "raw_frame_count": 15, "effective_sample_count": 1,
            "effective_structure_event_count": 5, "spatial_interval_m": 1.0,
            "temporal_window_frames": 5, "geometry_realizations": 3,
            "source_global_sequence_index": 188724, "teacher_row": 2,
            "structure_event_counts": {
                "paired_geometry_rows": 3, "causal_frame_observations": 15,
                "unique_source_sequences": 1, "unique_source_primitives": 5,
                "repeated_primitive_instances": 15,
                "directed_attachment_positive_entries": 24,
                "disconnected_overlap_positive_entries": 18,
                "temporal_dustbin_entries": 12,
            },
            "rule": "Read row 2 from the three sealed S10_3d_complex_C04 geometry shards. The three rows share source sequence 188724 and contain five causal frames spaced at 1 m; no row is sampled or selected using a model result.",
        },
        "split": {
            "world_disjoint": True, "trajectory_disjoint": True,
            "fit": "Only S10_3d_complex_C04 is read by this audit.",
            "selection": "C07 is listed to preserve the future split but zero rows are read.",
            "development_transfer": "C08 zero rows read.",
            "strict_test": "All C09/C10 worlds and M-TARE benchmark worlds remain unread.",
            "historical_pollution_audit": "No historical prediction, graph, planner result, failed checkpoint or test-world statistic chooses the real readiness row, model weights, loss or threshold.",
        },
        "teacher": {
            "source": "Exact program-construction primitive identity and relations propagated through qualified P1a ray exits and materialized by sealed P1b; no human event class or predicted exit generates a label.",
            "labels": "Primitive presence, three axis controls, endpoint half-axes/exponents, five-frame visibility, endpoint attachment, disconnected overlap and relative causal odometry.",
            "student_forbidden_inputs": "World/parent/traversal/primitive identity, absolute pose, TNG, construction graph and future frames are unavailable to the model input.",
            "valid_mask": "Only Teacher slots with qualified P1a return support are valid; unused slots remain masked and are not interpreted as observed negative geometry.",
            "planner_consistency_plan": "No graph or planner runs in readiness. Edge execution semantics will be checked only after learned geometry and relation metrics pass C07/C08 offline gates.",
        },
        "leakage_audit": {
            "optimizer_step_count": 0, "model_inference_count": 0,
            "future_sensor_frames_excluded": True,
            "absolute_pose_not_retained_in_student_representation": True,
            "mtare_benchmark_excluded": True,
            "test_excluded_from_supervised_training": True,
            "test_excluded_from_ssl": True,
            "test_excluded_from_normalization": True,
            "test_excluded_from_teacher_calibration": True,
            "test_excluded_from_threshold_calibration": True,
            "test_excluded_from_augmentation_tuning": True,
            "test_excluded_from_checkpoint_selection": True,
        },
        "metrics_and_pre_registered_gates": {
            "interfaces": f"Exactly {EXPECTED_TESTS} geometry/model/reader tests pass; student inputs contain only normalized range, valid-return mask and five-frame relative odometry.",
            "model": "Parameter count is exactly 1,969,516; all typed output shapes are finite and 32-slot query permutation error is <=2e-5.",
            "losses": "Six equal-weight dimensionless loss families and their mean are finite on the real paired batch; target permutation and endpoint reversal change every loss by <=1e-6.",
            "optimization_readiness": "Every parameter tensor receives a finite nonzero gradient from one real-batch backward pass; optimizer steps and checkpoint writes remain zero.",
            "geometry_relations": "Attachment and overlap logits are exactly symmetric; causal point registration rotation error is <=2e-5 m; every real row contains attachment, overlap and dustbin examples.",
            "determinism": "Repeated evaluation is bit-exact under deterministic Torch and CUBLAS_WORKSPACE_CONFIG=:4096:8.",
            "resources": "Peak CUDA allocation <=2 GiB, wall time <=0.5 h, output <=0.05 GiB; zero C07/C08/C09/C10, graph or M-TARE reads.",
        },
        "estimated_cost": {
            "compute": "One RTX 5090 D process in the frozen Torch 2.9.0+cu129/Zarr 2.18.7 environment; tests plus one three-row forward/backward.",
            "wall_time_hours": 0.5, "host_ram_gb": 4, "gpu": 1,
            "gpu_memory_gb": 2, "disk_gb": 0.05,
        },
        "retention": "Keep Data Card/spec, unit log, real-batch contract, loss/gradient/symmetry metrics, environment, source hashes, RUN_STATE and SHA-256 seal.",
        "failure_policy": "Any source/environment drift, interface failure, nonfinite loss/gradient, missing gradient, permutation/reversal/asymmetry/determinism failure or resource excess stops training. Only a demonstrated system implementation defect may receive a separate corrective run; data, Teacher, model objective and thresholds may not be silently changed.",
    }
    tools = {
        "runner": RUNNER,
        "model": "src/mtare_topo/representation/primitive_relation_model.py",
        "losses": "src/mtare_topo/representation/primitive_relation_losses.py",
        "reader": "src/mtare_topo/data/primitive_relation_training.py",
        "geometry_field": "src/mtare_topo/teacher/swept_superellipse_field.py",
        "model_tests": "tests/v3/unit/test_primitive_relation_model.py",
        "reader_tests": "tests/v3/unit/test_primitive_relation_training.py",
        "field_tests": "tests/v3/unit/test_swept_superellipse_field.py",
        "variant_tests": "tests/v3/unit/test_geometry_variant_contract.py",
        "governance": "src/mtare_topo/governance.py",
        "preflight": "tools/v3/preflight.py", "create_run": "tools/v3/create_run.py",
    }
    if RUNNER != "tools/v3/run_primitive_relation_model_readiness_v1.py":
        tools["runner_base"] = "tools/v3/run_primitive_relation_model_readiness_v1.py"
    input_paths = [
        P1A / "RUN_STATE.json", P1A / "metrics/summary.json",
        P1A / "artifacts/task_manifest.json", P1A / "artifacts/evidence_sha256.txt",
        P1B / "RUN_STATE.json", P1B / "metrics/summary.json",
        P1B / "artifacts/task_manifest.json", P1B / "artifacts/evidence_sha256.txt",
        PROJECT_ROOT / "configs/v3/gate2/environments/phase3_torch290_cu129_zarr2187_v1r2.json",
        PROJECT_ROOT / "configs/v3/gate2/environments/phase3_torch290_cu129_zarr2187_pip_freeze.txt",
    ]
    if CORRECTIVE_OF is not None:
        predecessor_summary = json.loads((CORRECTIVE_OF / "metrics/summary.json").read_text())
        predecessor_state = json.loads((CORRECTIVE_OF / "RUN_STATE.json").read_text())
        failed_checks = [
            name for name, passed in predecessor_summary.get("checks", {}).items()
            if not passed
        ]
        if not (
            predecessor_state.get("state") == "COMPLETED"
            and predecessor_state.get("error") is None
            and predecessor_summary.get("scientific_pass") is False
            and predecessor_summary.get("error") is None
            and failed_checks == ["target_permutation_loss_error_le_1e6"]
            and predecessor_summary.get("target_permutation_loss_error")
            == 1.7344951629638672e-05
        ):
            raise RuntimeError("V1R requires the exact sealed V1 Hungarian tie-breaking failure")
        card["corrective_of"] = {
            "run": str(CORRECTIVE_OF.relative_to(PROJECT_ROOT)),
            "failure": "Teacher slot permutation changed the real-batch relation loss by 1.7344951629638672e-05 because Hungarian near-tie resolution depended on storage column order.",
            "scope": "Canonicalize active Teacher primitives by endpoint-reversal-invariant geometry and iteratively refined port/overlap signatures before the unchanged Hungarian objective. Preserve data, Teacher, six losses, weights and all tolerances.",
        }
        input_paths.extend((
            CORRECTIVE_OF / "RUN_STATE.json",
            CORRECTIVE_OF / "metrics/summary.json",
            CORRECTIVE_OF / "artifacts/evidence_sha256.txt",
        ))
    write(CARD, card)
    input_paths.insert(0, CARD)
    spec = {
        "schema_version": "v3_run_spec_v1", "gate": 3, "execution_phase": 3,
        "operation": "audit", "date": "20260830",
        "slug": SLUG, "seed": 0,
        "question": "Can a 32-slot causal LiDAR model learn explicit swept primitives and their attachment/overlap/temporal relations under a leakage-safe, permutation- and endpoint-reversal-invariant contract before training?",
        "method": "Circular range encoder plus five-frame causal registration/temporal aggregation and 32 exchangeable queries predicts swept-superellipse controls, endpoint shape/descriptors, uncertainty, pair relations and temporal correspondence; six dimensionless losses are Hungarian matched and equally weighted.",
        "baseline": "The subsequent formal training comparison is a non-learning robust superellipse/axis/relation estimator on the same causal scans; historical exit-only models and rule graphs are retained as downstream paper baselines but are not executed in this readiness audit.",
        "fallback": "If readiness fails, stop before training. Correct only an evidenced interface/environment defect in a separately sealed run; do not change data, Teacher, slots, loss families or tolerances.",
        "corrective_of": None if CORRECTIVE_OF is None else str(CORRECTIVE_OF.relative_to(PROJECT_ROOT)),
        "data_card": str(CARD.relative_to(PROJECT_ROOT)),
        "config_path": str(CARD.relative_to(PROJECT_ROOT)),
        "user_authorization": approval,
        "acceptance_criteria": list(card["metrics_and_pre_registered_gates"].values()),
        "expected_counts": {
            "independent_source_sequences": 1, "paired_geometry_rows": 3,
            "causal_frames": 15, "unique_primitives": 5,
            "directed_attachments": 24, "disconnected_overlaps": 18,
            "temporal_dustbins": 12, "unit_tests": EXPECTED_TESTS,
            "model_parameters": 1969516, "loss_families": 6,
            "optimizer_steps": 0, "checkpoint_writes": 0,
            "c07_c08_data_rows_read": 0, "c09_c10_worlds_read": 0,
            "graph_replays": 0, "mtare_worlds_read": 0,
        },
        "expected_evidence": [
            f"{EXPECTED_TESTS}-test log, exact real-batch source/tree hashes, six losses, all-parameter gradient audit, deterministic/permutation/reversal/symmetry/registration checks, environment, RUN_STATE and SHA-256 seal."
        ],
        "estimated_cost": card["estimated_cost"],
        "frozen_tools": {
            name: {"path": path, "sha256": sha(PROJECT_ROOT / path)}
            for name, path in tools.items()
        },
        "frozen_inputs": {
            str(path.relative_to(PROJECT_ROOT)): sha(path) for path in input_paths
        },
        "working_directory": str(PROJECT_ROOT),
        "command": [
            "/usr/bin/systemd-inhibit", "--what=sleep:shutdown",
            "--why=GSE primitive relation model readiness", "--mode=block",
            "/usr/bin/timeout", "--signal=INT", "--kill-after=60s", "1800s",
            "/usr/bin/env", "CUBLAS_WORKSPACE_CONFIG=:4096:8",
            f"PYTHONPATH={PROJECT_ROOT / 'src'}:{PROJECT_ROOT / 'tools/v3'}",
            PYTHON, RUNNER,
            "--spec", str(SPEC),
            "--run-dir", str(PROJECT_ROOT / f"results/gate3_semantics/{RUN_ID}"),
        ],
    }
    write(SPEC, spec)
    print(SPEC)


if __name__ == "__main__":
    main()
