#!/usr/bin/env python3
"""Freeze the evidence-driven sparse-port relation readiness audit."""

from __future__ import annotations

import json

from _bootstrap import PROJECT_ROOT
from freeze_primitive_relation_model_readiness_v1_spec import sha, write


CARD = PROJECT_ROOT / "configs/v3/gate3/data_cards/primitive_relation_sparse_port_readiness_v1.json"
SPEC = PROJECT_ROOT / "configs/v3/gate3/primitive_relation_sparse_port_readiness_v1.json"
RUN_ID = "gate3_20260831_primitive_relation_sparse_port_readiness_v1_seed0"
PYTHON = "/home/zeng-workstation/.local/share/mtare_topo_comm/envs/phase3_torch290_cu129_zarr2187_v1/bin/python"
P1A = PROJECT_ROOT / "results/gate3_semantics/gate3_20260830_primitive_relation_p1a_lossless_storage_corrective_v1_seed0"
P1B = PROJECT_ROOT / "results/gate3_semantics/gate3_20260830_primitive_relation_p1b_teacher_materialization_v1r_seed0"
ATTRIBUTION = PROJECT_ROOT / "results/gate3_semantics/gate3_20260831_primitive_relation_v1_failure_attribution_v1r2_seed0"
RUNNER = "tools/v3/run_primitive_relation_sparse_port_readiness_v1.py"
CARD_STATUS = "APPROVED_FOR_ONE_IMMUTABLE_PRIMITIVE_RELATION_SPARSE_PORT_READINESS_V1"


def main() -> None:
    p1a_summary = json.loads((P1A / "metrics/summary.json").read_text())
    p1b_summary = json.loads((P1B / "metrics/summary.json").read_text())
    attribution = json.loads((ATTRIBUTION / "metrics/attribution/summary.json").read_text())
    attribution_state = json.loads((ATTRIBUTION / "RUN_STATE.json").read_text())
    if not (
        p1a_summary.get("scientific_pass")
        and p1b_summary.get("scientific_pass")
        and p1b_summary.get("sequences") == 564378
        and attribution_state.get("state") == "COMPLETED"
        and attribution_state.get("error") is None
        and attribution.get("diagnosis") == "RELATION_HEAD_FAILS_EVEN_WITH_PROPOSAL_ORACLE"
        and attribution.get("decision") == "REQUIRE_SPARSE_PORT_RELATION_ARCHITECTURE_READINESS"
        and attribution.get("oracle_gate_seeds") == 0
    ):
        raise RuntimeError("sparse-port readiness requires the exact sealed V1 attribution")

    approval = {
        "status": "APPROVED",
        "approved_by": "user-standing-authorization",
        "approved_at": "2026-08-31T00:00:00+08:00",
        "authorized_gates": [3],
        "authorized_operations": ["audit"],
        "confirmation_reference": "User authorized continuous autonomous execution of the frozen primitive-relation paper plan without repeated routine approvals.",
        "scope": "One immutable zero-training sparse-port relation architecture readiness on one C01-C06 fit source sequence rendered with three paired geometries. No optimizer, checkpoint, threshold selection, graph, C07/C08/C09/C10 or M-TARE read.",
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
    gates = {
        "interfaces": "Exactly 42 geometry/V1/V2/reader tests pass; the V2 student still receives only normalized range, valid-return mask and five-frame relative odometry.",
        "model": "Parameter count is exactly 2,629,870 in 193 tensors; all typed geometry, relation and relation-uncertainty outputs are finite and 32-slot query permutation error is <=3e-5.",
        "sparse_set": "At a 3-of-32 synthetic target, every matched existence gradient is negative and every redundant-slot gradient is positive; the six registered loss-family names remain unchanged.",
        "losses": "Six equal-weight dimensionless loss families and their mean are finite on the paired real batch; target permutation and endpoint reversal change every loss by <=1e-6.",
        "optimization_readiness": "Every one of 193 parameter tensors receives a finite nonzero gradient from one real-batch backward pass; optimizer and checkpoint writes remain zero.",
        "relations": "Attachment/overlap logits are exactly symmetric, uncertainties lie in [0,1], and endpoint relative-geometry yaw-rotation error is <=2e-6.",
        "determinism": "Repeated evaluation is bit-exact with matmul and cuDNN TF32 disabled, highest FP32 precision, deterministic Torch and CUBLAS_WORKSPACE_CONFIG=:4096:8.",
        "resources": "Peak CUDA allocation <=4 GiB, wall time <=0.5 h and output <=0.05 GiB; zero C07/C08/C09/C10, graph or M-TARE reads.",
    }
    card = {
        "schema_version": "v3_data_card_v1",
        "card_id": "primitive_relation_sparse_port_readiness_v1",
        "status": CARD_STATUS,
        "approval": approval,
        "purpose": "Test whether the V1 attribution-supported sparse set and endpoint-context relation architecture is mathematically and computationally ready before any corrective training.",
        "corrective_of": {
            "run": str(ATTRIBUTION.relative_to(PROJECT_ROOT)),
            "evidence": "V1 activates about 2.066M proposals for 544,414 targets, expands relation pairs by about 14.04x, and has zero safe true attachments in all three seeds even under a Teacher proposal oracle.",
            "scope": "Preserve causal LiDAR, relative odometry, 32-slot capacity, swept geometry, Teacher, split and six loss families. Add calibrated cardinality/MDL inside primitive_set_parameters and replace the dense pair MLP with learned endpoint relative-geometry context and typed relation uncertainty.",
        },
        "source": {
            "raw_sources": [
                "sealed P1a corrected causal LiDAR/provenance shards",
                "sealed P1b 32-slot primitive geometry/relation Teacher shards",
                "sealed V1R2 corrected C07 failure attribution as the corrective trigger only",
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
            "independent_sampling_units": "One source topology sequence. Ellipse, rounded-rectangle and C1-mixed rows are paired repeated geometries, not independent samples.",
            "raw_frame_count": 15,
            "effective_sample_count": 1,
            "effective_structure_event_count": 5,
            "spatial_interval_m": 1.0,
            "temporal_window_frames": 5,
            "geometry_realizations": 3,
            "source_global_sequence_index": 188724,
            "teacher_row": 2,
            "structure_event_counts": {
                "paired_geometry_rows": 3,
                "causal_frame_observations": 15,
                "unique_source_sequences": 1,
                "unique_source_primitives": 5,
                "repeated_primitive_instances": 15,
                "directed_attachment_positive_entries": 24,
                "disconnected_overlap_positive_entries": 18,
                "temporal_dustbin_entries": 12,
            },
            "rule": "Read row 2 from the three sealed S10_3d_complex_C04 geometry shards. All share source sequence 188724 and five causal frames at 1 m spacing; selection is independent of every model result.",
        },
        "split": {
            "world_disjoint": True, "trajectory_disjoint": True,
            "fit": "Only one C04 source sequence is read.",
            "selection": "C07 zero rows read; V1R2 aggregate diagnosis is only a sealed corrective trigger and does not select a V2 weight or threshold.",
            "development_transfer": "C08 zero rows read.",
            "strict_test": "All C09/C10 and M-TARE benchmark worlds remain unread.",
            "historical_pollution_audit": "The V1 failure fixes the architectural fault class only. No V1 prediction, C07 row, threshold or checkpoint is supplied to the V2 forward/backward readiness batch.",
        },
        "teacher": {
            "source": "Exact program-construction primitive identity and relations propagated through qualified P1a ray exits and materialized by sealed P1b.",
            "labels": "Primitive presence/cardinality, three axis controls, endpoint half-axes/exponents, five-frame visibility, endpoint attachment and disconnected overlap.",
            "student_forbidden_inputs": "World/parent/traversal/primitive identity, absolute pose, TNG, construction graph, future frames and V1 outputs are unavailable to V2 forward.",
            "valid_mask": "Only qualified visible Teacher primitives are positive; unused 32-slot capacity is supervised as inactive and all relations involving inactive slots are negative.",
            "planner_consistency_plan": "No graph/planner is run. Learned relations must later pass C07/C08 before traversal-only graph edges are evaluated.",
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
        "metrics_and_pre_registered_gates": gates,
        "estimated_cost": {
            "compute": "One RTX 5090 D process in the frozen Torch 2.9.0+cu129/Zarr 2.18.7 environment; 42 tests plus one three-row forward/backward and invariant audits.",
            "wall_time_hours": 0.5, "host_ram_gb": 4, "gpu": 1,
            "gpu_memory_gb": 4, "disk_gb": 0.05,
        },
        "retention": "Keep Data Card/spec, unit log, corrective trigger, real-batch hashes, sparse gradients, six losses, all-parameter gradient audit, relation invariants, environment, RUN_STATE and seal.",
        "failure_policy": "Any drift, nonfinite/missing/zero gradient, sparse-gradient reversal, permutation/reversal/rotation/asymmetry/determinism failure or resource excess stops before training. Do not change data, Teacher, 32-slot capacity, six families or tolerances inside the run.",
    }
    write(CARD, card)

    tools = {
        "runner": RUNNER,
        "model_v2": "src/mtare_topo/representation/primitive_relation_sparse_port_model.py",
        "losses_v2": "src/mtare_topo/representation/primitive_relation_sparse_port_losses.py",
        "model_v1": "src/mtare_topo/representation/primitive_relation_model.py",
        "losses_v1": "src/mtare_topo/representation/primitive_relation_losses.py",
        "reader": "src/mtare_topo/data/primitive_relation_training.py",
        "runner_helpers": "tools/v3/run_primitive_relation_model_readiness_v1.py",
        "v2_tests": "tests/v3/unit/test_primitive_relation_sparse_port.py",
        "v1_tests": "tests/v3/unit/test_primitive_relation_model.py",
        "reader_tests": "tests/v3/unit/test_primitive_relation_training.py",
        "field_tests": "tests/v3/unit/test_swept_superellipse_field.py",
        "variant_tests": "tests/v3/unit/test_geometry_variant_contract.py",
        "governance": "src/mtare_topo/governance.py",
        "preflight": "tools/v3/preflight.py",
        "create_run": "tools/v3/create_run.py",
    }
    inputs = [
        CARD,
        P1A / "RUN_STATE.json", P1A / "metrics/summary.json",
        P1A / "artifacts/task_manifest.json", P1A / "artifacts/evidence_sha256.txt",
        P1B / "RUN_STATE.json", P1B / "metrics/summary.json",
        P1B / "artifacts/task_manifest.json", P1B / "artifacts/evidence_sha256.txt",
        ATTRIBUTION / "RUN_STATE.json",
        ATTRIBUTION / "metrics/attribution/summary.json",
        ATTRIBUTION / "artifacts/evidence_sha256.txt",
        PROJECT_ROOT / "configs/v3/gate2/environments/phase3_torch290_cu129_zarr2187_v1r2.json",
        PROJECT_ROOT / "configs/v3/gate2/environments/phase3_torch290_cu129_zarr2187_pip_freeze.txt",
    ]
    spec = {
        "schema_version": "v3_run_spec_v1", "gate": 3,
        "execution_phase": 3, "operation": "audit", "date": "20260831",
        "slug": "primitive_relation_sparse_port_readiness_v1", "seed": 0,
        "question": "Can the attribution-supported sparse set and endpoint-context relation Transformer satisfy the causal, set-invariant, relation-safe learning interface before any corrective training?",
        "method": "Preserve the five-frame circular LiDAR encoder and 32 swept-primitive slots; train calibrated inactive-slot BCE plus exact cardinality and redundant-description penalty inside primitive_set_parameters; encode endpoint position, outward tangent, section, shape and descriptor with a permutation-safe relation Transformer; predict symmetric attachment/overlap and typed relation uncertainty without rule thresholds.",
        "baseline": "Frozen V1 balanced-existence plus dense all-pair MLP, which passed geometry but activated almost all slots and produced zero safe attachments even under proposal oracle. No baseline is rerun in readiness.",
        "fallback": "If readiness fails, stop before training and correct only the exact model/loss/invariance/resource defect in a new sealed readiness. Do not read C07/C08, lower gates, add event rules or modify Teacher.",
        "corrective_of": str(ATTRIBUTION.relative_to(PROJECT_ROOT)),
        "data_card": str(CARD.relative_to(PROJECT_ROOT)),
        "config_path": str(CARD.relative_to(PROJECT_ROOT)),
        "user_authorization": approval,
        "acceptance_criteria": list(gates.values()),
        "expected_counts": {
            "independent_source_sequences": 1, "paired_geometry_rows": 3,
            "causal_frames": 15, "unique_primitives": 5,
            "directed_attachments": 24, "disconnected_overlaps": 18,
            "temporal_dustbins": 12, "unit_tests": 42,
            "model_parameters": 2629870, "parameter_tensors": 193,
            "loss_families": 6, "optimizer_steps": 0,
            "checkpoint_writes": 0, "c07_c08_data_rows_read": 0,
            "c09_c10_worlds_read": 0, "graph_replays": 0,
            "mtare_worlds_read": 0,
        },
        "expected_evidence": [
            "42-test log, exact attribution trigger and real-batch hashes, sparse-cardinality gradient directions, six losses, 193-tensor finite/nonzero gradient audit, deterministic/query-permutation/target-permutation/endpoint-reversal/rotation/symmetry/uncertainty checks, environment, RUN_STATE and SHA-256 seal."
        ],
        "estimated_cost": card["estimated_cost"],
        "frozen_tools": {
            name: {"path": path, "sha256": sha(PROJECT_ROOT / path)}
            for name, path in tools.items()
        },
        "frozen_inputs": {
            str(path.relative_to(PROJECT_ROOT)): sha(path) for path in inputs
        },
        "working_directory": str(PROJECT_ROOT),
        "command": [
            "/usr/bin/systemd-inhibit", "--what=sleep:shutdown",
            "--why=Primitive relation sparse-port readiness", "--mode=block",
            "/usr/bin/timeout", "--signal=INT", "--kill-after=60s", "1800s",
            "/usr/bin/env", "CUBLAS_WORKSPACE_CONFIG=:4096:8",
            f"PYTHONPATH={PROJECT_ROOT / 'src'}:{PROJECT_ROOT / 'tools/v3'}",
            PYTHON, RUNNER, "--spec", str(SPEC),
            "--run-dir", str(PROJECT_ROOT / f"results/gate3_semantics/{RUN_ID}"),
        ],
    }
    write(SPEC, spec)
    print(SPEC)


if __name__ == "__main__":
    main()
