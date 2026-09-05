#!/usr/bin/env python3
"""Freeze one zero-training O(E) composition-anchor model readiness run."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

from _bootstrap import PROJECT_ROOT


SOURCE_CARD = PROJECT_ROOT / "configs/v3/gate3/data_cards/primitive_predicted_geometry_association_diagnostic_v1.json"
CARD = PROJECT_ROOT / "configs/v3/gate3/data_cards/primitive_composition_anchor_model_readiness_v2.json"
SPEC = PROJECT_ROOT / "configs/v3/gate3/primitive_composition_anchor_model_readiness_v2.json"
RUN_ID = "gate3_20260903_primitive_composition_anchor_model_readiness_v2_seed0"
TASK = "S01_flat_tree_small_C01__c1_mixed"
P1A = PROJECT_ROOT / "results/gate3_semantics/gate3_20260830_primitive_relation_p1a_lossless_storage_corrective_v1_seed0"
P1B = PROJECT_ROOT / "results/gate3_semantics/gate3_20260830_primitive_relation_p1b_teacher_materialization_v1r_seed0"
SIDECAR = PROJECT_ROOT / "results/gate3_semantics/gate3_20260902_primitive_attachment_observability_sidecar_v1_seed0"
TRAINING = PROJECT_ROOT / "results/gate3_semantics/gate3_20260902_primitive_relation_observable_three_seed_training_v1_seed0"
ANCHOR_TEACHER = PROJECT_ROOT / "results/gate3_semantics/gate3_20260903_primitive_composition_anchor_teacher_readiness_v1_seed0"
DIAGNOSTIC = PROJECT_ROOT / "results/gate3_semantics/gate3_20260903_primitive_predicted_geometry_association_diagnostic_v1_seed0"
FAILED_V1 = PROJECT_ROOT / "results/gate3_semantics/gate3_20260903_primitive_composition_anchor_model_readiness_v1_seed0"
FAILED_V1R = PROJECT_ROOT / "results/gate3_semantics/gate3_20260903_primitive_composition_anchor_model_readiness_v1r_seed0"
FAILED_V1R2 = PROJECT_ROOT / "results/gate3_semantics/gate3_20260903_primitive_composition_anchor_model_readiness_v1r2_seed0"


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tree_hash(path: Path) -> str:
    digest = hashlib.sha256()
    for value in sorted(item for item in path.rglob("*") if item.is_file()):
        relative = value.relative_to(path).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(4, "little"))
        digest.update(relative)
        digest.update(bytes.fromhex(sha(value)))
    return digest.hexdigest()


def write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    if CARD.exists() or SPEC.exists():
        raise RuntimeError("composition-anchor model readiness card/spec already exists")
    diagnostic = json.loads((DIAGNOSTIC / "metrics/diagnostic/summary.json").read_text())
    anchor = json.loads((ANCHOR_TEACHER / "metrics/teacher_readiness/summary.json").read_text())
    if (
        diagnostic.get("decision")
        != "ALLOW_OE_COMPOSITION_ANCHOR_RESIDUAL_UNCERTAINTY_MODEL_READINESS"
        or diagnostic.get("proposal_oracle_geometry_safe_passing_seeds") != 0
        or anchor.get("scientific_pass") is not True
    ):
        raise RuntimeError("composition-anchor model readiness trigger is incomplete")

    card = copy.deepcopy(json.loads(SOURCE_CARD.read_text()))
    card.pop("corrective_of", None)
    failed = json.loads((FAILED_V1 / "metrics/summary.json").read_text())
    if (
        json.loads((FAILED_V1 / "RUN_STATE.json").read_text()).get("state") != "FAILED"
        or failed.get("error") != "ValueError: composition-anchor endpoint tangent is near vertical"
        or failed.get("optimizer_steps") != 0
        or failed.get("c07_rows_read") != 0
    ):
        raise RuntimeError("composition-anchor V1 system failure evidence is incomplete")
    failed_v1r = json.loads((FAILED_V1R / "metrics/summary.json").read_text())
    if (
        json.loads((FAILED_V1R / "RUN_STATE.json").read_text()).get("state") != "FAILED"
        or failed_v1r.get("error") != "AttributeError: 'numpy.ndarray' object has no attribute 'int'"
        or failed_v1r.get("optimizer_steps") != 0
        or failed_v1r.get("c07_rows_read") != 0
    ):
        raise RuntimeError("composition-anchor V1R post-compute failure evidence is incomplete")
    failed_v1r2 = json.loads((FAILED_V1R2 / "metrics/summary.json").read_text())
    failed_checks = failed_v1r2.get("checks", {})
    if (
        json.loads((FAILED_V1R2 / "RUN_STATE.json").read_text()).get("state") != "COMPLETED"
        or failed_v1r2.get("error") is not None
        or failed_v1r2.get("scientific_pass") is not False
        or failed_v1r2.get("decision") != "STOP_COMPOSITION_ANCHOR_MODEL_READINESS_FAILED"
        or failed_checks.get("all_anchor_head_gradients_finite_nonzero") is not True
        or failed_checks.get("degenerate_fallback_exercised_only_on_unmatched_unobserved_queries") is not False
        or failed_checks.get("conditional_yaw_equivariance_le_3e5") is not False
    ):
        raise RuntimeError("composition-anchor V1R2 scientific failure evidence is incomplete")
    card["card_id"] = "primitive_composition_anchor_model_readiness_v2"
    card["status"] = "APPROVED_FOR_ONE_IMMUTABLE_PRIMITIVE_COMPOSITION_ANCHOR_MODEL_READINESS_V2"
    card["supersedes"] = {
        "run": str(FAILED_V1R2.relative_to(PROJECT_ROOT)),
        "scientific_failure": (
            "An axis-anchored residual frame was unstable for 12 predicted endpoints, including one matched and observed endpoint; full-wrapper yaw and query-permutation contracts failed."
        ),
        "method_change": (
            "Use a current-sensor polar [radial,lateral,gravity] frame, remove tangent/segment/bend features and initialize the learned Gaussian temperature at softplus(-5). Target, backbone, output capacity and loss populations remain unchanged."
        ),
    }
    card["corrective_history"] = [{
        "run": str(FAILED_V1.relative_to(PROJECT_ROOT)),
        "failure": failed["error"],
        "resolution": "Sensor-radial yaw-equivariant fallback for unmatched collapsed queries, already retained unchanged in V1R2.",
    }]
    card["purpose"] = (
        "Implement and falsify-test the minimum O(E) relation correction allowed by the sealed C07 diagnosis: a frozen mature primitive backbone plus one local-frame composition-anchor residual and isotropic uncertainty per endpoint, with symmetric Gaussian compatibility and no identity or hand-written event rule."
    )
    card["approval"] = {
        "status": "APPROVED",
        "approved_by": "user-standing-authorization",
        "approved_at": "2026-09-03T01:40:00+08:00",
        "authorized_gates": [3],
        "authorized_operations": ["audit"],
        "scope": (
            "One immutable V2 zero-training readiness using the same predeclared C01 fit sequence only. Test the polar-frame head and decomposed incremental-versus-inherited equivariance evidence; write no checkpoint. Zero C07/C08/C09/C10, graph or M-TARE."
        ),
        "confirmation_reference": (
            "The user authorized uninterrupted strongest-evidence work under the frozen paper plan. The formal three-seed diagnostic triggered the pre-registered O(E) anchor-readiness branch."
        ),
    }
    card["source"] = {
        "license_or_allowed_use": card["source"]["license_or_allowed_use"],
        "raw_sources": [
            "one sealed P1a C01 fit five-frame LiDAR/odometry row and its current sensor pose",
            "the aligned sealed P1b primitive Teacher and endpoint-observability bits",
            "the realized C01 construction JSON supplying endpoint composition anchors",
            "the frozen seed0 observable-relation checkpoint as a mature no-gradient backbone",
            "sealed anchor-Teacher readiness and predicted-geometry necessity diagnosis",
        ],
        "worlds": ["C01 fit only: S01 flat-tree-small parent, c1-mixed paired geometry task, row 5"],
    }
    card["sampling"] = {
        "independent_sampling_units": "One predeclared fit-world sequence is an implementation proof only, not a performance estimate.",
        "raw_frame_count": 5,
        "effective_sample_count": 1,
        "effective_structure_event_count": 1,
        "raw_five_frame_sequence_count": 1,
        "spatial_interval_m": 1.0,
        "temporal_window_frames": 5,
        "maximum_primitive_slots": 32,
        "partition_sequences": {
            "fit_c01_real_readiness": 1,
            "selection_c07": 0,
            "development_transfer_c08": 0,
        },
        "structure_event_counts": {
            "fit_parent_worlds": 1,
            "fit_geometry_tasks": 1,
            "real_rows": 1,
            "active_primitives": 11,
            "observed_endpoints": 17,
            "directed_attachment_entries": 22,
        },
        "rule": (
            "Use exactly task S01_flat_tree_small_C01__c1_mixed row 5. Hungarian-align the frozen prediction to the existing primitive Teacher, reverse endpoint targets when required, and test the new polar-frame head without any optimizer step or checkpoint write."
        ),
    }
    card["split"] = {
        "fit": "Exactly one C01 row for implementation readiness; no parameter update and no metric/threshold selection.",
        "selection": "C07 is not read.",
        "development_transfer": "C08 is not read.",
        "strict_test": "C09/C10 and all M-TARE benchmark worlds remain forbidden.",
        "world_disjoint": True,
        "trajectory_disjoint": True,
        "historical_pollution_audit": (
            "The architecture branch was selected solely by the sealed three-seed C07 diagnosis. The real C01 row is fixed before execution and cannot select model capacity or acceptance."
        ),
    }
    card["teacher"] = {
        "source": (
            "PrimitiveConstructionGraph composition_anchor_xyz_m, transformed offline by the current sensor pose and aligned only for the loss."
        ),
        "labels": (
            "Per-observed-endpoint current-sensor-frame composition anchor, dual-observed physical attachment and disconnected-overlap hard negative."
        ),
        "valid_mask": (
            "Anchor regression and compatibility use only dual endpoint support from the sealed 0.25 m observability sidecar; hidden pairs remain unknown."
        ),
        "student_forbidden_inputs": (
            "Construction/world/node/primitive identity, absolute pose, Teacher anchor, attachment and future frames are absent from model forward. Only the frozen primitive prediction enters the new head."
        ),
        "planner_consistency_plan": (
            "No graph or planner run. Edges will remain traversal-only if later training and C07/C08 qualification pass."
        ),
        "unexplored_port_policy": "Absent from this zero-training perception readiness.",
    }
    card["leakage_audit"]["model_inference_count"] = 1
    card["leakage_audit"]["optimizer_step_count"] = 0
    card["metrics_and_pre_registered_gates"] = {
        "implementation": "Exactly 20 model/Teacher/observable unit tests pass and the selected real row/data hashes match.",
        "interface": "The head emits exactly 256 learned values per row, 7.75x fewer than 1984 independent pair values; forward has no Teacher identity.",
        "geometry": "Polar residual, scale and compatibility incremental query-permutation error and conditional yaw error are <=3e-5. Absolute anchors may inherit at most 1e-4 m from the frozen query decoder. The real row contains an observed degenerate axis endpoint but no observed endpoint with undefined sensor-radial direction.",
        "loss": "Anchor Gaussian NLL, uncertainty calibration, balanced compatibility and overlap-hard-negative losses are finite on a real row containing both positive and hard-negative pairs.",
        "gradient": "Every anchor-head parameter receives finite nonzero gradient; every frozen backbone parameter receives no gradient.",
        "resources": "Zero optimizer/checkpoint/C07/C08/graph/planner, CUDA allocation <=4GiB, wall <=0.5h and output <=0.1GiB.",
    }
    card["estimated_cost"] = {
        "compute": "One RTX 5090 forward/backward on one fixed real C01 sequence plus unit tests and plotting.",
        "gpu": 1,
        "gpu_memory_gb": 4,
        "host_ram_gb": 4,
        "wall_time_hours": 0.1,
        "disk_gb": 0.1,
    }
    card["failure_policy"] = (
        "Any source/data hash, target alignment, output complexity, symmetry/equivariance, real hard-negative, finite-loss/gradient, frozen-backbone, identity-isolation or resource failure stops the anchor model before training. Do not add pair rules, relax the safety gate or read C07/C08 to repair readiness."
    )
    card["retention"] = (
        "Retain real-row contract, losses, gradients/equivariance/resource metrics, clearly marked untrained PNG/PDF/SVG/source, environment, logs, RUN_STATE and seal. No new checkpoint or copied sensor data."
    )
    write(CARD, card)

    run_dir = PROJECT_ROOT / f"results/gate3_semantics/{RUN_ID}"
    spec = {
        "schema_version": "v3_run_spec_v1",
        "slug": "primitive_composition_anchor_model_readiness_v2",
        "date": "20260903",
        "seed": 0,
        "gate": 3,
        "execution_phase": 3,
        "operation": "audit",
        "working_directory": str(PROJECT_ROOT),
        "config_path": str(CARD.relative_to(PROJECT_ROOT)),
        "data_card": str(CARD.relative_to(PROJECT_ROOT)),
        "question": (
            "Does a sensor-polar O(E) composition-anchor head remain total and equivariant on the observed degenerate-axis endpoint while preserving identity isolation, hard-negative learning and the frozen backbone boundary?"
        ),
        "method": (
            "Frozen seed0 observable primitive backbone plus endpoint descriptor/range/shape/evidence invariants, sensor-polar radial/lateral/gravity residual frame, low-slope learnable Gaussian calibration, symmetric compatibility and balanced overlap-hard-negative loss."
        ),
        "baseline": (
            "Stopped independent O(E^2) pair head, stopped K=32 dense slot decoder and failed raw predicted-endpoint distance association."
        ),
        "fallback": card["failure_policy"],
        "user_authorization": card["approval"],
        "acceptance_criteria": list(card["metrics_and_pre_registered_gates"].values()),
        "expected_counts": {
            "fit_parent_worlds": 1,
            "fit_geometry_tasks": 1,
            "fit_rows_read": 1,
            "model_forward_rows": 4,
            "optimizer_steps": 0,
            "checkpoint_writes": 0,
            "c07_rows_read": 0,
            "c08_rows_read": 0,
            "c09_c10_worlds_read": 0,
            "graph_replays": 0,
            "mtare_worlds_read": 0,
        },
        "estimated_cost": card["estimated_cost"],
        "expected_evidence": [
            "Exact real C01 construction-anchor alignment and positive/hard-negative pair proof.",
            "O(E) output count, deterministic permutation/reversal/yaw contracts and identity-free interface.",
            "Explicit proof that an observed collapsed-axis endpoint is handled by a defined sensor-polar frame, with no observed radial degeneracy.",
            "Finite real-batch losses, all-head/no-backbone gradient boundary and resource evidence.",
            "Untrained readiness plot/source, environment, logs, RUN_STATE and SHA-256 seal.",
        ],
        "command": [
            "/usr/bin/systemd-inhibit",
            "--what=sleep:shutdown",
            "--why=Primitive composition anchor model readiness",
            "--mode=block",
            "/usr/bin/timeout",
            "--signal=INT",
            "--kill-after=60s",
            "1800s",
            "/usr/bin/env",
            "CUBLAS_WORKSPACE_CONFIG=:4096:8",
            f"PYTHONPATH={PROJECT_ROOT / 'src'}:{PROJECT_ROOT / 'tools/v3'}:{PROJECT_ROOT}",
            "/home/zeng-workstation/.local/share/mtare_topo_comm/envs/phase3_torch290_cu129_zarr2187_v1/bin/python",
            "tools/v3/run_primitive_composition_anchor_model_readiness_v1.py",
            "--spec",
            str(SPEC),
            "--run-dir",
            str(run_dir),
        ],
    }
    selected = {
        "sensor": tree_hash(P1A / "artifacts/dataset/fit" / f"{TASK}.zarr"),
        "teacher": tree_hash(P1B / "artifacts/teacher/fit" / f"{TASK}.zarr"),
        "sidecar": tree_hash(SIDECAR / "artifacts/endpoint_observability/fit" / f"{TASK}.zarr"),
        "construction": sha(P1A / "artifacts/constructions/fit" / f"{TASK}.json"),
    }
    spec["selected_real_data_hashes"] = selected
    inputs = [
        PROJECT_ROOT / "configs/v3/gate2/environments/phase3_torch290_cu129_zarr2187_pip_freeze.txt",
        PROJECT_ROOT / "configs/v3/gate2/environments/phase3_torch290_cu129_zarr2187_v1r2.json",
        PROJECT_ROOT / "docs/PRIMITIVE_RELATION_STRUCTURAL_GRAPH_PLAN_V1.md",
        CARD,
        *[
            root / name
            for root in (P1A, P1B, SIDECAR, TRAINING, ANCHOR_TEACHER, DIAGNOSTIC, FAILED_V1, FAILED_V1R, FAILED_V1R2)
            for name in ("RUN_STATE.json", "metrics/summary.json", "artifacts/evidence_sha256.txt")
        ],
        P1A / "artifacts/task_manifest.json",
        P1B / "artifacts/task_manifest.json",
        SIDECAR / "artifacts/task_manifest.json",
        TRAINING / "artifacts/models/seed0/selected.pt",
        ANCHOR_TEACHER / "metrics/teacher_readiness/summary.json",
        DIAGNOSTIC / "metrics/diagnostic/summary.json",
    ]
    spec["frozen_inputs"] = {
        str(path.relative_to(PROJECT_ROOT)): sha(path) for path in inputs
    }
    tools = {
        "anchor_model": "src/mtare_topo/representation/primitive_composition_anchor_model.py",
        "anchor_training": "src/mtare_topo/representation/primitive_composition_anchor_training.py",
        "anchor_model_tests": "tests/v3/unit/test_primitive_composition_anchor_model.py",
        "anchor_teacher": "src/mtare_topo/evaluation/primitive_composition_anchor_teacher.py",
        "anchor_teacher_tests": "tests/v3/unit/test_primitive_composition_anchor_teacher.py",
        "observable_model": "src/mtare_topo/representation/primitive_relation_observable_model.py",
        "observable_model_tests": "tests/v3/unit/test_primitive_relation_observable_model.py",
        "sparse_model": "src/mtare_topo/representation/primitive_relation_sparse_port_model.py",
        "base_model": "src/mtare_topo/representation/primitive_relation_model.py",
        "loss_alignment": "src/mtare_topo/representation/primitive_relation_losses.py",
        "batch_reader": "src/mtare_topo/data/primitive_relation_observable_batches.py",
        "base_batch_reader": "src/mtare_topo/data/primitive_relation_batches.py",
        "batch_conversion": "src/mtare_topo/representation/primitive_relation_observable_training.py",
        "runner": "tools/v3/run_primitive_composition_anchor_model_readiness_v1.py",
        "freezer": "tools/v3/freeze_primitive_composition_anchor_model_readiness_v1_spec.py",
        "governance": "src/mtare_topo/governance.py",
        "preflight": "tools/v3/preflight.py",
        "create_run": "tools/v3/create_run.py",
    }
    spec["frozen_tools"] = {
        name: {"path": path, "sha256": sha(PROJECT_ROOT / path)}
        for name, path in tools.items()
    }
    write(SPEC, spec)
    print(SPEC)


if __name__ == "__main__":
    main()
