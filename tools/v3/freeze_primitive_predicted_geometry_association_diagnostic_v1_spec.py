#!/usr/bin/env python3
"""Freeze the three-seed C07 predicted-geometry necessity diagnostic."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

from _bootstrap import PROJECT_ROOT


SOURCE_CARD = PROJECT_ROOT / "configs/v3/gate3/data_cards/primitive_relation_observable_failure_attribution_v1.json"
CARD = PROJECT_ROOT / "configs/v3/gate3/data_cards/primitive_predicted_geometry_association_diagnostic_v1.json"
SPEC = PROJECT_ROOT / "configs/v3/gate3/primitive_predicted_geometry_association_diagnostic_v1.json"
RUN_ID = "gate3_20260903_primitive_predicted_geometry_association_diagnostic_v1_seed0"
P1A = PROJECT_ROOT / "results/gate3_semantics/gate3_20260830_primitive_relation_p1a_lossless_storage_corrective_v1_seed0"
P1B = PROJECT_ROOT / "results/gate3_semantics/gate3_20260830_primitive_relation_p1b_teacher_materialization_v1r_seed0"
SIDECAR = PROJECT_ROOT / "results/gate3_semantics/gate3_20260902_primitive_attachment_observability_sidecar_v1_seed0"
TRAINING = PROJECT_ROOT / "results/gate3_semantics/gate3_20260902_primitive_relation_observable_three_seed_training_v1_seed0"
CORRECTIVE = PROJECT_ROOT / "results/gate3_semantics/gate3_20260902_primitive_relation_observable_c07_tf32_evidence_corrective_v1_seed0"
ATTRIBUTION = PROJECT_ROOT / "results/gate3_semantics/gate3_20260902_primitive_relation_observable_failure_attribution_v1_seed0"
ANCHOR = PROJECT_ROOT / "results/gate3_semantics/gate3_20260903_primitive_composition_anchor_teacher_readiness_v1_seed0"


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
    if CARD.exists() or SPEC.exists():
        raise RuntimeError("predicted-geometry diagnostic card/spec already exists")
    for root in (P1A, P1B, SIDECAR, TRAINING, CORRECTIVE, ATTRIBUTION, ANCHOR):
        if json.loads((root / "RUN_STATE.json").read_text()).get("state") != "COMPLETED":
            raise RuntimeError(f"predicted-geometry source is incomplete: {root.name}")
    training = json.loads((TRAINING / "metrics/summary.json").read_text())
    attribution = json.loads((ATTRIBUTION / "metrics/summary.json").read_text())
    anchor = json.loads((ANCHOR / "metrics/teacher_readiness/summary.json").read_text())
    if training.get("optimizer_steps") != 240_624 or training.get("c08_rows_read") != 0:
        raise RuntimeError("predicted-geometry training source contract drift")
    if (
        attribution.get("model_forward_rows") != 193_932
        or attribution.get("decision") != "STOP_DIRECT_PAIR_RELATION_HEAD_AND_REASSESS_ARCHITECTURE"
    ):
        raise RuntimeError("direct-pair stop evidence is incomplete")
    if (
        anchor.get("scientific_pass") is not True
        or anchor.get("axis_endpoint_baseline", {}).get("c07_transfer", {}).get("precision") != 1.0
        or anchor.get("axis_endpoint_baseline", {}).get("c07_transfer", {}).get("true_positive") != 442_606
    ):
        raise RuntimeError("composition-anchor readiness evidence is incomplete")

    card = copy.deepcopy(json.loads(SOURCE_CARD.read_text()))
    card.pop("corrective_of", None)
    card["card_id"] = "primitive_predicted_geometry_association_diagnostic_v1"
    card["status"] = "APPROVED_FOR_ONE_IMMUTABLE_PRIMITIVE_PREDICTED_GEOMETRY_ASSOCIATION_DIAGNOSTIC_V1"
    card["purpose"] = (
        "Determine whether the actual predicted primitive axis endpoints from the three frozen observable-relation models already recover safe physical connections on world-held-out C07, or whether an additional O(E) learned composition-anchor residual and uncertainty head is scientifically necessary."
    )
    card["approval"] = {
        "status": "APPROVED",
        "approved_by": "user-standing-authorization",
        "approved_at": "2026-09-03T00:30:00+08:00",
        "authorized_gates": [3],
        "authorized_operations": ["audit"],
        "scope": (
            "One immutable zero-training C07-only diagnostic over 10 independent parents, 30 paired geometry tasks, 64644 five-frame rows per frozen seed and 193932 total model-forward rows. Zero C08/C09/C10, graph, M-TARE, optimizer or checkpoint writes."
        ),
        "confirmation_reference": (
            "The user authorized uninterrupted strongest-evidence execution under the frozen primitive-relation paper plan. The near-perfect Teacher-geometry baseline makes this necessity diagnostic mandatory before adding another learned head."
        ),
    }
    card["source"] = {
        "license_or_allowed_use": card["source"]["license_or_allowed_use"],
        "raw_sources": [
            "sealed P1a corrected C07 five-frame 16x720 range/valid data and relative odometry",
            "sealed P1b C07 32-slot construction Teacher and endpoint-observability sidecar",
            "three frozen observable-relation selected checkpoints and corrected TF32-off C07 existence thresholds",
            "sealed direct-pair failure attribution for exact learned-score reproduction",
            "sealed composition-anchor readiness fit-only endpoint-distance threshold",
        ],
        "worlds": [
            "C07 selection only: 10 independent topology parents and 30 paired geometry tasks",
            "C01-C06 are not read; their already-sealed scalar distance threshold is imported unchanged",
        ],
    }
    card["sampling"] = {
        "independent_sampling_units": (
            "10 world-disjoint C07 topology parents; three geometry realizations are paired repeated measures. The same 64644 five-frame rows are evaluated independently by each of three frozen checkpoints."
        ),
        "raw_frame_count": 87_300,
        "effective_sample_count": 10,
        "effective_structure_event_count": 64_644,
        "raw_five_frame_sequence_count": 64_644,
        "spatial_interval_m": 1.0,
        "temporal_window_frames": 5,
        "maximum_primitive_slots": 32,
        "partition_sequences": {
            "fit_c01_c06": 0,
            "selection_c07_per_seed": 64_644,
            "selection_c07_three_seed_forward_rows": 193_932,
            "development_transfer_c08": 0,
        },
        "structure_event_counts": {
            "c07_parent_worlds": 10,
            "c07_geometry_tasks": 30,
            "c07_rows_per_seed": 64_644,
            "frozen_seeds": 3,
            "model_forward_rows": 193_932,
            "c07_observable_positive_pairs_per_seed": 442_936,
        },
        "rule": (
            "Read every C07 row once per frozen seed. Under both the deployed existence mask and the Teacher-aligned proposal-oracle mask, rank cross-primitive dual-observed endpoint pairs by negative Euclidean distance between predicted axis endpoints and separately by the frozen learned pair score. Preserve all tied scores, unmatched active proposals as negatives and hidden matched endpoints as unknown."
        ),
    }
    card["split"] = {
        "fit": "No fit row is read. The scalar 0.499973 m Teacher-fit endpoint-distance threshold is imported unchanged from the sealed readiness run.",
        "selection": (
            "C07: 10 parents/30 tasks/64644 rows per seed. Exact best-F1 and precision>=0.98 prefixes are diagnostic only and may select the next architecture class, not a deployment threshold."
        ),
        "development_transfer": "C08 directory, manifests, arrays and prior results remain unopened.",
        "strict_test": "C09/C10 and all M-TARE benchmark worlds remain forbidden.",
        "world_disjoint": True,
        "trajectory_disjoint": True,
        "historical_pollution_audit": (
            "Only sealed C07 model outputs and prior fit-only scalar threshold are used. C08, online graph and planner evidence cannot influence masks, scores, gates or the decision."
        ),
    }
    card["teacher"] = {
        "source": (
            "Sealed P1b construction provenance supplies evaluation-only primitive alignment and physical endpoint attachment; the endpoint-observability sidecar defines which matched endpoints are visible."
        ),
        "labels": (
            "Dual-observed physical endpoint attachment and disconnected angular-overlap hard negatives. Proposal-oracle primitive activity is an attribution mask only."
        ),
        "valid_mask": (
            "A pair is supervised only when both matched endpoints have direct 0.25 m support. Hidden matched pairs are unknown; active unmatched predictions remain negatives."
        ),
        "student_forbidden_inputs": (
            "Teacher primitive identity, attachment, observability, construction/world/node identity, absolute pose and future frames never enter model forward or endpoint coordinates."
        ),
        "planner_consistency_plan": (
            "No graph or planner runs. If deployed predicted geometry is already safely usable, stop the anchor head; if only proposal-oracle geometry is safe, repair proposal consolidation; otherwise permit only an O(E) anchor residual/uncertainty readiness."
        ),
        "unexplored_port_policy": "Execution state is absent from this perception-only diagnostic.",
    }
    card["leakage_audit"]["model_inference_count"] = 193_932
    card["leakage_audit"]["optimizer_step_count"] = 0
    card["metrics_and_pre_registered_gates"] = {
        "implementation": "All predicted-geometry, attribution and metric unit tests pass under deterministic TF32-off execution.",
        "population": "Exactly 64644 rows and 30 tasks per seed, three selected checkpoints unchanged, 193932 total forward rows and zero C08/C09/C10/graph/M-TARE.",
        "reproduction": "Frozen learned-pair best-F1 TP/FP/FN, threshold, precision, recall and F1 reproduce the sealed attribution for deployed and proposal-oracle masks at all three seeds.",
        "geometry": "Report predicted-endpoint distance best-F1, nonempty precision>=0.98 prefix, fixed Teacher-fit threshold transfer and overlap-hard-negative false positives under both masks.",
        "decision": (
            "At least two deployed-safe seeds stops the anchor head and qualifies geometry composition; otherwise at least two proposal-oracle-safe seeds stops anchor-only work and redirects to proposal consolidation; otherwise permit O(E) anchor residual/uncertainty readiness."
        ),
        "resources": "Wall time <=3h, CUDA reserved <=16GiB, host RSS <=16GiB, output <=0.2GiB and zero optimizer/checkpoint update.",
    }
    card["estimated_cost"] = {
        "compute": "One RTX 5090; one streaming complete C07 pass for each of three frozen checkpoints plus CPU exact ranking.",
        "gpu": 1,
        "gpu_memory_gb": 16,
        "host_ram_gb": 16,
        "wall_time_hours": 1.0,
        "disk_gb": 0.2,
    }
    card["failure_policy"] = (
        "Any checkpoint, population, formal-score reproduction, input/tool/environment, numerical, resource, tie-handling or split-isolation drift fails closed. Do not change masks, thresholds, checkpoints, Teacher, C08 data or planner to force a preferred diagnosis."
    )
    card["retention"] = (
        "Retain per-seed/per-task geometry and learned-score metrics, exact safe selections, fixed-threshold transfer, overlap-hard-negative counts, PNG/PDF/SVG/source, environment, commands, logs, RUN_STATE and SHA-256 seal. Do not retain raw pair-score arrays."
    )
    write(CARD, card)

    run_dir = PROJECT_ROOT / f"results/gate3_semantics/{RUN_ID}"
    spec = {
        "schema_version": "v3_run_spec_v1",
        "slug": "primitive_predicted_geometry_association_diagnostic_v1",
        "date": "20260903",
        "seed": 0,
        "gate": 3,
        "execution_phase": 3,
        "operation": "audit",
        "working_directory": str(PROJECT_ROOT),
        "config_path": str(CARD.relative_to(PROJECT_ROOT)),
        "data_card": str(CARD.relative_to(PROJECT_ROOT)),
        "question": (
            "Do the actual predicted primitive axis endpoints from three frozen models already recover safe C07 physical connections under deployed or proposal-oracle masks, making a learned composition-anchor head unnecessary?"
        ),
        "method": (
            "Three deterministic TF32-off full-C07 inference passes; negative predicted-endpoint distance versus frozen learned pair scores under deployed and proposal-oracle masks; exact tie-safe best-F1 and precision>=0.98 selection plus unchanged fit-only distance threshold transfer."
        ),
        "baseline": (
            "The frozen independent pair head and the fit-only Teacher-axis distance threshold from the sealed composition-anchor readiness proof."
        ),
        "fallback": card["failure_policy"],
        "user_authorization": card["approval"],
        "acceptance_criteria": list(card["metrics_and_pre_registered_gates"].values()),
        "expected_counts": {
            "c07_parent_worlds": 10,
            "c07_geometry_tasks": 30,
            "c07_rows_per_seed": 64_644,
            "frozen_seeds": 3,
            "model_forward_rows": 193_932,
            "optimizer_steps": 0,
            "c08_rows_read": 0,
            "c09_c10_worlds_read": 0,
            "graph_replays": 0,
            "mtare_worlds_read": 0,
        },
        "estimated_cost": card["estimated_cost"],
        "expected_evidence": [
            "Exact sealed learned-pair reproduction and complete C07 population proof.",
            "Per-seed deployed/proposal-oracle predicted-geometry best and safe selections.",
            "Unchanged fit-only distance threshold transfer and overlap-hard-negative diagnostics.",
            "One pre-registered architecture decision, plots, environment, commands, logs, RUN_STATE and seal.",
        ],
        "command": [
            "/usr/bin/systemd-inhibit",
            "--what=sleep:shutdown",
            "--why=Primitive predicted geometry association diagnostic",
            "--mode=block",
            "/usr/bin/timeout",
            "--signal=INT",
            "--kill-after=60s",
            "10800s",
            "/usr/bin/env",
            "CUBLAS_WORKSPACE_CONFIG=:4096:8",
            f"PYTHONPATH={PROJECT_ROOT / 'src'}:{PROJECT_ROOT / 'tools/v3'}:{PROJECT_ROOT}",
            "/home/zeng-workstation/.local/share/mtare_topo_comm/envs/phase3_torch290_cu129_zarr2187_v1/bin/python",
            "tools/v3/run_primitive_predicted_geometry_association_diagnostic_v1.py",
            "--spec",
            str(SPEC),
            "--run-dir",
            str(run_dir),
        ],
    }
    inputs = [
        PROJECT_ROOT / "configs/v3/gate2/environments/phase3_torch290_cu129_zarr2187_pip_freeze.txt",
        PROJECT_ROOT / "configs/v3/gate2/environments/phase3_torch290_cu129_zarr2187_v1r2.json",
        PROJECT_ROOT / "docs/PRIMITIVE_RELATION_STRUCTURAL_GRAPH_PLAN_V1.md",
        CARD,
        *[
            root / name
            for root in (P1A, P1B, SIDECAR, TRAINING, CORRECTIVE, ATTRIBUTION, ANCHOR)
            for name in ("RUN_STATE.json", "metrics/summary.json", "artifacts/evidence_sha256.txt")
        ],
        P1A / "artifacts/task_manifest.json",
        P1B / "artifacts/task_manifest.json",
        SIDECAR / "artifacts/task_manifest.json",
        TRAINING / "config/source_integrity_after.json",
        ANCHOR / "metrics/teacher_readiness/summary.json",
        *[TRAINING / f"artifacts/models/seed{seed}/selected.pt" for seed in range(3)],
        *[CORRECTIVE / f"metrics/source_evaluation/c07_seed{seed}.json" for seed in range(3)],
    ]
    spec["frozen_inputs"] = {
        str(path.relative_to(PROJECT_ROOT)): sha(path) for path in inputs
    }
    tools = {
        "predicted_geometry": "src/mtare_topo/evaluation/primitive_predicted_geometry_association.py",
        "predicted_geometry_tests": "tests/v3/unit/test_primitive_predicted_geometry_association.py",
        "attribution_module": "src/mtare_topo/evaluation/primitive_relation_observable_failure_attribution.py",
        "attribution_tests": "tests/v3/unit/test_primitive_relation_observable_failure_attribution.py",
        "ranked_selection": "src/mtare_topo/evaluation/primitive_relation_sparse_port_failure_attribution.py",
        "metrics": "src/mtare_topo/evaluation/primitive_relation_metrics.py",
        "metric_tests": "tests/v3/unit/test_primitive_relation_metrics.py",
        "model": "src/mtare_topo/representation/primitive_relation_observable_model.py",
        "base_model": "src/mtare_topo/representation/primitive_relation_sparse_port_model.py",
        "loss_alignment": "src/mtare_topo/representation/primitive_relation_losses.py",
        "batch_reader": "src/mtare_topo/data/primitive_relation_observable_batches.py",
        "base_batch_reader": "src/mtare_topo/data/primitive_relation_batches.py",
        "training_conversion": "src/mtare_topo/representation/primitive_relation_observable_training.py",
        "executor": "tools/v3/execute_primitive_predicted_geometry_association_diagnostic_v1.py",
        "runner": "tools/v3/run_primitive_predicted_geometry_association_diagnostic_v1.py",
        "freezer": "tools/v3/freeze_primitive_predicted_geometry_association_diagnostic_v1_spec.py",
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
