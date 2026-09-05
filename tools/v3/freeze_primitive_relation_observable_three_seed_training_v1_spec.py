#!/usr/bin/env python3
"""Freeze relation-only three-seed training after corrected C07 baseline."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

from _bootstrap import PROJECT_ROOT


CARD = PROJECT_ROOT / "configs/v3/gate3/data_cards/primitive_relation_observable_three_seed_training_v1.json"
SPEC = PROJECT_ROOT / "configs/v3/gate3/primitive_relation_observable_three_seed_training_v1.json"
SOURCE_CARD = PROJECT_ROOT / "configs/v3/gate3/data_cards/primitive_relation_sparse_port_three_seed_training_v1r.json"
RUN_ID = "gate3_20260902_primitive_relation_observable_three_seed_training_v1_seed0"
PYTHON = "/home/zeng-workstation/.local/share/mtare_topo_comm/envs/phase3_torch290_cu129_zarr2187_v1/bin/python"
P1A = PROJECT_ROOT / "results/gate3_semantics/gate3_20260830_primitive_relation_p1a_lossless_storage_corrective_v1_seed0"
P1B = PROJECT_ROOT / "results/gate3_semantics/gate3_20260830_primitive_relation_p1b_teacher_materialization_v1r_seed0"
SIDECAR = PROJECT_ROOT / "results/gate3_semantics/gate3_20260902_primitive_attachment_observability_sidecar_v1_seed0"
READINESS = PROJECT_ROOT / "results/gate3_semantics/gate3_20260902_primitive_relation_observable_readiness_v1r_seed0"
BASELINE = PROJECT_ROOT / "results/gate3_semantics/gate3_20260902_primitive_relation_observable_nonlearning_c07_v1_seed0"
V2 = PROJECT_ROOT / "results/gate3_semantics/gate3_20260901_primitive_relation_sparse_port_three_seed_training_v1r_seed0"
ATTRIBUTION = PROJECT_ROOT / "results/gate3_semantics/gate3_20260902_primitive_relation_sparse_port_failure_attribution_evidence_corrective_v1_seed0"


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
        raise RuntimeError("observable training card/spec already exists")
    for source in (P1A, P1B, SIDECAR, READINESS, BASELINE, ATTRIBUTION):
        state = json.loads((source / "RUN_STATE.json").read_text())
        summary = json.loads((source / "metrics/summary.json").read_text())
        if (
            state.get("state") != "COMPLETED" or state.get("error") is not None
            or not summary.get("scientific_pass")
        ):
            raise RuntimeError(f"observable training prerequisite failed: {source.name}")
    v2_state = json.loads((V2 / "RUN_STATE.json").read_text())
    v2_summary = json.loads((V2 / "metrics/summary.json").read_text())
    if (
        v2_state.get("state") != "COMPLETED" or v2_state.get("error") is not None
        or v2_summary.get("scientific_pass") is not False
        or int(v2_summary.get("c08_rows_read", -1)) != 0
    ):
        raise RuntimeError("V2 source checkpoint contract drift")

    approval = {
        "status": "APPROVED", "approved_by": "user-standing-authorization",
        "approved_at": "2026-09-02T20:00:00+08:00",
        "authorized_gates": [3], "authorized_operations": ["training"],
        "scope": "One immutable three-seed relation-only training on all C01-C06 fit sequences with C07 selection, using frozen V2 geometry checkpoints and the sealed endpoint-observability sidecar. Zero C08+, graph or M-TARE.",
        "confirmation_reference": "The user authorized uninterrupted best-in-plan execution without repeated routine approvals.",
    }
    gates = {
        "population": "Every seed reads exactly 426552 fit rows and 64644 C07 rows per epoch for three epochs; seeds are 0/1/2 and total optimizer steps are 240624.",
        "transfer": "Each seed transfers its own V2 selected geometry/temporal state exactly; only 742149 parameters in 64 relation/evidence tensors train and the frozen-state digest is unchanged.",
        "objective": "Only the mean of the registered port-relations and uncertainty-calibration families drives updates; sparse cardinality and all six families remain reported without adding a seventh loss.",
        "geometry": "C07 sampled-surface error and continuous geometry macro improve at least 10% over the corrected same-input non-learning baseline, with primitive F1 and target coverage not below baseline.",
        "relations": "C07 observable attachment F1 improves at least 5 points over the corrected non-learning baseline and the learned safe score yields at least one true attachment at precision >=0.98.",
        "seeds": "At least two of three independent seeds pass every geometry, primitive, coverage, relation and nonempty-safe gate in the same direction.",
        "isolation": "C08 rows, C09/C10 worlds, graph replays and M-TARE reads remain exactly zero; no threshold, checkpoint or method choice uses them.",
        "resources": "Wall time <=24 h, host RSS/GPU process/PyTorch allocation each <=16 GiB and result <=6 GiB.",
    }
    card = copy.deepcopy(json.loads(SOURCE_CARD.read_text()))
    card.update({
        "card_id": "primitive_relation_observable_three_seed_training_v1",
        "status": "APPROVED_FOR_ONE_IMMUTABLE_PRIMITIVE_RELATION_OBSERVABLE_THREE_SEED_TRAINING_V1",
        "approval": approval,
        "purpose": "Test whether correcting endpoint observability and learning deployment-time endpoint evidence converts the proven V2 geometry representation into reliable physical primitive connections without retraining the backbone.",
        "metrics_and_pre_registered_gates": gates,
        "estimated_cost": {
            "compute": "One RTX 5090 D; three serial seeds, three full fit epochs per seed, C07 loss selection after every epoch and one final three-seed C07 evaluation.",
            "wall_time_hours": 8.0, "host_ram_gb": 16, "gpu": 1,
            "gpu_memory_gb": 16, "disk_gb": 6,
        },
        "retention": "Keep all three selected/epoch checkpoints, histories, commands, resource monitors, complete C07 metrics, paper figures, environment, logs, RUN_STATE and seal.",
        "failure_policy": "Any source, population, frozen-state, gradient, optimizer-step, resource or isolation failure fails the run. A scientific C07 failure stops before C08/graph; do not add epochs, tune thresholds on C08, select seeds or add rules.",
    })
    card["source"]["raw_sources"] = [
        "sealed P1a C01-C07 causal LiDAR shards",
        "sealed P1b construction primitive/relation Teacher shards",
        "sealed C01-C07 endpoint-observability sidecars",
        "three V2 selected checkpoints used only for geometry/temporal transfer",
        "sealed corrected same-input non-learning C07 baseline",
    ]
    card["sampling"].update({
        "independent_sampling_units": "60 C01-C06 topology parents for fit and 10 disjoint C07 parents for selection; three geometry realizations are paired repeated measures within a parent.",
        "raw_frame_count": 659_940, "effective_sample_count": 163_732,
        "effective_structure_event_count": 491_196,
        "spatial_interval_m": 1.0, "temporal_window_frames": 5,
        "geometry_realizations": 3,
        "rule": "For each seed run exactly three deterministic full fit epochs. Evaluate all C07 rows after each epoch and select the minimum mean(port_relations, uncertainty_calibration). Final thresholds and science gates use C07 once after all checkpoints are frozen.",
    })
    card["sampling"]["structure_event_counts"] = {
        "fit_parent_worlds": 60, "c07_parent_worlds": 10,
        "fit_geometry_tasks": 180, "c07_geometry_tasks": 30,
        "fit_sequences_per_epoch": 426_552,
        "c07_sequences_per_epoch": 64_644,
        "fit_batches_per_epoch": 26_736,
        "c07_batches_per_epoch": 522,
        "epochs_per_seed": 3, "seeds": 3,
        "optimizer_steps_per_seed": 80_208,
        "optimizer_steps_total": 240_624,
        "fit_observable_positive_attachments": 2_782_487,
        "c07_observable_positive_attachments": 442_936,
    }
    card["teacher"] = {
        "source": "P1b construction geometry/relations plus the sealed 0.25 m per-window endpoint-support validity sidecar.",
        "labels": "Swept primitive geometry, temporal visibility, attachment, disconnected overlap and endpoint observability.",
        "valid_mask": "A matched physical attachment trains/scores only when both endpoints are observed. Hidden physical connections are unknown, never negative; unmatched sparse queries remain valid negative regularization.",
        "student_forbidden_inputs": "Endpoint support bits are loss/evaluator targets only. Construction identity, world/TNG, absolute pose, true endpoints and future frames never enter forward.",
        "planner_consistency_plan": "The model predicts endpoint evidence at deployment. No graph runs here; only a later actual traversal can commit an edge.",
    }
    card["leakage_audit"].update({
        "optimizer_step_count": 240_624,
        "model_inference_count": 3 * 3 * 64_644 + 3 * 2 * 64_644,
        "test_excluded_from_checkpoint_selection": True,
    })
    card["split"].update({
        "fit": "All C01-C06 parents and three paired geometries provide gradients.",
        "selection": "All C07 parents select one checkpoint per seed and final thresholds; C07 never changes the data, support band, architecture, epoch count or gates.",
        "development_transfer": "C08 is not opened by this run even if C07 passes; the decision only authorizes a separately sealed C08 endpoint-observability sidecar and zero-adaptation transfer.",
        "strict_test": "C09/C10 and M-TARE benchmark worlds remain unread.",
        "historical_pollution_audit": "V2 C07 failures justify only the frozen observability correction. V2 relation weights and predictions are discarded; only geometry/temporal tensors transfer.",
    })
    write(CARD, card)

    inputs = [
        CARD,
        *[
            source / name
            for source in (P1A, P1B, SIDECAR, READINESS, BASELINE, V2, ATTRIBUTION)
            for name in ("RUN_STATE.json", "metrics/summary.json", "artifacts/evidence_sha256.txt")
        ],
        P1A / "artifacts/task_manifest.json",
        P1B / "artifacts/task_manifest.json",
        SIDECAR / "artifacts/task_manifest.json",
        ATTRIBUTION / "metrics/attribution/summary.json",
        *[V2 / f"artifacts/models/seed{seed}/selected.pt" for seed in (0, 1, 2)],
        PROJECT_ROOT / "configs/v3/gate2/environments/phase3_torch290_cu129_zarr2187_v1r2.json",
        PROJECT_ROOT / "configs/v3/gate2/environments/phase3_torch290_cu129_zarr2187_pip_freeze.txt",
    ]
    tools = {
        "runner": "tools/v3/run_primitive_relation_observable_three_seed_training_v1.py",
        "freezer": "tools/v3/freeze_primitive_relation_observable_three_seed_training_v1_spec.py",
        "trainer": "tools/v3/train_primitive_relation_observable_v1.py",
        "evaluator": "tools/v3/evaluate_primitive_relation_observable_three_seed_v1.py",
        "old_runner_helper": "tools/v3/run_primitive_relation_sparse_port_three_seed_training_v1.py",
        "resource_monitor_helper": "tools/v3/run_primitive_relation_sparse_port_three_seed_training_v1r.py",
        "observable_reader": "src/mtare_topo/data/primitive_relation_observable_batches.py",
        "sidecar": "src/mtare_topo/data/primitive_attachment_observability_sidecar.py",
        "loss_base": "src/mtare_topo/representation/primitive_relation_losses.py",
        "loss_sparse": "src/mtare_topo/representation/primitive_relation_sparse_port_losses.py",
        "loss_observable": "src/mtare_topo/representation/primitive_relation_observable_training.py",
        "schedule": "src/mtare_topo/representation/primitive_relation_observable_schedule.py",
        "model": "src/mtare_topo/representation/primitive_relation_observable_model.py",
        "initialization": "src/mtare_topo/representation/primitive_relation_observable_initialization.py",
        "metrics": "src/mtare_topo/evaluation/primitive_relation_metrics.py",
        "test_observability": "tests/v3/unit/test_primitive_attachment_observability.py",
        "test_sidecar": "tests/v3/unit/test_primitive_attachment_observability_sidecar.py",
        "test_reader": "tests/v3/unit/test_primitive_relation_observable_batches.py",
        "test_model": "tests/v3/unit/test_primitive_relation_observable_model.py",
        "test_initialization": "tests/v3/unit/test_primitive_relation_observable_initialization.py",
        "test_schedule": "tests/v3/unit/test_primitive_relation_observable_schedule.py",
        "test_metrics": "tests/v3/unit/test_primitive_relation_metrics.py",
        "test_evaluator": "tests/v3/unit/test_evaluate_primitive_relation_observable_three_seed_v1.py",
        "test_resource": "tests/v3/unit/test_primitive_relation_sparse_port_resource_monitor_v1r.py",
        "test_sparse": "tests/v3/unit/test_primitive_relation_sparse_port.py",
        "test_v1": "tests/v3/unit/test_primitive_relation_model.py",
        "governance": "src/mtare_topo/governance.py",
        "preflight": "tools/v3/preflight.py", "create_run": "tools/v3/create_run.py",
    }
    expected = {
        "fit_parent_worlds": 60, "c07_parent_worlds": 10,
        "fit_tasks": 180, "c07_tasks": 30,
        "fit_rows_per_seed_epoch": 426_552,
        "c07_rows_per_seed_epoch": 64_644,
        "epochs_per_seed": 3, "seeds": 3,
        "optimizer_steps_per_seed": 80_208,
        "optimizer_steps_total": 240_624,
        "model_parameters": 2_635_631,
        "trainable_parameters": 742_149, "trainable_tensors": 64,
        "observable_c07_positive_attachments": 442_936,
        "unit_tests": 53, "c08_rows_read": 0,
        "c09_c10_worlds_read": 0, "graph_replays": 0,
        "mtare_worlds_read": 0,
    }
    run_dir = PROJECT_ROOT / f"results/gate3_semantics/{RUN_ID}"
    spec = {
        "schema_version": "v3_run_spec_v1", "gate": 3,
        "execution_phase": 3, "operation": "training", "date": "20260902",
        "slug": "primitive_relation_observable_three_seed_training_v1", "seed": 0,
        "question": "Can dual-endpoint-observable supervision and learned endpoint evidence produce reliable physical primitive connections while preserving frozen V2 geometry on C07?",
        "method": "For each seed transfer its exact V2 geometry/temporal checkpoint, reset relation/evidence modules, freeze the backbone, train three complete epochs on corrected C01-C06 attachment validity, select by C07 relation+uncertainty loss, then apply the unchanged geometry/F1/nonempty-98%-precision gates.",
        "baseline": "The sealed same-input robust superellipse fitter evaluated on the identical C07 dual-endpoint-observable attachment population; old unmasked V2 is the label-policy ablation.",
        "fallback": "Scientific failure stops before C08 and graph. No extra epoch, seed selection, threshold lowering, rule connection or planner compensation is allowed.",
        "corrective_of": str(ATTRIBUTION.relative_to(PROJECT_ROOT)),
        "data_card": str(CARD.relative_to(PROJECT_ROOT)),
        "config_path": str(CARD.relative_to(PROJECT_ROOT)),
        "user_authorization": approval,
        "acceptance_criteria": list(gates.values()),
        "expected_counts": expected, "estimated_cost": card["estimated_cost"],
        "expected_evidence": [
            "Three source-bound initialization reports and frozen-state hashes.",
            "Nine complete fit epochs, nine C07 selection passes and 240624 optimizer steps.",
            "Per-seed corrected C07 geometry/relation/evidence/safe metrics and 2-of-3 decision.",
            "Checkpoints, histories, resources, paper figure, logs, RUN_STATE and seal.",
        ],
        "frozen_inputs": {
            str(path.relative_to(PROJECT_ROOT)): sha(path) for path in inputs
        },
        "frozen_tools": {
            name: {"path": path, "sha256": sha(PROJECT_ROOT / path)}
            for name, path in tools.items()
        },
        "working_directory": str(PROJECT_ROOT),
        "command": [
            "/usr/bin/systemd-inhibit", "--what=sleep:shutdown",
            "--why=Observable primitive relation three-seed training",
            "--mode=block", "/usr/bin/timeout", "--signal=INT",
            "--kill-after=120s", "86400s", "/usr/bin/env",
            "CUBLAS_WORKSPACE_CONFIG=:4096:8",
            f"PYTHONPATH={PROJECT_ROOT / 'src'}:{PROJECT_ROOT / 'tools/v3'}:{PROJECT_ROOT}",
            PYTHON, "tools/v3/run_primitive_relation_observable_three_seed_training_v1.py",
            "--spec", str(SPEC), "--run-dir", str(run_dir),
        ],
    }
    write(SPEC, spec)
    print(SPEC)


if __name__ == "__main__":
    main()
