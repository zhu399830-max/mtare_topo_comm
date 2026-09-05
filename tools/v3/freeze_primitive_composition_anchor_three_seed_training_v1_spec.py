#!/usr/bin/env python3
"""Freeze the formal three-seed composition-anchor training Data Card/spec."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

from _bootstrap import PROJECT_ROOT


SOURCE_CARD = PROJECT_ROOT / "configs/v3/gate3/data_cards/primitive_composition_anchor_training_readiness_v1.json"
CARD = PROJECT_ROOT / "configs/v3/gate3/data_cards/primitive_composition_anchor_three_seed_training_v1.json"
SPEC = PROJECT_ROOT / "configs/v3/gate3/primitive_composition_anchor_three_seed_training_v1.json"
RUN_ID = "gate3_20260903_primitive_composition_anchor_three_seed_training_v1_seed0"

P1A = PROJECT_ROOT / "results/gate3_semantics/gate3_20260830_primitive_relation_p1a_lossless_storage_corrective_v1_seed0"
P1B = PROJECT_ROOT / "results/gate3_semantics/gate3_20260830_primitive_relation_p1b_teacher_materialization_v1r_seed0"
OBSERVABILITY = PROJECT_ROOT / "results/gate3_semantics/gate3_20260902_primitive_attachment_observability_sidecar_v1_seed0"
SOURCE_MODELS = PROJECT_ROOT / "results/gate3_semantics/gate3_20260902_primitive_relation_observable_three_seed_training_v1_seed0"
ANCHORS = PROJECT_ROOT / "results/gate3_semantics/gate3_20260903_primitive_composition_anchor_target_sidecar_v1r_seed0"
READINESS = PROJECT_ROOT / "results/gate3_semantics/gate3_20260903_primitive_composition_anchor_training_readiness_v1_seed0"
MODEL_READINESS = PROJECT_ROOT / "results/gate3_semantics/gate3_20260903_primitive_composition_anchor_model_readiness_v2_seed0"
DIAGNOSTIC = PROJECT_ROOT / "results/gate3_semantics/gate3_20260903_primitive_predicted_geometry_association_diagnostic_v1_seed0"
BASELINE_C07 = PROJECT_ROOT / "results/gate3_semantics/gate3_20260902_primitive_relation_observable_nonlearning_c07_v1_seed0"


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    if CARD.exists() or SPEC.exists():
        raise RuntimeError("composition-anchor three-seed card/spec already exists")
    card = copy.deepcopy(json.loads(SOURCE_CARD.read_text(encoding="utf-8")))
    card["card_id"] = "primitive_composition_anchor_three_seed_training_v1"
    card["status"] = "APPROVED_FOR_ONE_IMMUTABLE_PRIMITIVE_COMPOSITION_ANCHOR_THREE_SEED_TRAINING_V1"
    card["purpose"] = (
        "Test whether a small sensor-polar endpoint composition-anchor head can convert the "
        "frozen five-frame primitive representation into reliable physical connections on unseen C07 topology."
    )
    card["approval"] = {
        "status": "APPROVED", "approved_by": "user-standing-authorization",
        "approved_at": "2026-09-03T15:30:00+08:00", "authorized_gates": [3],
        "authorized_operations": ["training"],
        "scope": (
            "One immutable head-only seeds 0/1/2 run: 3 fit epochs and 10233 optimizer steps per seed, "
            "C07 loss selection and final C07 gate; no C08+, graph or M-TARE."
        ),
        "confirmation_reference": (
            "The user authorized uninterrupted best-in-plan execution; the formal target sidecar and "
            "batch-128 readiness both passed and explicitly allow this single training contract."
        ),
    }
    card["sampling"].update({
        "effective_sample_count": 426_552,
        "effective_structure_event_count": 3_826_561,
        "independent_sampling_units": (
            "60 disjoint C01-C06 topology parents for fit and 10 disjoint C07 parents for selection; "
            "three geometry realizations are paired repeated measures within each parent."
        ),
        "rule": (
            "For each seed, load its own frozen observable selected checkpoint; jointly train only the "
            "22278-parameter anchor/scale/compatibility head for exactly three shuffled fit epochs at batch128. "
            "Evaluate all C07 rows after every epoch and select minimum mean C07 anchor total loss."
        ),
        "temporal_window_frames": 5,
        "structure_event_counts": {
            "fit_parent_worlds": 60, "fit_geometry_tasks": 180,
            "fit_sequences_per_epoch": 426_552, "fit_batches_per_epoch": 3_411,
            "fit_observable_positive_attachments": 2_782_487,
            "c07_parent_worlds": 10, "c07_geometry_tasks": 30,
            "c07_sequences_per_epoch": 64_644, "c07_batches_per_epoch": 522,
            "c07_observable_positive_attachments": 442_936,
            "epochs_per_seed": 3, "seeds": 3,
            "optimizer_steps_per_seed": 10_233, "optimizer_steps_total": 30_699,
            "model_forward_rows_total": 4_614_696,
            "epoch_checkpoints": 9, "selected_checkpoint_copies": 3,
            "c08_rows_read": 0, "c09_c10_worlds_read": 0,
        },
    })
    card["split"] = {
        "fit": "All C01-C06 parents and all three paired geometry realizations provide gradients.",
        "selection": (
            "All C07 parents select one checkpoint per seed by mean anchor total loss; the final exact "
            "ranked C07 threshold is a validation metric and cannot alter data, model, loss, epochs or gates."
        ),
        "development_transfer": "C08 remains unopened even if C07 passes; transfer requires a separate frozen run.",
        "strict_test": "C08/C09/C10 and all M-TARE benchmark worlds remain unread.",
        "world_disjoint": True, "trajectory_disjoint": True,
        "historical_pollution_audit": (
            "Earlier C07 relation failures motivated the pre-registered composition-anchor target only. "
            "No C07 result from this run may trigger architecture, Teacher or threshold changes in-place."
        ),
    }
    card["teacher"] = {
        "source": (
            "Sealed construction-program composition anchors in current-sensor coordinates, exact P1b primitive "
            "slots and the sealed dual-endpoint observability validity sidecar."
        ),
        "labels": "Per-endpoint shared 3-D anchor, relation target, disconnected-overlap hard negative and endpoint support.",
        "student_forbidden_inputs": (
            "Construction/world/node/primitive identity, absolute pose, TNG, future frames, support bits at forward, "
            "C08+ and planner state never enter the student model."
        ),
        "valid_mask": (
            "Only matched active endpoints with current-window support enter anchor regression; physical relation "
            "metrics use dual-endpoint-observed pairs and hidden pairs remain unknown."
        ),
        "planner_consistency_plan": (
            "A future graph may use high-confidence connection proposals, but only physical traversal can commit an edge."
        ),
    }
    card["leakage_audit"].update({
        "model_inference_count": 4_614_696,
        "optimizer_step_count": 30_699,
    })
    card["metrics_and_pre_registered_gates"] = {
        "population": (
            "Every seed reads exactly 426552 fit rows and 64644 C07 rows in each of three epochs; "
            "seeds are 0/1/2 and total optimizer steps are 30699."
        ),
        "freeze": (
            "Each seed loads its same-seed source checkpoint; exactly 22278 parameters in 8 head tensors train, "
            "2635631 backbone parameters and their state digest remain unchanged."
        ),
        "selection": "One checkpoint per seed is selected only by minimum mean C07 composition-anchor total loss.",
        "relations": (
            "On the same 442936-positive observable C07 population, deployed attachment F1 improves at least "
            "0.05 absolute over baseline 0.0063801323 and safe score yields TP>0 at precision>=0.98."
        ),
        "anchor": (
            "Mean observed-endpoint anchor error improves at least 10% over the same frozen model's raw predicted endpoint."
        ),
        "seeds": "At least two of three seeds pass relation, nonempty-safe, anchor and frozen-backbone gates together.",
        "resources": (
            "Wall time <=18 h; host RSS <=4 GiB; GPU process and PyTorch allocation <=16 GiB; result <=2 GiB."
        ),
        "isolation": "C08 rows, C09/C10 worlds, graph replays and M-TARE reads remain exactly zero.",
        "decision": (
            "PASS allows a separately frozen zero-adaptation C08 relation transfer and graph-readiness run; "
            "FAIL stops this method before C08/graph and triggers C07-only attribution."
        ),
    }
    card["estimated_cost"] = {
        "compute": (
            "One RTX 5090 D; three serial seeds, three full batch-128 fit epochs per seed, "
            "C07 loss after every epoch and one final three-seed C07 evaluation."
        ),
        "gpu": 1, "gpu_memory_gb": 16, "host_ram_gb": 4,
        "wall_time_hours": 18, "disk_gb": 2,
    }
    card["failure_policy"] = (
        "Any input, environment, population, freeze, gradient, resource, output or isolation drift fails closed. "
        "A valid C07 scientific failure is sealed without changing batch, epoch, Teacher, threshold, model or gate; "
        "do not read C08 or build a graph."
    )
    card["retention"] = (
        "Retain all nine epoch checkpoints and three selected copies until the paper evidence/ablation map is frozen, "
        "plus histories, C07 per-seed metrics, figure, logs, environment, resource monitors, RUN_STATE and seal."
    )
    _write(CARD, card)

    run_dir = PROJECT_ROOT / f"results/gate3_semantics/{RUN_ID}"
    spec = {
        "schema_version": "v3_run_spec_v1",
        "slug": "primitive_composition_anchor_three_seed_training_v1",
        "date": "20260903", "seed": 0, "gate": 3, "execution_phase": 3,
        "operation": "training", "working_directory": str(PROJECT_ROOT),
        "config_path": str(CARD.relative_to(PROJECT_ROOT)),
        "data_card": str(CARD.relative_to(PROJECT_ROOT)),
        "question": (
            "Can the 22278-parameter sensor-polar composition-anchor head produce nonempty precision>=0.98 "
            "physical connections and beat the same-input baseline on all unseen C07 rows in at least 2/3 seeds?"
        ),
        "method": (
            "Freeze each seed's observable five-frame primitive backbone; train only endpoint anchor, uncertainty "
            "and compatibility layers using shared construction anchors and dual-endpoint observability."
        ),
        "baseline": (
            "The frozen same-input non-learning robust primitive relation baseline with C07 attachment F1 "
            "0.0063801323, plus each frozen model's uncorrected raw endpoint error."
        ),
        "fallback": (
            "No in-run fallback. A valid scientific failure stops before C08/graph and permits only a separately "
            "frozen C07-only failure attribution."
        ),
        "user_authorization": card["approval"],
        "acceptance_criteria": list(card["metrics_and_pre_registered_gates"].values()),
        "expected_counts": {
            "fit_rows_per_seed_epoch": 426_552, "fit_tasks": 180,
            "c07_rows_per_seed_epoch": 64_644, "c07_tasks": 30,
            "epochs_per_seed": 3, "seeds": 3,
            "optimizer_steps_per_seed": 10_233, "optimizer_steps_total": 30_699,
            "model_forward_rows_total": 4_614_696,
            "tests": 37, "head_parameters": 22_278,
            "c08_rows_read": 0, "c09_c10_worlds_read": 0,
            "graph_replays": 0, "mtare_worlds_read": 0,
        },
        "estimated_cost": card["estimated_cost"],
        "expected_evidence": [
            "Three same-seed frozen-backbone training histories and exact selected checkpoints.",
            "C07 relation precision/recall/F1, safe TP, anchor errors, hard-negative and proposal-oracle diagnostics.",
            "37 tests, resource monitors, input hashes, raw logs, comparison PNG/PDF/SVG, RUN_STATE and seal.",
        ],
        "command": [
            "/usr/bin/systemd-inhibit", "--what=sleep:shutdown",
            "--why=Composition anchor three-seed training", "--mode=block",
            "/usr/bin/timeout", "--signal=INT", "--kill-after=120s", "64800s",
            "/usr/bin/env",
            f"PYTHONPATH={PROJECT_ROOT / 'src'}:{PROJECT_ROOT / 'tools/v3'}:{PROJECT_ROOT}",
            "PYTHONHASHSEED=0", "CUBLAS_WORKSPACE_CONFIG=:4096:8",
            "/home/zeng-workstation/.local/share/mtare_topo_comm/envs/phase3_torch290_cu129_zarr2187_v1/bin/python",
            "tools/v3/run_primitive_composition_anchor_three_seed_training_v1.py",
            "--spec", str(SPEC), "--run-dir", str(run_dir),
        ],
    }
    inputs = [CARD]
    for source in (
        P1A, P1B, OBSERVABILITY, SOURCE_MODELS, ANCHORS, READINESS,
        MODEL_READINESS, DIAGNOSTIC, BASELINE_C07,
    ):
        for name in ("RUN_STATE.json", "metrics/summary.json", "artifacts/evidence_sha256.txt"):
            path = source / name
            if path.is_file():
                inputs.append(path)
    inputs.extend((
        P1A / "artifacts/task_manifest.json",
        P1B / "artifacts/task_manifest.json",
        OBSERVABILITY / "artifacts/task_manifest.json",
        ANCHORS / "artifacts/materialized/task_manifest.json",
        DIAGNOSTIC / "metrics/diagnostic/summary.json",
        BASELINE_C07 / "metrics/summary.json",
        *[SOURCE_MODELS / f"artifacts/models/seed{seed}/selected.pt" for seed in range(3)],
    ))
    spec["frozen_inputs"] = {
        str(path.relative_to(PROJECT_ROOT)): _sha(path) for path in inputs
    }
    tools = {
        "batch_module": "src/mtare_topo/data/primitive_composition_anchor_batches.py",
        "sidecar_module": "src/mtare_topo/data/primitive_composition_anchor_sidecar.py",
        "model": "src/mtare_topo/representation/primitive_composition_anchor_model.py",
        "training": "src/mtare_topo/representation/primitive_composition_anchor_training.py",
        "train_seed": "tools/v3/train_primitive_composition_anchor_v1.py",
        "evaluator": "tools/v3/evaluate_primitive_composition_anchor_three_seed_v1.py",
        "runner": "tools/v3/run_primitive_composition_anchor_three_seed_training_v1.py",
        "freezer": "tools/v3/freeze_primitive_composition_anchor_three_seed_training_v1_spec.py",
        "resource_monitor": "tools/v3/run_primitive_relation_sparse_port_three_seed_training_v1r.py",
        "batch_tests": "tests/v3/unit/test_primitive_composition_anchor_batches.py",
        "sidecar_tests": "tests/v3/unit/test_primitive_composition_anchor_sidecar.py",
        "teacher_tests": "tests/v3/unit/test_primitive_composition_anchor_teacher.py",
        "model_tests": "tests/v3/unit/test_primitive_composition_anchor_model.py",
        "evaluation_tests": "tests/v3/unit/test_evaluate_primitive_composition_anchor_three_seed_v1.py",
        "resource_tests": "tests/v3/unit/test_primitive_relation_sparse_port_resource_monitor_v1r.py",
        "governance": "src/mtare_topo/governance.py",
        "preflight": "tools/v3/preflight.py", "create_run": "tools/v3/create_run.py",
    }
    spec["frozen_tools"] = {
        name: {"path": path, "sha256": _sha(PROJECT_ROOT / path)}
        for name, path in tools.items()
    }
    _write(SPEC, spec)
    print(SPEC)


if __name__ == "__main__":
    main()
