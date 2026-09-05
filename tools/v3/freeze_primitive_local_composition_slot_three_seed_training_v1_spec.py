#!/usr/bin/env python3
"""Freeze the local composition-slot three-seed training/C07 gate."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SOURCE_CARD = ROOT / "configs/v3/gate3/data_cards/primitive_composition_anchor_three_seed_training_v1.json"
CARD = ROOT / "configs/v3/gate3/data_cards/primitive_local_composition_slot_three_seed_training_v1.json"
SPEC = ROOT / "configs/v3/gate3/primitive_local_composition_slot_three_seed_training_v1.json"
RUN_ID = "gate3_20260903_primitive_local_composition_slot_three_seed_training_v1_seed0"
CARD_STATUS = "APPROVED_FOR_ONE_IMMUTABLE_PRIMITIVE_LOCAL_COMPOSITION_SLOT_THREE_SEED_TRAINING_V1"
PYTHON = "/home/zeng-workstation/.local/share/mtare_topo_comm/envs/phase3_torch290_cu129_zarr2187_v1/bin/python"
P1A = ROOT / "results/gate3_semantics/gate3_20260830_primitive_relation_p1a_lossless_storage_corrective_v1_seed0"
P1B = ROOT / "results/gate3_semantics/gate3_20260830_primitive_relation_p1b_teacher_materialization_v1r_seed0"
OBS = ROOT / "results/gate3_semantics/gate3_20260902_primitive_attachment_observability_sidecar_v1_seed0"
MODELS = ROOT / "results/gate3_semantics/gate3_20260902_primitive_relation_observable_three_seed_training_v1_seed0"
FEASIBILITY = ROOT / "results/gate3_semantics/gate3_20260903_local_composition_slot_teacher_feasibility_v1_seed0"
READINESS = ROOT / "results/gate3_semantics/gate3_20260903_primitive_local_composition_slot_readiness_v1r_seed0"
BASELINE = ROOT / "results/gate3_semantics/gate3_20260902_primitive_relation_observable_nonlearning_c07_v1_seed0"
THRESHOLDS = ROOT / "results/gate3_semantics/gate3_20260903_primitive_predicted_geometry_association_diagnostic_v1_seed0"


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    if CARD.exists() or SPEC.exists():
        raise RuntimeError("local composition-slot training card/spec already exists")
    readiness = _load(READINESS / "metrics/summary.json")
    if readiness.get("decision") != "ALLOW_PRIMITIVE_LOCAL_COMPOSITION_SLOT_THREE_SEED_TRAINING_DATA_CARD":
        raise RuntimeError("local composition-slot training requires the exact readiness PASS")
    source = _load(SOURCE_CARD); card = copy.deepcopy(source)
    card["card_id"] = "primitive_local_composition_slot_three_seed_training_v1"
    card["purpose"] = "Train seeds 0/1/2 of the readiness-qualified endpoint-to-32-slot/dustbin relation head on C01-C06, select checkpoints and one structured refusal threshold on full C07, and decide whether C08 may open."
    card["teacher_source"] = "Sealed P1b construction attachments masked by the sealed dual-endpoint observability sidecar, then losslessly decomposed into exchangeable local composition cliques. Hidden connections remain unknown; C07 is selection/evaluation only."
    card["sampling"]["raw_frame_count"] = 659_940
    card["sampling"]["effective_sample_count"] = 491_196
    card["sampling"]["effective_structure_event_count"] = 3_225_423
    card["sampling"]["independent_sampling_units"] = "70 disjoint topology parents: C01-C06 fit=60 parents/180 geometry tasks; C07 selection=10 parents/30 tasks. Three geometry variants remain paired repeated measures within each parent."
    card["sampling"]["rule"] = "Each seed sees every 426552 fit sequence once per epoch for three epochs and every 64644 C07 sequence once per epoch for selection. C07 provides no gradient. Final C07 uses one exact-ranked structured refusal threshold per seed and the fixed existence threshold inherited from the frozen geometry model."
    card["sampling"]["structure_event_counts"] = {
        "fit_parent_worlds": 60, "fit_tasks": 180, "fit_sequences": 426552,
        "fit_observable_positive_attachments": 2782487,
        "c07_parent_worlds": 10, "c07_tasks": 30, "c07_sequences": 64644,
        "c07_observable_positive_attachments": 442936,
        "unique_sequences": 491196, "seeds": 3, "epochs_per_seed": 3,
        "optimizer_steps_per_seed": 10233, "optimizer_steps_total": 30699,
        "c08_rows_read": 0, "graph_replays": 0, "mtare_worlds_read": 0,
    }
    card["estimated_cost"] = {
        "compute": "Three sequential head-only CUDA trainings, each 3 fit epochs plus 3 C07 loss passes, then one full three-seed C07 structured-link evaluation.",
        "disk_gb": 2, "gpu": 1, "gpu_memory_gb": 16,
        "host_ram_gb": 4, "wall_time_hours": 30,
    }
    card["failure_policy"] = "Fail closed on population/input drift, non-finite or zero head gradient, backbone mutation, resource cap, missing selected checkpoint, C08+ read or evaluator failure. A valid 0/3 or 1/3 scientific result stops the slot model before C08/graph; do not add epochs, tune multiple thresholds, choose a seed, alter Teacher or add connection rules."
    card["retention"] = "Keep three selected checkpoints and epoch histories, per-seed/per-task C07 metrics, exact safe thresholds, slot-collapse/overlap diagnostics, comparison PNG/PDF/SVG, commands, environment, logs, RUN_STATE and SHA-256 seal."
    card["metrics_and_pre_registered_gates"] = {
        "training": "Each seed performs exactly 3 epochs and 10233 updates on all 426552 fit sequences; C07 has 64644 rows per selection pass and zero gradient.",
        "selection": "Checkpoint is the minimum mean C07 local composition-slot total loss. Final rejection uses exactly one structured endpoint-confidence threshold selected by exact ranking on C07.",
        "relation": "On the same 442936 observable C07 true pairs, each passing seed improves attachment F1 over the frozen 0.0063801323 non-learning baseline by at least 0.05 and has a nonempty TP result at precision>=0.98.",
        "structure": "Score must be finite/exact symmetric and threshold decoding must be same-slot transitive. Report used slots, accepted clusters, maximum cluster size, oversized clusters, collapse rows and overlap hard-negative false positives without hiding them in averages.",
        "seeds": "At least 2 of 3 seeds must pass every relation/structure/frozen-backbone check; ensemble or seed selection cannot replace this requirement.",
        "resources": "CUDA allocated/reserved/process each <=16 GiB, host RSS <=4 GiB, result <=2 GiB and wall time <=30 h.",
        "isolation": "C08/C09/C10, graph and M-TARE reads are zero. Only a valid C07 2/3 PASS may authorize a separate one-shot C08 run.",
    }
    approval = {
        "approved_at": "2026-09-03T15:05:00+08:00", "approved_by": "user-standing-authorization",
        "authorized_gates": [3],
        "authorized_operations": ["training", "checkpoint_selection", "threshold_calibration"],
        "confirmation_reference": "The user requested uninterrupted automatic best-in-plan execution without repeated approvals.",
        "scope": "One immutable three-seed local composition-slot training/full-C07 gate; no C08, graph or M-TARE.",
        "status": "APPROVED",
    }
    card["approval"] = approval; card["status"] = CARD_STATUS; _write(CARD, card)

    run_dir = ROOT / "results/gate3_semantics" / RUN_ID
    spec = {
        "schema_version": "v3_run_spec_v1", "slug": "primitive_local_composition_slot_three_seed_training_v1",
        "date": "20260903", "seed": 0, "gate": 3, "execution_phase": 3,
        "operation": "training", "working_directory": str(ROOT),
        "config_path": str(CARD.relative_to(ROOT)), "data_card": str(CARD.relative_to(ROOT)),
        "question": "Can the 32-slot/dustbin relation head learn lossless local compositions from causal LiDAR and retain nonempty precision>=0.98 on full unseen C07 in at least two of three seeds?",
        "method": "Freeze each seed's observable five-frame primitive backbone and train only the 1001507-parameter set endpoint/32-slot head with Hungarian clique assignment, dustbin, slot-presence and overlap-refusal losses; deploy with one structured transitive confidence threshold.",
        "baseline": "Frozen non-learning robust superellipse association on the identical C07 observable-pair population (F1=0.0063801323); failed global composition anchors remain a representation ablation.",
        "fallback": card["failure_policy"], "user_authorization": approval,
        "acceptance_criteria": list(card["metrics_and_pre_registered_gates"].values()),
        "expected_counts": {
            "fit_parent_worlds": 60, "fit_tasks": 180, "fit_rows": 426552,
            "fit_positive_pairs": 2782487, "c07_parent_worlds": 10,
            "c07_tasks": 30, "c07_rows": 64644, "c07_positive_pairs": 442936,
            "seeds": 3, "epochs_per_seed": 3, "steps_per_seed": 10233,
            "optimizer_steps": 30699, "head_parameters": 1001507,
            "frozen_parameters": 2635631, "tests": 40,
            "c08_rows": 0, "graph_replays": 0, "mtare_worlds": 0,
        },
        "estimated_cost": card["estimated_cost"],
        "expected_evidence": [
            "Three seed histories/checkpoints with exact steps, populations, frozen backbone hashes and resources.",
            "Full C07 per-seed and 30-task structured attachment metrics on 442936 true pairs.",
            "Best-F1 and precision>=0.98 nonempty threshold evidence plus overlap and slot-collapse diagnostics.",
            "Comparison PNG/PDF/SVG, source/environment hashes, logs, RUN_STATE and seal.",
        ],
        "command": [
            "/usr/bin/systemd-inhibit", "--what=sleep:shutdown",
            "--why=Primitive local composition slot three-seed training", "--mode=block",
            "/usr/bin/timeout", "--signal=INT", "--kill-after=120s", "108000s",
            "/usr/bin/env", f"PYTHONPATH={ROOT / 'src'}:{ROOT / 'tools/v3'}:{ROOT}",
            "PYTHONHASHSEED=0", "CUBLAS_WORKSPACE_CONFIG=:4096:8", PYTHON,
            "tools/v3/run_primitive_local_composition_slot_three_seed_training_v1.py",
            "--spec", str(SPEC), "--run-dir", str(run_dir),
        ],
    }
    inputs = [
        CARD,
        FEASIBILITY / "RUN_STATE.json", FEASIBILITY / "metrics/summary.json",
        FEASIBILITY / "metrics/teacher_feasibility/summary.json", FEASIBILITY / "artifacts/evidence_sha256.txt",
        READINESS / "RUN_STATE.json", READINESS / "metrics/summary.json", READINESS / "artifacts/evidence_sha256.txt",
        P1A / "RUN_STATE.json", P1A / "metrics/summary.json", P1A / "artifacts/task_manifest.json", P1A / "artifacts/evidence_sha256.txt",
        P1B / "RUN_STATE.json", P1B / "metrics/summary.json", P1B / "artifacts/task_manifest.json", P1B / "artifacts/evidence_sha256.txt",
        OBS / "RUN_STATE.json", OBS / "metrics/summary.json", OBS / "artifacts/task_manifest.json", OBS / "artifacts/evidence_sha256.txt",
        MODELS / "RUN_STATE.json", MODELS / "metrics/summary.json", MODELS / "artifacts/evidence_sha256.txt",
        *[MODELS / f"artifacts/models/seed{seed}/selected.pt" for seed in (0, 1, 2)],
        BASELINE / "RUN_STATE.json", BASELINE / "metrics/summary.json", BASELINE / "artifacts/evidence_sha256.txt",
        THRESHOLDS / "RUN_STATE.json", THRESHOLDS / "metrics/summary.json",
        THRESHOLDS / "metrics/diagnostic/summary.json", THRESHOLDS / "artifacts/evidence_sha256.txt",
    ]
    spec["frozen_inputs"] = {str(path.relative_to(ROOT)): _sha(path) for path in inputs}
    tools = {
        "model": "src/mtare_topo/representation/primitive_local_composition_slot_model.py",
        "training_loss": "src/mtare_topo/representation/primitive_local_composition_slot_training.py",
        "decoding": "src/mtare_topo/representation/primitive_local_composition_slot_decoding.py",
        "teacher": "src/mtare_topo/evaluation/local_composition_slot_teacher.py",
        "trainer": "tools/v3/train_primitive_local_composition_slot_v1.py",
        "evaluator": "tools/v3/evaluate_primitive_local_composition_slot_three_seed_v1.py",
        "runner": "tools/v3/run_primitive_local_composition_slot_three_seed_training_v1.py",
        "freezer": "tools/v3/freeze_primitive_local_composition_slot_three_seed_training_v1_spec.py",
        "model_tests": "tests/v3/unit/test_primitive_local_composition_slot_model.py",
        "loss_tests": "tests/v3/unit/test_primitive_local_composition_slot_training.py",
        "decode_tests": "tests/v3/unit/test_primitive_local_composition_slot_decoding.py",
        "teacher_tests": "tests/v3/unit/test_local_composition_slot_teacher.py",
        "evaluator_tests": "tests/v3/unit/test_evaluate_primitive_local_composition_slot_three_seed_v1.py",
        "trainer_tests": "tests/v3/unit/test_train_primitive_local_composition_slot_v1.py",
        "observable_tests": "tests/v3/unit/test_primitive_relation_observable_model.py",
        "resource_tests": "tests/v3/unit/test_primitive_relation_sparse_port_resource_monitor_v1r.py",
        "resource_monitor": "tools/v3/run_primitive_relation_sparse_port_three_seed_training_v1r.py",
        "governance": "src/mtare_topo/governance.py",
        "preflight": "tools/v3/preflight.py", "create_run": "tools/v3/create_run.py",
    }
    spec["frozen_tools"] = {name: {"path": path, "sha256": _sha(ROOT / path)} for name, path in tools.items()}
    _write(SPEC, spec); print(CARD); print(SPEC)


if __name__ == "__main__":
    main()
