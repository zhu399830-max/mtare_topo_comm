#!/usr/bin/env python3
"""Freeze the endpoint relation metric three-seed training/C07 gate."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SOURCE_CARD = ROOT / "configs/v3/gate3/data_cards/primitive_local_composition_slot_three_seed_training_v1.json"
CARD = ROOT / "configs/v3/gate3/data_cards/primitive_endpoint_relation_metric_three_seed_training_v1.json"
SPEC = ROOT / "configs/v3/gate3/primitive_endpoint_relation_metric_three_seed_training_v1.json"
RUN_ID = "gate3_20260904_primitive_endpoint_relation_metric_three_seed_training_v1_seed0"
CARD_STATUS = "APPROVED_FOR_ONE_IMMUTABLE_PRIMITIVE_ENDPOINT_RELATION_METRIC_THREE_SEED_TRAINING_V1"
PYTHON = "/home/zeng-workstation/.local/share/mtare_topo_comm/envs/phase3_torch290_cu129_zarr2187_v1/bin/python"
P1A = ROOT / "results/gate3_semantics/gate3_20260830_primitive_relation_p1a_lossless_storage_corrective_v1_seed0"
P1B = ROOT / "results/gate3_semantics/gate3_20260830_primitive_relation_p1b_teacher_materialization_v1r_seed0"
OBS = ROOT / "results/gate3_semantics/gate3_20260902_primitive_attachment_observability_sidecar_v1_seed0"
SOURCE_MODELS = ROOT / "results/gate3_semantics/gate3_20260902_primitive_relation_observable_three_seed_training_v1_seed0"
READINESS = ROOT / "results/gate3_semantics/gate3_20260904_primitive_endpoint_relation_metric_readiness_v1_seed0"
ATTRIBUTION = ROOT / "results/gate3_semantics/gate3_20260904_primitive_local_composition_slot_failure_attribution_v1_seed0"
BASELINE = ROOT / "results/gate3_semantics/gate3_20260902_primitive_relation_observable_nonlearning_c07_v1_seed0"
THRESHOLDS = ROOT / "results/gate3_semantics/gate3_20260903_primitive_predicted_geometry_association_diagnostic_v1_seed0"


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""): digest.update(block)
    return digest.hexdigest()


def _load(path: Path): return json.loads(path.read_text(encoding="utf-8"))
def _write(path: Path, value) -> None: path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    if CARD.exists() or SPEC.exists(): raise RuntimeError("endpoint metric training card/spec already exists")
    readiness = _load(READINESS / "metrics/summary.json")
    if readiness.get("decision") != "ALLOW_PRIMITIVE_ENDPOINT_RELATION_METRIC_THREE_SEED_TRAINING_DATA_CARD" or readiness.get("scientific_pass") is not True: raise RuntimeError("training requires exact readiness PASS")
    card = copy.deepcopy(_load(SOURCE_CARD))
    card.update({
        "card_id": "primitive_endpoint_relation_metric_three_seed_training_v1", "status": CARD_STATUS,
        "purpose": "Train seeds0/1/2 of the no-arbitrary-slot endpoint relation metric on all C01-C06 sequences, select by full C07 mean metric loss, freeze one exact relation threshold per seed and decide whether C08 may open.",
        "teacher_source": "Sealed P1b local composition cliques with the sealed dual-endpoint observability mask. Same-clique endpoint pairs are positives; different-clique, singleton and disconnected-overlap pairs are negatives. No arbitrary slot target or dustbin class is used.",
        "estimated_cost": {"compute": "Three sequential frozen-backbone CUDA trainings, each 3 complete fit epochs and C07 selection passes, followed by one full three-seed C07 pair/complete-link evaluation.", "disk_gb": 2, "gpu": 1, "gpu_memory_gb": 16, "host_ram_gb": 4, "wall_time_hours": 30},
        "failure_policy": "Fail closed on population/hash/environment/gradient/backbone/resource/evaluator drift. A valid 0/3 or1/3 scientific result stops before C08 and graph; do not add epochs, tune multiple thresholds, pick a seed, alter Teacher or add semantic connection rules.",
        "retention": "Keep three selected checkpoints/histories, per-seed/per-task C07 pair and complete-link metrics, PNG/PDF/SVG, commands, environment, logs, RUN_STATE and seal.",
    })
    card["sampling"]["rule"] = "Each seed sees all 426552 fit sequences once per epoch for three epochs and all 64644 C07 sequences once per epoch for checkpoint selection. C07 supplies no gradient. Final C07 chooses exactly one tie-safe relation threshold per seed."
    card["sampling"]["structure_event_counts"].update({"seeds": 3, "epochs_per_seed": 3, "optimizer_steps_per_seed": 10233, "optimizer_steps_total": 30699, "c08_rows_read": 0, "graph_replays": 0})
    card["teacher"] = {"source": "Sealed construction-program composition cliques plus endpoint observability.", "labels": "Same-clique pair positives and different-clique/singleton/disconnected-overlap negatives after frozen primitive matching.", "valid_mask": "Dual-observed cross-primitive endpoint pairs; hidden positives remain in the objective recall denominator but not in pair loss.", "student_forbidden_inputs": "Teacher/world/node/primitive identity, absolute pose, future frames, C08+ and planner state never enter forward.", "planner_consistency_plan": "Accepted complete-link clusters are local relation proposals only; physical traversal remains the only future edge commit."}
    card["metrics_and_pre_registered_gates"] = {
        "training": "Each seed performs exactly3 epochs/10233 updates on426552 fit rows; C07 has64644 rows per selection pass and zero gradient.",
        "selection": "Select minimum mean C07 total metric loss and one exact tie-safe pair threshold; no factor/ensemble/seed selection.",
        "relation": "On442936 observable C07 true pairs, each passing seed improves F1 over0.0063801323 by>=0.05 and has TP>0 at precision>=0.98.",
        "cluster": "At the same safe threshold complete-link retains TP>0 and precision>=0.98, with zero disconnected-overlap FP and zero clusters larger than Teacher maximum4.",
        "seeds": "At least2/3 seeds pass every relation/cluster/frozen-backbone check.",
        "resources": "CUDA allocated/reserved/process<=16GiB, host RSS<=4GiB, result<=2GiB, wall<=30h.",
        "isolation": "C08/C09/C10, graph and M-TARE reads are zero.",
    }
    approval = {"approved_at": "2026-09-04T13:20:00+08:00", "approved_by": "user-standing-authorization", "authorized_gates": [3], "authorized_operations": ["training", "checkpoint_selection", "threshold_calibration"], "confirmation_reference": "The user requested uninterrupted automatic best-plan execution; the formal readiness explicitly authorized this Data Card.", "scope": "One immutable three-seed endpoint relation metric training/full-C07 gate; no C08, graph or M-TARE.", "status": "APPROVED"}
    card["approval"] = approval; _write(CARD, card)
    run_dir = ROOT / "results/gate3_semantics" / RUN_ID
    spec = {"schema_version": "v3_run_spec_v1", "slug": "primitive_endpoint_relation_metric_three_seed_training_v1", "date": "20260904", "seed": 0, "gate": 3, "execution_phase": 3, "operation": "training", "working_directory": str(ROOT), "config_path": str(CARD.relative_to(ROOT)), "data_card": str(CARD.relative_to(ROOT)), "question": "Can the no-slot endpoint relation metric safely recover local composition on full unseen C07 in at least two of three seeds?", "method": "Freeze each seed's observable primitive backbone; train only a426818-parameter endpoint set encoder and normalized64-D relation metric with balanced pair, compactness, separation and overlap losses; deploy one symmetric pair threshold plus deterministic complete-link consistency.", "baseline": "Frozen non-learning association F1=0.0063801323 and the stopped arbitrary-slot head as a representation ablation.", "fallback": card["failure_policy"], "user_authorization": approval, "acceptance_criteria": list(card["metrics_and_pre_registered_gates"].values()), "expected_counts": {"fit_parent_worlds": 60, "fit_tasks": 180, "fit_rows": 426552, "c07_parent_worlds": 10, "c07_tasks": 30, "c07_rows": 64644, "c07_positive_pairs": 442936, "seeds": 3, "epochs_per_seed": 3, "steps_per_seed": 10233, "optimizer_steps": 30699, "head_parameters": 426818, "frozen_parameters": 2635631, "tests": 22, "c08_rows": 0, "graph_replays": 0}, "estimated_cost": card["estimated_cost"], "expected_evidence": ["Three seed histories/checkpoints with exact populations, steps, frozen hashes and resources.", "Full C07 per-seed/per-task pair metrics and safe complete-link cluster metrics.", "PNG/PDF/SVG, commands, environment, logs, RUN_STATE and SHA-256 seal."], "command": ["/usr/bin/systemd-inhibit", "--what=sleep:shutdown", "--why=Endpoint relation metric three-seed training", "--mode=block", "/usr/bin/timeout", "--signal=INT", "--kill-after=120s", "108000s", "/usr/bin/env", f"PYTHONPATH={ROOT / 'src'}:{ROOT / 'tools/v3'}:{ROOT}", "PYTHONHASHSEED=0", "CUBLAS_WORKSPACE_CONFIG=:4096:8", PYTHON, "tools/v3/run_primitive_endpoint_relation_metric_three_seed_training_v1.py", "--spec", str(SPEC), "--run-dir", str(run_dir)]}
    inputs = [CARD, SOURCE_CARD, READINESS / "RUN_STATE.json", READINESS / "metrics/summary.json", READINESS / "artifacts/evidence_sha256.txt", ATTRIBUTION / "RUN_STATE.json", ATTRIBUTION / "metrics/summary.json", ATTRIBUTION / "artifacts/evidence_sha256.txt", P1A / "RUN_STATE.json", P1A / "metrics/summary.json", P1A / "artifacts/task_manifest.json", P1A / "artifacts/evidence_sha256.txt", P1B / "RUN_STATE.json", P1B / "metrics/summary.json", P1B / "artifacts/task_manifest.json", P1B / "artifacts/evidence_sha256.txt", OBS / "RUN_STATE.json", OBS / "metrics/summary.json", OBS / "artifacts/task_manifest.json", OBS / "artifacts/evidence_sha256.txt", SOURCE_MODELS / "RUN_STATE.json", SOURCE_MODELS / "metrics/summary.json", SOURCE_MODELS / "artifacts/evidence_sha256.txt", *[SOURCE_MODELS / f"artifacts/models/seed{seed}/selected.pt" for seed in range(3)], BASELINE / "RUN_STATE.json", BASELINE / "metrics/summary.json", BASELINE / "artifacts/evidence_sha256.txt", THRESHOLDS / "RUN_STATE.json", THRESHOLDS / "metrics/diagnostic/summary.json", THRESHOLDS / "artifacts/evidence_sha256.txt"]
    spec["frozen_inputs"] = {str(path.relative_to(ROOT)): _sha(path) for path in inputs}
    tools = {"model": "src/mtare_topo/representation/primitive_endpoint_relation_metric_model.py", "loss": "src/mtare_topo/representation/primitive_endpoint_relation_metric_training.py", "decoder": "src/mtare_topo/representation/primitive_endpoint_relation_metric_decoding.py", "base_trainer": "tools/v3/train_primitive_local_composition_slot_v1.py", "trainer": "tools/v3/train_primitive_endpoint_relation_metric_v1.py", "evaluator": "tools/v3/evaluate_primitive_endpoint_relation_metric_three_seed_v1.py", "runner": "tools/v3/run_primitive_endpoint_relation_metric_three_seed_training_v1.py", "freezer": "tools/v3/freeze_primitive_endpoint_relation_metric_three_seed_training_v1_spec.py", "model_tests": "tests/v3/unit/test_primitive_endpoint_relation_metric.py", "trainer_tests": "tests/v3/unit/test_train_primitive_endpoint_relation_metric_v1.py", "evaluator_tests": "tests/v3/unit/test_evaluate_primitive_endpoint_relation_metric_three_seed_v1.py", "target_tests": "tests/v3/unit/test_primitive_local_composition_slot_training.py", "backbone_tests": "tests/v3/unit/test_primitive_relation_observable_model.py", "resource_monitor": "tools/v3/run_primitive_relation_sparse_port_three_seed_training_v1r.py", "governance": "src/mtare_topo/governance.py", "preflight": "tools/v3/preflight.py", "create_run": "tools/v3/create_run.py"}
    spec["frozen_tools"] = {name: {"path": path, "sha256": _sha(ROOT / path)} for name, path in tools.items()}; _write(SPEC, spec); print(CARD); print(SPEC)


if __name__ == "__main__": main()
