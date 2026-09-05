#!/usr/bin/env python3
"""Freeze the one-shot C07 composition-anchor failure-attribution run."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SOURCE_CARD = ROOT / "configs/v3/gate3/data_cards/primitive_composition_anchor_c07_evaluation_corrective_v1.json"
SOURCE_SPEC = ROOT / "configs/v3/gate3/primitive_composition_anchor_c07_evaluation_corrective_v1.json"
FORMAL = ROOT / "results/gate3_semantics/gate3_20260903_primitive_composition_anchor_c07_evaluation_corrective_v1_seed0"
TRAINING = ROOT / "results/gate3_semantics/gate3_20260903_primitive_composition_anchor_three_seed_training_v1_seed0"
CARD = ROOT / "configs/v3/gate3/data_cards/primitive_composition_anchor_failure_attribution_v1.json"
SPEC = ROOT / "configs/v3/gate3/primitive_composition_anchor_failure_attribution_v1.json"
RUN_ID = "gate3_20260903_primitive_composition_anchor_failure_attribution_v1_seed0"
CARD_STATUS = "APPROVED_FOR_ONE_IMMUTABLE_PRIMITIVE_COMPOSITION_ANCHOR_FAILURE_ATTRIBUTION_V1"
PYTHON = "/home/zeng-workstation/.local/share/mtare_topo_comm/envs/phase3_torch290_cu129_zarr2187_v1/bin/python"


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
        raise RuntimeError("composition-anchor attribution card/spec already exists")
    formal_state = _load(FORMAL / "RUN_STATE.json")
    formal_summary = _load(FORMAL / "metrics/evaluation/summary.json")
    if (
        formal_state.get("state") != "COMPLETED"
        or formal_summary.get("decision") != "STOP_COMPOSITION_ANCHOR_BEFORE_C08_AND_GRAPH"
        or formal_summary.get("c07", {}).get("passing_seeds") != 0
    ):
        raise RuntimeError("composition-anchor attribution requires the exact formal 0/3 failure")
    source_card = _load(SOURCE_CARD); source_spec = _load(SOURCE_SPEC)
    card = copy.deepcopy(source_card)
    card["card_id"] = "primitive_composition_anchor_failure_attribution_v1"
    card["purpose"] = (
        "Identify whether the frozen C07 failure comes from proposal existence, predicted anchor "
        "coordinates, anchor scale/compatibility, endpoint evidence, or the construction Teacher."
    )
    card["estimated_cost"] = {
        "compute": "One RTX 5090 D; eight read-only score conditions over all 64644 C07 rows for each of three seeds.",
        "disk_gb": 0.5, "gpu": 1, "gpu_memory_gb": 16, "host_ram_gb": 8, "wall_time_hours": 4,
    }
    card["failure_policy"] = (
        "Fail closed on input, seal, environment, population, metric reproduction, resource or isolation drift. "
        "Do not read C08+, retrain, change thresholds or build a graph. The pre-registered attribution decision "
        "selects only the next minimal C07 method interface."
    )
    card["retention"] = (
        "Retain per-seed counterfactual metrics, endpoint-error strata, per-task CSV/JSON, paper PNG/PDF/SVG, "
        "logs, environment, RUN_STATE and SHA-256 seal."
    )
    card["sampling"]["effective_sample_count"] = 64_644
    card["sampling"]["effective_structure_event_count"] = 442_936
    card["sampling"]["raw_frame_count"] = 323_220
    card["sampling"]["independent_sampling_units"] = (
        "10 disjoint C07 topology parents and 30 paired geometry tasks; the same complete frozen population is "
        "evaluated by three independently trained sealed checkpoints."
    )
    card["sampling"]["rule"] = (
        "Every C07 sequence exactly once per seed, deterministic batch128, with no shuffle, optimizer, "
        "adaptation or threshold selection."
    )
    card["sampling"]["structure_event_counts"] = {
        "c07_parent_worlds": 10, "c07_geometry_tasks": 30,
        "c07_sequences_per_seed": 64_644, "c07_observable_positive_attachments": 442_936,
        "frozen_seeds": 3, "model_forward_rows": 193_932,
        "optimizer_steps": 0, "c08_rows_read": 0, "c09_c10_worlds_read": 0,
    }
    card["leakage_audit"]["model_inference_count"] = 193_932
    card["leakage_audit"]["optimizer_step_count"] = 0
    card["metrics_and_pre_registered_gates"] = {
        "population": "Exactly 30 C07 tasks, 64644 rows and 442936 observable positive attachments per seed.",
        "reproduction": "The deployed full-model best-F1 TP/FP/FN and F1 exactly reproduce the sealed formal evaluation for each seed.",
        "counterfactuals": "Compare deployed, Teacher proposal, predicted-anchor distance, raw-endpoint distance, Teacher-anchor Gaussian/safe and Teacher-anchor distance without fitting new thresholds beyond exact ranked diagnostic summaries.",
        "strata": "Report endpoint error by Teacher-anchor range, required correction, observable attachment cluster size, absolute axis slope and geometry family.",
        "decision": "Use the fixed decision tree in diagnose_attribution; require the same primary diagnosis in at least two of three seeds.",
        "resources": "Wall time <=4 h; host RSS <=8 GiB; GPU process memory <=16 GiB; output <=0.5 GiB.",
        "isolation": "Optimizer steps, C08 rows, C09/C10 worlds, graph replays and M-TARE reads are all zero.",
    }
    approval = {
        "approved_at": "2026-09-03T13:10:00+08:00", "approved_by": "user-standing-authorization",
        "authorized_gates": [3], "authorized_operations": ["audit"], "status": "APPROVED",
        "confirmation_reference": "The user requested uninterrupted automatic best-in-plan execution without repeated approvals.",
        "scope": "One immutable zero-training C07-only attribution over three sealed checkpoints; no C08+, graph or M-TARE.",
    }
    card["approval"] = approval; card["status"] = CARD_STATUS
    _write(CARD, card)
    run_dir = ROOT / "results/gate3_semantics" / RUN_ID
    spec = {
        "schema_version": "v3_run_spec_v1", "slug": "primitive_composition_anchor_failure_attribution_v1",
        "date": "20260903", "seed": 0, "gate": 3, "execution_phase": 3, "operation": "audit",
        "working_directory": str(ROOT), "config_path": str(CARD.relative_to(ROOT)),
        "data_card": str(CARD.relative_to(ROOT)),
        "question": "Which observable component causes the three-seed composition-anchor C07 safety failure?",
        "method": "Frozen three-seed controlled replacement of proposal, anchor coordinates, scale compatibility and endpoint evidence, plus endpoint-error stratification.",
        "baseline": "The exact sealed deployed C07 result and raw predicted-endpoint distance under the same proposal oracle.",
        "fallback": card["failure_policy"], "user_authorization": approval,
        "acceptance_criteria": list(card["metrics_and_pre_registered_gates"].values()),
        "expected_counts": {
            "frozen_seeds": 3, "c07_parent_worlds": 10, "c07_tasks": 30,
            "c07_rows_per_seed": 64_644, "model_forward_rows": 193_932,
            "observable_positive_attachments": 442_936, "optimizer_steps": 0,
            "c08_rows_read": 0, "c09_c10_worlds_read": 0, "graph_replays": 0, "mtare_worlds_read": 0,
        },
        "estimated_cost": card["estimated_cost"],
        "expected_evidence": [
            "Exact reproduction of the sealed deployed result for all three seeds.",
            "Eight controlled score conditions and a pre-registered primary-bottleneck decision.",
            "Endpoint error strata and 90 task-seed rows.",
            "Paper PNG/PDF/SVG, source hashes, logs, RUN_STATE and SHA-256 seal.",
        ],
        "command": [
            "/usr/bin/systemd-inhibit", "--what=sleep:shutdown", "--why=Composition anchor C07 failure attribution",
            "--mode=block", "/usr/bin/timeout", "--signal=INT", "--kill-after=120s", "14400s",
            "/usr/bin/env", f"PYTHONPATH={ROOT / 'src'}:{ROOT / 'tools/v3'}:{ROOT}",
            "PYTHONHASHSEED=0", "CUBLAS_WORKSPACE_CONFIG=:4096:8", PYTHON,
            "tools/v3/run_primitive_composition_anchor_failure_attribution_v1.py",
            "--spec", str(SPEC), "--run-dir", str(run_dir),
        ],
    }
    frozen_inputs = dict(source_spec["frozen_inputs"])
    inputs = [
        CARD, SOURCE_CARD, SOURCE_SPEC,
        FORMAL / "RUN_STATE.json", FORMAL / "metrics/summary.json",
        FORMAL / "metrics/evaluation/summary.json", FORMAL / "artifacts/evidence_sha256.txt",
    ]
    for seed in range(3):
        inputs.extend((
            TRAINING / f"artifacts/models/seed{seed}/selected.pt",
            TRAINING / f"artifacts/models/seed{seed}/summary.json",
        ))
    frozen_inputs.update({str(path.relative_to(ROOT)): _sha(path) for path in inputs})
    spec["frozen_inputs"] = frozen_inputs
    tools = {
        "runner": "tools/v3/run_primitive_composition_anchor_failure_attribution_v1.py",
        "freezer": "tools/v3/freeze_primitive_composition_anchor_failure_attribution_v1_spec.py",
        "executor": "tools/v3/execute_primitive_composition_anchor_failure_attribution_v1.py",
        "diagnostic": "src/mtare_topo/evaluation/primitive_composition_anchor_failure_attribution.py",
        "diagnostic_tests": "tests/v3/unit/test_primitive_composition_anchor_failure_attribution.py",
        "model": "src/mtare_topo/representation/primitive_composition_anchor_model.py",
        "model_tests": "tests/v3/unit/test_primitive_composition_anchor_model.py",
        "evaluation_tests": "tests/v3/unit/test_evaluate_primitive_composition_anchor_three_seed_v1.py",
        "batch_module": "src/mtare_topo/data/primitive_composition_anchor_batches.py",
        "training_module": "src/mtare_topo/representation/primitive_composition_anchor_training.py",
        "governance": "src/mtare_topo/governance.py",
        "preflight": "tools/v3/preflight.py", "create_run": "tools/v3/create_run.py",
    }
    spec["frozen_tools"] = {
        name: {"path": path, "sha256": _sha(ROOT / path)} for name, path in tools.items()
    }
    _write(SPEC, spec)
    print(CARD); print(SPEC)


if __name__ == "__main__":
    main()
