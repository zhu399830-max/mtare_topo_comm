#!/usr/bin/env python3
"""Freeze the one C07-only sparse-port relation failure attribution."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

from _bootstrap import PROJECT_ROOT


SOURCE_CARD = PROJECT_ROOT / "configs/v3/gate3/data_cards/primitive_relation_v1_failure_attribution_v1r2.json"
CARD = PROJECT_ROOT / "configs/v3/gate3/data_cards/primitive_relation_sparse_port_failure_attribution_v1.json"
SPEC = PROJECT_ROOT / "configs/v3/gate3/primitive_relation_sparse_port_failure_attribution_v1.json"
RUN_ID = "gate3_20260902_primitive_relation_sparse_port_failure_attribution_v1_seed0"
P1A = PROJECT_ROOT / "results/gate3_semantics/gate3_20260830_primitive_relation_p1a_lossless_storage_corrective_v1_seed0"
P1B = PROJECT_ROOT / "results/gate3_semantics/gate3_20260830_primitive_relation_p1b_teacher_materialization_v1r_seed0"
TRAINING = PROJECT_ROOT / "results/gate3_semantics/gate3_20260901_primitive_relation_sparse_port_three_seed_training_v1r_seed0"
BASELINE = PROJECT_ROOT / "results/gate3_semantics/gate3_20260830_primitive_relation_nonlearning_c07_v1_seed0"


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    if CARD.exists() or SPEC.exists():
        raise RuntimeError("sparse-port failure attribution card/spec already exists")
    state = json.loads((TRAINING / "RUN_STATE.json").read_text())
    summary = json.loads((TRAINING / "metrics/summary.json").read_text())
    if state.get("state") != "COMPLETED" or state.get("error") is not None:
        raise RuntimeError("sparse-port source run is not cleanly completed")
    if summary.get("scientific_pass") is not False or summary.get("c08_rows_read") != 0:
        raise RuntimeError("sparse-port source scientific decision drift")
    evaluation = summary.get("evaluation", {})
    if evaluation.get("decision") != "STOP_SPARSE_PORT_BEFORE_C08_AND_GRAPH":
        raise RuntimeError("sparse-port stop decision drift")
    if evaluation.get("c07", {}).get("passing_seeds") != 0:
        raise RuntimeError("sparse-port source is not the frozen 0/3 result")

    card = copy.deepcopy(json.loads(SOURCE_CARD.read_text()))
    card["card_id"] = "primitive_relation_sparse_port_failure_attribution_v1"
    card["status"] = "APPROVED_FOR_ONE_IMMUTABLE_PRIMITIVE_RELATION_SPARSE_PORT_FAILURE_ATTRIBUTION_V1"
    card["purpose"] = (
        "Attribute the sealed V2 sparse-port C07 failure to primitive proposal identity/cardinality, independent pair decoding, relation-score ranking, or uncertainty calibration without changing any checkpoint or reading C08."
    )
    card["approval"] = {
        "status": "APPROVED",
        "approved_by": "user-standing-authorization",
        "approved_at": "2026-09-02T02:40:00+08:00",
        "authorized_gates": [3],
        "authorized_operations": ["audit"],
        "scope": (
            "One immutable zero-training C07-only audit: 10 parents, 30 paired geometry tasks, 64644 sequences per frozen seed, 193932 inference rows total; zero C08/C09/C10, graph, M-TARE, optimizer or checkpoint write."
        ),
        "confirmation_reference": (
            "The user repeatedly delegated automatic selection of the strongest in-plan option. The sealed V2 0/3 C07 result requires the pre-registered C07-only failure attribution before any new method or transfer-domain read."
        ),
    }
    card["source"]["raw_sources"] = [
        "sealed P1a corrected C07 five-frame 16x720 range/valid and relative odometry",
        "sealed P1b C07 32-slot visible primitive, endpoint attachment, disconnected-overlap and temporal Teacher",
        "three selected sparse-port V2 checkpoints and three sealed formal C07 metrics",
    ]
    card["sampling"]["rule"] = (
        "Read every C07 sequence exactly once per frozen seed. Compare deployed existence, Teacher-count top-k, and Teacher-aligned proposal masks; evaluate unchanged independent, best-link-union and reciprocal-best-link decoders with raw and risk-adjusted frozen scores. Exact ranked thresholds preserve score ties. No raw pair score is retained."
    )
    counts = card["sampling"]["structure_event_counts"]
    counts.pop("corrective_evaluation_inference_rows", None)
    counts["attribution_inference_rows"] = 193_932
    counts["model_inference_rows"] = 193_932
    card["leakage_audit"]["model_inference_count"] = 193_932
    card["metrics_and_pre_registered_gates"] = {
        "implementation": "Eighteen frozen unit tests pass; formal V2 attachment counts are exactly reproduced before diagnostics; matmul and cuDNN TF32 remain disabled.",
        "population": "Exactly 64644 rows and 30 tasks per seed, three selected epochs unchanged, 193932 total inference rows, and zero C08/C09/C10/graph/M-TARE.",
        "proposal": "Report deployed, Teacher-cardinality and Teacher-aligned masks with slot/pair populations and exact raw/risk-adjusted attachment selections.",
        "structure": "Report independent, best-link-union and reciprocal-best-link decoding. Best-link variants may use only the frozen learned relation score and candidate mask; no geometry threshold or relation Teacher enters decoding.",
        "calibration": "Use tie-safe exact ranked thresholds and precision>=0.98 nonempty selection to test whether the frozen 0.05--0.95 grid or relation uncertainty hid a valid score prefix.",
        "stratification": "Report per-task topology/geometry attachment F1, positive/negative score quantiles and disconnected-overlap hard-negative false attachments.",
        "decision": "Resolve exactly one diagnosis in priority order: coarse-grid miss, structured decoding missing, uncertainty calibration suppression, proposal/cardinality dominance, local-link-only ranking, or relation-score failure even with proposal oracle. Oracle results are diagnostic only and never deployment performance.",
        "resources": "Wall time <=3h, CUDA reserved <=16GiB, host RSS <=16GiB and output <=0.2GiB; zero optimizer/checkpoint update.",
    }
    card["split"]["selection"] = (
        "No checkpoint or method selection and no deployment claim. Exact thresholds and oracle masks diagnose the already-failed C07 model only; any proposed correction requires a separate readiness and new formal training contract."
    )
    card["teacher"]["valid_mask"] = (
        "Teacher primitive mask supplies either only visible cardinality or an evaluation-only proposal oracle after unchanged Hungarian alignment. Teacher attachments remain metric targets and never enter model inference or best-link decoding."
    )
    card["teacher"]["planner_consistency_plan"] = (
        "No graph or planner run. The diagnosis only selects whether a new structured relation/calibration readiness is scientifically defensible or the current relation head must stop."
    )
    card["estimated_cost"] = {
        "compute": "One RTX 5090; one streaming C07 pass for each of three frozen sparse-port checkpoints plus CPU exact ranking.",
        "gpu": 1, "gpu_memory_gb": 16, "host_ram_gb": 16,
        "wall_time_hours": 1.5, "disk_gb": 0.2,
    }
    card["failure_policy"] = (
        "Any formal-count reproduction, population, checkpoint, input/tool/environment, resource, tie handling or isolation drift fails closed. Do not modify thresholds, checkpoint, model, data, Teacher, C08 or planner to resolve the diagnosis."
    )
    card["retention"] = (
        "Retain per-seed and per-task attribution, exact score selections, mask/pair populations, error strata, PNG/PDF/SVG/source, environment, commands, logs, RUN_STATE and SHA-256 seal. Do not retain raw pair-score arrays."
    )
    write(CARD, card)

    run_dir = PROJECT_ROOT / f"results/gate3_semantics/{RUN_ID}"
    spec = {
        "schema_version": "v3_run_spec_v1",
        "slug": "primitive_relation_sparse_port_failure_attribution_v1",
        "date": "20260902", "seed": 0, "gate": 3, "execution_phase": 3,
        "operation": "audit", "working_directory": str(PROJECT_ROOT),
        "config_path": str(CARD.relative_to(PROJECT_ROOT)),
        "data_card": str(CARD.relative_to(PROJECT_ROOT)),
        "question": (
            "Why do all three frozen sparse-port seeds recover C07 geometry but fail non-vacuous precision>=0.98 endpoint attachment, and which single relation-layer mechanism must change before any C08 or graph work?"
        ),
        "method": (
            "One TF32-off inference pass per frozen seed on all C07 rows, comparing deployed, Teacher-cardinality and proposal-oracle candidate masks; independent and learned-score-only best-link decoding; raw versus uncertainty-adjusted exact ranked precision-recall and hard-negative strata."
        ),
        "baseline": (
            "Sealed same-input non-learning attachment F1=0.005819 and the exact sealed V2 deployed C07 attachment metrics for seeds 0/1/2."
        ),
        "fallback": (
            "Any failed check stops the attribution. No C08, retraining, threshold relaxation, hand-written attachment rule, graph or planner compensation is allowed."
        ),
        "user_authorization": card["approval"],
        "acceptance_criteria": list(card["metrics_and_pre_registered_gates"].values()),
        "expected_counts": {
            "unit_tests": 18, "c07_parent_worlds": 10, "c07_geometry_tasks": 30,
            "c07_rows_per_seed": 64_644, "frozen_seeds": 3,
            "attribution_inference_rows": 193_932, "model_inference_rows": 193_932,
            "optimizer_steps": 0, "c08_rows_read": 0, "c09_c10_worlds_read": 0,
            "graph_replays": 0, "mtare_worlds_read": 0,
        },
        "estimated_cost": card["estimated_cost"],
        "expected_evidence": [
            "Exact formal attachment reproduction and complete C07 population proof.",
            "Per-seed deployed/cardinality/proposal masks under three score-only decoders.",
            "Tie-safe exact F1 and nonempty precision>=0.98 selections for raw and risk-adjusted scores.",
            "Per-task and disconnected-overlap failure strata plus one resolved diagnosis.",
            "PNG/PDF/SVG/source, environment, commands, logs, RUN_STATE and SHA-256 seal.",
        ],
        "command": [
            "/usr/bin/systemd-inhibit", "--what=sleep:shutdown",
            "--why=Sparse-port relation C07 failure attribution", "--mode=block",
            "/usr/bin/timeout", "--signal=INT", "--kill-after=60s", "10800s",
            "/usr/bin/env", "CUBLAS_WORKSPACE_CONFIG=:4096:8",
            f"PYTHONPATH={PROJECT_ROOT / 'src'}:{PROJECT_ROOT / 'tools/v3'}",
            "/home/zeng-workstation/.local/share/mtare_topo_comm/envs/phase3_torch290_cu129_zarr2187_v1/bin/python",
            "tools/v3/run_primitive_relation_sparse_port_failure_attribution_v1.py",
            "--spec", str(SPEC), "--run-dir", str(run_dir),
        ],
    }
    inputs = [
        PROJECT_ROOT / "configs/v3/gate2/environments/phase3_torch290_cu129_zarr2187_pip_freeze.txt",
        PROJECT_ROOT / "configs/v3/gate2/environments/phase3_torch290_cu129_zarr2187_v1r2.json",
        CARD,
        *[root / name for root in (P1A, P1B, BASELINE) for name in (
            "RUN_STATE.json", "metrics/summary.json", "artifacts/evidence_sha256.txt",
        )],
        P1A / "artifacts/task_manifest.json", P1B / "artifacts/task_manifest.json",
        TRAINING / "RUN_STATE.json", TRAINING / "metrics/summary.json",
        TRAINING / "artifacts/evidence_sha256.txt", TRAINING / "config/source_integrity_after.json",
        *[TRAINING / f"artifacts/models/seed{seed}/selected.pt" for seed in range(3)],
        *[TRAINING / f"metrics/evaluation/c07_seed{seed}.json" for seed in range(3)],
    ]
    spec["frozen_inputs"] = {str(path.relative_to(PROJECT_ROOT)): sha(path) for path in inputs}
    tools = {
        "attribution_module": "src/mtare_topo/evaluation/primitive_relation_sparse_port_failure_attribution.py",
        "attribution_tests": "tests/v3/unit/test_primitive_relation_sparse_port_failure_attribution.py",
        "legacy_attribution_module": "src/mtare_topo/evaluation/primitive_relation_failure_attribution.py",
        "legacy_attribution_tests": "tests/v3/unit/test_primitive_relation_failure_attribution.py",
        "metrics": "src/mtare_topo/evaluation/primitive_relation_metrics.py",
        "metric_tests": "tests/v3/unit/test_primitive_relation_metrics.py",
        "sparse_evaluation_tests": "tests/v3/unit/test_evaluate_primitive_relation_sparse_port_three_seed_v1.py",
        "model": "src/mtare_topo/representation/primitive_relation_sparse_port_model.py",
        "loss_alignment": "src/mtare_topo/representation/primitive_relation_losses.py",
        "batch_reader": "src/mtare_topo/data/primitive_relation_batches.py",
        "training_conversion": "src/mtare_topo/representation/primitive_relation_training.py",
        "executor": "tools/v3/execute_primitive_relation_sparse_port_failure_attribution_v1.py",
        "runner": "tools/v3/run_primitive_relation_sparse_port_failure_attribution_v1.py",
        "freezer": "tools/v3/freeze_primitive_relation_sparse_port_failure_attribution_v1_spec.py",
        "governance": "src/mtare_topo/governance.py",
        "preflight": "tools/v3/preflight.py", "create_run": "tools/v3/create_run.py",
    }
    spec["frozen_tools"] = {
        name: {"path": path, "sha256": sha(PROJECT_ROOT / path)} for name, path in tools.items()
    }
    write(SPEC, spec)
    print(SPEC)


if __name__ == "__main__":
    main()
