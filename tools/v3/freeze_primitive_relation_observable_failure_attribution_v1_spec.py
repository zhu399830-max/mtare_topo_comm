#!/usr/bin/env python3
"""Freeze one C07-only observable primitive-relation failure attribution."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

from _bootstrap import PROJECT_ROOT


SOURCE_CARD = PROJECT_ROOT / "configs/v3/gate3/data_cards/primitive_relation_observable_c07_tf32_corrective_v1r.json"
CARD = PROJECT_ROOT / "configs/v3/gate3/data_cards/primitive_relation_observable_failure_attribution_v1.json"
SPEC = PROJECT_ROOT / "configs/v3/gate3/primitive_relation_observable_failure_attribution_v1.json"
RUN_ID = "gate3_20260902_primitive_relation_observable_failure_attribution_v1_seed0"
P1A = PROJECT_ROOT / "results/gate3_semantics/gate3_20260830_primitive_relation_p1a_lossless_storage_corrective_v1_seed0"
P1B = PROJECT_ROOT / "results/gate3_semantics/gate3_20260830_primitive_relation_p1b_teacher_materialization_v1r_seed0"
SIDECAR = PROJECT_ROOT / "results/gate3_semantics/gate3_20260902_primitive_attachment_observability_sidecar_v1_seed0"
TRAINING = PROJECT_ROOT / "results/gate3_semantics/gate3_20260902_primitive_relation_observable_three_seed_training_v1_seed0"
CORRECTIVE = PROJECT_ROOT / "results/gate3_semantics/gate3_20260902_primitive_relation_observable_c07_tf32_evidence_corrective_v1_seed0"
BASELINE = PROJECT_ROOT / "results/gate3_semantics/gate3_20260902_primitive_relation_observable_nonlearning_c07_v1_seed0"


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
        raise RuntimeError("observable failure attribution card/spec already exists")
    training_state = json.loads((TRAINING / "RUN_STATE.json").read_text())
    training_summary = json.loads((TRAINING / "metrics/summary.json").read_text())
    corrected_state = json.loads((CORRECTIVE / "RUN_STATE.json").read_text())
    corrected_summary = json.loads((CORRECTIVE / "metrics/summary.json").read_text())
    if (
        training_state.get("state") != "COMPLETED"
        or training_summary.get("optimizer_steps") != 240_624
        or training_summary.get("c08_rows_read") != 0
    ):
        raise RuntimeError("observable training source is incomplete")
    if (
        corrected_state.get("state") != "COMPLETED"
        or corrected_summary.get("evidence_scientific_pass") is not True
        or corrected_summary.get("observable_relation_model_scientific_pass") is not False
        or corrected_summary.get("scientific_decision") != "STOP_OBSERVABLE_RELATION_BEFORE_C08_AND_GRAPH"
    ):
        raise RuntimeError("corrected observable C07 stop evidence is incomplete")

    card = copy.deepcopy(json.loads(SOURCE_CARD.read_text()))
    card["card_id"] = "primitive_relation_observable_failure_attribution_v1"
    card["status"] = "APPROVED_FOR_ONE_IMMUTABLE_PRIMITIVE_RELATION_OBSERVABLE_FAILURE_ATTRIBUTION_V1"
    card["purpose"] = (
        "Attribute the compliant sealed 0-of-3 C07 endpoint-relation failure to primitive proposal errors, dual-endpoint evidence, pair-space combinatorics, learned relation ranking or confidence calibration without changing checkpoints or reading C08."
    )
    card["approval"] = {
        "status": "APPROVED", "approved_by": "user-standing-authorization",
        "approved_at": "2026-09-02T17:10:00+08:00",
        "authorized_gates": [3], "authorized_operations": ["audit"],
        "scope": (
            "One immutable zero-training C07-only audit: 10 parents, 30 geometry tasks, 64644 sequences per frozen seed and 193932 model-forward rows total; zero C08/C09/C10, graph, M-TARE, optimizer or checkpoint write."
        ),
        "confirmation_reference": (
            "The user delegated automatic strongest-evidence execution. The compliant 0-of-3 stop result requires the frozen-plan C07 failure attribution before any method revision or transfer-domain access."
        ),
    }
    card["source"]["raw_sources"] = [
        "sealed P1a corrected C07 five-frame 16x720 range/valid data and relative odometry",
        "sealed P1b C07 32-slot construction Teacher and endpoint-observability sidecar",
        "three selected observable-relation checkpoints and corrected TF32-off C07 metrics",
        "sealed same-input observable non-learning C07 baseline",
    ]
    card["sampling"]["rule"] = (
        "Read every C07 sequence exactly once per frozen seed. Compare deployed existence, Teacher-count top-k and Teacher-aligned proposal masks; evaluate unchanged independent and learned-score-only best-link decoders with raw, uncertainty-adjusted and learned endpoint-evidence safe scores. Exact ranked thresholds preserve ties; raw pair scores are not retained."
    )
    counts = card["sampling"]["structure_event_counts"]
    counts["attribution_model_forward_rows"] = 193_932
    counts["model_forward_rows"] = 193_932
    counts["passes_per_seed"] = 1
    card["leakage_audit"]["model_inference_count"] = 193_932
    card["metrics_and_pre_registered_gates"] = {
        "implementation": "All frozen observable-attribution, legacy relation metric and corrected-evaluator tests pass; formal attachment and safe counts reproduce exactly under deterministic TF32-off.",
        "population": "Exactly 64644 rows and 30 tasks per seed, three selected epochs unchanged, 193932 total forward rows and zero C08/C09/C10/graph/M-TARE.",
        "proposal": "Report deployed, Teacher-cardinality and Teacher-aligned proposal masks with matched/missed/redundant slots, raw pair spaces and observable eligible pairs.",
        "evidence": "Report endpoint-evidence exact F1 and nonempty precision>=0.98 prefixes for all slots and proposal-oracle matched endpoints.",
        "relation": "Report independent and frozen-score-only best-link-union relation ranking for deployed/cardinality/oracle masks; no geometry threshold, identity or Teacher relation enters decoding.",
        "calibration": "Compare raw, relation-uncertainty-adjusted and full learned-safe scores using tie-safe exact ranking and nonempty precision>=0.98 selection.",
        "decision": "Resolve one diagnosis in priority order: safe-score suppression, grid miss, structured decoding, uncertainty suppression, endpoint-evidence suppression, proposal dominance, local-ranking-only or relation-score failure even with observable proposal oracle.",
        "resources": "Wall time <=3h, CUDA reserved <=16GiB, host RSS <=16GiB, output <=0.2GiB and zero optimizer/checkpoint update.",
    }
    card["split"]["selection"] = (
        "No checkpoint, threshold or method selection and no deployment claim. Exact thresholds and oracle masks diagnose the already-failed C07 model only; any correction requires a separate readiness and training contract."
    )
    card["teacher"]["valid_mask"] = (
        "The frozen dual-endpoint-observed mask excludes hidden matched pairs as unknown while retaining unmatched proposals as negatives. Teacher cardinality and aligned proposal identity are evaluation-only diagnostics and never enter model forward."
    )
    card["teacher"]["planner_consistency_plan"] = (
        "No graph or planner run. The diagnosis only decides whether a plan-compliant learned candidate/matching/calibration revision is defensible or the direct pair head must stop."
    )
    card["estimated_cost"] = {
        "compute": "One RTX 5090; one streaming complete C07 pass for each of three frozen checkpoints plus CPU exact ranking.",
        "gpu": 1, "gpu_memory_gb": 16, "host_ram_gb": 16,
        "wall_time_hours": 1.0, "disk_gb": 0.2,
    }
    card["failure_policy"] = (
        "Any formal-count reproduction, population, checkpoint, input/tool/environment, numerical contract, resource, tie handling or isolation drift fails closed. Do not modify threshold, checkpoint, model, Teacher, C08 or planner to resolve the diagnosis."
    )
    card["retention"] = (
        "Retain per-seed/per-task attribution, exact score selections, slot/pair populations, endpoint evidence, PNG/PDF/SVG/source, environment, commands, logs, RUN_STATE and SHA-256 seal. Do not retain raw pair-score arrays."
    )
    write(CARD, card)

    run_dir = PROJECT_ROOT / f"results/gate3_semantics/{RUN_ID}"
    spec = {
        "schema_version": "v3_run_spec_v1",
        "slug": "primitive_relation_observable_failure_attribution_v1",
        "date": "20260902", "seed": 0, "gate": 3, "execution_phase": 3,
        "operation": "audit", "working_directory": str(PROJECT_ROOT),
        "config_path": str(CARD.relative_to(PROJECT_ROOT)),
        "data_card": str(CARD.relative_to(PROJECT_ROOT)),
        "question": (
            "Why do all three TF32-off observable-relation seeds recover C07 geometry but produce no nonzero precision>=0.98 attachment, and which single relation-layer mechanism may change before any C08 or graph work?"
        ),
        "method": (
            "One TF32-off inference pass per frozen seed on all C07 rows; deployed/cardinality/proposal masks, exact raw/uncertainty/evidence score ranking, frozen-score-only best-link decoding, endpoint-evidence and hard-negative strata."
        ),
        "baseline": (
            "Sealed observable non-learning attachment F1=0.006380 and the exact corrected deployed metrics for seeds 0/1/2."
        ),
        "fallback": card["failure_policy"], "user_authorization": card["approval"],
        "acceptance_criteria": list(card["metrics_and_pre_registered_gates"].values()),
        "expected_counts": {
            "c07_parent_worlds": 10, "c07_geometry_tasks": 30,
            "c07_rows_per_seed": 64_644, "frozen_seeds": 3,
            "attribution_model_forward_rows": 193_932, "model_forward_rows": 193_932,
            "optimizer_steps": 0, "c08_rows_read": 0, "c09_c10_worlds_read": 0,
            "graph_replays": 0, "mtare_worlds_read": 0,
        },
        "estimated_cost": card["estimated_cost"],
        "expected_evidence": [
            "Exact corrected formal attachment/safe reproduction and full C07 population proof.",
            "Per-seed candidate, endpoint-evidence, pair-space, oracle and calibration diagnostics.",
            "Per-task strata, one resolved diagnosis and a plan-compliant next-method boundary.",
            "PNG/PDF/SVG/source, environment, commands, logs, RUN_STATE and SHA-256 seal.",
        ],
        "command": [
            "/usr/bin/systemd-inhibit", "--what=sleep:shutdown",
            "--why=Observable primitive relation C07 failure attribution", "--mode=block",
            "/usr/bin/timeout", "--signal=INT", "--kill-after=60s", "10800s",
            "/usr/bin/env", "CUBLAS_WORKSPACE_CONFIG=:4096:8",
            f"PYTHONPATH={PROJECT_ROOT / 'src'}:{PROJECT_ROOT / 'tools/v3'}:{PROJECT_ROOT}",
            "/home/zeng-workstation/.local/share/mtare_topo_comm/envs/phase3_torch290_cu129_zarr2187_v1/bin/python",
            "tools/v3/run_primitive_relation_observable_failure_attribution_v1.py",
            "--spec", str(SPEC), "--run-dir", str(run_dir),
        ],
    }
    inputs = [
        PROJECT_ROOT / "configs/v3/gate2/environments/phase3_torch290_cu129_zarr2187_pip_freeze.txt",
        PROJECT_ROOT / "configs/v3/gate2/environments/phase3_torch290_cu129_zarr2187_v1r2.json",
        CARD,
        *[root / name for root in (P1A, P1B, SIDECAR, BASELINE, TRAINING, CORRECTIVE) for name in (
            "RUN_STATE.json", "metrics/summary.json", "artifacts/evidence_sha256.txt",
        )],
        P1A / "artifacts/task_manifest.json",
        P1B / "artifacts/task_manifest.json",
        SIDECAR / "artifacts/task_manifest.json",
        TRAINING / "config/source_integrity_after.json",
        *[TRAINING / f"artifacts/models/seed{seed}/selected.pt" for seed in range(3)],
        *[CORRECTIVE / f"metrics/source_evaluation/c07_seed{seed}.json" for seed in range(3)],
    ]
    spec["frozen_inputs"] = {
        str(path.relative_to(PROJECT_ROOT)): sha(path) for path in inputs
    }
    tools = {
        "attribution_module": "src/mtare_topo/evaluation/primitive_relation_observable_failure_attribution.py",
        "attribution_tests": "tests/v3/unit/test_primitive_relation_observable_failure_attribution.py",
        "sparse_attribution_module": "src/mtare_topo/evaluation/primitive_relation_sparse_port_failure_attribution.py",
        "sparse_attribution_tests": "tests/v3/unit/test_primitive_relation_sparse_port_failure_attribution.py",
        "legacy_attribution_module": "src/mtare_topo/evaluation/primitive_relation_failure_attribution.py",
        "legacy_attribution_tests": "tests/v3/unit/test_primitive_relation_failure_attribution.py",
        "metrics": "src/mtare_topo/evaluation/primitive_relation_metrics.py",
        "metric_tests": "tests/v3/unit/test_primitive_relation_metrics.py",
        "model": "src/mtare_topo/representation/primitive_relation_observable_model.py",
        "base_model": "src/mtare_topo/representation/primitive_relation_sparse_port_model.py",
        "loss_alignment": "src/mtare_topo/representation/primitive_relation_losses.py",
        "batch_reader": "src/mtare_topo/data/primitive_relation_observable_batches.py",
        "base_batch_reader": "src/mtare_topo/data/primitive_relation_batches.py",
        "training_conversion": "src/mtare_topo/representation/primitive_relation_observable_training.py",
        "corrective_evaluator": "tools/v3/evaluate_primitive_relation_observable_three_seed_c07_tf32_corrective_v1.py",
        "corrective_tests": "tests/v3/unit/test_evaluate_primitive_relation_observable_three_seed_c07_tf32_corrective_v1.py",
        "base_evaluator_tests": "tests/v3/unit/test_evaluate_primitive_relation_observable_three_seed_v1.py",
        "executor": "tools/v3/execute_primitive_relation_observable_failure_attribution_v1.py",
        "runner": "tools/v3/run_primitive_relation_observable_failure_attribution_v1.py",
        "freezer": "tools/v3/freeze_primitive_relation_observable_failure_attribution_v1_spec.py",
        "governance": "src/mtare_topo/governance.py",
        "preflight": "tools/v3/preflight.py", "create_run": "tools/v3/create_run.py",
    }
    spec["frozen_tools"] = {
        name: {"path": path, "sha256": sha(PROJECT_ROOT / path)}
        for name, path in tools.items()
    }
    write(SPEC, spec)
    print(SPEC)


if __name__ == "__main__":
    main()
