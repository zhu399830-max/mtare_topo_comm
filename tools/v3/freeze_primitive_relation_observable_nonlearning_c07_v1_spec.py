#!/usr/bin/env python3
"""Freeze the C07 same-input baseline on corrected observable relations."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

from _bootstrap import PROJECT_ROOT


CARD = PROJECT_ROOT / "configs/v3/gate3/data_cards/primitive_relation_observable_nonlearning_c07_v1.json"
SPEC = PROJECT_ROOT / "configs/v3/gate3/primitive_relation_observable_nonlearning_c07_v1.json"
SOURCE_CARD = PROJECT_ROOT / "configs/v3/gate3/data_cards/primitive_relation_nonlearning_c07_v1.json"
RUN_ID = "gate3_20260902_primitive_relation_observable_nonlearning_c07_v1_seed0"
PYTHON = "/home/zeng-workstation/.local/share/mtare_topo_comm/envs/phase3_torch290_cu129_zarr2187_v1/bin/python"
P1A = PROJECT_ROOT / "results/gate3_semantics/gate3_20260830_primitive_relation_p1a_lossless_storage_corrective_v1_seed0"
P1B = PROJECT_ROOT / "results/gate3_semantics/gate3_20260830_primitive_relation_p1b_teacher_materialization_v1r_seed0"
SIDECAR = PROJECT_ROOT / "results/gate3_semantics/gate3_20260902_primitive_attachment_observability_sidecar_v1_seed0"
READINESS = PROJECT_ROOT / "results/gate3_semantics/gate3_20260902_primitive_relation_observable_readiness_v1r_seed0"


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
        raise RuntimeError("observable baseline card/spec already exists")
    for source in (P1A, P1B, SIDECAR, READINESS):
        state = json.loads((source / "RUN_STATE.json").read_text())
        summary = json.loads((source / "metrics/summary.json").read_text())
        if (
            state.get("state") != "COMPLETED" or state.get("error") is not None
            or not summary.get("scientific_pass")
        ):
            raise RuntimeError(f"observable baseline prerequisite failed: {source.name}")
    approval = {
        "status": "APPROVED", "approved_by": "user-standing-authorization",
        "approved_at": "2026-09-02T19:00:00+08:00",
        "authorized_gates": [3], "authorized_operations": ["audit"],
        "scope": "One immutable C07-only rerun of the already frozen same-input non-learning baseline, changing only attachment scoring to the sealed dual-endpoint observability population. Zero tuning, C08+, training, graph or M-TARE.",
        "confirmation_reference": "The user authorized uninterrupted best-in-plan execution without repeated routine approvals.",
    }
    gates = {
        "population": "Exactly 10 C07 parents, 30 paired tasks, 64644 sequences and 323220 causal frame references; all sensor, Teacher and sidecar tree hashes match.",
        "method": "The previously frozen robust baseline configuration is reused unchanged; only five causal scans and relative odometry enter predict().",
        "observable_relation": "Attachment target positives equal exactly 442936; 82141 hidden physical positives are excluded from both positive and negative metric population.",
        "metrics": "Report the same primitive, geometry, overlap and temporal metrics plus observable attachment P/R/F1 for direct comparison with the corrected learned model.",
        "resources": "Complete within 2 CPU hours, <=4 GiB RAM and <=0.25 GiB output; zero optimizer/model/checkpoint/C08+/graph/M-TARE.",
    }
    card = copy.deepcopy(json.loads(SOURCE_CARD.read_text()))
    card.update({
        "card_id": "primitive_relation_observable_nonlearning_c07_v1",
        "status": "APPROVED_FOR_ONE_IMMUTABLE_PRIMITIVE_RELATION_OBSERVABLE_NONLEARNING_C07_V1",
        "approval": approval,
        "purpose": "Freeze the fair same-input non-learning comparison on the corrected locally observable attachment population before relation-only learned training.",
        "metrics_and_pre_registered_gates": gates,
        "estimated_cost": {
            "compute": "One CPU process; same frozen robust fitter over all C07 rows with endpoint-observability-aware scoring.",
            "wall_time_hours": 1.0, "host_ram_gb": 4, "gpu": 0,
            "disk_gb": .25,
        },
        "retention": "Keep per-task hashes/timings, complete aggregate metrics, three-format paper figure, environment, test log, RUN_STATE and seal.",
        "failure_policy": "Any source, population, 442936 observable-positive count, scorer, environment or resource drift fails closed. Do not tune the baseline or read C08.",
    })
    card["source"]["raw_sources"].append(
        "sealed C07 endpoint-observability sidecars used only by the scorer"
    )
    card["teacher"] = {
        "source": "P1b construction targets plus the sealed 0.25 m endpoint-support sidecar, both used only after baseline.predict returns.",
        "valid_mask": "A matched physical attachment is scored only when both attached endpoints have current-window ray support. Hidden pairs are excluded, not relabeled negative; unmatched predicted pairs remain false-positive candidates.",
        "planner_consistency_plan": "No graph/planner executes. This exact corrected metric population is reused for learned C07 scoring.",
        "student_forbidden_inputs": "World, topology, primitive identity, construction graph, endpoint-support mask, absolute pose and future frames never enter baseline.predict.",
    }
    card["sampling"]["structure_event_counts"].update({
        "observable_positive_attachments": 442_936,
        "hidden_positive_attachments_excluded": 82_141,
    })
    write(CARD, card)

    inputs = [
        CARD,
        *[
            source / name
            for source in (P1A, P1B, SIDECAR, READINESS)
            for name in ("RUN_STATE.json", "metrics/summary.json", "artifacts/evidence_sha256.txt")
        ],
        P1A / "artifacts/task_manifest.json",
        P1B / "artifacts/task_manifest.json",
        SIDECAR / "artifacts/task_manifest.json",
        PROJECT_ROOT / "configs/v3/gate2/environments/phase3_torch290_cu129_zarr2187_v1r2.json",
        PROJECT_ROOT / "configs/v3/gate2/environments/phase3_torch290_cu129_zarr2187_pip_freeze.txt",
    ]
    tools = {
        "runner": "tools/v3/run_primitive_relation_observable_nonlearning_c07_v1.py",
        "freezer": "tools/v3/freeze_primitive_relation_observable_nonlearning_c07_v1_spec.py",
        "plot_helper": "tools/v3/run_primitive_relation_nonlearning_c07_v1.py",
        "baseline": "src/mtare_topo/semantics/primitive_relation_nonlearning.py",
        "exit_baseline": "src/mtare_topo/semantics/range_exit_baseline.py",
        "observable_reader": "src/mtare_topo/data/primitive_relation_observable_batches.py",
        "torch_batch": "src/mtare_topo/representation/primitive_relation_observable_training.py",
        "metrics": "src/mtare_topo/evaluation/primitive_relation_metrics.py",
        "loss": "src/mtare_topo/representation/primitive_relation_losses.py",
        "baseline_tests": "tests/v3/unit/test_primitive_relation_nonlearning.py",
        "metric_tests": "tests/v3/unit/test_primitive_relation_metrics.py",
        "reader_tests": "tests/v3/unit/test_primitive_relation_observable_batches.py",
        "governance": "src/mtare_topo/governance.py",
        "preflight": "tools/v3/preflight.py", "create_run": "tools/v3/create_run.py",
    }
    run_dir = PROJECT_ROOT / f"results/gate3_semantics/{RUN_ID}"
    spec = {
        "schema_version": "v3_run_spec_v1", "gate": 3,
        "execution_phase": 3, "operation": "audit", "date": "20260902",
        "slug": "primitive_relation_observable_nonlearning_c07_v1", "seed": 0,
        "question": "How well does the frozen same-input robust fitter recover locally observable physical endpoint attachments on all C07 development topologies?",
        "method": "Rerun the frozen robust swept-superellipse baseline unchanged and score its outputs with the shared reversal-safe evaluator after masking only matched attachment pairs that lack dual endpoint ray support.",
        "baseline": "This run is the corrected non-learning baseline; the previous unmasked C07 result remains a label-policy ablation.",
        "fallback": "Any failure stops before learned training. Do not reuse the old unmasked attachment F1 as the corrected comparison or alter the fitter.",
        "corrective_of": "results/gate3_semantics/gate3_20260830_primitive_relation_nonlearning_c07_v1_seed0",
        "data_card": str(CARD.relative_to(PROJECT_ROOT)),
        "config_path": str(CARD.relative_to(PROJECT_ROOT)),
        "user_authorization": approval,
        "acceptance_criteria": list(gates.values()),
        "expected_counts": {
            "independent_parent_worlds": 10, "paired_geometry_tasks": 30,
            "rows": 64_644, "causal_frame_references": 323_220,
            "observable_positive_attachments": 442_936,
            "hidden_positive_attachments_excluded": 82_141,
            "unit_tests": 17, "optimizer_steps": 0,
            "model_inference_frames": 0, "checkpoint_writes": 0,
            "c08_rows_read": 0, "c09_c10_worlds_read": 0,
            "graph_replays": 0, "mtare_worlds_read": 0,
        },
        "expected_evidence": [
            "Thirty source-bound task records and complete corrected metrics.",
            "Exact 442936 observable positives and 82141 hidden positives excluded.",
            "Three-format paper figure, environment, tests, RUN_STATE and seal.",
        ],
        "estimated_cost": card["estimated_cost"],
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
            "--why=Observable non-learning C07 baseline", "--mode=block",
            "/usr/bin/timeout", "--signal=INT", "--kill-after=60s", "7200s",
            "/usr/bin/env", f"PYTHONPATH={PROJECT_ROOT / 'src'}:{PROJECT_ROOT / 'tools/v3'}",
            PYTHON, "tools/v3/run_primitive_relation_observable_nonlearning_c07_v1.py",
            "--spec", str(SPEC), "--run-dir", str(run_dir),
        ],
    }
    write(SPEC, spec)
    print(SPEC)


if __name__ == "__main__":
    main()
