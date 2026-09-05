#!/usr/bin/env python3
"""Freeze the zero-inference TF32-off C07 evidence completion corrective."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

from _bootstrap import PROJECT_ROOT


SOURCE_CARD = PROJECT_ROOT / "configs/v3/gate3/data_cards/primitive_relation_observable_c07_tf32_corrective_v1r.json"
SOURCE_SPEC = PROJECT_ROOT / "configs/v3/gate3/primitive_relation_observable_c07_tf32_corrective_v1r.json"
SOURCE = PROJECT_ROOT / "results/gate3_semantics/gate3_20260902_primitive_relation_observable_c07_tf32_corrective_v1r_seed0"
CARD = PROJECT_ROOT / "configs/v3/gate3/data_cards/primitive_relation_observable_c07_tf32_evidence_corrective_v1.json"
SPEC = PROJECT_ROOT / "configs/v3/gate3/primitive_relation_observable_c07_tf32_evidence_corrective_v1.json"
RUN_ID = "gate3_20260902_primitive_relation_observable_c07_tf32_evidence_corrective_v1_seed0"


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


def source_sealed_files() -> list[Path]:
    result = []
    seal = SOURCE / "artifacts/evidence_sha256.txt"
    for line in seal.read_text(encoding="utf-8").splitlines():
        _, relative = line.split("  ", 1)
        result.append(PROJECT_ROOT / relative)
    if len(result) != 28:
        raise RuntimeError(f"expected 28 source seal files, got {len(result)}")
    return result


def main() -> None:
    if CARD.exists() or SPEC.exists():
        raise RuntimeError("TF32 evidence corrective card/spec already exists")
    state = json.loads((SOURCE / "RUN_STATE.json").read_text())
    outer = json.loads((SOURCE / "metrics/summary.json").read_text())
    evaluation = json.loads((SOURCE / "metrics/evaluation/summary.json").read_text())
    if (
        state.get("state") != "COMPLETED"
        or outer.get("model_forward_rows") != 387_864
        or evaluation.get("scientific_pass") is not False
        or evaluation.get("c07", {}).get("passing_seeds") != 0
    ):
        raise RuntimeError("source TF32 corrective scientific evidence is incomplete")

    card = copy.deepcopy(json.loads(SOURCE_CARD.read_text()))
    card["card_id"] = "primitive_relation_observable_c07_tf32_evidence_corrective_v1"
    card["status"] = "APPROVED_FOR_ONE_IMMUTABLE_PRIMITIVE_RELATION_OBSERVABLE_C07_TF32_EVIDENCE_CORRECTIVE_V1"
    card["purpose"] = (
        "Validate and seal the already-computed TF32-off C07 evidence after the source wrapper called len() on the integer returned by _seal. No sensor, Teacher, checkpoint or model is loaded and no inference is repeated."
    )
    card["approval"] = {
        "status": "APPROVED", "approved_by": "user-standing-authorization",
        "approved_at": "2026-09-02T16:30:00+08:00", "authorized_gates": [3],
        "authorized_operations": ["audit"],
        "scope": (
            "One immutable zero-inference audit of the complete 28-file TF32 corrective source seal, 387864 completed source forward rows, three C07 seed outputs, numerical contract, resources, isolation and unchanged upstream inputs; copy only compact evidence and figures."
        ),
        "confirmation_reference": (
            "The user delegated automatic strongest-evidence execution. The scientific computation and source seal completed before a post-seal reporting TypeError, so evidence-only completion avoids repeating identical GPU work."
        ),
    }
    card["source"]["raw_sources"] = [
        "sealed TF32-off C07 corrective V1R run with 387864 completed forward rows",
        "sealed three-seed evaluation summary, per-seed metrics, old/new comparison and figures",
        "source run spec with all upstream frozen input/tool hashes",
    ]
    card["sampling"]["rule"] = (
        "No sensor, Teacher or checkpoint loader is constructed. Read only sealed source evidence; verify all 28 seal entries, three 64644-row seed outputs, numerical contract, source resources/isolation, upstream hashes and byte-identical compact copies."
    )
    counts = card["sampling"]["structure_event_counts"]
    counts["source_model_forward_rows"] = 387_864
    counts["new_model_forward_rows"] = 0
    card["leakage_audit"]["model_inference_count"] = 0
    card["metrics_and_pre_registered_gates"] = {
        "implementation": "All evidence-audit and original TF32 corrective evaluator unit tests pass.",
        "source_boundary": "Source computation is complete, error null and system status PASS; its frozen runner must contain the exact post-seal integer/len defect.",
        "source_evidence": "All 28 source seal entries, three seed outputs, summary, comparison, logs and three-format figure exist and are hash-valid.",
        "scientific_result": "The TF32-off contract is exact; 0/3 seeds pass and all seeds have zero safe attachment true positives, preserving STOP before C08/graph.",
        "resources": "Source host/GPU caps pass; corrective wall time <=10min, output <=20MiB and zero new inference/optimizer/C08/C09/C10/graph/M-TARE.",
        "integrity": "Every source upstream input/tool remains unchanged; copied compact evidence is byte-identical and receives a new immutable seal.",
    }
    card["split"]["selection"] = "No selection or thresholding; validate already-computed sealed scientific evidence only."
    card["teacher"]["valid_mask"] = "Teacher is not loaded; sealed evaluation metrics are read only as evidence files."
    card["teacher"]["planner_consistency_plan"] = "No graph/planner work; preserve the source stop decision."
    card["estimated_cost"] = {
        "compute": "CPU-only hash, schema and byte-copy audit; zero model inference.",
        "gpu": 0, "gpu_memory_gb": 0, "host_ram_gb": 4,
        "wall_time_hours": 0.1, "disk_gb": 0.02,
    }
    card["failure_policy"] = (
        "Any source seal, population, numerical contract, result, resource, isolation, upstream hash or byte-copy mismatch fails closed. Do not rerun inference or repair the source run in place."
    )
    card["retention"] = (
        "Retain source provenance, byte-identical evaluation metrics/figures/logs, tests, integrity snapshots, RUN_STATE and new seal. Preserve the source run unchanged."
    )
    write(CARD, card)

    run_dir = PROJECT_ROOT / f"results/gate3_semantics/{RUN_ID}"
    spec = {
        "schema_version": "v3_run_spec_v1",
        "slug": "primitive_relation_observable_c07_tf32_evidence_corrective_v1",
        "date": "20260902", "seed": 0, "gate": 3, "execution_phase": 3,
        "operation": "audit", "working_directory": str(PROJECT_ROOT),
        "config_path": str(CARD.relative_to(PROJECT_ROOT)),
        "data_card": str(CARD.relative_to(PROJECT_ROOT)),
        "question": "Can the complete sealed TF32-off C07 scientific result be formally finalized without repeating 387864 model forward rows?",
        "method": "CPU-only verification of all source seal entries, numerical/result/resource/isolation contracts and byte-identical compact copy into one new immutable run.",
        "baseline": "The complete TF32 corrective V1R source run and its 28-file immutable seal; no new model comparison.",
        "fallback": card["failure_policy"], "user_authorization": card["approval"],
        "acceptance_criteria": list(card["metrics_and_pre_registered_gates"].values()),
        "expected_counts": {
            "source_seal_files": 28, "source_model_forward_rows": 387_864,
            "new_model_forward_rows": 0, "c07_parent_worlds": 10,
            "c07_tasks": 30, "c07_rows_per_seed": 64_644, "frozen_seeds": 3,
            "optimizer_steps": 0, "c08_rows_read": 0, "c09_c10_worlds_read": 0,
            "graph_replays": 0, "mtare_worlds_read": 0,
        },
        "estimated_cost": card["estimated_cost"],
        "expected_evidence": [
            "Exact 28-file source seal and post-seal wrapper-defect verification.",
            "Three complete 64644-row TF32-off seed results and valid 0-of-3 scientific stop decision.",
            "Upstream inputs/tools unchanged, resources and isolation pass, byte-identical compact copies.",
            "Source provenance, tests, logs, RUN_STATE, figures and new SHA-256 seal.",
        ],
        "command": [
            "/usr/bin/timeout", "--signal=INT", "--kill-after=30s", "600s",
            "/usr/bin/env", f"PYTHONPATH={PROJECT_ROOT / 'src'}:{PROJECT_ROOT / 'tools/v3'}:{PROJECT_ROOT}",
            "/home/zeng-workstation/.local/share/mtare_topo_comm/envs/phase3_torch290_cu129_zarr2187_v1/bin/python",
            "tools/v3/run_primitive_relation_observable_c07_tf32_evidence_corrective_v1.py",
            "--spec", str(SPEC), "--run-dir", str(run_dir),
        ],
    }
    inputs = [CARD, SOURCE_CARD, SOURCE_SPEC, *source_sealed_files()]
    spec["frozen_inputs"] = {
        str(path.relative_to(PROJECT_ROOT)): sha(path) for path in inputs
    }
    tools = {
        "runner": "tools/v3/run_primitive_relation_observable_c07_tf32_evidence_corrective_v1.py",
        "freezer": "tools/v3/freeze_primitive_relation_observable_c07_tf32_evidence_corrective_v1_spec.py",
        "evidence_tests": "tests/v3/unit/test_primitive_relation_observable_c07_tf32_evidence_corrective_v1.py",
        "source_runner": "tools/v3/run_primitive_relation_observable_c07_tf32_corrective_v1.py",
        "corrective_evaluator": "tools/v3/evaluate_primitive_relation_observable_three_seed_c07_tf32_corrective_v1.py",
        "corrective_tests": "tests/v3/unit/test_evaluate_primitive_relation_observable_three_seed_c07_tf32_corrective_v1.py",
        "base_evaluator_tests": "tests/v3/unit/test_evaluate_primitive_relation_observable_three_seed_v1.py",
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
