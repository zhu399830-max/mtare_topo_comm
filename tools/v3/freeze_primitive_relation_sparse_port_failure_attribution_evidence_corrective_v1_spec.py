#!/usr/bin/env python3
"""Freeze the zero-inference evidence completion corrective."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

from _bootstrap import PROJECT_ROOT


SOURCE_CARD = PROJECT_ROOT / "configs/v3/gate3/data_cards/primitive_relation_sparse_port_failure_attribution_v1r.json"
SOURCE_SPEC = PROJECT_ROOT / "configs/v3/gate3/primitive_relation_sparse_port_failure_attribution_v1r.json"
SOURCE = PROJECT_ROOT / "results/gate3_semantics/gate3_20260902_primitive_relation_sparse_port_failure_attribution_v1r_seed0"
CARD = PROJECT_ROOT / "configs/v3/gate3/data_cards/primitive_relation_sparse_port_failure_attribution_evidence_corrective_v1.json"
SPEC = PROJECT_ROOT / "configs/v3/gate3/primitive_relation_sparse_port_failure_attribution_evidence_corrective_v1.json"
RUN_ID = "gate3_20260902_primitive_relation_sparse_port_failure_attribution_evidence_corrective_v1_seed0"


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
        raise RuntimeError("attribution evidence corrective card/spec already exists")
    state = json.loads((SOURCE / "RUN_STATE.json").read_text())
    outer = json.loads((SOURCE / "metrics/summary.json").read_text())
    attribution = json.loads((SOURCE / "metrics/attribution/summary.json").read_text())
    if state.get("state") != "FAILED" or state.get("error") != "NameError: name 'SEEDS' is not defined":
        raise RuntimeError("evidence corrective requires the sealed post-compute NameError")
    if outer.get("model_inference_rows") != 193_932 or attribution.get("scientific_pass") is not True:
        raise RuntimeError("source attribution did not complete full scientific evidence")
    if attribution.get("diagnosis") != "RELATION_SCORE_FAILS_EVEN_WITH_PROPOSAL_ORACLE":
        raise RuntimeError("source attribution diagnosis drift")

    card = copy.deepcopy(json.loads(SOURCE_CARD.read_text()))
    card["card_id"] = "primitive_relation_sparse_port_failure_attribution_evidence_corrective_v1"
    card["status"] = "APPROVED_FOR_ONE_IMMUTABLE_PRIMITIVE_RELATION_SPARSE_PORT_FAILURE_ATTRIBUTION_EVIDENCE_CORRECTIVE_V1"
    card["purpose"] = (
        "Complete evidence validation and sealing for the already-computed V1R attribution after its outer runner raised an undefined-SEEDS NameError only while constructing the required-file list. No model or data inference is repeated."
    )
    card["approval"] = {
        "status": "APPROVED", "approved_by": "user-standing-authorization",
        "approved_at": "2026-09-02T03:20:00+08:00", "authorized_gates": [3],
        "authorized_operations": ["audit"],
        "scope": (
            "One immutable zero-inference audit of the sealed V1R output: validate 23 source seal entries, 193932 completed source inference rows, three seed JSON files, full diagnostics, resources, isolation and unchanged upstream inputs; copy only compact evidence and figures into a new sealed run."
        ),
        "confirmation_reference": (
            "The user delegated automatic strongest-evidence execution. Full attribution finished before a post-compute required-file NameError, so read-only evidence completion is stronger and cheaper than repeating identical inference."
        ),
    }
    card["source"]["raw_sources"] = [
        "sealed sparse-port failure-attribution V1R run with 193932 completed C07 inference rows",
        "sealed V1R attribution summary, three seed outputs, per-task table and three-format figure",
        "V1R frozen run spec and all upstream frozen-input hashes",
    ]
    card["sampling"]["rule"] = (
        "No sensor, Teacher or checkpoint loader is constructed. Read only the sealed V1R evidence files; verify every source seal entry, all nine attribution outputs, resource/isolation fields and every upstream frozen-input hash; copy compact evidence byte-for-byte."
    )
    counts = card["sampling"]["structure_event_counts"]
    counts["source_attribution_inference_rows"] = 193_932
    counts["new_attribution_inference_rows"] = 0
    counts["model_inference_rows"] = 0
    card["leakage_audit"]["model_inference_count"] = 0
    card["metrics_and_pre_registered_gates"] = {
        "implementation": "Twenty frozen tests pass, including formal batch rank and required three-seed file enumeration.",
        "source_failure_boundary": "Source V1R must be FAILED only with NameError undefined SEEDS after attribution summary reports PASS and 193932 completed inference rows.",
        "source_evidence": "All 23 source seal entries, three seed JSON files, per-task table, summary/source and PNG/PDF/SVG must exist, be nonempty and hash-valid.",
        "scientific_diagnosis": "All seven diagnostic conditions have zero passing seeds; diagnosis and decision exactly equal RELATION_SCORE_FAILS_EVEN_WITH_PROPOSAL_ORACLE and STOP_CURRENT_RELATION_HEAD_AND_REASSESS_METHOD.",
        "resources": "Source host RSS and CUDA reserved remain <=16GiB; corrective wall time <=10min, output <=20MiB, and zero new model inference/optimizer/C08/C09/C10/graph/M-TARE.",
        "integrity": "Every upstream input frozen by V1R is unchanged; all copied compact outputs are byte-identical; new evidence receives its own immutable SHA-256 seal.",
    }
    card["split"]["selection"] = "No selection or thresholding occurs; this run validates already-computed sealed diagnostic evidence only."
    card["teacher"]["valid_mask"] = "Teacher is not loaded. Existing sealed attribution labels are read only as evidence files."
    card["teacher"]["planner_consistency_plan"] = "No graph/planner work; preserve the source decision to stop the current relation head."
    card["estimated_cost"] = {
        "compute": "CPU-only hash and evidence audit; zero model inference.",
        "gpu": 0, "gpu_memory_gb": 0, "host_ram_gb": 4,
        "wall_time_hours": 0.1, "disk_gb": 0.02,
    }
    card["failure_policy"] = (
        "Any source seal, file, population, diagnosis, resource, isolation, upstream hash or byte-copy mismatch fails closed. Do not rerun inference or repair results inside the corrective."
    )
    card["retention"] = (
        "Retain source provenance, byte-identical attribution summary/seed/task/figures, tests, integrity snapshots, environment, RUN_STATE and new seal. Preserve both failed source runs unchanged."
    )
    write(CARD, card)

    run_dir = PROJECT_ROOT / f"results/gate3_semantics/{RUN_ID}"
    spec = {
        "schema_version": "v3_run_spec_v1", "slug": "primitive_relation_sparse_port_failure_attribution_evidence_corrective_v1",
        "date": "20260902", "seed": 0, "gate": 3, "execution_phase": 3,
        "operation": "audit", "working_directory": str(PROJECT_ROOT),
        "config_path": str(CARD.relative_to(PROJECT_ROOT)), "data_card": str(CARD.relative_to(PROJECT_ROOT)),
        "question": "Can the fully computed but post-compute-failed V1R attribution be validated and sealed without repeating inference?",
        "method": "CPU-only verification of the complete sealed V1R evidence, upstream hashes, resource/isolation fields and byte-identical compact copy into one new immutable run.",
        "baseline": "The source V1R run and its 23-file immutable seal; no model comparison or new metric computation.",
        "fallback": card["failure_policy"], "user_authorization": card["approval"],
        "acceptance_criteria": list(card["metrics_and_pre_registered_gates"].values()),
        "expected_counts": {
            "unit_tests": 20, "source_seal_files": 23,
            "source_model_inference_rows": 193_932, "new_model_inference_rows": 0,
            "frozen_seeds": 3, "optimizer_steps": 0, "c08_rows_read": 0,
            "c09_c10_worlds_read": 0, "graph_replays": 0, "mtare_worlds_read": 0,
        },
        "estimated_cost": card["estimated_cost"],
        "expected_evidence": [
            "Exact source failure-boundary and 23-file seal verification.",
            "All three seed diagnostics, per-task table and three-format figure byte-identical.",
            "Upstream frozen inputs unchanged and resource/isolation contracts pass.",
            "Source provenance, logs, RUN_STATE and new SHA-256 seal.",
        ],
        "command": [
            "/usr/bin/timeout", "--signal=INT", "--kill-after=30s", "600s",
            "/usr/bin/env", f"PYTHONPATH={PROJECT_ROOT / 'src'}:{PROJECT_ROOT / 'tools/v3'}",
            "/home/zeng-workstation/.local/share/mtare_topo_comm/envs/phase3_torch290_cu129_zarr2187_v1/bin/python",
            "tools/v3/run_primitive_relation_sparse_port_failure_attribution_evidence_corrective_v1.py",
            "--spec", str(SPEC), "--run-dir", str(run_dir),
        ],
    }
    source_names = [
        "RUN_STATE.json", "metrics/summary.json", "artifacts/evidence_sha256.txt",
        "logs/01_attribution.log", "logs/failure_traceback.log",
        "metrics/attribution/summary.json", "metrics/attribution/per_task.csv",
        "metrics/attribution/figure_source.json",
        *[f"metrics/attribution/seed{seed}.json" for seed in range(3)],
        *[f"metrics/attribution/primitive_relation_sparse_port_failure_attribution.{suffix}" for suffix in ("png", "pdf", "svg")],
    ]
    inputs = [CARD, SOURCE_CARD, SOURCE_SPEC, *[SOURCE / name for name in source_names]]
    spec["frozen_inputs"] = {str(path.relative_to(PROJECT_ROOT)): sha(path) for path in inputs}
    tools = {
        "runner": "tools/v3/run_primitive_relation_sparse_port_failure_attribution_evidence_corrective_v1.py",
        "freezer": "tools/v3/freeze_primitive_relation_sparse_port_failure_attribution_evidence_corrective_v1_spec.py",
        "corrected_source_runner": "tools/v3/run_primitive_relation_sparse_port_failure_attribution_v1.py",
        "attribution_module": "src/mtare_topo/evaluation/primitive_relation_sparse_port_failure_attribution.py",
        "attribution_tests": "tests/v3/unit/test_primitive_relation_sparse_port_failure_attribution.py",
        "legacy_attribution_tests": "tests/v3/unit/test_primitive_relation_failure_attribution.py",
        "metric_tests": "tests/v3/unit/test_primitive_relation_metrics.py",
        "sparse_evaluation_tests": "tests/v3/unit/test_evaluate_primitive_relation_sparse_port_three_seed_v1.py",
        "governance": "src/mtare_topo/governance.py", "preflight": "tools/v3/preflight.py",
        "create_run": "tools/v3/create_run.py",
    }
    spec["frozen_tools"] = {
        name: {"path": path, "sha256": sha(PROJECT_ROOT / path)} for name, path in tools.items()
    }
    write(SPEC, spec)
    print(SPEC)


if __name__ == "__main__":
    main()
