#!/usr/bin/env python3
"""Freeze the C01--C07-only P1b attachment observability audit."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

from _bootstrap import PROJECT_ROOT


SOURCE_CARD = PROJECT_ROOT / "configs/v3/gate3/data_cards/primitive_relation_sparse_port_three_seed_training_v1r.json"
CARD = PROJECT_ROOT / "configs/v3/gate3/data_cards/primitive_attachment_teacher_observability_v1.json"
SPEC = PROJECT_ROOT / "configs/v3/gate3/primitive_attachment_teacher_observability_v1.json"
RUN_ID = "gate3_20260902_primitive_attachment_teacher_observability_v1_seed0"
P1A = PROJECT_ROOT / "results/gate3_semantics/gate3_20260830_primitive_relation_p1a_lossless_storage_corrective_v1_seed0"
P1B = PROJECT_ROOT / "results/gate3_semantics/gate3_20260830_primitive_relation_p1b_teacher_materialization_v1r_seed0"
ATTRIBUTION = PROJECT_ROOT / "results/gate3_semantics/gate3_20260902_primitive_relation_sparse_port_failure_attribution_evidence_corrective_v1_seed0"


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
        raise RuntimeError("attachment observability card/spec already exists")
    for run in (P1A, P1B, ATTRIBUTION):
        state = json.loads((run / "RUN_STATE.json").read_text())
        if state.get("state") != "COMPLETED" or state.get("error") is not None:
            raise RuntimeError(f"required source run is not complete: {run.name}")
    attribution = json.loads((ATTRIBUTION / "metrics/summary.json").read_text())
    if attribution.get("diagnosis") != "RELATION_SCORE_FAILS_EVEN_WITH_PROPOSAL_ORACLE":
        raise RuntimeError("source relation diagnosis drift")
    manifest = json.loads((P1B / "artifacts/task_manifest.json").read_text())["tasks"]
    rows = [row for row in manifest if row["partition"] in {"fit", "c07"}]
    counts = {
        "geometry_tasks": len(rows),
        "parent_worlds": len({row["world"] for row in rows}),
        "raw_frames": sum(int(row["frames"]) for row in rows),
        "sequences": sum(int(row["sequences"]) for row in rows),
        "fit_sequences": sum(int(row["sequences"]) for row in rows if row["partition"] == "fit"),
        "c07_sequences": sum(int(row["sequences"]) for row in rows if row["partition"] == "c07"),
        "undirected_attachment_labels": sum(int(row["directed_attachment_labels"]) for row in rows) // 2,
    }
    expected = {
        "geometry_tasks": 210, "parent_worlds": 70, "raw_frames": 659_940,
        "sequences": 491_196, "fit_sequences": 426_552, "c07_sequences": 64_644,
        "undirected_attachment_labels": 3_826_561,
    }
    if counts != expected:
        raise RuntimeError(f"attachment observability population drift: {counts}")

    card = copy.deepcopy(json.loads(SOURCE_CARD.read_text()))
    card["card_id"] = "primitive_attachment_teacher_observability_v1"
    card["status"] = "APPROVED_FOR_ONE_IMMUTABLE_PRIMITIVE_ATTACHMENT_TEACHER_OBSERVABILITY_V1"
    card["purpose"] = (
        "Determine whether every P1b physical endpoint-attachment label has direct five-frame LiDAR support at the labelled endpoints, or whether the failed relation head was supervised with hidden construction connectivity."
    )
    card["approval"] = {
        "status": "APPROVED", "approved_by": "user-standing-authorization",
        "approved_at": "2026-09-02T14:00:00+08:00", "authorized_gates": [3],
        "authorized_operations": ["audit"],
        "scope": "One immutable zero-training C01--C07 endpoint-support audit over 70 development parents, 210 paired geometry tasks and 491196 five-frame sequences; no C08/C09/C10, model inference, graph or M-TARE.",
        "confirmation_reference": "The user explicitly authorized automatic selection of the strongest in-plan action and requested continuous execution. The sealed C07 oracle attribution requires Teacher observability reassessment before another relation model.",
    }
    card["source"]["raw_sources"] = [
        "sealed P1a corrected C01--C07 sensor pose and construction-program primitive endpoints",
        "sealed P1b C01--C07 five-frame cropped primitive axes and endpoint-neighbor labels",
        "sealed V2 relation failure attribution establishing zero safe true attachment for all three seeds",
    ]
    card["sampling"] = {
        "raw_frame_count": counts["raw_frames"],
        "effective_sample_count": counts["sequences"],
        "effective_structure_event_count": counts["undirected_attachment_labels"],
        "spatial_interval_m": 1.0,
        "independent_sampling_units": "70 program-generated topology parents; three geometry realizations are paired repeated measures within each parent.",
        "rule": "Read each C01--C07 P1b five-frame sequence once. For every unique positive endpoint pair, measure the distance from each true realized endpoint to the nearest observed cropped-axis endpoint. Report all fixed 0.10/0.25/0.50/1.00/2.00 m bands; 0.25 m is the pre-registered primary band inherited from the Teacher LOS margin.",
        "structure_event_counts": {
            **counts, "model_inference_rows": 0, "optimizer_steps": 0,
            "c08_rows_read": 0, "c09_c10_worlds_read": 0,
        },
    }
    card["teacher"] = {
        "source": "P1b construction-program endpoint attachment after five-frame primitive visibility aggregation.",
        "valid_mask": "The audit tests the missing mask itself: both labelled endpoints must lie within 0.25 m of actual five-frame ray-supported cropped axes. Construction identity is offline audit metadata and never a student input.",
        "planner_consistency_plan": "If window-level labels fail but >=95% of physical connection identities have at least one supported window in both fit and C07, mask unsupported windows and learn only local evidence; a topological edge is still committed only after real traversal. Otherwise redefine the relation target as provisional evidence plus traversal commit.",
        "student_forbidden_inputs": "No model is executed. Future students remain forbidden from construction/TNG identity, absolute pose, future scans and test worlds.",
    }
    card["split"]["fit"] = "All C01--C06 rows are audited without fitting a model or threshold."
    card["split"]["selection"] = "C07 is reported separately; it does not tune the fixed support bands or the 0.99/0.95 decision thresholds."
    card["split"]["development_transfer"] = "C08 remains forbidden until a future relation model passes C07."
    card["split"]["strict_test"] = "C08/C09/C10 and all M-TARE benchmark worlds remain unread."
    card["split"]["historical_pollution_audit"] = "Only C01--C07 populations already exposed to development are used to diagnose the failed Teacher/model contract. No transfer or test claim is made."
    card["metrics_and_pre_registered_gates"] = {
        "population": "Exactly 70 parents, 210 geometry tasks, 659940 source frames, 491196 sequences and 3826561 unique positive endpoint-pair labels; zero C08+.",
        "measurement": "For each label, report maximum of the two endpoint support gaps, observed endpoint separation, both/one/neither support at all five fixed bands, and physical connection identity coverage.",
        "teacher_valid": "Current Teacher is endpoint-observable only if both-endpoint fraction and physical-identity coverage are each >=0.99 in both fit and C07 at 0.25 m.",
        "correctable": "If current Teacher fails but physical-identity coverage is >=0.95 in both splits, freeze the diagnosis WINDOW_LEVEL_HIDDEN_ENDPOINT_SUPERVISION_CORRECTABLE_BY_OBSERVABILITY_MASK.",
        "fundamental": "If either split has physical-identity coverage <0.95, freeze PHYSICAL_ATTACHMENT_IDENTITY_NOT_RELIABLY_OBSERVABLE_IN_FIVE_FRAMES and replace direct attachment classification with provisional evidence plus traversal commit.",
        "isolation": "Five unit tests pass; no model inference, optimizer, checkpoint, C08/C09/C10, graph or M-TARE read/write.",
        "resources": "CPU wall time <=2 h, host RSS <=16 GiB, output <=0.2 GiB and zero GPU requirement.",
    }
    card["estimated_cost"] = {
        "compute": "Serial CPU/Zarr read-only audit over 210 already-materialized shards; no GPU.",
        "gpu": 0, "host_ram_gb": 16, "wall_time_hours": 1.0, "disk_gb": 0.2,
    }
    card["failure_policy"] = "Any population, shard, input/tool, support transform, symmetry, resource or isolation drift fails closed. Do not change the support bands, decision thresholds, data, Teacher or model inside this run."
    card["retention"] = "Retain split/task summaries, Teacher-contract evidence, CDF/support-band PNG/PDF/SVG, environment, logs, RUN_STATE and SHA-256 seal; do not retain per-label raw arrays."
    write(CARD, card)

    run_dir = PROJECT_ROOT / f"results/gate3_semantics/{RUN_ID}"
    spec = {
        "schema_version": "v3_run_spec_v1", "slug": "primitive_attachment_teacher_observability_v1",
        "date": "20260902", "seed": 0, "gate": 3, "execution_phase": 3,
        "operation": "audit", "working_directory": str(PROJECT_ROOT),
        "config_path": str(CARD.relative_to(PROJECT_ROOT)), "data_card": str(CARD.relative_to(PROJECT_ROOT)),
        "question": "Did P1b supervise physical endpoint attachments that the five causal LiDAR frames did not directly observe?",
        "method": "Compare each labelled realized endpoint with the nearest five-frame ray-supported cropped-axis endpoint, with fixed multi-band sensitivity and physical-identity coverage reported separately for C01--C06 and C07.",
        "baseline": "Current P1b rule: any two primitives visible anywhere in the five-frame aggregate are labelled attached when their hidden construction endpoints share a node, without an endpoint-support condition.",
        "fallback": "Mask unsupported window labels and learn observable local connection evidence if identities remain covered; otherwise use provisional evidence and let physical traversal alone commit graph edges.",
        "user_authorization": card["approval"],
        "acceptance_criteria": list(card["metrics_and_pre_registered_gates"].values()),
        "expected_counts": {**expected, "unit_tests": 5, "model_inference_rows": 0, "optimizer_steps": 0, "c08_rows_read": 0, "c09_c10_worlds_read": 0, "graph_replays": 0, "mtare_worlds_read": 0},
        "estimated_cost": card["estimated_cost"],
        "expected_evidence": [
            "Exact population and split isolation proof.",
            "Both/one/neither endpoint support at five fixed bands and maximum-gap quantiles.",
            "Physical attachment identity coverage and one pre-registered diagnosis.",
            "Per-task JSON, Teacher-contract evidence, PNG/PDF/SVG, logs, RUN_STATE and SHA-256 seal.",
        ],
        "command": [
            "/usr/bin/systemd-inhibit", "--what=sleep:shutdown",
            "--why=P1b attachment Teacher observability audit", "--mode=block",
            "/usr/bin/timeout", "--signal=INT", "--kill-after=60s", "7200s",
            "/usr/bin/env", f"PYTHONPATH={PROJECT_ROOT / 'src'}:{PROJECT_ROOT / 'tools/v3'}",
            "/tmp/mtare_gate4_meshing_sidecar_v1/bin/python",
            "tools/v3/run_primitive_attachment_teacher_observability_v1.py",
            "--spec", str(SPEC), "--run-dir", str(run_dir),
        ],
    }
    inputs = [
        CARD,
        *[run / name for run in (P1A, P1B, ATTRIBUTION) for name in (
            "RUN_STATE.json", "metrics/summary.json", "artifacts/evidence_sha256.txt",
        )],
        P1A / "artifacts/task_manifest.json", P1B / "artifacts/task_manifest.json",
    ]
    spec["frozen_inputs"] = {str(path.relative_to(PROJECT_ROOT)): sha(path) for path in inputs}
    tools = {
        "audit_module": "src/mtare_topo/evaluation/primitive_attachment_observability.py",
        "audit_tests": "tests/v3/unit/test_primitive_attachment_observability.py",
        "relation_targets": "src/mtare_topo/data/primitive_relation_targets.py",
        "relation_target_tests": "tests/v3/unit/test_primitive_relation_targets.py",
        "materialization": "src/mtare_topo/data/primitive_relation_materialization.py",
        "runner": "tools/v3/run_primitive_attachment_teacher_observability_v1.py",
        "freezer": "tools/v3/freeze_primitive_attachment_teacher_observability_v1_spec.py",
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
