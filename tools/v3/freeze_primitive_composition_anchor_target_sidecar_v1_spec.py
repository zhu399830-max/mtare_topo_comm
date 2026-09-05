#!/usr/bin/env python3
"""Freeze the exact fit/C07 composition-anchor target sidecar card/spec."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

from _bootstrap import PROJECT_ROOT


SOURCE_CARD = PROJECT_ROOT / "configs/v3/gate3/data_cards/primitive_attachment_observability_sidecar_v1.json"
CARD = PROJECT_ROOT / "configs/v3/gate3/data_cards/primitive_composition_anchor_target_sidecar_v1r.json"
SPEC = PROJECT_ROOT / "configs/v3/gate3/primitive_composition_anchor_target_sidecar_v1r.json"
RUN_ID = "gate3_20260903_primitive_composition_anchor_target_sidecar_v1r_seed0"
P1A = PROJECT_ROOT / "results/gate3_semantics/gate3_20260830_primitive_relation_p1a_lossless_storage_corrective_v1_seed0"
P1B = PROJECT_ROOT / "results/gate3_semantics/gate3_20260830_primitive_relation_p1b_teacher_materialization_v1r_seed0"
OBSERVABILITY = PROJECT_ROOT / "results/gate3_semantics/gate3_20260902_primitive_attachment_observability_sidecar_v1_seed0"
ANCHOR_TEACHER = PROJECT_ROOT / "results/gate3_semantics/gate3_20260903_primitive_composition_anchor_teacher_readiness_v1_seed0"
MODEL_READINESS = PROJECT_ROOT / "results/gate3_semantics/gate3_20260903_primitive_composition_anchor_model_readiness_v2_seed0"
FAILED_V1 = PROJECT_ROOT / "results/gate3_semantics/gate3_20260903_primitive_composition_anchor_target_sidecar_v1_seed0"


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
        raise RuntimeError("composition-anchor target sidecar card/spec already exists")
    source = json.loads(SOURCE_CARD.read_text(encoding="utf-8"))
    teacher_summary = json.loads(
        (ANCHOR_TEACHER / "metrics/teacher_readiness/summary.json").read_text(encoding="utf-8")
    )
    model_summary = json.loads((MODEL_READINESS / "metrics/summary.json").read_text(encoding="utf-8"))
    observability_summary = json.loads((OBSERVABILITY / "metrics/summary.json").read_text(encoding="utf-8"))
    if (
        teacher_summary.get("scientific_pass") is not True
        or model_summary.get("scientific_pass") is not True
        or observability_summary.get("scientific_pass") is not True
    ):
        raise RuntimeError("composition-anchor target sidecar source decision drift")
    fit_pairs = int(observability_summary["split"]["fit"]["positive_attachment_pairs"])
    c07_pairs = int(observability_summary["split"]["c07"]["positive_attachment_pairs"])
    if (fit_pairs, c07_pairs) != (3_301_484, 525_077):
        raise RuntimeError("composition-anchor target pair population drift")

    card = copy.deepcopy(source)
    card["card_id"] = "primitive_composition_anchor_target_sidecar_v1r"
    card["status"] = "APPROVED_FOR_ONE_IMMUTABLE_PRIMITIVE_COMPOSITION_ANCHOR_TARGET_SIDECAR_V1R"
    card["purpose"] = (
        "Materialize exact construction-program endpoint composition anchors in each five-frame "
        "sequence's current sensor frame for head-only O(E) relation learning."
    )
    card["approval"] = {
        "status": "APPROVED",
        "approved_by": "user-standing-authorization",
        "approved_at": "2026-09-03T14:00:00+08:00",
        "authorized_gates": [3],
        "authorized_operations": ["teacher_generation"],
        "scope": (
            "One immutable C01--C07 composition-anchor target sidecar for 210 geometry tasks "
            "and 491196 sequences; no LiDAR duplication, model update, C08+, graph or M-TARE."
        ),
        "confirmation_reference": (
            "The user authorized uninterrupted best-in-plan execution. The sealed sensor-polar "
            "O(E) model readiness explicitly allowed this exact target-sidecar step."
        ),
    }
    card["sampling"] = {
        "effective_sample_count": 491_196,
        "effective_structure_event_count": fit_pairs + c07_pairs,
        "independent_sampling_units": (
            "70 program-generated topology parents; three geometry realizations are paired "
            "repeated measures within each parent."
        ),
        "raw_frame_count": 659_940,
        "spatial_interval_m": 1.0,
        "rule": (
            "For every immutable P1b fit/C07 sequence, gather the two construction anchors of "
            "each active primitive in the exact P1b slot order, transform them using only the "
            "current sensor xyz/yaw, store float32 [32,2,3], and write exact zeros for inactive slots."
        ),
        "structure_event_counts": {
            "parent_worlds": 70,
            "geometry_tasks": 210,
            "fit_tasks": 180,
            "c07_tasks": 30,
            "sequences": 491_196,
            "fit_sequences": 426_552,
            "c07_sequences": 64_644,
            "fit_attachment_anchor_pairs": fit_pairs,
            "c07_attachment_anchor_pairs": c07_pairs,
            "c08_rows_read": 0,
            "c09_c10_worlds_read": 0,
            "model_inference_rows": 0,
            "optimizer_steps": 0,
        },
    }
    card["teacher"] = {
        "source": (
            "Sealed P1a base_construction endpoint composition_anchor_xyz_m, exact P1b "
            "primitive_index/primitive_mask/frame_row/source sequence order, and sealed P1a "
            "current sensor xyz/yaw."
        ),
        "coordinate_contract": (
            "World anchors are translated by current sensor_xyz_m and rotated by negative current "
            "yaw into current-sensor XYZ. Pitch/roll are absent from the frozen sensor contract."
        ),
        "identity_contract": (
            "Node and primitive identities are used only offline to validate construction ordering. "
            "The sidecar contains only numeric anchor targets and source sequence indices."
        ),
        "valid_mask": (
            "Only endpoint anchors belonging to P1b primitive_mask=1 slots enter regression or "
            "relation losses. Inactive slots are stored as exact zero and remain loss-masked."
        ),
        "student_forbidden_inputs": (
            "Construction node/primitive identity, world coordinates, sensor world pose, TNG, "
            "future frames, C08+ and topology labels never enter the student forward pass."
        ),
        "planner_consistency_plan": (
            "This sidecar authorizes head-only relation learning only. Graph edges will remain "
            "traversal-verified and cannot be created from Teacher identity."
        ),
    }
    card["leakage_audit"]["optimizer_step_count"] = 0
    card["leakage_audit"]["model_inference_count"] = 0
    card["trajectories"] = [
        row for row in card["trajectories"] if row.get("split") in {"train", "validation"}
        and "C08" not in str(row.get("id", ""))
    ]
    card["metrics_and_pre_registered_gates"] = {
        "population": (
            "Exactly 60 fit parents/180 tasks/426552 rows and 10 C07 parents/30 tasks/64644 "
            "rows; total 70/210/491196; C08+ zero."
        ),
        "alignment": (
            "Every source_global_sequence_index equals both P1b and endpoint-observability "
            "sidecars; primitive slot order is validated against each construction document."
        ),
        "geometry": (
            f"All {fit_pairs + c07_pairs} true attachment pairs receive bit-exact shared anchors; "
            "active targets are finite and inactive slots are exact zero."
        ),
        "precision": (
            "Float32 materialization differs from a float64 re-derivation by at most 0.0001 m; "
            "every shard is reopened and fully rederived exactly."
        ),
        "integrity": (
            "All 210 sidecars bind construction, P1b and observability hashes and receive array "
            "and tree SHA-256 evidence; 22 unit/model/Teacher tests pass."
        ),
        "resources": (
            "CPU wall time <=2 h, host RSS <=4 GiB, output <=1 GiB, zero GPU/model inference/"
            "optimizer/checkpoint/C08+/graph/M-TARE."
        ),
        "decision": (
            "PASS allows one head-only three-seed training Data Card/spec; it is not learned "
            "performance and does not unlock C08 or graph replay."
        ),
    }
    card["estimated_cost"] = {
        "compute": "Serial CPU/Zarr transform, compression, reopen and full re-derivation; no GPU.",
        "gpu": 0,
        "host_ram_gb": 4,
        "wall_time_hours": 2,
        "disk_gb": 1,
    }
    card["failure_policy"] = (
        "Any source hash, task/row/order, construction membership, transform, inactive-zero, "
        "shared-anchor, deterministic replay or resource drift fails closed; do not read C08+ "
        "or start training."
    )
    card["corrective_scope"] = (
        "V1 failed only because a paper-figure histogram treated anchor norm >=80 m as invalid. "
        "V1R preserves every numeric target and all scientific gates, records the true maximum, "
            "and counts values beyond the 80 m display range in an explicit overflow bin."
    )
    card["retention"] = (
        "Retain compact per-task targets, task manifest, summary, source hashes, environment, "
        "logs, paper-ready PNG/PDF/SVG, RUN_STATE and SHA-256 seal."
    )
    _write(CARD, card)

    run_dir = PROJECT_ROOT / f"results/gate3_semantics/{RUN_ID}"
    expected = {
        "geometry_tasks": 210,
        "sequences": 491_196,
        "fit_sequences": 426_552,
        "c07_sequences": 64_644,
        "fit_attachment_pairs": fit_pairs,
        "c07_attachment_pairs": c07_pairs,
        "unit_tests": 22,
        "optimizer_steps": 0,
        "model_inference_rows": 0,
        "c08_rows_read": 0,
        "c09_c10_worlds_read": 0,
        "graph_replays": 0,
        "mtare_worlds_read": 0,
    }
    spec = {
        "schema_version": "v3_run_spec_v1",
        "slug": "primitive_composition_anchor_target_sidecar_v1r",
        "date": "20260903",
        "seed": 0,
        "gate": 3,
        "execution_phase": 3,
        "operation": "teacher_generation",
        "working_directory": str(PROJECT_ROOT),
        "config_path": str(CARD.relative_to(PROJECT_ROOT)),
        "data_card": str(CARD.relative_to(PROJECT_ROOT)),
        "question": (
            "Can every fit/C07 construction anchor be materialized in exact P1b slot and current-"
            "sensor order without identity leakage or storage ambiguity before head-only training?"
        ),
        "method": (
            "Gather one shared construction anchor per endpoint in P1b slot order, apply the frozen "
            "current-sensor SE(2)+z transform, store float32 [32,2,3], and fully rederive every shard."
        ),
        "baseline": (
            "On-the-fly world-coordinate Teacher gathering used only in readiness, which is correct "
            "for one row but is not an immutable, hashed, training-aligned 491196-row artifact."
        ),
        "fallback": (
            "Any failure stops before training. Do not modify P1a/P1b, substitute predicted anchors, "
            "drop rows, read C08+, or use construction identities in the model."
        ),
        "user_authorization": card["approval"],
        "acceptance_criteria": list(card["metrics_and_pre_registered_gates"].values()),
        "expected_counts": expected,
        "estimated_cost": card["estimated_cost"],
        "expected_evidence": [
            "210 compressed sidecars with [N,32,2,3] anchors and sequence identity.",
            "Exact shared-anchor, transform, inactive-zero, full-rederive and source-hash audit.",
            "Task manifest, summary, 22 tests, environment, logs, figures, RUN_STATE and seal.",
        ],
        "command": [
            "/usr/bin/systemd-inhibit", "--what=sleep:shutdown",
            "--why=Primitive composition anchor target sidecar", "--mode=block",
            "/usr/bin/timeout", "--signal=INT", "--kill-after=60s", "7200s",
            "/usr/bin/env",
            f"PYTHONPATH={PROJECT_ROOT / 'src'}:{PROJECT_ROOT / 'tools/v3'}",
            "/home/zeng-workstation/.local/share/mtare_topo_comm/envs/phase3_torch290_cu129_zarr2187_v1/bin/python",
            "tools/v3/run_primitive_composition_anchor_target_sidecar_v1.py",
            "--spec", str(SPEC), "--run-dir", str(run_dir),
        ],
    }
    inputs = [
        CARD,
        *[
            run / name for run in (
                P1A, P1B, OBSERVABILITY, ANCHOR_TEACHER, MODEL_READINESS, FAILED_V1,
            )
            for name in ("RUN_STATE.json", "metrics/summary.json", "artifacts/evidence_sha256.txt")
            if (run / name).exists()
        ],
        P1A / "artifacts/task_manifest.json",
        P1B / "artifacts/task_manifest.json",
        OBSERVABILITY / "artifacts/task_manifest.json",
        ANCHOR_TEACHER / "metrics/teacher_readiness/summary.json",
    ]
    spec["frozen_inputs"] = {
        str(path.relative_to(PROJECT_ROOT)): _sha(path) for path in inputs
    }
    tools = {
        "sidecar_module": "src/mtare_topo/data/primitive_composition_anchor_sidecar.py",
        "sidecar_tests": "tests/v3/unit/test_primitive_composition_anchor_sidecar.py",
        "teacher_module": "src/mtare_topo/evaluation/primitive_composition_anchor_teacher.py",
        "teacher_tests": "tests/v3/unit/test_primitive_composition_anchor_teacher.py",
        "model_tests": "tests/v3/unit/test_primitive_composition_anchor_model.py",
        "executor": "tools/v3/execute_primitive_composition_anchor_target_sidecar_v1.py",
        "runner": "tools/v3/run_primitive_composition_anchor_target_sidecar_v1.py",
        "freezer": "tools/v3/freeze_primitive_composition_anchor_target_sidecar_v1_spec.py",
        "governance": "src/mtare_topo/governance.py",
        "preflight": "tools/v3/preflight.py",
        "create_run": "tools/v3/create_run.py",
    }
    spec["frozen_tools"] = {
        name: {"path": path, "sha256": _sha(PROJECT_ROOT / path)}
        for name, path in tools.items()
    }
    _write(SPEC, spec)
    print(SPEC)


if __name__ == "__main__":
    main()
