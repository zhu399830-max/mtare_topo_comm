#!/usr/bin/env python3
"""Freeze the fit/C07 Primitive Connection Hypergraph Teacher audit."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

from _bootstrap import PROJECT_ROOT


SOURCE_CARD = PROJECT_ROOT / "configs/v3/gate3/data_cards/primitive_relation_p1b_teacher_materialization_v1r.json"
CARD = PROJECT_ROOT / "configs/v3/gate3/data_cards/primitive_connection_hypergraph_teacher_feasibility_v1.json"
SPEC = PROJECT_ROOT / "configs/v3/gate3/primitive_connection_hypergraph_teacher_feasibility_v1.json"
RUN_ID = "gate3_20260902_primitive_connection_hypergraph_teacher_feasibility_v1_seed0"
P1B = PROJECT_ROOT / "results/gate3_semantics/gate3_20260830_primitive_relation_p1b_teacher_materialization_v1r_seed0"
SIDECAR = PROJECT_ROOT / "results/gate3_semantics/gate3_20260902_primitive_attachment_observability_sidecar_v1_seed0"
ATTRIBUTION = PROJECT_ROOT / "results/gate3_semantics/gate3_20260902_primitive_relation_observable_failure_attribution_v1_seed0"


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tree_sha(path: Path) -> str:
    digest = hashlib.sha256()
    files = sorted(value for value in path.rglob("*") if value.is_file())
    for value in files:
        relative = value.relative_to(path).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(4, "little"))
        digest.update(relative)
        digest.update(bytes.fromhex(sha(value)))
    return digest.hexdigest()


def dataset_trees() -> dict[str, str]:
    result = {}
    roots = {
        "p1b": P1B / "artifacts/teacher",
        "sidecar": SIDECAR / "artifacts/endpoint_observability",
    }
    for source, root in roots.items():
        for split in ("fit", "c07"):
            for path in sorted(value for value in (root / split).glob("*.zarr") if value.is_dir()):
                result[f"{source}/{split}/{path.name}"] = tree_sha(path)
    return result


def write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    if CARD.exists() or SPEC.exists():
        raise RuntimeError("connection hypergraph Teacher card/spec already exists")
    for source, expected in (
        (P1B, "PASS_PRIMITIVE_RELATION_P1B_TEACHER_MATERIALIZATION_V1R"),
        (SIDECAR, "PASS_PRIMITIVE_ATTACHMENT_OBSERVABILITY_SIDECAR_V1"),
    ):
        state = json.loads((source / "RUN_STATE.json").read_text())
        if state.get("state") != "COMPLETED" or state.get("overall_status") != expected:
            raise RuntimeError(f"required sealed Teacher source is incomplete: {source.name}")
    attribution = json.loads((ATTRIBUTION / "metrics/summary.json").read_text())
    if (
        attribution.get("evidence_scientific_pass") is not True
        or attribution.get("diagnosis") != "RELATION_SCORE_FAILS_EVEN_WITH_OBSERVABLE_PROPOSAL_ORACLE"
        or attribution.get("decision") != "STOP_DIRECT_PAIR_RELATION_HEAD_AND_REASSESS_ARCHITECTURE"
    ):
        raise RuntimeError("direct-pair stop evidence is incomplete")

    card = copy.deepcopy(json.loads(SOURCE_CARD.read_text()))
    card["card_id"] = "primitive_connection_hypergraph_teacher_feasibility_v1"
    card["status"] = "APPROVED_FOR_ONE_IMMUTABLE_PRIMITIVE_CONNECTION_HYPERGRAPH_TEACHER_FEASIBILITY_V1"
    card["purpose"] = (
        "Before any replacement training, determine whether the frozen construction Teacher defines an unambiguous bounded union of endpoint connection clusters suitable for a learned permutation-invariant hyperedge decoder after the independent pair head failed even with proposal oracle."
    )
    card["approval"] = {
        "status": "APPROVED", "approved_by": "user-standing-authorization",
        "approved_at": "2026-09-02T22:25:00+08:00",
        "authorized_gates": [3], "authorized_operations": ["audit"],
        "scope": (
            "One immutable CPU-only Teacher feasibility audit over every C01-C06 fit and C07 selection row: 70 parents, 210 paired geometry shards and 491196 five-frame rows; zero sensor read, model inference, optimizer, C08/C09/C10, graph or M-TARE."
        ),
        "confirmation_reference": (
            "The user explicitly requested uninterrupted automatic execution toward the frozen paper plan. The sealed direct-pair oracle failure makes this zero-training Teacher proof the plan's strongest-evidence next step."
        ),
    }
    card["source"] = {
        "license_or_allowed_use": card["source"]["license_or_allowed_use"],
        "raw_sources": [
            "sealed P1b compact primitive_mask and endpoint_neighbor arrays for fit/C07 only",
            "sealed 0.25 m endpoint_observed sidecars for the same fit/C07 rows",
            "sealed direct-pair C07 oracle-failure decision as architecture-change trigger",
        ],
        "worlds": [
            "C01-C06 fit: 60 independent topology parents, three paired geometry realizations",
            "C07 selection: 10 independent topology parents, three paired geometry realizations",
        ],
    }
    card["sampling"] = {
        "independent_sampling_units": (
            "70 world-disjoint topology parents; each has three paired geometry realizations. Rows are five-frame causal sequences and are not counted as independent worlds."
        ),
        "raw_frame_count": 659_940,
        "effective_sample_count": 70,
        "effective_structure_event_count": 491_196,
        "raw_five_frame_sequence_count": 491_196,
        "spatial_interval_m": 1.0,
        "temporal_window_frames": 5,
        "maximum_primitive_slots": 32,
        "partition_sequences": {
            "fit_c01_c06": 426_552, "selection_c07": 64_644,
            "development_transfer_c08": 0,
        },
        "structure_event_counts": {
            "fit_parent_worlds": 60, "fit_geometry_shards": 180,
            "fit_rows": 426_552, "c07_parent_worlds": 10,
            "c07_geometry_shards": 30, "c07_rows": 64_644,
            "total_rows": 491_196,
        },
        "rule": (
            "Read every compact fit/C07 Teacher row once. Validate neighbor range, unique canonical storage, active cross-primitive destinations, symmetry and clique transitivity; induce the same clusters on dual-endpoint-observed masks. Select decoder cluster capacity only from fit maximum with a frozen 25% margin and candidates 4/8/16/32, then audit C07 overflow without adjustment."
        ),
    }
    card["split"] = {
        "fit": "C01-C06: 60 parents, 180 paired geometry shards, 426552 rows; sole source for capacity selection.",
        "selection": "C07: 10 parents, 30 paired geometry shards, 64644 untouched rows; audit frozen capacity with no adjustment.",
        "development_transfer": "C08 directory, manifest records and arrays are not opened by the executor.",
        "strict_test": "C09/C10 and all M-TARE benchmark worlds are forbidden.",
        "world_disjoint": True, "trajectory_disjoint": True,
        "historical_pollution_audit": (
            "The architecture change is triggered only by the sealed C07 proposal-oracle failure. No C08 result, model output, graph metric or planner outcome selects capacity or acceptance."
        ),
    }
    card["teacher"] = {
        "source": (
            "Exact program construction compositions. P1b stores up to three same-node neighbor endpoints per visible primitive endpoint; sidecar marks direct endpoint support at the frozen 0.25 m band."
        ),
        "labels": (
            "Physical endpoint attachment graph, induced observable attachment graph, connection-cluster membership, cluster size/count and singleton/hidden censoring state."
        ),
        "valid_mask": (
            "A physical connection is observable only when both endpoints have direct support. Hidden endpoints remain unknown, never negative; a lone observed member of a multi-endpoint physical cluster is reported as censored singleton and cannot commit an edge."
        ),
        "student_forbidden_inputs": (
            "No sensor frame, world/primitive/node identity, TNG, absolute pose, future frame, checkpoint or model output is read. Construction identity is used only to define the already-materialized Teacher relation."
        ),
        "planner_consistency_plan": (
            "No graph or planner runs. If Teacher feasibility passes, a separate readiness contract may implement learned endpoint-to-exchangeable-cluster assignment; execution-verified edge policy remains unchanged."
        ),
        "unexplored_port_policy": "Not part of this audit or perception Teacher.",
    }
    card["leakage_audit"] = {
        "absolute_pose_not_retained_in_student_representation": True,
        "future_sensor_frames_excluded": True,
        "model_inference_count": 0, "optimizer_step_count": 0,
        "mtare_benchmark_excluded": True,
        "test_excluded_from_supervised_training": True,
        "test_excluded_from_ssl": True,
        "test_excluded_from_normalization": True,
        "test_excluded_from_augmentation_tuning": True,
        "test_excluded_from_teacher_calibration": True,
        "test_excluded_from_threshold_calibration": True,
        "test_excluded_from_checkpoint_selection": True,
    }
    card["metrics_and_pre_registered_gates"] = {
        "population": "Exactly 60/10 parents, 180/30 paired shards and 426552/64644 fit/C07 rows; zero C08/C09/C10.",
        "integrity": "Every source/destination is active, cross-primitive, in range, unique, canonical, symmetric and unchanged before/after.",
        "cluster_semantics": "Every attachment component is a complete clique, hence each endpoint has exactly one or zero connection-cluster membership; induced observed relations remain clique-consistent.",
        "capacity": "Choose only from fit using ceil(1.25*fit maximum) and candidates 4/8/16/32; fit and untouched C07 must both have zero overflow.",
        "observability": "Reproduce exactly 442936 C07 dual-endpoint-observable undirected positives and report censored singleton/full-hidden cluster populations.",
        "complexity": "At frozen 32-primitive input contract, fixed endpoint-to-cluster decoder outputs must be fewer than the independent pair head; active and positive-label ratios are reported without overclaiming per-row compression.",
        "resources": "CPU only, wall time <=1h, host RSS <=4GiB, output <=0.1GiB and zero model/optimizer/graph/planner work.",
    }
    card["estimated_cost"] = {
        "compute": "CPU-only streaming read and SHA-256 audit of 355 MiB P1b plus 10.5 MiB sidecar fit/C07 shards.",
        "gpu": 0, "gpu_memory_gb": 0, "host_ram_gb": 4,
        "wall_time_hours": 0.25, "disk_gb": 0.1,
    }
    card["failure_policy"] = (
        "Any population, provenance, symmetry, clique, observability, capacity, compression, resource or isolation failure stops the hypergraph candidate. Do not repair labels, merge clusters, change the 25% margin/candidate set, read C08 or add geometric rules."
    )
    card["retention"] = (
        "Retain per-task/split cluster populations, frozen capacity, complexity statistics, PNG/PDF/SVG/source, environment, command, logs, RUN_STATE and SHA-256 seal; retain no copied Teacher rows."
    )
    write(CARD, card)

    run_dir = PROJECT_ROOT / f"results/gate3_semantics/{RUN_ID}"
    spec = {
        "schema_version": "v3_run_spec_v1",
        "slug": "primitive_connection_hypergraph_teacher_feasibility_v1",
        "date": "20260902", "seed": 0, "gate": 3, "execution_phase": 3,
        "operation": "audit", "working_directory": str(PROJECT_ROOT),
        "config_path": str(CARD.relative_to(PROJECT_ROOT)),
        "data_card": str(CARD.relative_to(PROJECT_ROOT)),
        "question": (
            "Does the frozen C01-C07 dual-endpoint-observable construction Teacher define an unambiguous bounded set of endpoint connection clusters suitable for a learned permutation-invariant hyperedge decoder?"
        ),
        "method": (
            "CPU-only exhaustive union-of-cliques and observability audit over all 491196 fit/C07 rows, followed by fit-only 25%-margin capacity freeze and untouched C07 overflow proof."
        ),
        "baseline": (
            "The sealed independent endpoint-pair relation head, whose proposal-oracle relation ranking still produced zero nonempty precision>=0.98 true connections in all three seeds."
        ),
        "fallback": card["failure_policy"],
        "user_authorization": card["approval"],
        "acceptance_criteria": list(card["metrics_and_pre_registered_gates"].values()),
        "expected_counts": {
            "fit_parent_worlds": 60, "fit_geometry_tasks": 180,
            "fit_rows": 426_552, "c07_parent_worlds": 10,
            "c07_geometry_tasks": 30, "c07_rows": 64_644,
            "total_rows": 491_196, "c07_observable_positive_pairs": 442_936,
            "model_forward_rows": 0, "optimizer_steps": 0,
            "c08_rows_read": 0, "c09_c10_worlds_read": 0,
            "graph_replays": 0, "mtare_worlds_read": 0,
        },
        "estimated_cost": card["estimated_cost"],
        "expected_evidence": [
            "Per-task and split union-of-cliques, symmetry, active/cross-primitive and observability proof.",
            "Fit-only frozen cluster capacity with fit/C07 overflow counts.",
            "Pair-head versus endpoint-cluster output complexity and censoring populations.",
            "PNG/PDF/SVG/source, environment, commands, logs, RUN_STATE and SHA-256 seal.",
        ],
        "command": [
            "/usr/bin/systemd-inhibit", "--what=sleep:shutdown",
            "--why=Primitive connection hypergraph Teacher feasibility", "--mode=block",
            "/usr/bin/timeout", "--signal=INT", "--kill-after=60s", "3600s",
            "/usr/bin/env",
            f"PYTHONPATH={PROJECT_ROOT / 'src'}:{PROJECT_ROOT / 'tools/v3'}:{PROJECT_ROOT}",
            "/home/zeng-workstation/.local/share/mtare_topo_comm/envs/phase3_torch290_cu129_zarr2187_v1/bin/python",
            "tools/v3/run_primitive_connection_hypergraph_teacher_feasibility_v1.py",
            "--spec", str(SPEC), "--run-dir", str(run_dir),
        ],
    }
    inputs = [
        PROJECT_ROOT / "configs/v3/gate2/environments/phase3_torch290_cu129_zarr2187_pip_freeze.txt",
        PROJECT_ROOT / "configs/v3/gate2/environments/phase3_torch290_cu129_zarr2187_v1r2.json",
        PROJECT_ROOT / "docs/PRIMITIVE_RELATION_STRUCTURAL_GRAPH_PLAN_V1.md",
        CARD,
        *[
            root / name
            for root in (P1B, SIDECAR, ATTRIBUTION)
            for name in ("RUN_STATE.json", "metrics/summary.json", "artifacts/evidence_sha256.txt")
        ],
    ]
    spec["frozen_inputs"] = {
        str(path.relative_to(PROJECT_ROOT)): sha(path) for path in inputs
    }
    tools = {
        "teacher_audit_module": "src/mtare_topo/evaluation/primitive_connection_hypergraph_teacher.py",
        "teacher_audit_tests": "tests/v3/unit/test_primitive_connection_hypergraph_teacher.py",
        "relation_storage": "src/mtare_topo/data/primitive_relation_storage.py",
        "relation_storage_tests": "tests/v3/unit/test_primitive_relation_storage.py",
        "observability_sidecar": "src/mtare_topo/data/primitive_attachment_observability_sidecar.py",
        "observability_sidecar_tests": "tests/v3/unit/test_primitive_attachment_observability_sidecar.py",
        "relation_targets": "src/mtare_topo/data/primitive_relation_targets.py",
        "relation_targets_tests": "tests/v3/unit/test_primitive_relation_targets.py",
        "executor": "tools/v3/execute_primitive_connection_hypergraph_teacher_feasibility_v1.py",
        "runner": "tools/v3/run_primitive_connection_hypergraph_teacher_feasibility_v1.py",
        "freezer": "tools/v3/freeze_primitive_connection_hypergraph_teacher_feasibility_v1_spec.py",
        "governance": "src/mtare_topo/governance.py",
        "preflight": "tools/v3/preflight.py",
        "create_run": "tools/v3/create_run.py",
    }
    spec["frozen_tools"] = {
        name: {"path": path, "sha256": sha(PROJECT_ROOT / path)}
        for name, path in tools.items()
    }
    spec["frozen_dataset_trees"] = dataset_trees()
    if len(spec["frozen_dataset_trees"]) != 420:
        raise RuntimeError("expected exactly 210 paired fit/C07 Teacher/sidecar trees")
    write(SPEC, spec)
    print(SPEC)


if __name__ == "__main__":
    main()
