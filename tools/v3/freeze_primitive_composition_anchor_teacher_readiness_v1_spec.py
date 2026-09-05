#!/usr/bin/env python3
"""Freeze the fit/C07 O(E) composition-anchor Teacher readiness audit."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

from _bootstrap import PROJECT_ROOT


SOURCE_CARD = PROJECT_ROOT / "configs/v3/gate3/data_cards/primitive_connection_hypergraph_teacher_feasibility_v1.json"
CARD = PROJECT_ROOT / "configs/v3/gate3/data_cards/primitive_composition_anchor_teacher_readiness_v1.json"
SPEC = PROJECT_ROOT / "configs/v3/gate3/primitive_composition_anchor_teacher_readiness_v1.json"
RUN_ID = "gate3_20260903_primitive_composition_anchor_teacher_readiness_v1_seed0"
P1A = PROJECT_ROOT / "results/gate3_semantics/gate3_20260830_primitive_relation_p1a_lossless_storage_corrective_v1_seed0"
P1B = PROJECT_ROOT / "results/gate3_semantics/gate3_20260830_primitive_relation_p1b_teacher_materialization_v1r_seed0"
SIDECAR = PROJECT_ROOT / "results/gate3_semantics/gate3_20260902_primitive_attachment_observability_sidecar_v1_seed0"
HYPERGRAPH = PROJECT_ROOT / "results/gate3_semantics/gate3_20260902_primitive_connection_hypergraph_teacher_feasibility_v1_seed0"


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
        digest.update(len(relative).to_bytes(4, "little")); digest.update(relative)
        digest.update(bytes.fromhex(sha(value)))
    return digest.hexdigest()


def dataset_contract() -> dict[str, str]:
    result = {}
    sensor_root = P1A / "artifacts/dataset"
    construction_root = P1A / "artifacts/constructions"
    teacher_root = P1B / "artifacts/teacher"
    sidecar_root = SIDECAR / "artifacts/endpoint_observability"
    for split in ("fit", "c07"):
        tasks = sorted(value.stem for value in (teacher_root / split).glob("*.zarr") if value.is_dir())
        for task in tasks:
            result[f"construction/{split}/{task}.json"] = sha(
                construction_root / split / f"{task}.json"
            )
            sensor = sensor_root / split / f"{task}.zarr"
            for array in ("sensor_xyz_m", "yaw_deg"):
                result[f"sensor_pose/{split}/{task}/{array}"] = tree_sha(sensor / array)
            result[f"p1b/{split}/{task}"] = tree_sha(teacher_root / split / f"{task}.zarr")
            result[f"sidecar/{split}/{task}"] = tree_sha(sidecar_root / split / f"{task}.zarr")
    return result


def write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    if CARD.exists() or SPEC.exists():
        raise RuntimeError("composition-anchor readiness card/spec already exists")
    for source in (P1A, P1B, SIDECAR):
        if json.loads((source / "RUN_STATE.json").read_text()).get("state") != "COMPLETED":
            raise RuntimeError(f"composition-anchor source is incomplete: {source.name}")
    hypergraph = json.loads((HYPERGRAPH / "metrics/teacher_feasibility/summary.json").read_text())
    if (
        hypergraph.get("scientific_pass") is not False
        or hypergraph.get("decision") != "STOP_PRIMITIVE_CONNECTION_HYPERGRAPH_CANDIDATE"
        or hypergraph.get("checks", {}).get("fit_teacher_union_of_cliques") is not True
        or hypergraph.get("checks", {}).get("c07_teacher_union_of_cliques") is not True
        or hypergraph.get("checks", {}).get("fixed_decoder_output_compression") is not False
    ):
        raise RuntimeError("dense hypergraph stop evidence is incomplete")

    card = copy.deepcopy(json.loads(SOURCE_CARD.read_text()))
    card["card_id"] = "primitive_composition_anchor_teacher_readiness_v1"
    card["status"] = "APPROVED_FOR_ONE_IMMUTABLE_PRIMITIVE_COMPOSITION_ANCHOR_TEACHER_READINESS_V1"
    card["purpose"] = (
        "After the direct pair score and dense K-slot decoder failed, prove whether exact construction-node anchors give every observable primitive endpoint one unique causal O(E) relation target, preserve stacked/nonincident separation and expose transferable geometric connection evidence before implementing or training a replacement model."
    )
    card["approval"] = {
        "status": "APPROVED", "approved_by": "user-standing-authorization",
        "approved_at": "2026-09-03T00:18:00+08:00",
        "authorized_gates": [3], "authorized_operations": ["audit"],
        "scope": (
            "One immutable CPU-only C01-C06 fit plus C07 selection composition-anchor Teacher/readiness audit over 70 parents, 210 paired geometry tasks and 491196 five-frame rows. Read only construction JSON, current sensor pose/yaw, P1b targets and endpoint-observability masks; zero LiDAR range, model inference, optimizer, C08/C09/C10, graph or M-TARE."
        ),
        "confirmation_reference": (
            "The user authorized uninterrupted strongest-evidence execution toward the frozen paper objective. The valid connection-clique Teacher plus failed dense K-slot complexity gate uniquely motivates an O(E) per-endpoint shared-anchor readiness proof."
        ),
    }
    card["source"] = {
        "license_or_allowed_use": card["source"]["license_or_allowed_use"],
        "raw_sources": [
            "sealed P1a fit/C07 realized construction JSON with exact endpoint node anchors",
            "sealed P1a fit/C07 current sensor_xyz_m and yaw_deg arrays only; zero range/valid returns",
            "sealed P1b fit/C07 primitive indices, cropped axis controls, endpoint neighbors and overlap labels",
            "sealed fit/C07 endpoint-observability sidecars",
            "sealed dense K-slot hypergraph scientific stop evidence",
        ],
        "worlds": [
            "C01-C06 fit: 60 independent topology parents, 180 paired geometry tasks",
            "C07 selection: 10 independent topology parents, 30 paired geometry tasks",
        ],
    }
    card["sampling"] = {
        "independent_sampling_units": (
            "70 world-disjoint topology parents; three geometry realizations are paired repeated measures. Five-frame rows are dependent observations, not independent worlds."
        ),
        "raw_frame_count": 659_940,
        "effective_sample_count": 70,
        "effective_structure_event_count": 491_196,
        "raw_five_frame_sequence_count": 491_196,
        "spatial_interval_m": 1.0, "temporal_window_frames": 5,
        "maximum_primitive_slots": 32,
        "partition_sequences": {
            "fit_c01_c06": 426_552, "selection_c07": 64_644,
            "development_transfer_c08": 0,
        },
        "structure_event_counts": {
            "fit_parent_worlds": 60, "fit_geometry_tasks": 180,
            "fit_rows": 426_552, "c07_parent_worlds": 10,
            "c07_geometry_tasks": 30, "c07_rows": 64_644,
            "total_rows": 491_196, "c07_observable_positive_pairs": 442_936,
        },
        "rule": (
            "For every active primitive endpoint, validate exactly one construction-operation membership and transform its shared world anchor into the current sensor frame. On dual-endpoint-observed pairs, report target-anchor equality/separation, endpoint-to-anchor residual and overlap hard negatives. Select one non-learning observed-axis distance threshold on fit at precision>=0.98, transfer it unchanged to C07, and compare fixed O(E) anchor output size against the old pair head."
        ),
    }
    card["split"] = {
        "fit": "C01-C06: 60 parents/180 tasks/426552 rows; sole source for the non-learning axis-distance threshold.",
        "selection": "C07: 10 parents/30 tasks/64644 rows; threshold transferred unchanged and no method adaptation.",
        "development_transfer": "C08 directory, manifest records, arrays and results remain unopened.",
        "strict_test": "C09/C10 and all M-TARE benchmark worlds remain forbidden.",
        "world_disjoint": True, "trajectory_disjoint": True,
        "historical_pollution_audit": (
            "Only the sealed direct-pair oracle failure and dense-slot complexity failure select this representation class. No C08, graph or planner result selects targets, baseline threshold or acceptance."
        ),
    }
    card["teacher"] = {
        "source": (
            "PrimitiveConstructionGraph endpoint composition_anchor_xyz_m and node membership, transformed offline by the sealed current sensor pose/yaw."
        ),
        "labels": (
            "Per-active-endpoint current-sensor-frame composition anchor; shared-anchor attachment; endpoint anchor residual; disconnected angular-overlap hard negative; endpoint evidence validity."
        ),
        "valid_mask": (
            "Anchor regression is valid only for active endpoints with direct 0.25 m support. A pair is supervised only when both endpoints are observed; hidden endpoints remain unknown."
        ),
        "student_forbidden_inputs": (
            "Construction/world/node/primitive identity, absolute pose, sensor_xyz/yaw and future frames are Teacher-only. The future model may see only five causal range/valid scans and relative odometry."
        ),
        "planner_consistency_plan": (
            "No graph or planner runs. If ready, a separate model readiness must implement per-endpoint anchor mean/uncertainty and differentiable calibrated shared-anchor probability; no fixed distance/angle connection rule becomes the main method."
        ),
        "unexplored_port_policy": "Execution state only; absent from this Teacher and future perception loss.",
    }
    card["metrics_and_pre_registered_gates"] = {
        "population": "Exactly 70 parents, 210 tasks and 491196 fit/C07 rows; current-pose rows only, zero sensor range and zero C08/C09/C10.",
        "identity": "Every primitive endpoint has exactly one composition membership; base/realized primitive order, endpoint direction, node identity and anchor agree.",
        "target_geometry": "All true connection endpoint anchors coincide within 1e-9 m; all distinct and overlap-hard-negative anchors remain separated by more than 1e-6 m.",
        "observability": "Exactly reproduce 442936 C07 observable positive pairs and report endpoint-anchor residual/censoring distributions.",
        "baseline": "A fit-only exact threshold on negative observed-axis endpoint separation has a nonempty precision>=0.98 prefix and transfers unchanged to C07 with precision>=0.98 and nonzero true positives.",
        "complexity": "Anchor xyz plus uncertainty requires 256 values at the fixed 32-primitive contract versus 1984 independent unordered cross-primitive pair scores, at least 7x fewer learned outputs.",
        "resources": "CPU only, wall time <=1h, host RSS <=4GiB, output <=0.1GiB and zero model/optimizer/graph/planner work.",
    }
    card["estimated_cost"] = {
        "compute": "CPU-only streaming construction/pose/P1b/sidecar audit and exact fit ranking over about 40.7M observable endpoint pairs.",
        "gpu": 0, "gpu_memory_gb": 0, "host_ram_gb": 4,
        "wall_time_hours": 0.25, "disk_gb": 0.1,
    }
    card["failure_policy"] = (
        "Any endpoint membership ambiguity, coordinate/provenance drift, shared-anchor inconsistency, distinct/hard-negative anchor collapse, empty or non-transferable safe baseline, output-complexity failure, resource breach or split leakage stops the composition-anchor candidate. Do not change support band, threshold on C07, relation labels or read C08 to repair it."
    )
    card["retention"] = (
        "Retain per-task/split anchor distributions, fit threshold and untouched C07 transfer, complexity proof, PNG/PDF/SVG/source, environment, commands, logs, RUN_STATE and SHA-256 seal; retain no copied Teacher rows or pair arrays."
    )
    write(CARD, card)

    run_dir = PROJECT_ROOT / f"results/gate3_semantics/{RUN_ID}"
    spec = {
        "schema_version": "v3_run_spec_v1",
        "slug": "primitive_composition_anchor_teacher_readiness_v1",
        "date": "20260903", "seed": 0, "gate": 3, "execution_phase": 3,
        "operation": "audit", "working_directory": str(PROJECT_ROOT),
        "config_path": str(CARD.relative_to(PROJECT_ROOT)),
        "data_card": str(CARD.relative_to(PROJECT_ROOT)),
        "question": (
            "Can each observable primitive endpoint receive one unique causal current-sensor-frame construction-anchor target whose shared-anchor geometry separates true connection clusters from stacked/nonincident endpoints, enabling an O(E) learned relation output without hand-written connection rules?"
        ),
        "method": (
            "Exhaustive fit/C07 construction-anchor identity and geometry audit, fit-only precision>=0.98 observed-axis distance baseline transfer, and fixed O(E) output-complexity proof."
        ),
        "baseline": (
            "Observed Teacher primitive endpoint separation with no learned residual, plus the stopped 1984-output independent pair head and 2048-output K=32 dense slot head."
        ),
        "fallback": card["failure_policy"], "user_authorization": card["approval"],
        "acceptance_criteria": list(card["metrics_and_pre_registered_gates"].values()),
        "expected_counts": {
            "fit_parent_worlds": 60, "fit_geometry_tasks": 180,
            "fit_rows": 426_552, "c07_parent_worlds": 10,
            "c07_geometry_tasks": 30, "c07_rows": 64_644,
            "total_rows": 491_196, "c07_observable_positive_pairs": 442_936,
            "sensor_range_rows_read": 0, "model_forward_rows": 0,
            "optimizer_steps": 0, "c08_rows_read": 0,
            "c09_c10_worlds_read": 0, "graph_replays": 0, "mtare_worlds_read": 0,
        },
        "estimated_cost": card["estimated_cost"],
        "expected_evidence": [
            "Every endpoint's unique composition-node anchor and causal current-sensor transform proof.",
            "Same-cluster equality and distinct/overlap-hard-negative separation distributions.",
            "Fit-only safe endpoint-distance threshold and unchanged C07 transfer metrics.",
            "7.75x learned-output compression, per-task tables, plots, logs, RUN_STATE and seal.",
        ],
        "command": [
            "/usr/bin/systemd-inhibit", "--what=sleep:shutdown",
            "--why=Primitive composition anchor Teacher readiness", "--mode=block",
            "/usr/bin/timeout", "--signal=INT", "--kill-after=60s", "3600s",
            "/usr/bin/env",
            f"PYTHONPATH={PROJECT_ROOT / 'src'}:{PROJECT_ROOT / 'tools/v3'}:{PROJECT_ROOT}",
            "/home/zeng-workstation/.local/share/mtare_topo_comm/envs/phase3_torch290_cu129_zarr2187_v1/bin/python",
            "tools/v3/run_primitive_composition_anchor_teacher_readiness_v1.py",
            "--spec", str(SPEC), "--run-dir", str(run_dir),
        ],
    }
    inputs = [
        PROJECT_ROOT / "configs/v3/gate2/environments/phase3_torch290_cu129_zarr2187_pip_freeze.txt",
        PROJECT_ROOT / "configs/v3/gate2/environments/phase3_torch290_cu129_zarr2187_v1r2.json",
        PROJECT_ROOT / "docs/PRIMITIVE_RELATION_STRUCTURAL_GRAPH_PLAN_V1.md", CARD,
        *[
            root / name for root in (P1A, P1B, SIDECAR)
            for name in ("RUN_STATE.json", "metrics/summary.json", "artifacts/evidence_sha256.txt")
        ],
        HYPERGRAPH / "RUN_STATE.json",
        HYPERGRAPH / "metrics/teacher_feasibility/summary.json",
        HYPERGRAPH / "artifacts/evidence_sha256.txt",
    ]
    spec["frozen_inputs"] = {str(path.relative_to(PROJECT_ROOT)): sha(path) for path in inputs}
    tools = {
        "anchor_teacher": "src/mtare_topo/evaluation/primitive_composition_anchor_teacher.py",
        "anchor_teacher_tests": "tests/v3/unit/test_primitive_composition_anchor_teacher.py",
        "hypergraph_teacher": "src/mtare_topo/evaluation/primitive_connection_hypergraph_teacher.py",
        "hypergraph_tests": "tests/v3/unit/test_primitive_connection_hypergraph_teacher.py",
        "observability": "src/mtare_topo/evaluation/primitive_attachment_observability.py",
        "observability_tests": "tests/v3/unit/test_primitive_attachment_observability.py",
        "sidecar": "src/mtare_topo/data/primitive_attachment_observability_sidecar.py",
        "sidecar_tests": "tests/v3/unit/test_primitive_attachment_observability_sidecar.py",
        "ranked_selection": "src/mtare_topo/evaluation/primitive_relation_sparse_port_failure_attribution.py",
        "executor": "tools/v3/execute_primitive_composition_anchor_teacher_readiness_v1.py",
        "runner": "tools/v3/run_primitive_composition_anchor_teacher_readiness_v1.py",
        "freezer": "tools/v3/freeze_primitive_composition_anchor_teacher_readiness_v1_spec.py",
        "governance": "src/mtare_topo/governance.py",
        "preflight": "tools/v3/preflight.py", "create_run": "tools/v3/create_run.py",
    }
    spec["frozen_tools"] = {
        name: {"path": path, "sha256": sha(PROJECT_ROOT / path)}
        for name, path in tools.items()
    }
    spec["frozen_dataset_contract"] = dataset_contract()
    if len(spec["frozen_dataset_contract"]) != 1050:
        raise RuntimeError("composition-anchor frozen data contract must contain 1050 fit/C07 entries")
    write(SPEC, spec); print(SPEC)


if __name__ == "__main__":
    main()
