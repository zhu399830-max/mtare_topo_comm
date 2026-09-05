#!/usr/bin/env python3
"""Freeze one zero-training observable-relation readiness run."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

from _bootstrap import PROJECT_ROOT


CARD = PROJECT_ROOT / "configs/v3/gate3/data_cards/primitive_relation_observable_readiness_v1r.json"
SPEC = PROJECT_ROOT / "configs/v3/gate3/primitive_relation_observable_readiness_v1r.json"
RUN_ID = "gate3_20260902_primitive_relation_observable_readiness_v1r_seed0"
PYTHON = "/home/zeng-workstation/.local/share/mtare_topo_comm/envs/phase3_torch290_cu129_zarr2187_v1/bin/python"
RUNNER = "tools/v3/run_primitive_relation_observable_readiness_v1.py"
P1A = PROJECT_ROOT / "results/gate3_semantics/gate3_20260830_primitive_relation_p1a_lossless_storage_corrective_v1_seed0"
P1B = PROJECT_ROOT / "results/gate3_semantics/gate3_20260830_primitive_relation_p1b_teacher_materialization_v1r_seed0"
SIDECAR = PROJECT_ROOT / "results/gate3_semantics/gate3_20260902_primitive_attachment_observability_sidecar_v1_seed0"
AUDIT = PROJECT_ROOT / "results/gate3_semantics/gate3_20260902_primitive_attachment_teacher_observability_v1_seed0"
V2 = PROJECT_ROOT / "results/gate3_semantics/gate3_20260901_primitive_relation_sparse_port_three_seed_training_v1r_seed0"
ATTRIBUTION = PROJECT_ROOT / "results/gate3_semantics/gate3_20260902_primitive_relation_sparse_port_failure_attribution_evidence_corrective_v1_seed0"
OLD_CARD = PROJECT_ROOT / "configs/v3/gate3/data_cards/primitive_relation_sparse_port_readiness_v1.json"
CARD_STATUS = "APPROVED_FOR_ONE_IMMUTABLE_PRIMITIVE_RELATION_OBSERVABLE_READINESS_V1R"


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
        raise RuntimeError("observable readiness card/spec already exists")
    source_summary = json.loads((SIDECAR / "metrics/summary.json").read_text())
    audit_summary = json.loads((AUDIT / "metrics/summary.json").read_text())
    attribution = json.loads((ATTRIBUTION / "metrics/attribution/summary.json").read_text())
    if not (
        source_summary.get("scientific_pass")
        and audit_summary.get("diagnosis")
        == "WINDOW_LEVEL_HIDDEN_ENDPOINT_SUPERVISION_CORRECTABLE_BY_OBSERVABILITY_MASK"
        and attribution.get("diagnosis")
        == "RELATION_SCORE_FAILS_EVEN_WITH_PROPOSAL_ORACLE"
    ):
        raise RuntimeError("observable readiness corrective evidence drift")

    approval = {
        "status": "APPROVED", "approved_by": "user-standing-authorization",
        "approved_at": "2026-09-02T18:00:00+08:00",
        "authorized_gates": [3], "authorized_operations": ["audit"],
        "scope": "One immutable zero-training readiness using one C01 fit sequence, the three frozen V2 geometry checkpoints and the sealed endpoint-observability sidecar. Zero C07/C08/C09/C10, optimizer, checkpoint selection, graph or M-TARE.",
        "confirmation_reference": "The user authorized uninterrupted best-in-plan execution without repeated routine approvals.",
    }
    gates = {
        "interfaces": "Exactly 42 old/new unit tests pass and the student input remains five causal normalized range/valid scans plus relative odometry only.",
        "transfer": "All three V2 checkpoints transfer exactly 133 frozen geometry/temporal tensors and reset exactly 64 relation/evidence tensors.",
        "boundary": "The model has exactly 2635631 parameters in 197 tensors; only 742149 parameters in 64 relation/evidence tensors are trainable.",
        "real_batch": "The fixed C01 row contains 11 active primitives, 17 observed endpoints, supported connections and hidden connections; all six losses and their mean are finite.",
        "mask": "Toggling a hidden positive attachment changes no loss, while the sparse-cardinality objective remains bit exact.",
        "gradient": "Every trainable tensor has a finite nonzero gradient and every frozen tensor has no gradient after one real-batch backward pass.",
        "invariance": "Repeated inference is bit exact; query permutation error <=3e-5 and target permutation/endpoint reversal loss errors <=1e-6.",
        "resources": "Peak CUDA allocation <=4 GiB, wall time <=0.5 h and output <=0.05 GiB; zero optimizer/checkpoint write/C07+/graph/M-TARE reads.",
    }
    card = copy.deepcopy(json.loads(OLD_CARD.read_text()))
    card.update({
        "card_id": "primitive_relation_observable_readiness_v1r",
        "status": CARD_STATUS, "approval": approval,
        "purpose": "Prove that the corrected window-observability supervision reaches the reversal-aware sparse relation loss and that frozen V2 geometry can support relation-only corrective training.",
        "corrective_of": {
            "runs": [str(V2.relative_to(PROJECT_ROOT)), str(ATTRIBUTION.relative_to(PROJECT_ROOT)), str(AUDIT.relative_to(PROJECT_ROOT))],
            "evidence": "V2 geometry passed but safe relation TP was zero; the follow-up audit found 15.7% of physical window positives lacked support at both attached endpoints while more than 95% of physical identities were observable somewhere.",
            "scope": "Keep causal input, 32 slots, geometry checkpoints, six loss families, splits and C07 gates. Add Teacher-only endpoint validity and a learned deployment evidence head; reset relation layers only.",
        },
        "metrics_and_pre_registered_gates": gates,
        "estimated_cost": {
            "compute": "One RTX 5090 D process; 42 tests, three CPU checkpoint-transfer audits and one C01 real-row forward/backward.",
            "wall_time_hours": .5, "host_ram_gb": 4, "gpu": 1,
            "gpu_memory_gb": 4, "disk_gb": .05,
        },
        "retention": "Keep card/spec, test log, three transfer reports, real-row contract, losses, gradient/invariance checks, environment, RUN_STATE and SHA-256 seal.",
        "failure_policy": "Any input, transfer, mask, sparse-objective, gradient, invariance, determinism or resource failure stops before training. Do not alter data, Teacher band, model input, gates or thresholds inside the run.",
    })
    card["source"]["raw_sources"] = [
        "sealed P1a corrected C01 causal LiDAR shard",
        "sealed P1b construction Teacher shard",
        "sealed 0.25 m endpoint-observability sidecar",
        "three frozen V2 selected geometry checkpoints",
    ]
    card["worlds"].update({
        "train": ["S01_flat_tree_small_C01"],
        "validation": ["S01_flat_tree_small_C07"],
        "checkpoint_selection": [],
    })
    card["trajectories"] = [{
        "id": "S01_flat_tree_small_C01_c1_mixed_source_sequence_5",
        "world": "S01_flat_tree_small_C01", "split": "train",
        "independent": True, "duration_s": 4.0, "distance_m": 4.0,
        "spatial_coverage_m": 4.0,
    }]
    card["sampling"].update({
        "independent_sampling_units": "One fixed C01 source sequence selected before model execution because it contains both supported and hidden physical endpoint connections.",
        "raw_frame_count": 5, "effective_sample_count": 1,
        "effective_structure_event_count": 11, "spatial_interval_m": 1.0,
        "temporal_window_frames": 5, "geometry_realizations": 1,
        "source_global_sequence_index": 5, "teacher_row": 5,
        "rule": "Read row 5 only from S01_flat_tree_small_C01__c1_mixed in the fit partition. It has 11 active primitives, 17 supported endpoints, 8 supported and 3 hidden undirected positive attachments.",
    })
    card["sampling"]["structure_event_counts"] = {
        "paired_geometry_rows": 1, "causal_frame_observations": 5,
        "unique_source_sequences": 1, "active_primitives": 11,
        "observed_endpoints": 17, "supported_positive_attachments": 8,
        "hidden_positive_attachments": 3,
    }
    card["split"].update({
        "fit": "One C01 row is read for readiness only.",
        "selection": "C07 zero rows read; no threshold or checkpoint is selected.",
        "development_transfer": "C08 zero rows read.",
        "strict_test": "C09/C10 and all formal M-TARE worlds remain unread.",
        "historical_pollution_audit": "The failed V2 run supplies frozen geometry/temporal weights only. Its relation weights, predictions and C07 outputs are not transferred.",
    })
    card["teacher"] = {
        "source": "P1b construction labels plus the sealed per-window endpoint ray-support sidecar at the pre-registered 0.25 m band.",
        "labels": "Swept primitive parameters, presence, temporal visibility, attachment, disconnected overlap and endpoint observability.",
        "valid_mask": "A physical attachment between two matched primitives enters relation loss only if both endpoints are observed. Hidden physical connections are unknown, never negative. Unmatched sparse queries remain valid negative regularization.",
        "student_forbidden_inputs": "Endpoint support bits are loss-side Teacher targets only. World identity, construction graph, TNG, absolute pose, endpoint truth and future frames never enter forward.",
        "planner_consistency_plan": "No graph is run. Deployment must use the learned endpoint-evidence probability; graph edges remain traversal-only after C07/C08 pass.",
    }
    card["leakage_audit"].update({
        "optimizer_step_count": 0, "model_inference_count": 1,
        "test_excluded_from_checkpoint_selection": True,
    })
    write(CARD, card)

    inputs = [
        CARD,
        *[
            run / name
            for run in (P1A, P1B, SIDECAR, AUDIT, V2, ATTRIBUTION)
            for name in ("RUN_STATE.json", "metrics/summary.json", "artifacts/evidence_sha256.txt")
        ],
        P1A / "artifacts/task_manifest.json",
        P1B / "artifacts/task_manifest.json",
        SIDECAR / "artifacts/task_manifest.json",
        ATTRIBUTION / "metrics/attribution/summary.json",
        *[V2 / f"artifacts/models/seed{seed}/selected.pt" for seed in (0, 1, 2)],
        PROJECT_ROOT / "configs/v3/gate2/environments/phase3_torch290_cu129_zarr2187_v1r2.json",
        PROJECT_ROOT / "configs/v3/gate2/environments/phase3_torch290_cu129_zarr2187_pip_freeze.txt",
    ]
    tools = {
        "runner": RUNNER,
        "freezer": "tools/v3/freeze_primitive_relation_observable_readiness_v1_spec.py",
        "observable_reader": "src/mtare_topo/data/primitive_relation_observable_batches.py",
        "sidecar": "src/mtare_topo/data/primitive_attachment_observability_sidecar.py",
        "loss_base": "src/mtare_topo/representation/primitive_relation_losses.py",
        "loss_sparse": "src/mtare_topo/representation/primitive_relation_sparse_port_losses.py",
        "loss_observable": "src/mtare_topo/representation/primitive_relation_observable_training.py",
        "model_observable": "src/mtare_topo/representation/primitive_relation_observable_model.py",
        "initialization": "src/mtare_topo/representation/primitive_relation_observable_initialization.py",
        "test_observability": "tests/v3/unit/test_primitive_attachment_observability.py",
        "test_sidecar": "tests/v3/unit/test_primitive_attachment_observability_sidecar.py",
        "test_reader": "tests/v3/unit/test_primitive_relation_observable_batches.py",
        "test_model": "tests/v3/unit/test_primitive_relation_observable_model.py",
        "test_initialization": "tests/v3/unit/test_primitive_relation_observable_initialization.py",
        "test_sparse": "tests/v3/unit/test_primitive_relation_sparse_port.py",
        "test_v1": "tests/v3/unit/test_primitive_relation_model.py",
        "test_metrics": "tests/v3/unit/test_primitive_relation_metrics.py",
        "governance": "src/mtare_topo/governance.py",
        "preflight": "tools/v3/preflight.py", "create_run": "tools/v3/create_run.py",
    }
    expected = {
        "independent_source_sequences": 1, "causal_frames": 5,
        "active_primitives": 11, "observed_endpoints": 17,
        "supported_positive_attachments": 8,
        "hidden_positive_attachments": 3, "unit_tests": 42,
        "source_checkpoints": 3, "model_parameters": 2_635_631,
        "parameter_tensors": 197, "trainable_parameters": 742_149,
        "trainable_tensors": 64, "loss_families": 6,
        "optimizer_steps": 0, "checkpoint_writes": 0,
        "c07_rows_read": 0, "c08_rows_read": 0,
        "c09_c10_worlds_read": 0, "graph_replays": 0,
        "mtare_worlds_read": 0,
    }
    run_dir = PROJECT_ROOT / f"results/gate3_semantics/{RUN_ID}"
    spec = {
        "schema_version": "v3_run_spec_v1", "gate": 3,
        "execution_phase": 3, "operation": "audit", "date": "20260902",
        "slug": "primitive_relation_observable_readiness_v1r", "seed": 0,
        "question": "Can corrected endpoint observability, learned endpoint evidence and exact V2 geometry transfer form a deterministic relation-only training interface before C07 selection?",
        "method": "Transfer each seed's proven V2 geometry/temporal tensors exactly, reset only relation and endpoint-evidence modules, mask physical attachment supervision by dual endpoint ray support, preserve the sparse cardinality objective, and audit one real C01 backward pass.",
        "baseline": "Failed V2 sparse-port model with unmasked window-level physical attachment labels and zero safe true positives in all three seeds.",
        "fallback": "Any failure stops before training; correct only the demonstrated interface defect in a new immutable readiness. Do not change data, support band, C07 gates or add rules.",
        "corrective_of": str(ATTRIBUTION.relative_to(PROJECT_ROOT)),
        "data_card": str(CARD.relative_to(PROJECT_ROOT)),
        "config_path": str(CARD.relative_to(PROJECT_ROOT)),
        "user_authorization": approval,
        "acceptance_criteria": list(gates.values()),
        "expected_counts": expected, "estimated_cost": card["estimated_cost"],
        "expected_evidence": [
            "42-test log and exact source hashes.",
            "Three checkpoint transfer/freeze reports.",
            "One real supported+hidden relation batch with sparse-loss, mask, gradient and invariance audits.",
            "Environment, command, RUN_STATE and SHA-256 seal.",
        ],
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
            "--why=Observable primitive relation readiness", "--mode=block",
            "/usr/bin/timeout", "--signal=INT", "--kill-after=60s", "1800s",
            "/usr/bin/env", "CUBLAS_WORKSPACE_CONFIG=:4096:8",
            f"PYTHONPATH={PROJECT_ROOT / 'src'}:{PROJECT_ROOT / 'tools/v3'}",
            PYTHON, RUNNER, "--spec", str(SPEC), "--run-dir", str(run_dir),
        ],
    }
    write(SPEC, spec)
    print(SPEC)


if __name__ == "__main__":
    main()
