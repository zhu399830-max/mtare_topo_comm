#!/usr/bin/env python3
"""Freeze the local composition-slot zero-training readiness run."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SOURCE_CARD = ROOT / "configs/v3/gate3/data_cards/primitive_composition_anchor_training_readiness_v1.json"
CARD = ROOT / "configs/v3/gate3/data_cards/primitive_local_composition_slot_readiness_v1.json"
SPEC = ROOT / "configs/v3/gate3/primitive_local_composition_slot_readiness_v1.json"
RUN_ID = "gate3_20260903_primitive_local_composition_slot_readiness_v1_seed0"
CARD_STATUS = "APPROVED_FOR_ONE_IMMUTABLE_PRIMITIVE_LOCAL_COMPOSITION_SLOT_READINESS_V1"
PYTHON = "/home/zeng-workstation/.local/share/mtare_topo_comm/envs/phase3_torch290_cu129_zarr2187_v1/bin/python"
P1A = ROOT / "results/gate3_semantics/gate3_20260830_primitive_relation_p1a_lossless_storage_corrective_v1_seed0"
P1B = ROOT / "results/gate3_semantics/gate3_20260830_primitive_relation_p1b_teacher_materialization_v1r_seed0"
OBS = ROOT / "results/gate3_semantics/gate3_20260902_primitive_attachment_observability_sidecar_v1_seed0"
MODELS = ROOT / "results/gate3_semantics/gate3_20260902_primitive_relation_observable_three_seed_training_v1_seed0"
FEASIBILITY = ROOT / "results/gate3_semantics/gate3_20260903_local_composition_slot_teacher_feasibility_v1_seed0"
TASK = "S01_flat_tree_small_C01__c1_mixed.zarr"


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _tree_hash(path: Path) -> str:
    digest = hashlib.sha256()
    for value in sorted(item for item in path.rglob("*") if item.is_file()):
        relative = value.relative_to(path).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(4, "little")); digest.update(relative)
        digest.update(bytes.fromhex(_sha(value)))
    return digest.hexdigest()


def _load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    if CARD.exists() or SPEC.exists():
        raise RuntimeError("local composition-slot readiness card/spec already exists")
    feasibility = _load(FEASIBILITY / "metrics/teacher_feasibility/summary.json")
    if (
        feasibility.get("decision") != "ALLOW_LOCAL_COMPOSITION_SLOT_MODEL_READINESS"
        or feasibility.get("capacity", {}).get("selected_capacity") != 32
        or feasibility.get("capacity", {}).get("c07_overflow_rows") != 0
    ):
        raise RuntimeError("local composition-slot readiness requires the sealed Teacher PASS")
    source = _load(SOURCE_CARD); card = copy.deepcopy(source)
    card["card_id"] = "primitive_local_composition_slot_readiness_v1"
    card["purpose"] = "Verify the permutation-safe endpoint-to-32-slot/dustbin relation model, loss, real-batch gradients, ambiguity interface, memory and development split isolation before any optimizer step."
    card["teacher_source"] = "Sealed P1b construction endpoint attachments aligned to frozen primitive queries and decomposed into the lossless local composition cliques proven by the prior fit/C07 feasibility audit; endpoint observability masks hidden relations as unknown."
    card["sampling"] = {
        "raw_frame_count": 172,
        "effective_sample_count": 128,
        "effective_structure_event_count": 128,
        "spatial_interval_m": 1.0,
        "temporal_window_frames": 5,
        "independent_sampling_units": "One predeclared C01 fit parent/task and its first 128 five-frame sequences are an implementation/resource proof only; the bound population remains 70 disjoint parents, 210 geometry tasks and 491196 sequences.",
        "rule": "Bind exact fit/C07 populations 426552/64644 without C07 model forward; read rows 0:128 of S01_flat_tree_small_C01__c1_mixed once through the frozen seed0 backbone and untrained slot head. Those sequences reference exactly 172 unique raw frames.",
        "structure_event_counts": {
            "bound_parent_worlds": 70,
            "bound_geometry_tasks": 210,
            "bound_sequences": 491196,
            "fit_tasks": 180,
            "fit_sequences": 426552,
            "c07_tasks": 30,
            "c07_sequences": 64644,
            "readiness_fit_sequences": 128,
            "readiness_unique_raw_frames": 172,
            "model_forward_rows": 128,
            "c07_model_forward_rows": 0,
            "optimizer_steps": 0,
            "checkpoint_writes": 0,
        },
    }
    card["trajectories"] = list(source["trajectories"][:2])
    card["estimated_cost"] = {
        "compute": "One deterministic CUDA batch-128 forward/backward plus direct-head invariance probes and 26 CPU tests.",
        "disk_gb": 0.05, "gpu": 1, "gpu_memory_gb": 16,
        "host_ram_gb": 4, "wall_time_hours": 0.1,
    }
    card["failure_policy"] = "Fail closed on target losslessness drift, non-finite or missing gradient, backbone mutation, query/slot permutation failure, yaw failure, CUDA/RSS excess, source drift, C07 forward, optimizer/checkpoint write, C08+ or graph/M-TARE read. Correct implementation defects only; do not train or change Teacher/capacity."
    card["retention"] = "Keep config, environment, source hashes, test log, real-batch losses/counts, invariance/resource metrics, untrained interface PNG/PDF/SVG, RUN_STATE and SHA-256 seal."
    card["metrics_and_pre_registered_gates"] = {
        "population": "Exact fit/C07 loader populations are 180/30 tasks and 426552/64644 sequences; C07 model forward is zero.",
        "teacher": "The real batch contains nonempty clusters, dustbin endpoints and disconnected-overlap negatives, with maximum cluster count <=32.",
        "model": "Frozen backbone=2635631 parameters; trainable local slot head=1001507 parameters in 81 tensors.",
        "gradient": "All 81 head parameter tensors have finite nonzero gradients; every backbone gradient is absent and its state hash is unchanged.",
        "symmetry": "Direct head repeat is bit exact; slot permutation and Hungarian-matched loss errors <=3e-6; yaw error <=3e-5; safe score is finite and exactly symmetric.",
        "resources": "Batch128 CUDA allocated/reserved/process each <=16 GiB and host RSS <=4 GiB.",
        "forbidden": "Zero optimizer, checkpoint, C07 model forward, C08/C09/C10, graph and M-TARE reads.",
    }
    approval = {
        "approved_at": "2026-09-03T14:35:00+08:00",
        "approved_by": "user-standing-authorization",
        "authorized_gates": [3],
        "authorized_operations": ["audit"],
        "confirmation_reference": "The user requested uninterrupted automatic best-in-plan execution without repeated approvals.",
        "scope": "One immutable zero-training local composition-slot readiness run; no training, C08, graph or M-TARE.",
        "status": "APPROVED",
    }
    card["approval"] = approval; card["status"] = CARD_STATUS; _write(CARD, card)

    run_dir = ROOT / "results/gate3_semantics" / RUN_ID
    spec = {
        "schema_version": "v3_run_spec_v1", "slug": "primitive_local_composition_slot_readiness_v1",
        "date": "20260903", "seed": 0, "gate": 3, "execution_phase": 3,
        "operation": "audit", "working_directory": str(ROOT),
        "config_path": str(CARD.relative_to(ROOT)), "data_card": str(CARD.relative_to(ROOT)),
        "question": "Can a permutation-safe endpoint-to-32-slot/dustbin head consume the frozen causal LiDAR primitive representation with lossless Teacher alignment, finite full-head gradients, deterministic invariants and bounded batch128 memory?",
        "method": "Frozen observable five-frame primitive backbone plus a set-transformer endpoint encoder, 32 exchangeable composition queries, categorical slot/dustbin assignment, Hungarian cluster matching and ambiguity-aware same-slot score.",
        "baseline": "The sealed lossless O(E^2) observable attachment matrix and the frozen global-anchor failure; slot targets must reconstruct the former without reintroducing the latter's absolute coordinate regression.",
        "fallback": card["failure_policy"], "user_authorization": approval,
        "acceptance_criteria": list(card["metrics_and_pre_registered_gates"].values()),
        "expected_counts": {
            "fit_tasks": 180, "fit_rows": 426552, "c07_tasks": 30, "c07_rows": 64644,
            "readiness_rows": 128, "unique_raw_frames": 172, "tests": 26,
            "backbone_parameters": 2635631, "head_parameters": 1001507, "head_parameter_tensors": 81,
            "model_forward_rows": 128, "c07_model_forward_rows": 0,
            "optimizer_steps": 0, "checkpoint_writes": 0, "c08_rows": 0,
            "graph_replays": 0, "mtare_worlds": 0,
        },
        "estimated_cost": card["estimated_cost"],
        "expected_evidence": [
            "26 unit tests and exact source/environment hashes.",
            "Real batch128 target cluster/dustbin/overlap counts and five finite loss terms.",
            "81/81 trainable gradients, frozen backbone hash, repeat/permutation/yaw/symmetry metrics.",
            "CUDA allocated/reserved/process and host RSS evidence.",
            "Untrained assignment-vs-Teacher PNG/PDF/SVG, RUN_STATE and SHA-256 seal.",
        ],
        "selected_real_data_hashes": {
            "sensor": _tree_hash(P1A / "artifacts/dataset/fit" / TASK),
            "teacher": _tree_hash(P1B / "artifacts/teacher/fit" / TASK),
            "observability": _tree_hash(OBS / "artifacts/endpoint_observability/fit" / TASK),
        },
        "command": [
            "/usr/bin/systemd-inhibit", "--what=sleep:shutdown",
            "--why=Primitive local composition slot readiness", "--mode=block",
            "/usr/bin/timeout", "--signal=INT", "--kill-after=60s", "1800s",
            "/usr/bin/env", f"PYTHONPATH={ROOT / 'src'}:{ROOT / 'tools/v3'}:{ROOT}",
            "PYTHONHASHSEED=0", "CUBLAS_WORKSPACE_CONFIG=:4096:8", PYTHON,
            "tools/v3/run_primitive_local_composition_slot_readiness_v1.py",
            "--spec", str(SPEC), "--run-dir", str(run_dir),
        ],
    }
    inputs = [
        CARD,
        FEASIBILITY / "RUN_STATE.json",
        FEASIBILITY / "metrics/summary.json",
        FEASIBILITY / "metrics/teacher_feasibility/summary.json",
        FEASIBILITY / "artifacts/evidence_sha256.txt",
        P1A / "RUN_STATE.json", P1A / "metrics/summary.json", P1A / "artifacts/evidence_sha256.txt",
        P1B / "RUN_STATE.json", P1B / "metrics/summary.json", P1B / "artifacts/evidence_sha256.txt",
        OBS / "RUN_STATE.json", OBS / "metrics/summary.json", OBS / "artifacts/evidence_sha256.txt",
        MODELS / "RUN_STATE.json", MODELS / "metrics/summary.json", MODELS / "artifacts/evidence_sha256.txt",
        *[MODELS / f"artifacts/models/seed{seed}/selected.pt" for seed in (0, 1, 2)],
    ]
    spec["frozen_inputs"] = {str(path.relative_to(ROOT)): _sha(path) for path in inputs}
    tools = {
        "model": "src/mtare_topo/representation/primitive_local_composition_slot_model.py",
        "training": "src/mtare_topo/representation/primitive_local_composition_slot_training.py",
        "teacher": "src/mtare_topo/evaluation/local_composition_slot_teacher.py",
        "model_tests": "tests/v3/unit/test_primitive_local_composition_slot_model.py",
        "training_tests": "tests/v3/unit/test_primitive_local_composition_slot_training.py",
        "teacher_tests": "tests/v3/unit/test_local_composition_slot_teacher.py",
        "observable_tests": "tests/v3/unit/test_primitive_relation_observable_model.py",
        "runner": "tools/v3/run_primitive_local_composition_slot_readiness_v1.py",
        "freezer": "tools/v3/freeze_primitive_local_composition_slot_readiness_v1_spec.py",
        "governance": "src/mtare_topo/governance.py",
        "preflight": "tools/v3/preflight.py",
        "create_run": "tools/v3/create_run.py",
    }
    spec["frozen_tools"] = {
        name: {"path": path, "sha256": _sha(ROOT / path)} for name, path in tools.items()
    }
    _write(SPEC, spec); print(CARD); print(SPEC)


if __name__ == "__main__":
    main()
