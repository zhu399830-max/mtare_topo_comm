#!/usr/bin/env python3
"""Freeze the zero-optimizer batch-128 anchor training readiness run."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

from _bootstrap import PROJECT_ROOT


SOURCE_CARD = PROJECT_ROOT / "configs/v3/gate3/data_cards/primitive_composition_anchor_target_sidecar_v1r.json"
CARD = PROJECT_ROOT / "configs/v3/gate3/data_cards/primitive_composition_anchor_training_readiness_v1.json"
SPEC = PROJECT_ROOT / "configs/v3/gate3/primitive_composition_anchor_training_readiness_v1.json"
RUN_ID = "gate3_20260903_primitive_composition_anchor_training_readiness_v1_seed0"
P1A = PROJECT_ROOT / "results/gate3_semantics/gate3_20260830_primitive_relation_p1a_lossless_storage_corrective_v1_seed0"
P1B = PROJECT_ROOT / "results/gate3_semantics/gate3_20260830_primitive_relation_p1b_teacher_materialization_v1r_seed0"
OBSERVABILITY = PROJECT_ROOT / "results/gate3_semantics/gate3_20260902_primitive_attachment_observability_sidecar_v1_seed0"
SOURCE_MODELS = PROJECT_ROOT / "results/gate3_semantics/gate3_20260902_primitive_relation_observable_three_seed_training_v1_seed0"
ANCHORS = PROJECT_ROOT / "results/gate3_semantics/gate3_20260903_primitive_composition_anchor_target_sidecar_v1r_seed0"
MODEL_READINESS = PROJECT_ROOT / "results/gate3_semantics/gate3_20260903_primitive_composition_anchor_model_readiness_v2_seed0"
DIAGNOSTIC = PROJECT_ROOT / "results/gate3_semantics/gate3_20260903_primitive_predicted_geometry_association_diagnostic_v1_seed0"


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    if CARD.exists() or SPEC.exists():
        raise RuntimeError("composition-anchor readiness card/spec already exists")
    card = copy.deepcopy(json.loads(SOURCE_CARD.read_text(encoding="utf-8")))
    card["card_id"] = "primitive_composition_anchor_training_readiness_v1"
    card["status"] = "APPROVED_FOR_ONE_IMMUTABLE_PRIMITIVE_COMPOSITION_ANCHOR_TRAINING_READINESS_V1"
    card["purpose"] = "Prove exact batch-128 loader, loss, frozen-backbone gradient and GPU resource contracts before the three-seed optimizer run."
    card["approval"] = {
        "status": "APPROVED", "approved_by": "user-standing-authorization",
        "approved_at": "2026-09-03T15:00:00+08:00", "authorized_gates": [3],
        "authorized_operations": ["audit"],
        "scope": "One zero-optimizer fit batch-128 forward/backward repeat plus full fit/C07 loader identity audit; 256 model-forward rows, no checkpoint, C08+, graph or M-TARE.",
        "confirmation_reference": "The user authorized uninterrupted best-in-plan execution; the sealed anchor sidecar decision requires head-only training next, and this is its zero-update resource/readiness proof.",
    }
    card["sampling"]["rule"] = "Audit all 180 fit and 30 C07 shard identities/counts, then read the first deterministic 128-row fit batch twice through frozen seed0 backbone and untrained anchor head."
    card["sampling"]["effective_sample_count"] = 128
    card["sampling"]["structure_event_counts"].update({
        "model_forward_rows": 256, "c07_model_forward_rows": 0,
        "optimizer_steps": 0, "checkpoint_writes": 0,
    })
    card["leakage_audit"]["model_inference_count"] = 256
    card["leakage_audit"]["optimizer_step_count"] = 0
    card["metrics_and_pre_registered_gates"] = {
        "population": "Loader identities exactly cover fit 60 parents/180 tasks/426552 rows and C07 10/30/64644; model forward is exactly two repeats of one 128-row fit batch.",
        "parameters": "Three source checkpoints bind exact hashes; real seed0 model has 2635631 frozen backbone and 22278 trainable head parameters in 8 tensors.",
        "gradients": "All 8 head gradients are finite and nonzero, all backbone gradients absent, all six losses finite, repeat outputs/losses exact.",
        "resources": "Peak CUDA allocation and GPU process memory each <=16 GiB; host RSS <=4 GiB; zero optimizer/checkpoint/C07 model forward/C08+/graph/M-TARE.",
        "decision": "PASS allows the exact three-seed training Data Card/spec; it is not learned performance and does not unlock C08.",
    }
    card["estimated_cost"] = {
        "compute": "One RTX 5090 D batch-128 forward/backward and repeat; zero optimizer.",
        "gpu": 1, "host_ram_gb": 4, "wall_time_hours": .2, "disk_gb": .1,
    }
    card["failure_policy"] = "Any source, count, loader, loss, gradient, repeat, environment or resource drift fails closed before training; do not reduce evidence or start optimizer work."
    card["retention"] = "Retain hashes, test/resource/loss metrics, environment, logs, RUN_STATE and seal."
    _write(CARD, card)

    run_dir = PROJECT_ROOT / f"results/gate3_semantics/{RUN_ID}"
    spec = {
        "schema_version": "v3_run_spec_v1", "slug": "primitive_composition_anchor_training_readiness_v1",
        "date": "20260903", "seed": 0, "gate": 3, "execution_phase": 3,
        "operation": "audit", "working_directory": str(PROJECT_ROOT),
        "config_path": str(CARD.relative_to(PROJECT_ROOT)), "data_card": str(CARD.relative_to(PROJECT_ROOT)),
        "question": "Can the sealed data/model interface train the 22278-parameter anchor head at batch 128 deterministically within 16 GiB while keeping the backbone exactly frozen?",
        "method": "Bind all four input layers, load each source checkpoint, and run one real 128-row seed0 forward/backward plus exact repeat without optimizer or checkpoint.",
        "baseline": "The earlier one-row model readiness, which proved mathematics but not full batch-128 data binding or resource headroom.",
        "fallback": "Any failure stops before optimizer work; a smaller batch requires a separately frozen corrective, never an in-run change.",
        "user_authorization": card["approval"],
        "acceptance_criteria": list(card["metrics_and_pre_registered_gates"].values()),
        "expected_counts": {
            "fit_rows": 426_552, "fit_tasks": 180, "c07_rows": 64_644, "c07_tasks": 30,
            "model_forward_rows": 256, "optimizer_steps": 0, "checkpoint_writes": 0,
            "tests": 33, "head_parameters": 22_278,
            "c08_rows_read": 0, "c09_c10_worlds_read": 0, "graph_replays": 0, "mtare_worlds_read": 0,
        },
        "estimated_cost": card["estimated_cost"],
        "expected_evidence": ["33 tests and exact four-layer loader binding.", "Real batch-128 finite loss/gradient, exact repeat and three checkpoint hashes.", "Environment, resource metrics, RUN_STATE and seal."],
        "command": [
            "/usr/bin/systemd-inhibit", "--what=sleep:shutdown", "--why=Composition anchor training readiness", "--mode=block",
            "/usr/bin/timeout", "--signal=INT", "--kill-after=60s", "1200s", "/usr/bin/env",
            f"PYTHONPATH={PROJECT_ROOT / 'src'}:{PROJECT_ROOT / 'tools/v3'}:{PROJECT_ROOT}",
            "CUBLAS_WORKSPACE_CONFIG=:4096:8", 
            "/home/zeng-workstation/.local/share/mtare_topo_comm/envs/phase3_torch290_cu129_zarr2187_v1/bin/python",
            "tools/v3/run_primitive_composition_anchor_training_readiness_v1.py",
            "--spec", str(SPEC), "--run-dir", str(run_dir),
        ],
    }
    inputs = [
        CARD,
        *[
            run / name for run in (P1A, P1B, OBSERVABILITY, SOURCE_MODELS, ANCHORS, MODEL_READINESS, DIAGNOSTIC)
            for name in ("RUN_STATE.json", "metrics/summary.json", "artifacts/evidence_sha256.txt")
            if (run / name).exists()
        ],
        P1A / "artifacts/task_manifest.json", P1B / "artifacts/task_manifest.json",
        OBSERVABILITY / "artifacts/task_manifest.json",
        ANCHORS / "artifacts/materialized/task_manifest.json",
        *[SOURCE_MODELS / f"artifacts/models/seed{seed}/selected.pt" for seed in range(3)],
    ]
    spec["frozen_inputs"] = {str(path.relative_to(PROJECT_ROOT)): _sha(path) for path in inputs}
    tools = {
        "batch_module": "src/mtare_topo/data/primitive_composition_anchor_batches.py",
        "sidecar_module": "src/mtare_topo/data/primitive_composition_anchor_sidecar.py",
        "model": "src/mtare_topo/representation/primitive_composition_anchor_model.py",
        "training": "src/mtare_topo/representation/primitive_composition_anchor_training.py",
        "train_seed": "tools/v3/train_primitive_composition_anchor_v1.py",
        "evaluator": "tools/v3/evaluate_primitive_composition_anchor_three_seed_v1.py",
        "runner": "tools/v3/run_primitive_composition_anchor_training_readiness_v1.py",
        "freezer": "tools/v3/freeze_primitive_composition_anchor_training_readiness_v1_spec.py",
        "batch_tests": "tests/v3/unit/test_primitive_composition_anchor_batches.py",
        "sidecar_tests": "tests/v3/unit/test_primitive_composition_anchor_sidecar.py",
        "teacher_tests": "tests/v3/unit/test_primitive_composition_anchor_teacher.py",
        "model_tests": "tests/v3/unit/test_primitive_composition_anchor_model.py",
        "evaluation_tests": "tests/v3/unit/test_evaluate_primitive_composition_anchor_three_seed_v1.py",
        "governance": "src/mtare_topo/governance.py", "preflight": "tools/v3/preflight.py", "create_run": "tools/v3/create_run.py",
    }
    spec["frozen_tools"] = {name: {"path": path, "sha256": _sha(PROJECT_ROOT / path)} for name, path in tools.items()}
    _write(SPEC, spec)
    print(SPEC)


if __name__ == "__main__":
    main()
