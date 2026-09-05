#!/usr/bin/env python3
"""Freeze the one-shot C07 evaluation corrective for composition anchors."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_SPEC = PROJECT_ROOT / (
    "configs/v3/gate3/primitive_composition_anchor_three_seed_training_v1.json"
)
SOURCE_CARD = PROJECT_ROOT / (
    "configs/v3/gate3/data_cards/primitive_composition_anchor_three_seed_training_v1.json"
)
SOURCE_RUN = PROJECT_ROOT / (
    "results/gate3_semantics/"
    "gate3_20260903_primitive_composition_anchor_three_seed_training_v1_seed0"
)
CARD = PROJECT_ROOT / (
    "configs/v3/gate3/data_cards/"
    "primitive_composition_anchor_c07_evaluation_corrective_v1.json"
)
SPEC = PROJECT_ROOT / (
    "configs/v3/gate3/primitive_composition_anchor_c07_evaluation_corrective_v1.json"
)
RUN_ID = "gate3_20260903_primitive_composition_anchor_c07_evaluation_corrective_v1_seed0"
CARD_STATUS = (
    "APPROVED_FOR_ONE_IMMUTABLE_PRIMITIVE_COMPOSITION_ANCHOR_"
    "C07_EVALUATION_CORRECTIVE_V1"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    if CARD.exists() or SPEC.exists():
        raise RuntimeError("composition-anchor C07 evaluation corrective card/spec already exists")
    source_spec = load(SOURCE_SPEC)
    source_card = load(SOURCE_CARD)
    source_state = load(SOURCE_RUN / "RUN_STATE.json")
    source_summary = load(SOURCE_RUN / "metrics/summary.json")
    expected_error = "RuntimeError: composition-anchor C07 evaluator produced no scientific result"
    if (
        source_state.get("state") != "FAILED"
        or source_state.get("error") != expected_error
        or source_summary.get("error") != expected_error
        or source_summary.get("optimizer_steps") != 30_699
        or len((SOURCE_RUN / "artifacts/evidence_sha256.txt").read_text().splitlines()) != 49
    ):
        raise RuntimeError("evaluation corrective requires the exact sealed post-training failure")

    card = copy.deepcopy(source_card)
    card["card_id"] = "primitive_composition_anchor_c07_evaluation_corrective_v1"
    card["purpose"] = (
        "Recover the missing C07 scientific decision from three already trained and sealed "
        "composition-anchor checkpoints after correcting only float32 multiplication grouping."
    )
    card["corrective_scope"] = (
        "The source completed all 30699 optimizer steps and three checkpoint selections. Its "
        "evaluator then rejected a finite mathematical score because left-associated float32 "
        "multiplication differed from its transpose by one ULP. This run groups endpoint "
        "evidence as an exactly symmetric outer product, performs zero optimizer steps, and "
        "repeats only the frozen C07 evaluation."
    )
    card["estimated_cost"] = {
        "compute": "One RTX 5090 D; one frozen pass over 64644 C07 rows for each of three seeds.",
        "disk_gb": 0.5, "gpu": 1, "gpu_memory_gb": 16,
        "host_ram_gb": 4, "wall_time_hours": 4,
    }
    card["failure_policy"] = (
        "Any source-seal, checkpoint, input, environment, population, symmetry, resource or "
        "isolation drift fails closed. A valid model scientific FAIL is retained as a valid "
        "corrective execution outcome and stops before C08/graph; no retraining or threshold change."
    )
    card["retention"] = (
        "Retain source failure provenance, corrected per-seed C07 metrics, numerical symmetry "
        "evidence, comparison PNG/PDF/SVG, logs, resource monitor, RUN_STATE and SHA-256 seal."
    )
    card["sampling"]["effective_sample_count"] = 64_644
    card["sampling"]["effective_structure_event_count"] = 442_936
    card["sampling"]["raw_frame_count"] = 64_644 * 5
    card["sampling"]["independent_sampling_units"] = (
        "10 disjoint C07 topology parents and 30 paired geometry tasks; each frozen seed is an "
        "independent trained model evaluated on the same sealed rows."
    )
    card["sampling"]["rule"] = (
        "Read every C07 sequence exactly once per frozen seed at batch128 with the original "
        "existence thresholds; no shuffle, adaptation, optimizer update or checkpoint selection."
    )
    card["sampling"]["structure_event_counts"] = {
        "c07_parent_worlds": 10, "c07_geometry_tasks": 30,
        "c07_sequences_per_seed": 64_644,
        "c07_observable_positive_attachments": 442_936,
        "frozen_seeds": 3, "model_forward_rows": 193_932,
        "optimizer_steps": 0, "c08_rows_read": 0,
        "c09_c10_worlds_read": 0,
    }
    card["trajectories"] = [source_card["trajectories"][1]]
    card["leakage_audit"]["model_inference_count"] = 193_932
    card["leakage_audit"]["optimizer_step_count"] = 0
    card["metrics_and_pre_registered_gates"] = {
        "source": "The sealed source has exactly 30699 optimizer steps, three valid selected checkpoints and only the known post-training evaluator failure.",
        "numerics": "All corrected safe scores are finite and bit-exact symmetric on every C07 batch; the legacy left-associated asymmetry is reproduced and quantified.",
        "population": "Each of seeds 0/1/2 reads exactly 64644 C07 rows and the evaluator verifies 442936 observable positives.",
        "science": "Use the unchanged gates: attachment F1 gain >=0.05, precision>=0.98 with TP>0, anchor MAE improvement>=10%, frozen backbone, and at least 2/3 passing seeds.",
        "resources": "Wall time <=4 h; host RSS <=4 GiB; GPU process memory <=16 GiB; output <=0.5 GiB.",
        "isolation": "Optimizer steps, C08 rows, C09/C10 worlds, graph replays and M-TARE reads are all zero.",
        "decision": "A valid PASS opens only a separately frozen C08 transfer/readiness step; a valid scientific FAIL stops before C08/graph and triggers C07-only attribution.",
    }
    approval = {
        "approved_at": "2026-09-03T12:30:00+08:00",
        "approved_by": "user-standing-authorization",
        "authorized_gates": [3],
        "authorized_operations": ["audit"],
        "confirmation_reference": (
            "The user explicitly authorized uninterrupted best-in-plan execution and requested "
            "automatic optimal choices without repeated approval."
        ),
        "scope": (
            "One immutable zero-training C07 evaluation corrective over the three sealed "
            "composition-anchor checkpoints; no C08+, graph or M-TARE."
        ),
        "status": "APPROVED",
    }
    card["approval"] = approval
    card["status"] = CARD_STATUS
    write(CARD, card)

    run_dir = PROJECT_ROOT / "results/gate3_semantics" / RUN_ID
    spec = {
        "schema_version": "v3_run_spec_v1",
        "slug": "primitive_composition_anchor_c07_evaluation_corrective_v1",
        "date": "20260903", "seed": 0, "gate": 3, "execution_phase": 3,
        "operation": "audit", "working_directory": str(PROJECT_ROOT),
        "config_path": str(CARD.relative_to(PROJECT_ROOT)),
        "data_card": str(CARD.relative_to(PROJECT_ROOT)),
        "question": (
            "Do the three sealed composition-anchor checkpoints satisfy the frozen C07 physical "
            "connection gates once mathematically identical safe scores are evaluated with exact "
            "float32 symmetry?"
        ),
        "method": (
            "Zero-training re-evaluation that groups the two endpoint evidence factors into a "
            "symmetric outer product before multiplying by symmetric anchor compatibility."
        ),
        "baseline": (
            "Frozen same-input non-learning C07 attachment F1 0.0063801323 and each model's "
            "unchanged raw predicted endpoint error."
        ),
        "fallback": card["failure_policy"],
        "user_authorization": approval,
        "acceptance_criteria": list(card["metrics_and_pre_registered_gates"].values()),
        "expected_counts": {
            "source_seal_files": 49, "frozen_seeds": 3,
            "c07_parent_worlds": 10, "c07_tasks": 30,
            "c07_rows_per_seed": 64_644, "model_forward_rows": 193_932,
            "optimizer_steps": 0, "c08_rows_read": 0,
            "c09_c10_worlds_read": 0, "graph_replays": 0,
            "mtare_worlds_read": 0,
        },
        "estimated_cost": card["estimated_cost"],
        "expected_evidence": [
            "Exact source failure, 49-file source seal and three selected-checkpoint contracts.",
            "Three complete C07 per-seed metrics under the original thresholds and scientific gates.",
            "Full-population legacy-versus-corrected symmetry diagnostics.",
            "Comparison PNG/PDF/SVG, resource monitor, input before/after hashes, RUN_STATE and seal.",
        ],
        "command": [
            "/usr/bin/systemd-inhibit", "--what=sleep:shutdown",
            "--why=Composition anchor C07 evaluation corrective", "--mode=block",
            "/usr/bin/timeout", "--signal=INT", "--kill-after=120s", "14400s",
            "/usr/bin/env",
            (
                f"PYTHONPATH={PROJECT_ROOT / 'src'}:{PROJECT_ROOT / 'tools/v3'}:"
                f"{PROJECT_ROOT}"
            ),
            "PYTHONHASHSEED=0", "CUBLAS_WORKSPACE_CONFIG=:4096:8",
            "/home/zeng-workstation/.local/share/mtare_topo_comm/envs/"
            "phase3_torch290_cu129_zarr2187_v1/bin/python",
            "tools/v3/run_primitive_composition_anchor_c07_evaluation_corrective_v1.py",
            "--spec", str(SPEC), "--run-dir", str(run_dir),
        ],
    }
    frozen_inputs = dict(source_spec["frozen_inputs"])
    additional_inputs = [
        CARD, SOURCE_SPEC, SOURCE_CARD,
        SOURCE_RUN / "RUN_STATE.json",
        SOURCE_RUN / "metrics/summary.json",
        SOURCE_RUN / "logs/04_c07_evaluation.log",
        SOURCE_RUN / "artifacts/evidence_sha256.txt",
    ]
    for seed in range(3):
        additional_inputs.extend((
            SOURCE_RUN / f"artifacts/models/seed{seed}/summary.json",
            SOURCE_RUN / f"artifacts/models/seed{seed}/selected.pt",
            SOURCE_RUN / f"metrics/seed{seed}_contract_checks.json",
        ))
    frozen_inputs.update({
        str(path.relative_to(PROJECT_ROOT)): sha256(path) for path in additional_inputs
    })
    spec["frozen_inputs"] = frozen_inputs
    tools = {
        "runner": "tools/v3/run_primitive_composition_anchor_c07_evaluation_corrective_v1.py",
        "freezer": "tools/v3/freeze_primitive_composition_anchor_c07_evaluation_corrective_v1_spec.py",
        "evaluator": "tools/v3/evaluate_primitive_composition_anchor_three_seed_v1.py",
        "model": "src/mtare_topo/representation/primitive_composition_anchor_model.py",
        "model_tests": "tests/v3/unit/test_primitive_composition_anchor_model.py",
        "evaluation_tests": "tests/v3/unit/test_evaluate_primitive_composition_anchor_three_seed_v1.py",
        "corrective_tests": "tests/v3/unit/test_primitive_composition_anchor_c07_evaluation_corrective_v1.py",
        "batch_module": "src/mtare_topo/data/primitive_composition_anchor_batches.py",
        "training_module": "src/mtare_topo/representation/primitive_composition_anchor_training.py",
        "governance": "src/mtare_topo/governance.py",
        "preflight": "tools/v3/preflight.py",
        "create_run": "tools/v3/create_run.py",
    }
    spec["frozen_tools"] = {
        name: {"path": path, "sha256": sha256(PROJECT_ROOT / path)}
        for name, path in tools.items()
    }
    write(SPEC, spec)
    print(CARD)
    print(SPEC)


if __name__ == "__main__":
    main()
