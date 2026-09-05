#!/usr/bin/env python3
"""Freeze the bounded last-encoder-block fallback Data Card and run spec."""

from __future__ import annotations

from copy import deepcopy
import hashlib

from _bootstrap import PROJECT_ROOT
from mtare_topo.governance import load_json, write_json


RUN_ID = "gate3_20260827_gse_causal_geometry_delta_last_block_training_v1_seed0"
CARD_PATH = PROJECT_ROOT / "configs/v3/gate3/data_cards/gse_causal_geometry_delta_last_block_training_v1.json"
SPEC_PATH = PROJECT_ROOT / "configs/v3/gate3/gse_causal_geometry_delta_last_block_training_v1.json"
SOURCE_CARD = PROJECT_ROOT / "configs/v3/gate3/data_cards/gse_causal_geometry_delta_multitask_training_v1.json"
DATASET = "results/gate2_representation/gate2_20260824_gse_deduplicated_dataset_export_v1_seed0"
TRAINING = "results/gate2_representation/gate2_20260824_gse_graph_three_seed_training_v1r_seed0"
TEACHER = "results/gate2_representation/gate2_20260827_gse_corrected_causal_teacher_manifest_v1r_seed0"
VERIFIER = "results/gate3_semantics/gate3_20260826_gse_exit_token_association_corrective_v2_seed0"
PURE_CLASSIFIER = "results/gate3_semantics/gate3_20260827_gse_corrected_causal_event_training_v1_seed0"
OBSERVABILITY = "results/gate3_semantics/gate3_20260827_gse_causal_geometry_delta_observability_v1_seed0"
OLD_DIRECTIONAL = "results/gate3_semantics/gate3_20260826_gse_directional_structural_event_training_v1_seed0"
FAILED_MULTITASK = "results/gate3_semantics/gate3_20260827_gse_causal_geometry_delta_multitask_training_v1_seed0"
PYTHON = "/home/zeng-workstation/.local/share/mtare_topo_comm/envs/phase3_torch290_cu129_zarr2187_v1/bin/python"


def _sha(path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    card = deepcopy(load_json(SOURCE_CARD))
    card.update(
        {
            "card_id": "gse_causal_geometry_delta_last_block_training_v1",
            "status": "APPROVED_FOR_ONE_IMMUTABLE_GSE_CAUSAL_GEOMETRY_DELTA_LAST_BLOCK_TRAINING_V1",
            "purpose": "Execute the single predeclared bounded fallback after the frozen-representation multitask run learned continuous geometry deltas but failed persistent structural-event triggering.",
        }
    )
    card["approval"] = {
        "status": "APPROVED",
        "approved_by": "user",
        "approved_at": "2026-08-27T22:10:00+08:00",
        "authorized_operations": ["training"],
        "authorized_gates": [3],
        "scope": "One immutable three-seed bounded adaptation: C01-C06 fit, C07-C08 checkpoint/threshold selection, six epochs, exactly 52236 optimizer steps updating only encoder[-1] plus a fresh multitask head; zero other-backbone/C09/C10/M-TARE updates or reads.",
        "confirmation_reference": "User explicitly instructed Codex to automatically choose the evidence-supported best in-scope option and not request routine approvals.",
    }
    card["source"].update(
        {
            "failed_frozen_multitask_run": FAILED_MULTITASK,
            "failed_frozen_multitask_status": "FAIL_GSE_CAUSAL_GEOMETRY_DELTA_MULTITASK_TRAINING_V1",
            "failed_frozen_multitask_seal_sha256": "aed1d39e051b7a917a2996ff4db301a2993d100dfc58007925c8f1e790c347e4",
            "partial_reuse": "READ_ONLY_SEALED_C01_C08_DATA_TEACHER_THREE_ORIGINAL_BACKBONES_FEATURE_BASELINES_OBSERVABILITY_AND_FAILED_COMPONENT_METRICS; NO_FAILED_HEAD_CHECKPOINT_REUSE",
        }
    )
    card["source"]["raw_sources"] = [
        "Sealed C01-C08 deduplicated Zarr: 252430 unique LiDAR frames and 188126 five-frame causal observations.",
        "Sealed corrected Teacher V1R: 1031 persistent change-point labels and continuous width/height/slope/curvature targets.",
        "Three sealed original GSE checkpoints supply initialization. Failed event/delta head checkpoints are comparison evidence only and are never loaded.",
        "Sealed observability PASS fixes 92845/29920 fit/selection geometry-valid four-step delta pairs and fit-only scales.",
        "The failed frozen-representation multitask run is read only for status, metrics and seal; it supplies no weights, scores, threshold or training example.",
        "No C09/C10/M-TARE sensor, Teacher, graph, model output or planner artifact is read.",
    ]
    card["sampling"]["optimization_draws"].update(
        {
            "total_optimizer_steps": 52236,
            "last_encoder_block_optimizer_steps": 52236,
            "frozen_backbone_optimizer_steps": 0,
        }
    )
    card["sampling"]["rule"] = (
        "Each epoch visits all 92845 C01-C06 geometry-valid lag pairs exactly once and adds 24000 identity-balanced event draws. "
        "Only encoder[-1] and a fresh zero-initialized multitask head update; C07-C08 is evaluated only at epochs 2/4/6 and invalid geometry is never imputed."
    )
    card["teacher"].update(
        {
            "student_input": "Only the existing five causal 16x720 range/mask frames. TNG, geometry targets, identity, pose, world, future observations and validation statistics are forbidden model inputs.",
            "planner_consistency_plan": "Continuous delta and structural events share an azimuth-preserving change encoder while the last LiDAR residual block may adapt. Event evidence may create a node only after the unchanged open-set gate; edges remain traversal-only.",
        }
    )
    card["methods"] = {
        "main": "Load each original sealed GSE seed, freeze all parameters, then unfreeze only encoder[-1] (96-to-128 residual block, 271104 parameters). Train it at 1e-4 with a fresh zero-initialized 444425-parameter causal delta/event head at 1e-3. Audit gradients every step and full frozen/updated parameter digests per seed.",
        "baseline": "Strong event baseline is the sealed old directional corrected-label ensemble macro-F1=0.6879041032. Geometry baseline is frozen metric-delta normalized MAE=0.5871666431. The failed fully frozen multitask decoder is a required ablation: macro-F1=0.6836686 and delta MAE=0.3551743.",
        "fallback": "No further automatic capacity expansion. If this bounded adaptation fails, stop the current event-representation route and re-evaluate the method before any graph/planner or full-backbone tuning.",
    }
    card["metrics_and_pre_registered_gates"] = {
        "event": "Unchanged ensemble corrected event gate: macro-F1>=0.7379041032; precision>=0.98, false accept<=0.01, recall>=0.40; junction/terminal/turn identity>=0.90/0.90/0.40; change-point>=7/17. Ensemble gain>=0.05, >=2 seeds gain>=0.05 and no seed regression.",
        "geometry": "Unchanged ensemble normalized delta MAE improves >=10% over frozen metric-delta baseline; >=2 seeds improve >=10%, and no seed regresses by more than 5%.",
        "selection": "One lexicographic checkpoint per seed using corrected event gate, change-point identity coverage, lower normalized delta MAE, macro-F1 and earlier epoch; C07-C08 only.",
        "trainability": "Exactly 271104 last-block plus 444425 fresh-head parameters are trainable. Every other backbone parameter has no gradient and identical before/after SHA-256 digest; the last block must change.",
        "nonvacuous": True,
    }
    card["acceptance"].update(
        {
            "baseline": "Old directional corrected-label macro-F1=0.6879041032; frozen three-seed metric-delta normalized MAE=0.5871666431.",
            "final_generalization_claim": False,
        }
    )
    card["split"].update(
        {
            "fit": "C01-C06: 60 worlds, 142184 observations, 92845 valid delta pairs. Only encoder[-1] and fresh multitask-head parameters update.",
            "checkpoint_and_threshold_selection": "C07-C08: 20 worlds, 45942 observations, 29920 valid delta pairs and 17 change-point identities. Epoch and event threshold selection only.",
            "future_validation": "C09/C10 and M-TARE remain unread. Passing permits a newly frozen offline topology evaluation; failing stops this representation route.",
        }
    )
    card["estimated_cost"] = {
        "compute": "One RTX 5090 D; three bounded last-block adaptations serially",
        "wall_time_hours": 6.0,
        "disk_gb": 3.0,
        "host_ram_gb": 8,
        "gpu_memory_gb": 16,
    }
    card["evidence"] = {
        "machine_metrics": "Exact data/lag counts, three original-backbone hashes, fresh-head initialization, parameter names/counts, per-step gradient boundary, before/after parameter digests, 18 epoch records, event/identity/open-set and delta metrics, optimizer/test read counts, environment and source seals.",
        "complete_visual_review": "Publish event-identity and delta-error figures only from the sealed final summary with CSV/JSON provenance; preserve useful failure figures.",
        "failure_policy": "Any data/split/source drift, failed-head weight reuse, trainability escape, frozen parameter change, absent last-block update, nonfinite/resource failure, C09/C10/M-TARE access or unmet scientific gate seals FAIL. No retry, capacity expansion or threshold relaxation.",
    }
    card["retention"] = "Retain three selected last-block+head checkpoints, epoch records, selection outputs, sampling/trainability contracts, config, raw log, RUN_STATE and complete seal."
    write_json(CARD_PATH, card)

    inputs = [
        f"{DATASET}/RUN_STATE.json", f"{DATASET}/metrics/summary.json", f"{DATASET}/artifacts/evidence_sha256.txt", f"{DATASET}/artifacts/sequence_manifest.jsonl",
        f"{TRAINING}/RUN_STATE.json", f"{TRAINING}/metrics/summary.json", f"{TRAINING}/artifacts/evidence_sha256.txt",
        f"{TEACHER}/RUN_STATE.json", f"{TEACHER}/metrics/summary.json", f"{TEACHER}/artifacts/evidence_sha256.txt", f"{TEACHER}/artifacts/teacher_observations.jsonl",
        f"{VERIFIER}/RUN_STATE.json", f"{VERIFIER}/metrics/summary.json", f"{VERIFIER}/artifacts/evidence_sha256.txt", f"{VERIFIER}/artifacts/pair_cache/pairs.npz",
        f"{PURE_CLASSIFIER}/RUN_STATE.json", f"{PURE_CLASSIFIER}/metrics/summary.json", f"{PURE_CLASSIFIER}/artifacts/evidence_sha256.txt",
        f"{OBSERVABILITY}/RUN_STATE.json", f"{OBSERVABILITY}/metrics/summary.json", f"{OBSERVABILITY}/artifacts/evidence_sha256.txt",
        f"{OLD_DIRECTIONAL}/RUN_STATE.json", f"{OLD_DIRECTIONAL}/metrics/summary.json", f"{OLD_DIRECTIONAL}/artifacts/evidence_sha256.txt", f"{OLD_DIRECTIONAL}/artifacts/training/ensemble_selection_outputs.npz",
        f"{FAILED_MULTITASK}/RUN_STATE.json", f"{FAILED_MULTITASK}/metrics/summary.json", f"{FAILED_MULTITASK}/artifacts/evidence_sha256.txt", f"{FAILED_MULTITASK}/artifacts/training/summary.json",
    ]
    for seed in (0, 1, 2):
        inputs.extend(
            [
                f"{TRAINING}/artifacts/models/seed{seed}/best.pt",
                f"{VERIFIER}/artifacts/models/seed{seed}/frozen_observation_features.npy",
                f"{OLD_DIRECTIONAL}/artifacts/training/seed{seed}/selection_outputs.npz",
            ]
        )
    tool_paths = {
        "data_card": str(CARD_PATH.relative_to(PROJECT_ROOT)),
        "multitask_head": "src/mtare_topo/representation/gse_causal_geometry_delta.py",
        "last_block_contract": "src/mtare_topo/representation/gse_last_block_adaptation.py",
        "directional_head": "src/mtare_topo/representation/gse_directional_structural_event.py",
        "corrected_event_contract": "src/mtare_topo/representation/gse_corrected_causal_event.py",
        "backbone": "src/mtare_topo/representation/gse_graph.py",
        "trainer": "tools/v3/train_gse_causal_geometry_delta_last_block_v1.py",
        "runner": "tools/v3/run_gse_causal_geometry_delta_last_block_training_v1.py",
        "freezer": "tools/v3/freeze_gse_causal_geometry_delta_last_block_training_spec_v1.py",
        "evidence_integrity": "src/mtare_topo/evaluation/gse_evidence_integrity.py",
        "governance": "src/mtare_topo/governance.py",
        "preflight": "tools/v3/preflight.py",
        "create_run": "tools/v3/create_run.py",
    }
    spec = {
        "schema_version": "v3_run_spec_v1",
        "gate": 3,
        "execution_phase": 3,
        "date": "20260827",
        "slug": "gse_causal_geometry_delta_last_block_training_v1",
        "seed": 0,
        "operation": "training",
        "question": "Can a bounded final-residual-block adaptation turn learned causal geometry deltas into safe persistent structural events without changing the data or graph contract?",
        "method": card["methods"]["main"],
        "baseline": card["methods"]["baseline"],
        "fallback": card["methods"]["fallback"],
        "data_card": str(CARD_PATH.relative_to(PROJECT_ROOT)),
        "config_path": str(CARD_PATH.relative_to(PROJECT_ROOT)),
        "user_authorization": deepcopy(card["approval"]),
        "acceptance_criteria": [
            card["metrics_and_pre_registered_gates"]["event"],
            card["metrics_and_pre_registered_gates"]["geometry"],
            card["metrics_and_pre_registered_gates"]["trainability"],
            "Exactly 52236 optimizer steps and zero C09/C10/M-TARE reads; complete immutable evidence and seal.",
        ],
        "expected_counts": {
            "fit_worlds": 60, "fit_observations": 142184, "fit_geometry_valid_lag_pairs": 92845,
            "selection_worlds": 20, "selection_observations": 45942, "selection_geometry_valid_lag_pairs": 29920,
            "seeds": 3, "epochs": 6, "optimizer_steps": 52236, "last_encoder_block_optimizer_steps": 52236,
            "frozen_backbone_optimizer_steps": 0, "trainable_parameters": 715529,
            "c09_worlds_read": 0, "strict_test_worlds_read": 0, "mtare_worlds_read": 0,
        },
        "hyperparameters": {
            "epochs": 6, "event_draws_per_epoch": 24000, "regression_batch_size": 32,
            "evaluation_batch_size": 48, "evaluate_every": 2, "head_learning_rate": 0.001,
            "last_encoder_block_learning_rate": 0.0001, "weight_decay": 0.0001,
            "gradient_clip": 5.0, "optimizer": "AdamW", "seeds": [0, 1, 2],
        },
        "expected_evidence": [
            "Three original-backbone source hashes, three selected last-block+head checkpoints, 18 epoch records, exact trainability and parameter update audits, selection outputs, event/delta metrics, environment, log, RUN_STATE and SHA-256 seal."
        ],
        "estimated_cost": {"compute": "One RTX 5090 D, serial three-seed bounded adaptation", "disk_gb": 3.0, "wall_time_hours": 6.0},
        "frozen_inputs": {path: _sha(PROJECT_ROOT / path) for path in inputs},
        "frozen_tools": {name: {"path": path, "sha256": _sha(PROJECT_ROOT / path)} for name, path in tool_paths.items()},
        "working_directory": str(PROJECT_ROOT),
        "command": [
            "/usr/bin/systemd-inhibit", "--what=sleep:shutdown", "--why=GSE bounded last-block geometry training", "--mode=block",
            "/usr/bin/timeout", "--signal=INT", "--kill-after=300s", "29400s", PYTHON,
            "tools/v3/run_gse_causal_geometry_delta_last_block_training_v1.py",
            "--spec", str(SPEC_PATH),
            "--run-dir", str(PROJECT_ROOT / "results/gate3_semantics" / RUN_ID),
        ],
    }
    write_json(SPEC_PATH, spec)
    print(CARD_PATH.relative_to(PROJECT_ROOT))
    print(SPEC_PATH.relative_to(PROJECT_ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
