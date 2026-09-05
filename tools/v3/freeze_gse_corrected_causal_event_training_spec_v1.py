#!/usr/bin/env python3
"""Freeze corrected causal event training Data Card and one-shot spec."""

from __future__ import annotations

from copy import deepcopy
import hashlib

from _bootstrap import PROJECT_ROOT
from mtare_topo.governance import load_json, write_json


RUN_ID = "gate3_20260827_gse_corrected_causal_event_training_v1_seed0"
CARD_PATH = PROJECT_ROOT / "configs/v3/gate3/data_cards/gse_corrected_causal_event_training_v1.json"
SPEC_PATH = PROJECT_ROOT / "configs/v3/gate3/gse_corrected_causal_event_training_v1.json"
SOURCE_CARD = PROJECT_ROOT / "configs/v3/gate3/data_cards/gse_directional_structural_event_training_v1.json"
DATASET = "results/gate2_representation/gate2_20260824_gse_deduplicated_dataset_export_v1_seed0"
TRAINING = "results/gate2_representation/gate2_20260824_gse_graph_three_seed_training_v1r_seed0"
VERIFIER = "results/gate3_semantics/gate3_20260826_gse_exit_token_association_corrective_v2_seed0"
TEACHER = "results/gate2_representation/gate2_20260827_gse_corrected_causal_teacher_manifest_v1r_seed0"
OLD_DIRECTIONAL = "results/gate3_semantics/gate3_20260826_gse_directional_structural_event_training_v1_seed0"


def _sha(path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _record(path: str) -> dict[str, str]:
    return {"path": path, "sha256": _sha(PROJECT_ROOT / path)}


def main() -> int:
    card = deepcopy(load_json(SOURCE_CARD))
    card.update(
        {
            "card_id": "gse_corrected_causal_event_training_v1",
            "status": "APPROVED_FOR_ONE_IMMUTABLE_GSE_CORRECTED_CAUSAL_EVENT_TRAINING_V1",
            "purpose": "Test whether the frozen five-frame azimuth-preserving GSE features can learn the corrected persistent bidirectional causal change-point Teacher without changing geometry, exit, place or association backbones.",
        }
    )
    card["approval"] = {
        "status": "APPROVED",
        "approved_by": "user",
        "approved_at": "2026-08-27T13:30:00+08:00",
        "authorized_operations": ["training"],
        "authorized_gates": [3],
        "scope": "One immutable three-seed corrected-causal directional event-head training: C01-C06 fit, C07-C08 selection, 36000 head steps, zero backbone update and zero C09/C10/M-TARE.",
        "confirmation_reference": "User explicitly authorized Codex to select the evidence-supported optimal in-scope implementation automatically without repeated routine approval prompts.",
    }
    sampling = card["sampling"]
    for stale in (
        "existing_pairs",
        "total_pairs",
        "open_set_pairs",
        "selection_pairs_all",
        "online_candidate_rule",
        "online_eligible_negative_pairs",
        "online_eligible_positive_pairs",
        "online_eligible_selection_pairs",
        "online_excluded_negative_pairs",
        "online_excluded_positive_pairs",
        "open_set_pair_rule",
    ):
        sampling.pop(stale, None)
    sampling.update(
        {
            "effective_structure_event_count": 37162,
            "fit_event_counts": {
                "corridor": 114617,
                "junction": 19743,
                "terminal": 5551,
                "turn": 1482,
                "geometry_transition": 791,
            },
            "selection_event_counts": {
                "corridor": 36347,
                "junction": 6865,
                "terminal": 1974,
                "turn": 516,
                "geometry_transition": 240,
            },
            "fit_event_identity_counts": {
                "junction": 417,
                "terminal": 375,
                "turn": 297,
                "geometry_transition": 59,
            },
            "selection_event_identity_counts": {
                "junction": 146,
                "terminal": 128,
                "turn": 95,
                "geometry_transition": 17,
            },
            "fit_association_identities": 1148,
            "selection_association_identities": 386,
            "fit_identity_valid_observations": 27567,
            "selection_identity_valid_observations": 9595,
            "fit_open_set_observations": 114617,
            "selection_open_set_observations": 36347,
            "node_gate_selection_negative": 36347,
            "node_gate_selection_positive": 9595,
            "node_gate_selection_observations": 45942,
            "structure_event_counts": {"identity_valid": 37162, "open_set": 150964},
            "identity_intersection": 0,
            "change_point_family_support": {
                "fit": ["S01", "S02", "S03", "S04", "S05", "S06", "S07", "S08", "S10"],
                "selection": ["S01", "S08", "S09", "S10"],
            },
            "independence_risk": "The 791/240 labels represent only 59/17 independent change-point identities; selection covers four topology families. Identity-balanced sampling and exact identity coverage with Wilson 95% interval are mandatory. This is a development learnability gate, not the final all-family generalization claim.",
            "hard_negative_source": "C01-C06 only: one minus the mean frozen three-seed corridor probability under the old event head. Former short transition artifacts are now corridor negatives and receive high hardness naturally; scores affect sampling only.",
            "rule": "All 188126 corrected C01-C08 observations remain the effective population. Each seed draws 24000 C01-C06 samples per epoch: corridor mass 0.5 split uniform/hardness, remaining mass equal across four structural classes and then identities. C07-C08 is evaluated only at epochs 3/6/9/12.",
        }
    )
    card["source"].update(
        {
            "corrected_teacher_run": TEACHER,
            "corrected_teacher_status": "PASS_GSE_CORRECTED_CAUSAL_TEACHER_MANIFEST_V1R",
            "corrected_teacher_seal_sha256": "3f840e1165709d04aab80374739551931783045515f1ba95456107e4b5867f49",
            "old_directional_baseline_run": OLD_DIRECTIONAL,
            "old_directional_corrected_label_macro_f1": 0.6879041031973032,
            "old_directional_corrected_label_change_point_identity_coverage": 0.0,
            "partial_reuse": "READ_ONLY_C01_C08_ZARR_THREE_FROZEN_GSE_BACKBONES_CORRECTED_TEACHER_AND_BASELINE_OUTPUTS",
        }
    )
    card["source"]["raw_sources"] = [
        "Sealed C01-C08 GSE deduplicated Zarr: 252430 unique frames and 188126 five-frame sequences.",
        "PASS corrected causal Teacher V1R: unchanged observations with 1031 persistent labels and 76 change-point identities.",
        "Three sealed GSE checkpoints; backbone, metric geometry, exit tokens, place descriptors and association parameters remain read-only.",
        "Sealed old directional outputs are recomputed against corrected labels as the strongest event baseline; no failed checkpoint is reused.",
        "No C09/C10/M-TARE sensor, Teacher, model, graph or planner artifact is read.",
    ]
    card["teacher"].update(
        {
            "source": "PASS corrected causal Teacher V1R. Junction/terminal stay objective TNG nodes, turns stay canonical edge clusters, and geometry transitions are only persistent bidirectional past-confirmed change-points back-projected to their causal boundary.",
            "planner_consistency_plan": "Binary structural evidence decides whether a stable event may create a node; conditional class names it. Identity is used only for fit balancing and selection coverage. Change-point detection may be delayed but is associated with the sealed back-projected boundary. Edges remain traversal-only.",
        }
    )
    card["methods"].update(
        {
            "main": "For each frozen GSE seed, train a new zero-initialized DirectionalStructuralEventHead on corrected labels. The circular change encoder consumes current, mean-past and delta azimuth features; separate residual heads predict structural evidence and conditional event class. Only head parameters update.",
            "baseline": "Strongest corrected-label baseline is the sealed old directional three-seed ensemble: macro-F1=0.6879041032 and 0/17 change-point identity coverage. Frozen original event heads are also recomputed and retained.",
            "fallback": "If corrected labels still cannot produce >=7/17 change-point identities and the unchanged open-set safety/gain gates, stop before C09. Do not unfreeze the backbone, tune the graph or weaken the Teacher in this run.",
        }
    )
    card["metrics_and_pre_registered_gates"] = {
        "aggregate": "Ensemble macro-F1>=0.7379041032; gain>=0.05 over old directional; at least 2/3 seed gains>=0.05 and no seed regresses; structural precision>=0.98, false accept<=0.01, recall>=0.40.",
        "identity": "junction/terminal>=0.90, turn>=0.40, corrected change-point>=7/17 identities; report exact count and Wilson 95% interval.",
        "per_family": "All ten families retain non-empty overall structural acceptance with precision>=0.95 and recall>=0.10. Change-point selection evidence is explicitly limited to the four families present; no all-family claim is permitted.",
        "nonvacuous": True,
    }
    card["acceptance"].update(
        {
            "baseline": "Old directional ensemble recomputed on corrected labels: macro-F1=0.6879041032, change-point identity coverage=0/17.",
            "event_macro_f1_absolute_min": 0.7379041031973032,
            "geometry_transition_correct_class_identity_coverage_min": 7 / 17,
            "geometry_transition_correct_class_identities_min": 7,
            "selection_geometry_transition_identities": 17,
            "seed_event_macro_f1_gain_min": 0.05,
            "seeds_reaching_gain_min": 2,
            "no_seed_regression": True,
            "final_generalization_claim": False,
        }
    )
    card["split"].update(
        {
            "fit": "C01-C06: 60 worlds, 142184 sequences, 59 corrected change-point identities. Only three new event heads update.",
            "checkpoint_and_threshold_selection": "C07-C08: 20 worlds, 45942 sequences, 17 corrected change-point identities across S01/S08/S09/S10. Select epochs and one non-vacuous threshold only here.",
            "future_validation": "C09 remains unread in this operation. Because historical C09 diagnostics motivated the Teacher audit, it cannot repair or select this model; final generalization requires a separately frozen evaluation and untouched C10.",
        }
    )
    card["evidence"].update(
        {
            "machine_metrics": "Exact corrected-label capacity; recomputed frozen and old-directional baselines; per-seed checkpoints/epochs; confusion/F1; threshold precision/false/recall; per-family metrics; identity counts/coverage and Wilson interval; gain checks; source seals; zero backbone/test access; log, RUN_STATE and full seal.",
            "failure_policy": "Any source/count/split/label/index drift, backbone gradient/update, C09/C10/M-TARE read, nonfinite/resource failure or unmet scientific gate seals FAIL. No retry, label change, threshold relaxation, family overclaim or graph tuning.",
        }
    )
    write_json(CARD_PATH, card)

    inputs = [
        f"{VERIFIER}/RUN_STATE.json",
        f"{VERIFIER}/metrics/summary.json",
        f"{VERIFIER}/artifacts/evidence_sha256.txt",
        f"{VERIFIER}/artifacts/pair_cache/pairs.npz",
        f"{DATASET}/RUN_STATE.json",
        f"{DATASET}/metrics/summary.json",
        f"{DATASET}/artifacts/evidence_sha256.txt",
        f"{DATASET}/artifacts/sequence_manifest.jsonl",
        f"{TRAINING}/RUN_STATE.json",
        f"{TRAINING}/metrics/summary.json",
        f"{TRAINING}/artifacts/evidence_sha256.txt",
        f"{TEACHER}/RUN_STATE.json",
        f"{TEACHER}/metrics/summary.json",
        f"{TEACHER}/artifacts/evidence_sha256.txt",
        f"{TEACHER}/artifacts/teacher_observations.jsonl",
        f"{OLD_DIRECTIONAL}/RUN_STATE.json",
        f"{OLD_DIRECTIONAL}/metrics/summary.json",
        f"{OLD_DIRECTIONAL}/artifacts/evidence_sha256.txt",
        f"{OLD_DIRECTIONAL}/artifacts/training/ensemble_selection_outputs.npz",
    ]
    for seed in (0, 1, 2):
        inputs.extend(
            [
                f"{VERIFIER}/artifacts/models/seed{seed}/frozen_observation_features.npy",
                f"{TRAINING}/artifacts/models/seed{seed}/best.pt",
                f"{OLD_DIRECTIONAL}/artifacts/training/seed{seed}/selection_outputs.npz",
            ]
        )
    tools = {
        "data_card": str(CARD_PATH.relative_to(PROJECT_ROOT)),
        "backbone_model": "src/mtare_topo/representation/gse_graph.py",
        "directional_head": "src/mtare_topo/representation/gse_directional_structural_event.py",
        "corrected_event_contract": "src/mtare_topo/representation/gse_corrected_causal_event.py",
        "trainer": "tools/v3/train_gse_corrected_causal_event_v1.py",
        "trainer_base": "tools/v3/train_gse_directional_structural_event_v1.py",
        "runner": "tools/v3/run_gse_corrected_causal_event_training_v1.py",
        "runner_base": "tools/v3/run_gse_directional_structural_event_training_v1.py",
        "dataset_reader": "src/mtare_topo/data/gse_training_dataset.py",
        "selector": "src/mtare_topo/representation/gse_open_set_association.py",
        "rare_event_metrics": "src/mtare_topo/representation/gse_rare_event_corrective.py",
        "evidence_integrity": "src/mtare_topo/evaluation/gse_evidence_integrity.py",
        "governance": "src/mtare_topo/governance.py",
        "preflight": "tools/v3/preflight.py",
        "create_run": "tools/v3/create_run.py",
        "freezer": "tools/v3/freeze_gse_corrected_causal_event_training_spec_v1.py",
    }
    run_dir = PROJECT_ROOT / "results/gate3_semantics" / RUN_ID
    spec = {
        "schema_version": "v3_run_spec_v1",
        "slug": "gse_corrected_causal_event_training_v1",
        "date": "20260827",
        "gate": 3,
        "execution_phase": 3,
        "operation": "training",
        "question": "Can identity-balanced five-frame directional evidence learn the corrected persistent causal change-points while preserving the one-percent open-set node contract?",
        "method": card["methods"]["main"],
        "baseline": card["methods"]["baseline"],
        "fallback": card["methods"]["fallback"],
        "data_card": str(CARD_PATH.relative_to(PROJECT_ROOT)),
        "config_path": str(CARD_PATH.relative_to(PROJECT_ROOT)),
        "seed": 0,
        "working_directory": str(PROJECT_ROOT),
        "estimated_cost": card["estimated_cost"],
        "expected_counts": {
            "fit_worlds": 60,
            "fit_observations": 142184,
            "selection_worlds": 20,
            "selection_observations": 45942,
            "fit_change_point_identities": 59,
            "selection_change_point_identities": 17,
            "seeds": 3,
            "optimizer_steps": 36000,
            "backbone_optimizer_steps": 0,
            "c09_worlds_read": 0,
            "strict_test_worlds_read": 0,
            "mtare_worlds_read": 0,
        },
        "hyperparameters": {
            "epochs": 12,
            "draws_per_epoch": 24000,
            "batch_size": 24,
            "evaluation_batch_size": 48,
            "evaluate_every": 3,
            "learning_rate": 0.001,
            "weight_decay": 0.0001,
            "optimizer": "AdamW",
            "gradient_clip": 5.0,
            "seeds": [0, 1, 2],
        },
        "acceptance_criteria": [
            "Exact corrected fit/selection event counts and identity counts; 59/17 change-point identities, with selection family support explicitly S01/S08/S09/S10.",
            "C07-C08 ensemble macro-F1>=0.7379041032 and gain>=0.05 over old directional corrected-label baseline; >=2/3 seed gains>=0.05 and no seed regresses.",
            "One non-vacuous threshold has precision>=0.98, false accept<=0.01, recall>=0.40; all ten families retain overall nonempty acceptance with precision>=0.95/recall>=0.10.",
            "Correct-class identity coverage junction/terminal>=0.90, turn>=0.40 and change-point>=7/17; report Wilson 95% interval without all-family claim.",
            "Exactly 36000 head optimizer steps, zero backbone update and zero C09/C10/M-TARE read; complete immutable evidence and seal.",
        ],
        "expected_evidence": [
            "Three checkpoints, 36 epoch records, sampling contract, corrected-label frozen/old-directional baselines, ensemble outputs, event/identity/open-set/gain metrics, capacity limitation, environment, raw log, RUN_STATE and SHA-256 seal."
        ],
        "frozen_inputs": {path: _sha(PROJECT_ROOT / path) for path in inputs},
        "frozen_tools": {name: _record(path) for name, path in tools.items()},
        "command": [
            "/usr/bin/systemd-inhibit",
            "--what=sleep:shutdown",
            "--why=GSE corrected causal event training",
            "--mode=block",
            "/usr/bin/timeout",
            "--signal=INT",
            "--kill-after=300s",
            "33000s",
            "/home/zeng-workstation/.local/share/mtare_topo_comm/envs/phase3_torch290_cu129_zarr2187_v1/bin/python",
            "tools/v3/run_gse_corrected_causal_event_training_v1.py",
            "--spec",
            str(SPEC_PATH),
            "--run-dir",
            str(run_dir),
        ],
        "user_authorization": {
            "status": "APPROVED",
            "approved_by": "user",
            "approved_at": "2026-08-27T13:30:00+08:00",
            "scope": card["approval"]["scope"],
            "confirmation_reference": card["approval"]["confirmation_reference"],
        },
    }
    write_json(SPEC_PATH, spec)
    print(CARD_PATH)
    print(SPEC_PATH)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
