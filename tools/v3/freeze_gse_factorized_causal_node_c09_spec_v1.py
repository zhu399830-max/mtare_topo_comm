#!/usr/bin/env python3
"""Freeze the Data Card and run spec for C09 causal decision-node qualification."""

from __future__ import annotations

from copy import deepcopy
import hashlib

from _bootstrap import PROJECT_ROOT
from mtare_topo.governance import load_json, write_json


RUN_ID = "gate3_20260828_gse_factorized_causal_node_c09_v1_seed0"
CARD = PROJECT_ROOT / "configs/v3/gate3/data_cards/gse_factorized_causal_node_c09_v1.json"
SPEC = PROJECT_ROOT / "configs/v3/gate3/gse_factorized_causal_node_c09_v1.json"
SOURCE_CARD = PROJECT_ROOT / "configs/v3/gate3/data_cards/gse_factorized_association_c09_qualification_v1.json"
DATASET = "results/gate2_representation/gate2_20260824_gse_deduplicated_dataset_export_v1_seed0"
TEACHER = "results/gate2_representation/gate2_20260824_gse_teacher_manifest_v1_seed0"
TRAINING = "results/gate2_representation/gate2_20260824_gse_graph_three_seed_training_v1r_seed0"
EPISODE = "results/gate3_semantics/gate3_20260827_gse_causal_episode_training_v1r_seed0"
PYTHON = "/home/zeng-workstation/.local/share/mtare_topo_comm/envs/phase3_torch290_cu129_zarr2187_v1/bin/python"


def _sha(path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    card = deepcopy(load_json(SOURCE_CARD))
    card.update({
        "card_id": "gse_factorized_causal_node_c09_v1",
        "status": "APPROVED_FOR_ONE_IMMUTABLE_GSE_FACTORIZED_CAUSAL_NODE_C09_V1",
        "purpose": "Qualify the already frozen 12-frame past-only junction/terminal trigger as the Factorized GSE-Graph decision-node generator on every C09 world.",
    })
    card["approval"] = {
        "status": "APPROVED", "approved_by": "user-standing-authorization",
        "approved_at": "2026-08-28T10:00:00+08:00", "authorized_gates": [3],
        "authorized_operations": ["audit"],
        "scope": "One immutable two-process GPU C09 inference/evaluation. Process 1 has no Teacher argument; process 2 evaluates frozen outputs. No threshold/model update, C10, M-TARE, graph or planner access.",
        "confirmation_reference": "User explicitly instructed automatic best-choice execution without routine approval prompts.",
    }
    card["sampling"] = {
        "raw_frame_count": 32_678,
        "effective_sample_count": 24_462,
        "effective_structure_event_count": 540,
        "spatial_interval_m": 1.0,
        "rule": "All 24462 observations from all directed C09 traversals; each model input uses the current scan plus up to eleven strictly past scans from the same traversal.",
        "independent_units": "Ten unseen procedural C09 topology worlds; structural episode and decision identity are reporting units.",
        "temporal_context": "Exactly 12 left-padded causal scan slots, minimum 5 and maximum 12 valid scans; 237678 valid reference cells and zero future cells.",
        "directed_traversals": 2_054,
        "true_decision_episodes": {"junction": 426, "terminal": 114},
        "true_decision_identities": {"junction": 71, "terminal": 59},
    }
    card["source"] = {
        "dataset_run": DATASET, "teacher_run": TEACHER,
        "base_training_run": TRAINING, "causal_episode_component_run": EPISODE,
        "raw_sources": [
            "Ten sealed C09 validation Zarr shards with 32678 unique LiDAR frames and a Teacher-free sequence/frame manifest.",
            "Three frozen GSE spatial encoders and their aligned C09 baseline event logits.",
            "Three frozen 12-frame causal episode checkpoints and the C07-C08-selected threshold 0.986.",
            "C09 Teacher rows opened only by the second evaluation process.",
        ],
        "license_or_allowed_use": "Local research use with sealed provenance.",
    }
    card["split"] = {
        "model_fit": "C01-C06 only in the sealed upstream runs; no fit occurs here.",
        "selection": "C07-C08 only selected checkpoints and structural threshold 0.986 upstream.",
        "validation": "Exactly all ten C09 worlds and all 24462 observations; no world, event, identity or frame selection.",
        "strict_test": "C10 and all M-TARE/closed-loop worlds remain unread.",
        "world_disjoint": True, "trajectory_disjoint": True,
        "historical_pollution_audit": "C09 was previously used for frozen perception and association validation, but never for causal-episode checkpoint or threshold selection. The decision-node-only reuse of an aggregate failed run is disclosed: junction/terminal selection precision/recall passed, while turn/transition caused the aggregate failure. C09 does not choose this restriction or any value.",
        "leakage_audit": "The runner enforces process separation. Inference receives dataset, base models and causal episode models only; it has no Teacher path. Teacher event/episode/identity are first opened after ensemble outputs and triggers exist.",
    }
    card["teacher"] = {
        "source": "Sealed TNG/spline/mesh C09 Teacher opened only for post-inference evaluation.",
        "event": "Contiguous junction and terminal Teacher rows form 426 and 114 decision episodes.",
        "association": "A trigger is correct only if its row belongs to a previously unmatched decision episode and its predicted event class is correct.",
        "valid_mask": "All C09 rows are retained. Only junction/terminal episodes enter decision-node recall; any predicted junction/terminal on corridor, turn or transition is a false trigger.",
        "student_forbidden_fields": ["event label", "episode id", "place identity", "world ID as classifier input", "absolute position", "future frames"],
        "planner_consistency_plan": "Only frozen junction/terminal triggers may request a structural node. Turn and geometry-transition outputs remain edge attributes and cannot create nodes.",
    }
    card["methods"] = {
        "main": "For each seed, encode all C09 LiDAR scans with its frozen spatial encoder, aggregate a 12-frame past-only history with its frozen causal GRU, average three event distributions, collapse contiguous responses at threshold 0.986, and retain only junction/terminal triggers.",
        "baseline": "The frozen five-frame GSE event output that produced unsafe raw graph candidates is the direct failure baseline; C07-C08 decision-trigger performance is the generalization reference.",
        "fallback": "If any pre-registered C09 decision-node gate fails, stop this 12-frame route and implement a separately selected five-frame decision gate on C07-C08; do not tune graph parameters or threshold on C09.",
    }
    card["metrics_and_pre_registered_gates"] = {
        "aggregate": "Nonvacuous decision triggers, class-correct precision>=0.98, false trigger fraction<=0.01 and episode recall>=0.60.",
        "junction": "Precision>=0.98, episode recall>=0.50 and decision-identity coverage>=0.80.",
        "terminal": "Precision>=0.98, episode recall>=0.75 and decision-identity coverage>=0.80.",
        "population": "Exactly 10 worlds, 32678 unique frames, 24462 observations, 2054 traversals, 237678 valid past-reference cells, 426 junction episodes/71 identities and 114 terminal episodes/59 identities.",
        "separation": "Inference Teacher/event/episode/identity reads and future-reference cells are exactly zero; evaluation begins only after frozen output materialization.",
        "forbidden": "Threshold/checkpoint selection, optimizer/model update, C10, strict test, M-TARE, graph and planner operations all zero.",
    }
    card["estimated_cost"] = {
        "compute": "GPU inference only: 98034 spatial-encoder frames plus 73386 causal-episode observations",
        "disk_gb": .25, "host_ram_gb": 8, "gpu_memory_gb": 8,
        "wall_time_hours": .5,
    }
    card["retention"] = "Retain three seed outputs, ensemble outputs, trigger rows, past-reference proof, metrics, PNG/PDF/SVG/source, environment, commands, logs, RUN_STATE and exact SHA-256 seal."
    card["failure_policy"] = "Any source/environment/process-separation/population/resource drift or pre-registered C09 gate failure seals FAIL. No retry, threshold change, frame masking, graph tuning, C10 read or planner compensation."
    card["evidence"] = {
        "machine_metrics": "Per-event decision trigger precision/recall/F1, identity coverage and exact input/inference counts.",
        "complete_visual_review": "C07-C08 selection versus C09 precision/recall in PNG/PDF/SVG with machine source.",
        "failure_policy": card["failure_policy"],
    }
    write_json(CARD, card)

    inputs = [
        f"{DATASET}/RUN_STATE.json", f"{DATASET}/metrics/summary.json",
        f"{DATASET}/artifacts/evidence_sha256.txt", f"{DATASET}/artifacts/sequence_manifest.jsonl",
        f"{DATASET}/artifacts/frame_manifest.jsonl", f"{DATASET}/artifacts/shard_manifest.json",
        f"{TEACHER}/RUN_STATE.json", f"{TEACHER}/metrics/summary.json",
        f"{TEACHER}/artifacts/evidence_sha256.txt", f"{TEACHER}/artifacts/teacher_observations.jsonl",
        f"{TRAINING}/RUN_STATE.json", f"{TRAINING}/metrics/summary.json",
        f"{TRAINING}/artifacts/evidence_sha256.txt",
        f"{EPISODE}/RUN_STATE.json", f"{EPISODE}/metrics/summary.json",
        f"{EPISODE}/metrics/ensemble/summary.json", f"{EPISODE}/artifacts/evidence_sha256.txt",
    ]
    for seed in range(3):
        inputs.extend([
            f"{TRAINING}/artifacts/models/seed{seed}/best.pt",
            f"{TRAINING}/artifacts/models/seed{seed}/validation_outputs.npz",
            f"{EPISODE}/artifacts/models/seed{seed}/best.pt",
        ])
    tools = {
        "data_card": str(CARD.relative_to(PROJECT_ROOT)),
        "method_plan": "docs/GSE_GRAPH_FACTORIZED_RESEARCH_PLAN_V2.md",
        "causal_detector": "src/mtare_topo/representation/gse_causal_episode_detector.py",
        "causal_cache": "src/mtare_topo/data/gse_causal_episode_cache.py",
        "causal_metrics": "src/mtare_topo/evaluation/gse_causal_episode_metrics.py",
        "factorized_graph": "src/mtare_topo/topology/factorized_gse_graph.py",
        "inference": "tools/v3/infer_gse_factorized_causal_nodes_c09_v1.py",
        "evaluation": "tools/v3/evaluate_gse_factorized_causal_nodes_c09_v1.py",
        "runner": "tools/v3/run_gse_factorized_causal_node_c09_v1.py",
        "freezer": "tools/v3/freeze_gse_factorized_causal_node_c09_spec_v1.py",
        "tests_detector": "tests/v3/unit/test_gse_causal_episode_detector.py",
        "tests_metrics": "tests/v3/unit/test_gse_causal_episode_metrics.py",
        "tests_graph": "tests/v3/unit/test_factorized_gse_graph.py",
        "governance": "src/mtare_topo/governance.py",
        "preflight": "tools/v3/preflight.py", "create_run": "tools/v3/create_run.py",
    }
    criteria = card["metrics_and_pre_registered_gates"]
    spec = {
        "schema_version": "v3_run_spec_v1", "gate": 3, "execution_phase": 3,
        "date": "20260828", "slug": "gse_factorized_causal_node_c09_v1", "seed": 0,
        "operation": "audit",
        "question": "Do frozen past-only causal episodes generate safe junction/terminal nodes on all C09 worlds?",
        "method": card["methods"]["main"], "baseline": card["methods"]["baseline"],
        "fallback": card["methods"]["fallback"],
        "data_card": str(CARD.relative_to(PROJECT_ROOT)),
        "config_path": str(CARD.relative_to(PROJECT_ROOT)),
        "user_authorization": deepcopy(card["approval"]),
        "acceptance_criteria": [
            criteria[name] for name in (
                "aggregate", "junction", "terminal", "population", "separation", "forbidden"
            )
        ],
        "expected_counts": {
            "validation_worlds": 10, "unique_lidar_frames": 32_678,
            "causal_observations": 24_462, "directed_traversals": 2_054,
            "valid_past_reference_cells": 237_678,
            "spatial_encoder_inference_frames": 98_034,
            "episode_detector_inference_observations": 73_386,
            "junction_episodes": 426, "terminal_episodes": 114,
            "optimizer_steps": 0, "model_updates": 0, "threshold_selection_steps": 0,
            "c10_worlds_read": 0, "mtare_worlds_read": 0,
        },
        "expected_evidence": [card["retention"]],
        "estimated_cost": card["estimated_cost"],
        "frozen_inputs": {path: _sha(PROJECT_ROOT / path) for path in inputs},
        "frozen_tools": {
            name: {"path": path, "sha256": _sha(PROJECT_ROOT / path)}
            for name, path in tools.items()
        },
        "working_directory": str(PROJECT_ROOT),
        "command": [
            "/usr/bin/timeout", "--signal=INT", "--kill-after=30s", "3800s", PYTHON,
            "tools/v3/run_gse_factorized_causal_node_c09_v1.py",
            "--spec", str(SPEC), "--run-dir",
            str(PROJECT_ROOT / "results/gate3_semantics" / RUN_ID),
        ],
    }
    write_json(SPEC, spec)
    print(CARD.relative_to(PROJECT_ROOT))
    print(SPEC.relative_to(PROJECT_ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
