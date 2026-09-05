#!/usr/bin/env python3
"""Freeze the operation-bound Data Card and one-run slope corrective spec."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import zarr

from _bootstrap import PROJECT_ROOT
from mtare_topo.data.gse_slope_corrective_dataset import (
    APPROVED_CORRECTIVE_PARENTS,
    EXPECTED_FIT_SEQUENCES,
    EXPECTED_FIT_WORLDS,
    EXPECTED_SELECTION_SEQUENCES,
    EXPECTED_SELECTION_WORLDS,
    corrective_partition,
)
from mtare_topo.governance import load_json, validate_data_card, write_json


SOURCE_CARD = PROJECT_ROOT / "configs/v3/gate2/data_cards/gse_graph_three_seed_training_v1.json"
CARD = PROJECT_ROOT / "configs/v3/gate2/data_cards/gse_slope_corrective_three_seed_training_v1.json"
SPEC = PROJECT_ROOT / "configs/v3/gate2/gse_slope_corrective_three_seed_training_v1.json"
DATASET = PROJECT_ROOT / "results/gate2_representation/gate2_20260824_gse_deduplicated_dataset_export_v1_seed0"
TRAINING = PROJECT_ROOT / "results/gate2_representation/gate2_20260824_gse_graph_three_seed_training_v1r_seed0"
FAILED_C09 = PROJECT_ROOT / "results/gate3_semantics/gate3_20260824_gse_perception_validation_v1_seed0"
RUN_ID = "gate2_20260824_gse_slope_corrective_three_seed_training_v1_seed0"
PYTHON = "/home/zeng-workstation/.local/share/mtare_topo_comm/envs/phase3_torch290_cu129_zarr2187_v1/bin/python"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _record(relative: str) -> dict[str, str]:
    return {"path": relative, "sha256": _sha256(PROJECT_ROOT / relative)}


def _train_only_inventory() -> dict:
    root = DATASET / "artifacts/dataset/train"
    paths = sorted(root.glob("*.zarr"))
    if {path.stem for path in paths} != set(APPROVED_CORRECTIVE_PARENTS):
        raise RuntimeError("corrective Data Card inventory world set drift")
    result = {
        "worlds": 0,
        "unique_frames": 0,
        "causal_sequences": 0,
        "directed_traversals": 0,
        "structure_event_counts": {name: 0 for name in ("corridor", "junction", "terminal", "turn", "geometry_transition")},
        "partition": {
            "fit": {"worlds": 0, "unique_frames": 0, "causal_sequences": 0},
            "selection": {"worlds": 0, "unique_frames": 0, "causal_sequences": 0},
        },
    }
    source_trajectories = {
        str(item["world"]): item
        for item in load_json(SOURCE_CARD)["trajectories"]
        if str(item.get("world")) in APPROVED_CORRECTIVE_PARENTS
    }
    if set(source_trajectories) != set(APPROVED_CORRECTIVE_PARENTS):
        raise RuntimeError("corrective traversal inventory world set drift")
    for path in paths:
        group = zarr.open_group(str(path), mode="r")
        parent = path.stem
        partition = corrective_partition(parent)
        frames = int(group["range_m"].shape[0])
        sequences = int(group["local_frame_references"].shape[0])
        events = np.asarray(group["event_index"][:], dtype=np.int64)
        counts = np.bincount(events, minlength=5)
        result["worlds"] += 1
        result["unique_frames"] += frames
        result["causal_sequences"] += sequences
        traversals = int(source_trajectories[parent].get("directed_traversal_count", 0))
        if traversals <= 0:
            raise RuntimeError(f"corrective directed traversal count invalid: {parent}")
        result["directed_traversals"] += traversals
        result["partition"][partition]["worlds"] += 1
        result["partition"][partition]["unique_frames"] += frames
        result["partition"][partition]["causal_sequences"] += sequences
        result["partition"][partition].setdefault("directed_traversals", 0)
        result["partition"][partition]["directed_traversals"] += traversals
        for index, name in enumerate(result["structure_event_counts"]):
            result["structure_event_counts"][name] += int(counts[index])
    if (
        result["worlds"] != 80
        or result["unique_frames"] != 252430
        or result["causal_sequences"] != 188126
        or result["directed_traversals"] != 16078
        or result["partition"]["fit"]["worlds"] != EXPECTED_FIT_WORLDS
        or result["partition"]["selection"]["worlds"] != EXPECTED_SELECTION_WORLDS
        or result["partition"]["fit"]["causal_sequences"] != EXPECTED_FIT_SEQUENCES
        or result["partition"]["selection"]["causal_sequences"] != EXPECTED_SELECTION_SEQUENCES
        or result["partition"]["fit"].get("directed_traversals") != 12106
        or result["partition"]["selection"].get("directed_traversals") != 3972
    ):
        raise RuntimeError(f"corrective Data Card inventory count drift: {result}")
    return result


def freeze() -> dict:
    if CARD.exists() or SPEC.exists():
        raise RuntimeError("slope corrective Data Card/spec already exists; refusing overwrite")
    source = load_json(SOURCE_CARD)
    inventory = _train_only_inventory()
    fit_worlds = sorted(parent for parent in APPROVED_CORRECTIVE_PARENTS if corrective_partition(parent) == "fit")
    selection_worlds = sorted(parent for parent in APPROVED_CORRECTIVE_PARENTS if corrective_partition(parent) == "selection")
    trajectory_by_world = {
        str(item["world"]): dict(item)
        for item in source["trajectories"]
        if str(item.get("world")) in APPROVED_CORRECTIVE_PARENTS
    }
    if set(trajectory_by_world) != set(APPROVED_CORRECTIVE_PARENTS):
        raise RuntimeError("source trajectory evidence does not cover corrective worlds")
    trajectories = []
    for parent in fit_worlds + selection_worlds:
        item = trajectory_by_world[parent]
        item["split"] = "train" if parent in fit_worlds else "validation"
        trajectories.append(item)
    strict = sorted(
        [f"{parent.rsplit('_C', 1)[0]}_C09" for parent in selection_worlds[::2]]
        + [f"{parent.rsplit('_C', 1)[0]}_C10" for parent in selection_worlds[::2]]
        + ["SEALED_MTARE_FINAL_WORLDS"]
    )
    card = {
        "schema_version": "v3_data_card_v1",
        "card_id": "gse_slope_corrective_three_seed_training_v1",
        "status": "APPROVED_FOR_ONE_IMMUTABLE_SLOPE_CORRECTIVE_TRAINING",
        "purpose": "Fit a bounded temporal residual on five deterministic causal LiDAR geometry estimates to correct only the failed GSE slope output, while freezing every successful GSE event, axis, width, height, curvature, place and exit output.",
        "approval": {
            "status": "APPROVED",
            "approved_by": "user",
            "approved_at": "2026-08-24T08:30:00+08:00",
            "scope": "One immutable Gate-2 corrective cache plus seeds 0/1/2 training run over exactly C01-C06 fit and C07-C08 internal selection worlds. C09/C10/M-TARE remain unread.",
            "authorized_operations": ["training", "normalization", "checkpoint_selection"],
            "authorized_gates": [2],
            "confirmation_reference": "The user granted standing authorization for the fixed GSE-Graph paper plan, requested continuous execution without repeated approval prompts, and explicitly required retention of paper-worthy figures.",
        },
        "source": {
            "raw_sources": [
                "Sealed PASS deduplicated GSE dataset V1; this operation opens only the 80 dataset/train Zarr shards for C01-C08.",
                "Sealed three-seed GSE V1R training PASS; all successful outputs remain frozen and are not read by the corrective learner.",
                "Sealed C09 perception FAIL identifying slope collapse; used only as pre-run rationale and never as corrective training or selection data.",
            ],
            "license_or_allowed_use": source["source"]["license_or_allowed_use"],
        },
        "worlds": {
            "train": fit_worlds,
            "validation": selection_worlds,
            "ssl": [],
            "normalization": fit_worlds,
            "teacher_calibration": [],
            "threshold_calibration": [],
            "augmentation_tuning": [],
            "checkpoint_selection": selection_worlds,
            "strict_test": strict,
        },
        "trajectories": trajectories,
        "sampling": {
            "raw_frame_count": inventory["unique_frames"],
            "effective_sample_count": inventory["causal_sequences"],
            "effective_structure_event_count": sum(inventory["structure_event_counts"].values()),
            "spatial_interval_m": 1.0,
            "spatial_interval_interpretation": "Each directed traversal contributes five strictly increasing causal frame references at the existing one-metre route spacing. Parent worlds, not overlapping sequences, are the independent units.",
            "structure_event_counts": inventory["structure_event_counts"],
            "rule": "Compute six deterministic geometry features for every unique C01-C08 train frame once. Fit normalization on C01-C06 only. Train on 142184 C01-C06 sequences; select by corrected slope MAE on 45942 C07-C08 sequences. No C09 manifest, shard or target is opened.",
            "exact_counts": inventory,
        },
        "teacher": {
            "source": "The existing sealed spline teacher geometry column order is width_m, height_m, slope_deg, curvature_per_m. Only slope_deg with its true validity mask is consumed; the five-frame physics prior comes from the frozen deterministic range geometry estimator.",
            "valid_mask": "Every used slope target must have geometry_valid_mask[:,2]==1, remain finite and lie within +/-45 degrees. Five local references must be strictly increasing, in bounds and exactly match frozen global frame references.",
            "planner_consistency_plan": "The corrective receives only six quantities computed independently from each of five causal scans. It receives no pose, world ID, route, TNG identity, future frame, C09/C10 or M-TARE information. Deployment replaces only slope_deg and adds slope error scale.",
        },
        "split": {
            "world_disjoint": True,
            "trajectory_disjoint": True,
            "historical_pollution_audit": "The original GSE backbone saw C01-C08, so its learned features are forbidden here. The corrective uses only deterministic scan geometry; C01-C06 and C07-C08 are disjoint module-level fit/selection worlds. C09 is reserved for one later complete corrective validation and C10/M-TARE remain strict test.",
        },
        "leakage_audit": {
            "test_excluded_from_supervised_training": True,
            "test_excluded_from_ssl": True,
            "test_excluded_from_normalization": True,
            "test_excluded_from_teacher_calibration": True,
            "test_excluded_from_threshold_calibration": True,
            "test_excluded_from_augmentation_tuning": True,
            "test_excluded_from_checkpoint_selection": True,
        },
        "estimated_cost": {
            "disk_gb": 2.0,
            "wall_time_hours": 2.0,
            "compute": "One RTX 5090 D; train-only CPU feature cache then three serial FP32 GRU residual seeds, batch 1024, at most 50 epochs with patience 8. Host RSS <=8 GiB and GPU allocation <=2 GiB.",
        },
        "evidence": {
            "machine_metrics": "Three best/last checkpoints, complete C07-C08 outputs, current-frame/five-frame/corrected MAE, per-world and per-family results, optimizer/resource counts, actual source paths and SHA-256 seal.",
            "complete_visual_review": "A paper bundle is published only after a later complete C09 PASS; this train-only run preserves source arrays needed to generate the learning and ablation plots.",
            "failure_policy": "Stop before C09 if mean improvement over the five-frame analytic prior is below 5%, fewer than two seeds improve by 5%, any seed regresses overall, any topology family regresses more than 5%, or any identity/index/seal/resource/leakage check fails.",
        },
        "training_contract": {
            "seeds": [0, 1, 2],
            "model": "Six-feature GRU(32) plus bounded +/-10 degree residual and positive error-scale head; final layer zero initialized so epoch 0 exactly equals the five-frame physics prior.",
            "optimizer": "AdamW",
            "learning_rate": 0.001,
            "weight_decay": 0.0001,
            "maximum_epochs": 50,
            "early_stopping_patience": 8,
            "batch_size": 1024,
            "gradient_clip_norm": 5.0,
            "normalization": "Mean/std from the 142184 C01-C06 fit sequences only.",
            "checkpoint_selection": "Minimum full C07-C08 corrected slope MAE independently per seed; epoch 0 analytic prior is an eligible checkpoint, but the aggregate run passes only with learned >=5% improvement.",
        },
    }
    report = validate_data_card(card)
    if not report.passed:
        raise RuntimeError(f"generated slope corrective Data Card invalid: {report.errors}")
    CARD.parent.mkdir(parents=True, exist_ok=True)
    write_json(CARD, card)
    tools = {
        "data_card": str(CARD.relative_to(PROJECT_ROOT)),
        "runner": "tools/v3/run_gse_slope_corrective_three_seed_training_v1.py",
        "cache_builder": "tools/v3/build_gse_slope_corrective_cache_v1.py",
        "seed_trainer": "tools/v3/train_gse_slope_corrective_v1.py",
        "corrective_dataset": "src/mtare_topo/data/gse_slope_corrective_dataset.py",
        "corrective_model": "src/mtare_topo/representation/gse_slope_corrective.py",
        "range_geometry_baseline": "src/mtare_topo/semantics/range_geometry_baseline.py",
        "sensor_contract": "src/mtare_topo/data/cano_sensor_smoke.py",
        "governance": "src/mtare_topo/governance.py",
        "preflight": "tools/v3/preflight.py",
    }
    input_paths = [
        DATASET / "RUN_STATE.json",
        DATASET / "metrics/summary.json",
        DATASET / "artifacts/evidence_sha256.txt",
        TRAINING / "RUN_STATE.json",
        TRAINING / "metrics/summary.json",
        TRAINING / "artifacts/evidence_sha256.txt",
        FAILED_C09 / "RUN_STATE.json",
        FAILED_C09 / "metrics/summary.json",
        FAILED_C09 / "artifacts/evidence_sha256.txt",
    ]
    if not all(path.is_file() for path in input_paths):
        raise RuntimeError("corrective predecessor evidence is incomplete")
    spec = {
        "schema_version": "v3_run_spec_v1",
        "gate": 2,
        "execution_phase": 3,
        "date": "20260824",
        "slug": "gse_slope_corrective_three_seed_training_v1",
        "operation": "training",
        "question": "Can a train-only learned temporal residual correct the collapsed GSE slope field by at least 5% beyond a five-frame analytic geometry prior without any topology-family regression or C09/C10/M-TARE access?",
        "method": "Build six deterministic features independently from each of five causal scans, average their slopes as the physics prior, and train a zero-initialized bounded GRU residual plus error-scale head on C01-C06. Select checkpoints on C07-C08 corrected slope MAE only; every other GSE output remains frozen.",
        "baseline": "The primary ablation is the identical five-frame analytic slope mean with no learned residual; the secondary diagnostic is the current fifth-frame analytic estimator. The failed original GSE slope head remains immutable failure evidence.",
        "user_authorization": card["approval"],
        "seed": 0,
        "data_card": str(CARD.relative_to(PROJECT_ROOT)),
        "config_path": str(CARD.relative_to(PROJECT_ROOT)),
        "working_directory": str(PROJECT_ROOT),
        "command": [
            "/usr/bin/systemd-inhibit",
            "--what=sleep:shutdown",
            "--why=GSE slope corrective three-seed training",
            "--mode=block",
            "/usr/bin/timeout",
            "--signal=INT",
            "--kill-after=300s",
            "7500s",
            PYTHON,
            "tools/v3/run_gse_slope_corrective_three_seed_training_v1.py",
            "--spec",
            str(SPEC),
            "--run-dir",
            str(PROJECT_ROOT / "results/gate2_representation" / RUN_ID),
        ],
        "acceptance_criteria": [
            "Exactly 60 C01-C06 fit worlds/142184 sequences and 20 C07-C08 selection worlds/45942 sequences are read from the 80 train shards; normalization uses fit only and C09/C10/M-TARE reads are zero.",
            "All three seeds produce finite best/last checkpoints and complete selection outputs; every selected seed is nonregressing against the five-frame analytic prior.",
            "Mean three-seed relative improvement over the five-frame prior is at least 5%, at least two seeds individually improve at least 5%, and no topology family regresses more than 5%.",
            "The source dataset seal verifies before and after, host RSS stays <=8 GiB, GPU allocation <=2 GiB, output <=2 GiB and wall time <=2 hours.",
        ],
        "stop_conditions": [
            "Any world, schema, five-frame causal reference, target column/mask, source seal, tool hash, environment, seed, sample count or selection rule drift.",
            "Any C09/C10/M-TARE access, use of frozen GSE backbone features, nonfinite value, resource overrun, family regression beyond 5%, or aggregate learned gain below the fixed gate.",
        ],
        "expected_evidence": [
            "Train-only cache manifest with all actual source paths, exact indices, fit-only normalization and 80 per-world NPZ files.",
            "Three seed histories, best/last checkpoints, complete C07-C08 outputs and per-world/per-family baseline-versus-corrected metrics.",
            "Aggregate gate summary, environment, raw logs, RUN_STATE and exact SHA-256 evidence seal; no paper PASS plot until later complete C09 PASS.",
        ],
        "estimated_cost": card["estimated_cost"],
        "frozen_tools": {name: _record(relative) for name, relative in tools.items()},
        "frozen_inputs": {str(path.relative_to(PROJECT_ROOT)): _sha256(path) for path in input_paths},
        "expected_counts": {
            "fit_worlds": 60,
            "selection_worlds": 20,
            "fit_sequences": 142184,
            "selection_sequences": 45942,
            "unique_frames": 252430,
            "seeds": 3,
            "c09_worlds_read": 0,
            "strict_test_worlds_read": 0,
            "mtare_worlds_read": 0,
        },
    }
    write_json(SPEC, spec)
    return {
        "data_card": str(CARD.relative_to(PROJECT_ROOT)),
        "data_card_sha256": _sha256(CARD),
        "spec": str(SPEC.relative_to(PROJECT_ROOT)),
        "spec_sha256": _sha256(SPEC),
        "inventory": inventory,
    }


if __name__ == "__main__":
    print(freeze())
