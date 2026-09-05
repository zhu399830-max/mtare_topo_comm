#!/usr/bin/env python3
"""Run the predeclared GSE and baseline validation topology replays."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import time
from typing import Any, Callable, Mapping

import numpy as np

from _bootstrap import PROJECT_ROOT
from mtare_topo.data.cano_sensor_smoke import MAX_RANGE_M
from mtare_topo.data.gse_replay_dataset import GSEReplayWorld, load_gse_replay_worlds
from mtare_topo.data.gse_training_dataset import GSESequenceDataset
from mtare_topo.evaluation.gse_topology_replay import (
    aggregate_gse_world_replays,
    gse_graph_parameter_grid,
    offline_topology_scientific_gate,
    replay_typed_observation_world,
    rule_graph_parameter_grid,
    select_baseline_graph_sweep,
    select_gse_graph_sweep,
    teacher_structural_graph,
)
from mtare_topo.evaluation.gse_evidence_integrity import verify_complete_run_seal
from mtare_topo.governance import load_json, write_json
from mtare_topo.semantics.exit_only_geometry_observation import exit_only_observation_from_arrays
from mtare_topo.semantics.gse_observation_adapter import geometric_semantic_observation_from_arrays
from mtare_topo.semantics.geometric_semantics import GeometricSemanticObservation
from mtare_topo.semantics.nonlearning_geometry_observation import nonlearning_geometry_observation
from mtare_topo.topology.gse_graph import GSEGraphConfig


TRAINING_STATUS = "PASS_GSE_GRAPH_THREE_SEED_TRAINING_V1R"
PERCEPTION_STATUS = "PASS_GSE_PERCEPTION_VALIDATION_V1"
CALIBRATION_STATUS = "PASS_GSE_VALIDATION_CALIBRATION_V1"
PASS_STATUS = "PASS_GSE_OFFLINE_TOPOLOGY_VALIDATION_V1"
FAIL_STATUS = "FAIL_GSE_OFFLINE_TOPOLOGY_VALIDATION_V1"
METHOD_REPLAY_COUNTS = {
    "gse_learned_association": 7290,
    "gse_rule_association_ablation": 7290,
    "exit_only_rule_graph": 7290,
    "nonlearning_geometry_rule_graph": 2430,
}
EXPECTED_WORLD_CONFIG_SEED_REPLAYS = 24300
EXPECTED_TYPED_OBSERVATION_UPDATES = 59442660


def _write_jsonl(path: Path, records) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, separators=(",", ":"), sort_keys=True) + "\n")


def _seed_contract(calibration_dir: Path, seed: int) -> tuple[dict[str, float], list[GSEGraphConfig]]:
    calibration = load_json(calibration_dir / f"seed{seed}_summary.json")
    event = calibration["event"]
    exits = calibration["exit_tokens"]
    place = calibration["place_association"]["selection"]
    exit_association = exits["descriptor_association"]["selection"]
    fixed = {
        "event_temperature": float(event["temperature"]["temperature"]),
        "event_probability_threshold": float(event["rejection_selection"]["threshold"]),
        "maximum_uncertainty": float(event["rejection_selection"]["maximum_uncertainty"]),
        "descriptor_minimum_similarity": float(place["threshold"]),
        "exit_presence_threshold": float(exits["presence_threshold"]["threshold"]),
        "exit_descriptor_minimum_similarity": float(exit_association["threshold"]),
        "exit_heading_tolerance_deg": float(exits["heading_error_deg"]["p95"]),
        "exit_width_log_tolerance": float(exits["opening_width_log_error"]["p95"]),
        "exit_vertical_profile_tolerance": float(exits["vertical_profile_error_m"]["p95"]),
    }
    if any(not np.isfinite(value) for value in fixed.values()):
        raise RuntimeError(f"seed {seed} calibration contains non-finite parameters")
    return fixed, gse_graph_parameter_grid(
        event_probability_threshold=fixed["event_probability_threshold"],
        maximum_uncertainty=fixed["maximum_uncertainty"],
        descriptor_minimum_similarity=fixed["descriptor_minimum_similarity"],
        exit_descriptor_minimum_similarity=fixed["exit_descriptor_minimum_similarity"],
        exit_heading_tolerance_deg=fixed["exit_heading_tolerance_deg"],
        exit_width_log_tolerance=fixed["exit_width_log_tolerance"],
        exit_vertical_profile_tolerance=fixed["exit_vertical_profile_tolerance"],
    )


def _load_archive(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as source:
        return {name: source[name] for name in source.files}


def _output_rows(arrays: Mapping[str, np.ndarray]) -> dict[int, int]:
    indices = np.asarray(arrays["global_sequence_index"], dtype=np.int64)
    result = {int(value): row for row, value in enumerate(indices)}
    if len(indices) != 24462 or len(result) != len(indices):
        raise RuntimeError("validation output indices are not a 24,462-row bijection")
    return result


def _gse_observations(
    training_run: Path,
    contracts: Mapping[int, Mapping[str, float]],
) -> dict[int, dict[int, GeometricSemanticObservation]]:
    result = {}
    for seed in (0, 1, 2):
        arrays = _load_archive(training_run / f"artifacts/models/seed{seed}/validation_outputs.npz")
        rows = _output_rows(arrays)
        result[seed] = {
            sequence: geometric_semantic_observation_from_arrays(
                arrays,
                row,
                event_temperature=float(contracts[seed]["event_temperature"]),
                exit_presence_threshold=float(contracts[seed]["exit_presence_threshold"]),
            )
            for sequence, row in rows.items()
        }
    return result


def _exit_only_observations(perception_run: Path) -> dict[int, dict[int, GeometricSemanticObservation]]:
    result = {}
    for seed in (0, 1, 2):
        arrays = _load_archive(
            perception_run / f"artifacts/exit_only_baseline/m1d_seed{seed}_validation_outputs.npz"
        )
        rows = _output_rows(arrays)
        result[seed] = {
            sequence: exit_only_observation_from_arrays(arrays, row)
            for sequence, row in rows.items()
        }
    return result


def _nonlearning_observations(dataset_run: Path) -> dict[int, dict[int, GeometricSemanticObservation]]:
    dataset = GSESequenceDataset(dataset_run, "validation", augment_azimuth=False)
    observations: dict[int, GeometricSemanticObservation] = {}
    for index in range(len(dataset)):
        sample = dataset[index]
        student = np.asarray(sample["student"], dtype=np.float32)
        sequence = int(sample["global_sequence_index"])
        observations[sequence] = nonlearning_geometry_observation(
            student[:, 0] * MAX_RANGE_M,
            student[:, 1],
        )
    if len(observations) != 24462:
        raise RuntimeError("non-learning observation materialization lost validation rows")
    return {0: observations}


def _structural_config(config: GSEGraphConfig) -> dict[str, Any]:
    values = config.to_dict()
    return {
        name: values[name]
        for name in (
            "stable_event_frames",
            "minimum_event_travel_m",
            "metric_anchor_interval_m",
            "association_radius_m",
            "ambiguity_similarity_margin",
        )
    }


def _run_sweep(
    *,
    method_id: str,
    worlds: Mapping[str, GSEReplayWorld],
    observations_by_seed: Mapping[int, Mapping[int, GeometricSemanticObservation]],
    grids_by_seed: Mapping[int, list[GSEGraphConfig]],
    output_dir: Path,
    association_mode: str,
    selector: Callable[[list[Mapping[str, Any]]], Mapping[str, Any]],
) -> dict[str, Any]:
    method_dir = output_dir / method_id
    method_dir.mkdir(parents=True, exist_ok=False)
    seeds = tuple(sorted(observations_by_seed))
    if set(seeds) != set(grids_by_seed) or any(len(grids_by_seed[seed]) != 243 for seed in seeds):
        raise RuntimeError(f"{method_id} seed/grid contract mismatch")
    rows = []
    sweep_path = method_dir / "parameter_sweep.jsonl"
    with sweep_path.open("w", encoding="utf-8") as stream:
        for grid_index in range(243):
            replay_results = []
            for seed in seeds:
                for world in worlds.values():
                    replay_results.append(
                        replay_typed_observation_world(
                            traversal_records=world.traversal_records,
                            teacher_observations=world.teacher_observations,
                            pose_by_sequence_index=world.pose_by_sequence_index,
                            observation_by_sequence_index=observations_by_seed[seed],
                            graph_config=grids_by_seed[seed][grid_index],
                            association_mode=association_mode,
                        )
                    )
            row = {
                "method_id": method_id,
                "grid_index": grid_index,
                "structural_config": _structural_config(grids_by_seed[seeds[0]][grid_index]),
                "aggregate": aggregate_gse_world_replays(replay_results),
            }
            rows.append(row)
            stream.write(json.dumps(row, separators=(",", ":"), sort_keys=True) + "\n")
            del replay_results
            gc.collect()
    selected = selector(rows)
    selected_index = int(selected["grid_index"])
    selected_results = []
    for seed in seeds:
        seed_dir = method_dir / "selected" / f"seed{seed}"
        seed_dir.mkdir(parents=True)
        write_json(
            seed_dir / "config.json",
            {
                "method_id": method_id,
                "seed": seed,
                "grid_index": selected_index,
                "association_mode": association_mode,
                "graph_config": grids_by_seed[seed][selected_index].to_dict(),
            },
        )
        for parent, world in worlds.items():
            result = replay_typed_observation_world(
                traversal_records=world.traversal_records,
                teacher_observations=world.teacher_observations,
                pose_by_sequence_index=world.pose_by_sequence_index,
                observation_by_sequence_index=observations_by_seed[seed],
                graph_config=grids_by_seed[seed][selected_index],
                association_mode=association_mode,
            )
            world_dir = seed_dir / parent
            world_dir.mkdir()
            for name in ("nodes", "edges", "decision_trace"):
                _write_jsonl(world_dir / f"{name}.jsonl", result[name])
            write_json(
                world_dir / "summary.json",
                {key: value for key, value in result.items() if key not in {"nodes", "edges", "decision_trace"}},
            )
            selected_results.append(result)
    aggregate = aggregate_gse_world_replays(selected_results)
    if aggregate != selected["aggregate"]:
        raise RuntimeError(f"{method_id} selected replay does not reproduce sweep aggregate")
    if sum(1 for line in sweep_path.read_text(encoding="utf-8").splitlines() if line.strip()) != 243:
        raise RuntimeError(f"{method_id} parameter sweep evidence is not exactly 243 rows")
    expected_files = {sweep_path.resolve(), (method_dir / "summary.json").resolve()}
    for seed in seeds:
        seed_dir = method_dir / "selected" / f"seed{seed}"
        expected_files.add((seed_dir / "config.json").resolve())
        for parent in worlds:
            world_dir = seed_dir / parent
            expected_files.update(
                (world_dir / name).resolve()
                for name in ("nodes.jsonl", "edges.jsonl", "decision_trace.jsonl", "summary.json")
            )
    summary = {
        "method_id": method_id,
        "seeds": list(seeds),
        "parameter_groups": 243,
        "world_seed_replays_in_sweep": 243 * len(seeds) * len(worlds),
        "selected_grid_index": selected_index,
        "selected_structural_config": selected["structural_config"],
        "selected_aggregate": aggregate,
        "association_mode": association_mode,
        "parameter_sweep_rows": 243,
        "selected_evidence_files": len(expected_files) - 2,
        "association_safe": bool(
            aggregate["association"]["merge_attempts"] > 0
            and aggregate["association"]["precision"] >= 0.98
            and aggregate["association"]["false_loop_merge_rate"] <= 0.01
        ),
    }
    write_json(method_dir / "summary.json", summary)
    actual_files = {path.resolve() for path in method_dir.rglob("*") if path.is_file()}
    if actual_files != expected_files:
        raise RuntimeError(f"{method_id} selected evidence file set is incomplete or undeclared")
    return summary


def _oracle_summary(worlds: Mapping[str, GSEReplayWorld]) -> dict[str, Any]:
    node_count = edge_count = 0
    for world in worlds.values():
        graph = teacher_structural_graph(world.teacher_observations, world.traversal_records)
        node_count += len(graph["node_ids"])
        edge_count += len(graph["edges"])
    return {
        "method_id": "gt_tng_oracle_diagnostic",
        "worlds": len(worlds),
        "node": {"precision": 1.0, "recall": 1.0, "f1": 1.0, "predicted": node_count, "teacher": node_count},
        "edge": {"precision": 1.0, "recall": 1.0, "f1": 1.0, "predicted": edge_count, "teacher": edge_count},
        "claim_boundary": "identity-aware diagnostic ceiling; not a deployable or theoretical upper bound",
    }


def evaluate(
    training_run: Path,
    perception_run: Path,
    dataset_run: Path,
    teacher_run: Path,
    output_dir: Path,
) -> dict[str, Any]:
    started = time.monotonic()
    paths = [path.resolve() for path in (training_run, perception_run, dataset_run, teacher_run, output_dir.parent)]
    for path in paths:
        path.relative_to(PROJECT_ROOT.resolve())
    training_run, perception_run, dataset_run, teacher_run, _ = paths
    output_dir = output_dir.resolve()
    source_status = {
        training_run: TRAINING_STATUS,
        perception_run: PERCEPTION_STATUS,
        dataset_run: "PASS_GSE_DEDUPLICATED_DATASET_EXPORT_V1",
        teacher_run: "PASS_GSE_TEACHER_MANIFEST_V1",
    }
    source_verification = {
        run.name: verify_complete_run_seal(PROJECT_ROOT, run, expected)
        for run, expected in source_status.items()
    }
    training_state = load_json(training_run / "RUN_STATE.json")
    training_summary = load_json(training_run / "metrics/summary.json")
    perception_state = load_json(perception_run / "RUN_STATE.json")
    perception_summary = load_json(perception_run / "metrics/summary.json")
    calibration_dir = perception_run / "artifacts/calibration"
    calibration_summary = load_json(calibration_dir / "summary.json")
    if (
        training_state.get("state") != "COMPLETED"
        or training_state.get("overall_status") != TRAINING_STATUS
        or training_summary.get("overall_status") != TRAINING_STATUS
        or perception_state.get("state") != "COMPLETED"
        or perception_state.get("overall_status") != PERCEPTION_STATUS
        or perception_summary.get("overall_status") != PERCEPTION_STATUS
        or calibration_summary.get("overall_status") != CALIBRATION_STATUS
        or any(
            source.get("strict_test_worlds_read") != 0
            or source.get("mtare_worlds_read") != 0
            for source in (training_summary, perception_summary, calibration_summary)
        )
    ):
        raise RuntimeError("offline topology sources are not completed development PASS evidence")
    if output_dir.exists():
        raise RuntimeError("offline topology output already exists")
    output_dir.mkdir(parents=True)
    worlds = load_gse_replay_worlds(dataset_run, teacher_run, split="validation")
    if len(worlds) != 10 or sum(len(world.teacher_observations) for world in worlds.values()) != 24462:
        raise RuntimeError("offline topology validation population drift")

    contracts: dict[int, dict[str, float]] = {}
    learned_grids: dict[int, list[GSEGraphConfig]] = {}
    for seed in (0, 1, 2):
        contracts[seed], learned_grids[seed] = _seed_contract(calibration_dir, seed)
    methods: dict[str, dict[str, Any]] = {}
    learned_observations = _gse_observations(training_run, contracts)
    methods["gse_learned_association"] = _run_sweep(
        method_id="gse_learned_association",
        worlds=worlds,
        observations_by_seed=learned_observations,
        grids_by_seed=learned_grids,
        output_dir=output_dir,
        association_mode="learned",
        selector=select_gse_graph_sweep,
    )
    rule_grids = {
        seed: rule_graph_parameter_grid(
            event_probability_threshold=contracts[seed]["event_probability_threshold"],
            maximum_uncertainty=contracts[seed]["maximum_uncertainty"],
        )
        for seed in (0, 1, 2)
    }
    methods["gse_rule_association_ablation"] = _run_sweep(
        method_id="gse_rule_association_ablation",
        worlds=worlds,
        observations_by_seed=learned_observations,
        grids_by_seed=rule_grids,
        output_dir=output_dir,
        association_mode="rule",
        selector=select_baseline_graph_sweep,
    )
    del learned_observations
    gc.collect()

    exit_observations = _exit_only_observations(perception_run)
    exit_grids = {seed: rule_graph_parameter_grid(event_probability_threshold=0.5) for seed in (0, 1, 2)}
    methods["exit_only_rule_graph"] = _run_sweep(
        method_id="exit_only_rule_graph",
        worlds=worlds,
        observations_by_seed=exit_observations,
        grids_by_seed=exit_grids,
        output_dir=output_dir,
        association_mode="rule",
        selector=select_baseline_graph_sweep,
    )
    del exit_observations
    gc.collect()

    nonlearning = _nonlearning_observations(dataset_run)
    methods["nonlearning_geometry_rule_graph"] = _run_sweep(
        method_id="nonlearning_geometry_rule_graph",
        worlds=worlds,
        observations_by_seed=nonlearning,
        grids_by_seed={0: rule_graph_parameter_grid(event_probability_threshold=0.5)},
        output_dir=output_dir,
        association_mode="rule",
        selector=select_baseline_graph_sweep,
    )
    del nonlearning
    gc.collect()

    if set(methods) != set(METHOD_REPLAY_COUNTS):
        raise RuntimeError("offline topology did not execute the exact four predeclared methods")
    for method_id, expected in METHOD_REPLAY_COUNTS.items():
        method = methods[method_id]
        if (
            method.get("parameter_groups") != 243
            or method.get("parameter_sweep_rows") != 243
            or method.get("world_seed_replays_in_sweep") != expected
        ):
            raise RuntimeError(f"{method_id} replay count or sweep evidence drift")
    world_config_seed_replays = sum(
        int(method["world_seed_replays_in_sweep"]) for method in methods.values()
    )
    typed_observation_updates = 24462 * 243 * sum(len(method["seeds"]) for method in methods.values())
    if (
        world_config_seed_replays != EXPECTED_WORLD_CONFIG_SEED_REPLAYS
        or typed_observation_updates != EXPECTED_TYPED_OBSERVATION_UPDATES
    ):
        raise RuntimeError("offline topology total replay/update count drift")

    oracle = _oracle_summary(worlds)
    write_json(output_dir / "gt_tng_oracle_diagnostic.json", oracle)
    gate = offline_topology_scientific_gate(methods)
    overall = PASS_STATUS if gate["passed"] else FAIL_STATUS
    summary = {
        "schema_version": "gse_offline_topology_validation_v1",
        "overall_status": overall,
        "validation_worlds": 10,
        "validation_sequences": 24462,
        "world_config_seed_replays": world_config_seed_replays,
        "typed_observation_updates": typed_observation_updates,
        "methods": methods,
        "oracle": oracle,
        "scientific_gate": gate,
        "selection_policy": {
            "gse": "association-safe first; maximize minimum node/edge F1, then mean F1, invariant error and lowest grid index",
            "baselines": "maximize minimum node/edge F1, then mean F1, invariant error and lowest grid index; unsafe association remains visible",
        },
        "source_verification": source_verification,
        "strict_test_worlds_read": 0,
        "mtare_worlds_read": 0,
        "optimizer_steps": 0,
        "model_updates": 0,
        "duration_seconds": time.monotonic() - started,
    }
    write_json(output_dir / "summary.json", summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-run", required=True, type=Path)
    parser.add_argument("--perception-run", required=True, type=Path)
    parser.add_argument("--dataset-run", required=True, type=Path)
    parser.add_argument("--teacher-run", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    result = evaluate(
        args.training_run,
        args.perception_run,
        args.dataset_run,
        args.teacher_run,
        args.output_dir,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["overall_status"] == PASS_STATUS else 2


if __name__ == "__main__":
    raise SystemExit(main())
