#!/usr/bin/env python3
"""C09 topology validation with the frozen distance-aware GSE ensemble."""

from __future__ import annotations

import argparse
import gc
import time
from pathlib import Path
from typing import Any

import numpy as np

from _bootstrap import PROJECT_ROOT
import evaluate_gse_offline_topology_v2 as base
from mtare_topo.data.gse_replay_dataset import load_gse_replay_worlds
from mtare_topo.evaluation.gse_evidence_integrity import (
    verify_complete_run_seal,
    verify_failed_component_run_seal,
)
from mtare_topo.evaluation.gse_topology_replay import (
    offline_topology_scientific_gate,
    rule_graph_parameter_grid,
    select_baseline_graph_sweep,
    select_gse_graph_sweep,
)
from mtare_topo.governance import load_json, write_json
from mtare_topo.representation.gse_exit_token_ensemble import (
    FROZEN_ENSEMBLE_THRESHOLD,
    FrozenExitTokenEnsembleRuntime,
)


PASS_STATUS = "PASS_GSE_OFFLINE_TOPOLOGY_VALIDATION_V3"
FAIL_STATUS = "FAIL_GSE_OFFLINE_TOPOLOGY_VALIDATION_V3"
VERIFIER_STATUS = "FAIL_GSE_EXIT_TOKEN_ASSOCIATION_CORRECTIVE_V2"
ENSEMBLE_STATUS = "PASS_GSE_DISTANCE_AWARE_ENSEMBLE_CALIBRATION_V1"


def _raw_gse_outputs(training_run: Path) -> dict[int, dict[str, np.ndarray]]:
    outputs = {
        seed: base._load_archive(
            training_run / f"artifacts/models/seed{seed}/validation_outputs.npz"
        )
        for seed in (0, 1, 2)
    }
    identities = [np.asarray(outputs[seed]["global_sequence_index"], dtype=np.int64) for seed in (0, 1, 2)]
    if any(len(values) != 24462 or len(np.unique(values)) != 24462 for values in identities):
        raise RuntimeError("C09 raw GSE output population drift")
    if any(set(values.tolist()) != set(identities[0].tolist()) for values in identities[1:]):
        raise RuntimeError("C09 GSE seed identity sets differ")
    return outputs


def evaluate(
    training_run: Path,
    perception_component_run: Path,
    corrected_perception_run: Path,
    dataset_run: Path,
    teacher_run: Path,
    verifier_run: Path,
    ensemble_run: Path,
    output_dir: Path,
    *,
    device: str = "cuda:0",
    node_calibration_run: Path | None = None,
    baselines_first: bool = False,
    pass_status: str = PASS_STATUS,
    fail_status: str = FAIL_STATUS,
    schema_version: str = "gse_offline_topology_validation_v3",
) -> dict[str, Any]:
    started = time.monotonic()
    inputs = (
        training_run,
        perception_component_run,
        corrected_perception_run,
        dataset_run,
        teacher_run,
        verifier_run,
        ensemble_run,
    )
    paths = [path.resolve() for path in (*inputs, output_dir.parent)]
    for path in paths:
        path.relative_to(PROJECT_ROOT.resolve())
    (
        training_run,
        perception_component_run,
        corrected_perception_run,
        dataset_run,
        teacher_run,
        verifier_run,
        ensemble_run,
        _,
    ) = paths
    output_dir = output_dir.resolve()
    source_verification = {
        training_run.name: verify_complete_run_seal(PROJECT_ROOT, training_run, base.TRAINING_STATUS),
        corrected_perception_run.name: verify_complete_run_seal(
            PROJECT_ROOT, corrected_perception_run, base.CORRECTED_PERCEPTION_STATUS
        ),
        dataset_run.name: verify_complete_run_seal(
            PROJECT_ROOT, dataset_run, "PASS_GSE_DEDUPLICATED_DATASET_EXPORT_V1"
        ),
        teacher_run.name: verify_complete_run_seal(
            PROJECT_ROOT, teacher_run, "PASS_GSE_TEACHER_MANIFEST_V1"
        ),
        ensemble_run.name: verify_complete_run_seal(PROJECT_ROOT, ensemble_run, ENSEMBLE_STATUS),
        perception_component_run.name: verify_failed_component_run_seal(
            PROJECT_ROOT, perception_component_run, base.PERCEPTION_COMPONENT_STATUS
        ),
        verifier_run.name: verify_failed_component_run_seal(
            PROJECT_ROOT, verifier_run, VERIFIER_STATUS
        ),
    }
    node_gate_enabled = node_calibration_run is not None
    node_selection = None
    if node_calibration_run is not None:
        node_calibration_run = node_calibration_run.resolve()
        node_calibration_run.relative_to(PROJECT_ROOT)
        source_verification[node_calibration_run.name] = verify_complete_run_seal(
            PROJECT_ROOT,
            node_calibration_run,
            "PASS_GSE_NODE_MATCHABILITY_ENSEMBLE_CALIBRATION_V1",
        )
        node_summary = load_json(node_calibration_run / "artifacts/calibration/summary.json")
        node_selection = node_summary.get("selection") or {}
        if (
            node_summary.get("overall_status") != "PASS_GSE_NODE_MATCHABILITY_ENSEMBLE_CALIBRATION_V1"
            or node_selection.get("threshold") != 0.982292910416921
            or node_summary.get("c09_worlds_read") != 0
            or node_summary.get("strict_test_worlds_read") != 0
            or node_summary.get("mtare_worlds_read") != 0
        ):
            raise RuntimeError("frozen node-matchability source contract drift")
    ensemble = load_json(ensemble_run / "artifacts/calibration/summary.json")
    selection = ensemble.get("selection") or {}
    if (
        ensemble.get("overall_status") != ENSEMBLE_STATUS
        or ensemble.get("scientific_pass") is not True
        or ensemble.get("method") != "arithmetic_mean_of_three_frozen_v2_seed_scores"
        or selection.get("threshold") != FROZEN_ENSEMBLE_THRESHOLD
        or ensemble.get("contract", {}).get("maximum_candidate_distance_m") != 16.0
        or any(
            ensemble.get(name) != 0
            for name in ("c09_worlds_read", "strict_test_worlds_read", "mtare_worlds_read")
        )
    ):
        raise RuntimeError("frozen distance-aware ensemble source contract drift")
    complete_gate = load_json(corrected_perception_run / "metrics/corrected_perception_gate.json")
    component_gate = load_json(perception_component_run / "metrics/perception_gate.json")
    calibration_dir = perception_component_run / "artifacts/calibration"
    if (
        complete_gate.get("passed") is not True
        or component_gate.get("event_gate", {}).get("passed") is not True
        or component_gate.get("association_gate", {}).get("passed") is not True
        or load_json(calibration_dir / "summary.json").get("overall_status") != base.CALIBRATION_STATUS
    ):
        raise RuntimeError("C09 perception source contract drift")
    if output_dir.exists():
        raise RuntimeError("offline topology V3 output already exists")
    output_dir.mkdir(parents=True)
    worlds = load_gse_replay_worlds(dataset_run, teacher_run, split="validation")
    if len(worlds) != 10 or sum(len(world.teacher_observations) for world in worlds.values()) != 24462:
        raise RuntimeError("offline topology C09 population drift")

    contracts = {}
    learned_grids = {}
    for seed in (0, 1, 2):
        contracts[seed], learned_grids[seed] = base._seed_contract(calibration_dir, seed)
    learned_observations = base._gse_observations(training_run, corrected_perception_run, contracts)
    raw_outputs = _raw_gse_outputs(training_run)
    verifier_directories = {
        seed: verifier_run / f"artifacts/models/seed{seed}" for seed in (0, 1, 2)
    }
    runtime = FrozenExitTokenEnsembleRuntime(
        outputs_by_seed=raw_outputs,
        model_directories=verifier_directories,
        device=device,
    )
    association_backends = {}
    association_pair_counts = {}
    for parent, world in worlds.items():
        sequence_keys = [int(row["global_sequence_index"]) for row in world.teacher_observations]
        xyz_by_key = {
            key: world.pose_by_sequence_index[key]["axis_xyz_m"] for key in sequence_keys
        }
        backend = runtime.prepare_world(sequence_keys, xyz_by_key)
        association_backends[parent] = backend
        association_pair_counts[parent] = backend.pair_count

    methods = {}
    rule_grids = {
        seed: rule_graph_parameter_grid(
            event_probability_threshold=contracts[seed]["event_probability_threshold"],
            maximum_uncertainty=contracts[seed]["maximum_uncertainty"],
        )
        for seed in (0, 1, 2)
    }
    exit_grids = {
        seed: rule_graph_parameter_grid(event_probability_threshold=0.5) for seed in (0, 1, 2)
    }

    def run_main() -> None:
        methods["gse_learned_association"] = base._run_sweep(
            method_id="gse_learned_association",
            worlds=worlds,
            observations_by_seed=learned_observations,
            grids_by_seed=learned_grids,
            output_dir=output_dir,
            association_mode=(
                "frozen_ensemble_with_open_set_node_gate"
                if node_gate_enabled else "frozen_exit_token_ensemble"
            ),
            selector=select_gse_graph_sweep,
            association_backends_by_world=association_backends,
            node_generation_backends_by_world=(association_backends if node_gate_enabled else None),
        )

    def run_baselines() -> None:
        methods["gse_rule_association_ablation"] = base._run_sweep(
            method_id="gse_rule_association_ablation",
            worlds=worlds,
            observations_by_seed=learned_observations,
            grids_by_seed=rule_grids,
            output_dir=output_dir,
            association_mode="rule",
            selector=select_baseline_graph_sweep,
        )
        exit_observations = base._exit_only_observations(perception_component_run)
        methods["exit_only_rule_graph"] = base._run_sweep(
            method_id="exit_only_rule_graph",
            worlds=worlds,
            observations_by_seed=exit_observations,
            grids_by_seed=exit_grids,
            output_dir=output_dir,
            association_mode="rule",
            selector=select_baseline_graph_sweep,
        )
        nonlearning = base._nonlearning_observations(dataset_run)
        methods["nonlearning_geometry_rule_graph"] = base._run_sweep(
            method_id="nonlearning_geometry_rule_graph",
            worlds=worlds,
            observations_by_seed=nonlearning,
            grids_by_seed={0: rule_graph_parameter_grid(event_probability_threshold=0.5)},
            output_dir=output_dir,
            association_mode="rule",
            selector=select_baseline_graph_sweep,
        )

    if baselines_first:
        run_baselines()
        run_main()
    else:
        run_main()
        run_baselines()
    del learned_observations, raw_outputs, runtime
    gc.collect()

    if set(methods) != set(base.METHOD_REPLAY_COUNTS):
        raise RuntimeError("V3 did not execute the four predeclared topology methods")
    for method_id, expected in base.METHOD_REPLAY_COUNTS.items():
        method = methods[method_id]
        if (
            method.get("parameter_groups") != 243
            or method.get("parameter_sweep_rows") != 243
            or method.get("world_seed_replays_in_sweep") != expected
        ):
            raise RuntimeError(f"{method_id} replay count drift")
    world_config_seed_replays = sum(
        int(method["world_seed_replays_in_sweep"]) for method in methods.values()
    )
    typed_observation_updates = 24462 * 243 * sum(len(method["seeds"]) for method in methods.values())
    if (
        world_config_seed_replays != base.EXPECTED_WORLD_CONFIG_SEED_REPLAYS
        or typed_observation_updates != base.EXPECTED_TYPED_OBSERVATION_UPDATES
    ):
        raise RuntimeError("V3 total replay/update count drift")

    oracle = base._oracle_summary(worlds)
    write_json(output_dir / "gt_tng_oracle_diagnostic.json", oracle)
    gate = offline_topology_scientific_gate(methods)
    overall = pass_status if gate["passed"] else fail_status
    pair_count = sum(association_pair_counts.values())
    summary = {
        "schema_version": schema_version,
        "overall_status": overall,
        "validation_worlds": 10,
        "validation_sequences": 24462,
        "world_config_seed_replays": world_config_seed_replays,
        "typed_observation_updates": typed_observation_updates,
        "methods": methods,
        "oracle": oracle,
        "scientific_gate": gate,
        "association_backend": {
            "name": "frozen_v2_exit_token_three_seed_equal_weight_ensemble",
            "threshold": FROZEN_ENSEMBLE_THRESHOLD,
            "maximum_candidate_distance_m": 16.0,
            "world_pair_counts": association_pair_counts,
            "unique_candidate_pairs": pair_count,
            "verifier_seed_pair_inferences": 3 * pair_count,
            "precomputation_causality": "each cached score consumes exactly two observations and distance; no graph state, future aggregate, Teacher identity, or label",
        },
        "node_generation_backend": (
            {
                "name": "frozen_v2_matchability_three_seed_equal_weight_ensemble",
                "threshold": float(node_selection["threshold"]),
                "source_run": node_calibration_run.name,
            }
            if node_gate_enabled else None
        ),
        "selection_policy": {
            "gse": "frozen verifier operating point; predeclared 243 graph grid selected association-safe first on C09",
            "baselines": "predeclared 243 graph grid; maximize topology F1 with unsafe association visible",
        },
        "source_verification": source_verification,
        "strict_test_worlds_read": 0,
        "c10_worlds_read": 0,
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
    parser.add_argument("--corrected-perception-run", required=True, type=Path)
    parser.add_argument("--dataset-run", required=True, type=Path)
    parser.add_argument("--teacher-run", required=True, type=Path)
    parser.add_argument("--verifier-run", required=True, type=Path)
    parser.add_argument("--ensemble-run", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    result = evaluate(
        args.training_run,
        args.perception_run,
        args.corrected_perception_run,
        args.dataset_run,
        args.teacher_run,
        args.verifier_run,
        args.ensemble_run,
        args.output_dir,
        device=args.device,
    )
    print(result["overall_status"])
    return 0 if result["overall_status"] == PASS_STATUS else 2


if __name__ == "__main__":
    raise SystemExit(main())
