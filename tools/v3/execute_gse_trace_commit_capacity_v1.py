#!/usr/bin/env python3
"""C01--C08 zero-training capacity proof for trace-committed GSE topology."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import time

import numpy as np

from mtare_topo.data.gse_action_set_cache import ActionSetNodeDataset
from mtare_topo.evaluation.gse_causal_episode_metrics import (
    evaluate_decision_mass_triggers,
    extract_decision_mass_triggers,
)
from mtare_topo.representation.gse_action_set_node import ActionSetNodeDetector
from mtare_topo.representation.gse_factorized_association import (
    FactorizedAssociationVerifier,
    factorized_pair_features,
    learned_geometry_profiles,
)
from mtare_topo.representation.gse_route_conditioned_node import (
    RouteConditionedNodeConfig,
    route_conditioned_action_scores,
    structured_decision_events,
)
from mtare_topo.topology.gse_trace_commit_replay import (
    ProposalTrigger,
    replay_trace_commits,
    score_trace_replay,
)


EXPECTED = 188_126
TOKEN_KEYS = (
    "exit_confidence", "exit_heading_unit", "exit_opening_width_m",
    "exit_vertical_profile", "exit_descriptor",
)
LEARNED_THRESHOLDS = (.80, .85, .90, .925, .95, .97, .98, .99, .995)
SEED_THRESHOLDS = (0.994343638420105, 0.9797766804695129, 0.9817055463790894)


def _read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _infer_action_ensemble(cache_dir: Path, model_run: Path) -> np.ndarray:
    import torch
    from torch.utils.data import DataLoader

    rows = np.arange(EXPECTED, dtype=np.int64)
    dataset = ActionSetNodeDataset(cache_dir, rows)

    def collate(samples: list[dict]) -> dict:
        return {
            "tokens": torch.from_numpy(np.stack([value["tokens"] for value in samples])),
            "mask": torch.from_numpy(np.stack([value["history_mask"] for value in samples])),
        }

    loader = DataLoader(dataset, batch_size=1024, shuffle=False, num_workers=0, collate_fn=collate)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    probabilities = []
    for seed in range(3):
        checkpoint = torch.load(
            model_run / f"artifacts/models/seed{seed}/best.pt",
            map_location=device, weights_only=False,
        )
        if checkpoint.get("schema_version") != "gse_action_set_node_checkpoint_v1":
            raise RuntimeError(f"action-set seed{seed} checkpoint drift")
        model = ActionSetNodeDetector().to(device)
        model.load_state_dict(checkpoint["model"], strict=True)
        model.eval()
        output = []
        with torch.inference_mode():
            for batch in loader:
                result = model(batch["tokens"].to(device), batch["mask"].to(device))
                output.append(result["decision_probability"].cpu().numpy())
        probabilities.append(np.concatenate(output))
    three = np.mean(np.stack(probabilities), axis=0)
    result = np.zeros((EXPECTED, 5), dtype=np.float32)
    result[:, :3] = three
    if not np.allclose(result.sum(axis=1), 1.0, atol=1e-5):
        raise RuntimeError("action-set ensemble simplex drift")
    return result


def _learned_trigger_rows(
    probability: np.ndarray, rows: np.ndarray, traversal: np.ndarray,
    sequence: np.ndarray, uncertainty: np.ndarray, threshold: float,
) -> tuple[list[int], dict[int, int]]:
    triggers = extract_decision_mass_triggers(
        probability[rows], traversal[rows], sequence[rows], uncertainty[rows],
        decision_threshold=threshold,
    )
    global_rows = [int(rows[value.row]) for value in triggers]
    events = {int(rows[value.row]): int(value.predicted_event_index) for value in triggers}
    return global_rows, events


def _structured_trigger_rows(
    token_paths: tuple[Path, Path, Path], rows: np.ndarray,
    traversal: np.ndarray, sequence: np.ndarray,
) -> tuple[list[int], dict[int, int]]:
    incoming = np.tile(np.asarray([0.0, -1.0], dtype=np.float32), (len(rows), 1))
    scores = {}
    for seed, path in enumerate(token_paths):
        with np.load(path, allow_pickle=False) as archive:
            confidence = np.asarray(archive["exit_confidence"][rows], dtype=np.float32)
            heading = np.asarray(archive["exit_heading_unit"][rows], dtype=np.float32)
        scores[seed] = route_conditioned_action_scores(
            confidence, heading, incoming, incoming_half_angle_deg=50.0,
        )
    config = RouteConditionedNodeConfig(.28, 50.0, 3, 2)
    predicted = structured_decision_events(
        scores, traversal[rows], sequence[rows], config,
    )
    active = predicted > 0
    continues = np.zeros(len(rows), dtype=np.bool_)
    continues[1:] = (
        active[1:] & active[:-1] & (predicted[1:] == predicted[:-1])
        & (traversal[rows][1:] == traversal[rows][:-1])
        & (sequence[rows][1:] == sequence[rows][:-1] + 1)
    )
    local = np.flatnonzero(active & ~continues)
    global_rows = [int(rows[value]) for value in local]
    return global_rows, {int(rows[value]): int(predicted[value]) for value in local}


def _candidate_pairs(
    trigger_rows: set[int], parent: np.ndarray, xyz: np.ndarray,
    association_valid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    from scipy.spatial import cKDTree

    pairs: set[tuple[int, int]] = set()
    for world in sorted(set(parent[list(trigger_rows)].tolist())):
        rows = np.asarray(sorted(
            value for value in trigger_rows
            if parent[value] == world and association_valid[value]
        ), dtype=np.int64)
        if len(rows) < 2:
            continue
        for left, right in cKDTree(xyz[rows]).query_pairs(4.0 + 1e-12):
            pairs.add(tuple(sorted((int(rows[left]), int(rows[right])))))
    ordered = sorted(pairs)
    left = np.asarray([value[0] for value in ordered], dtype=np.int64)
    right = np.asarray([value[1] for value in ordered], dtype=np.int64)
    distance = np.linalg.norm(xyz[left] - xyz[right], axis=1).astype(np.float32)
    return left, right, distance


def _score_association_pairs(
    left: np.ndarray, right: np.ndarray, capacity_run: Path,
    observation_paths: tuple[Path, Path, Path], token_paths: tuple[Path, Path, Path],
    history_references: np.ndarray, history_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    import torch

    scores = []
    accepted = []
    for seed in range(3):
        observation = np.load(observation_paths[seed], mmap_mode="r")
        profile = learned_geometry_profiles(observation, history_references, history_mask)
        with np.load(token_paths[seed], allow_pickle=False) as archive:
            token = {name: np.asarray(archive[name]) for name in TOKEN_KEYS}
            features = factorized_pair_features(
                observation, token, profile, left, right, include_route_geometry=True,
            )
        base = capacity_run / f"artifacts/models/seed{seed}/full_route_conditioned"
        with np.load(base / "normalization.npz", allow_pickle=False) as archive:
            mean = archive["mean"].astype(np.float32)
            std = archive["std"].astype(np.float32)
        checkpoint = torch.load(base / "best.pt", map_location="cpu", weights_only=False)
        model = FactorizedAssociationVerifier(include_route_geometry=True)
        model.load_state_dict(checkpoint["model"], strict=True)
        model.eval()
        with torch.inference_mode():
            score = torch.sigmoid(model(torch.from_numpy((features - mean) / std))).numpy()
        scores.append(score.astype(np.float32))
        accepted.append(score >= SEED_THRESHOLDS[seed])
        del observation, profile, token, features, model
    return np.stack(scores), np.sum(np.stack(accepted), axis=0) >= 2


def _proposal_objects(
    rows: list[int], events: dict[int, int], parent: np.ndarray, order: np.ndarray,
    traversal: np.ndarray, sequence: np.ndarray, xyz: np.ndarray,
    probability: np.ndarray, uncertainty: np.ndarray, identity: np.ndarray,
    position_uncertainty_m: np.ndarray,
) -> list[ProposalTrigger]:
    names = {1: "junction", 2: "terminal"}
    return [ProposalTrigger(
        row=row, world=str(parent[row]), order=int(order[row]),
        traversal_id=str(traversal[row]), sequence_index=int(sequence[row]),
        event=names[int(events[row])], confidence=float(probability[row, int(events[row])]),
        uncertainty=float(uncertainty[row]), xyz_m=tuple(float(value) for value in xyz[row]),
        teacher_identity=None if str(identity[row]) == "None" else str(identity[row]),
        position_uncertainty_m=float(position_uncertainty_m[row]),
    ) for row in rows]


def _truth(
    rows: np.ndarray, parent: np.ndarray, traversal: np.ndarray, sequence: np.ndarray,
    target: np.ndarray, identity: np.ndarray,
) -> tuple[set[tuple[str, str]], set[tuple[tuple[str, str], tuple[str, str]]]]:
    nodes = {
        (str(parent[row]), str(identity[row])) for row in rows
        if target[row] in (1, 2) and str(identity[row]) != "None"
    }
    relations = set()
    for trace in sorted(set(traversal[rows].tolist())):
        selected = [int(value) for value in rows if traversal[value] == trace]
        selected.sort(key=lambda value: (sequence[value], value))
        ordered = []
        for row in selected:
            if target[row] not in (1, 2) or str(identity[row]) == "None":
                continue
            node = (str(parent[row]), str(identity[row]))
            if not ordered or ordered[-1] != node:
                ordered.append(node)
        for left, right in zip(ordered, ordered[1:]):
            if left != right:
                relations.add(tuple(sorted((left, right))))
    return nodes, relations


def _run_method(
    proposal_rows: list[int], events: dict[int, int], *,
    parent: np.ndarray, order: np.ndarray, traversal: np.ndarray, sequence: np.ndarray,
    xyz: np.ndarray, probability: np.ndarray, uncertainty: np.ndarray,
    identity: np.ndarray, accepted_pairs: dict[tuple[int, int], bool],
    valid_rows: set[int], truth_nodes: set, truth_edges: set,
    commit_immediately: bool, position_uncertainty_m: np.ndarray,
) -> tuple[dict, dict]:
    proposals = _proposal_objects(
        proposal_rows, events, parent, order, traversal, sequence, xyz,
        probability, uncertainty, identity, position_uncertainty_m,
    )
    replay = replay_trace_commits(
        proposals, accepted_pairs, association_valid_rows=valid_rows,
        commit_immediately=commit_immediately,
    )
    metrics = score_trace_replay(replay, true_nodes=truth_nodes, true_trace_relations=truth_edges)
    metrics.update({
        "proposal_triggers": len(proposals),
        "provisional_hypotheses": len(replay["hypotheses"]),
        "verified_edges_raw": len(replay["edges"]),
    })
    return replay, metrics


def _write_replay(output: Path, replay: dict) -> None:
    with (output / "verified_nodes.jsonl").open("w", encoding="utf-8") as stream:
        for value in replay["hypotheses"]:
            if not value["committed"]:
                continue
            stream.write(json.dumps({
                "hypothesis_id": value["id"], "world": value["world"], "event": value["event"],
                "evidence_rows": [item.row for item in value["evidence"]],
                "approach_trace_ids": sorted({item.traversal_id for item in value["evidence"]}),
            }, sort_keys=True) + "\n")
    with (output / "verified_edges.jsonl").open("w", encoding="utf-8") as stream:
        for value in replay["edges"]:
            stream.write(json.dumps(value, sort_keys=True) + "\n")
    with (output / "decision_trace.jsonl").open("w", encoding="utf-8") as stream:
        for value in replay["decision_trace"]:
            stream.write(json.dumps(value, sort_keys=True) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher", required=True, type=Path)
    parser.add_argument("--sequence-manifest", required=True, type=Path)
    parser.add_argument("--pair-cache", required=True, type=Path)
    parser.add_argument("--action-cache", required=True, type=Path)
    parser.add_argument("--action-model-run", required=True, type=Path)
    parser.add_argument("--association-capacity-run", required=True, type=Path)
    parser.add_argument("--center-offsets", type=Path)
    parser.add_argument("--fixed-proposal-threshold", type=float)
    parser.add_argument("--output-dir", required=True, type=Path)
    for seed in range(3):
        parser.add_argument(f"--observation{seed}", required=True, type=Path)
        parser.add_argument(f"--tokens{seed}", required=True, type=Path)
    args = parser.parse_args()
    started = time.monotonic()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    rows = _read_jsonl(args.teacher.resolve())
    sequence_rows = _read_jsonl(args.sequence_manifest.resolve())
    if len(rows) != EXPECTED or len(sequence_rows) < EXPECTED:
        raise RuntimeError("trace-commit population drift")
    manifest_order = {int(value["global_sequence_index"]): int(value["world_sequence_row"]) for value in sequence_rows}
    global_index = np.asarray([int(value["global_sequence_index"]) for value in rows])
    order = np.asarray([manifest_order[int(value)] for value in global_index])
    parent = np.asarray([str(value["parent_id"]) for value in rows])
    traversal = np.asarray([str(value["traversal_id"]) for value in rows])
    sequence = np.asarray([int(value["sequence_index"]) for value in rows])
    identity = np.asarray([str(value["identity"]) for value in rows])
    target = np.asarray([{"junction": 1, "terminal": 2}.get(str(value["event"]), 0) for value in rows], dtype=np.int8)
    with np.load(args.pair_cache.resolve(), allow_pickle=False) as archive:
        if not np.array_equal(archive["compact_to_global_sequence_index"], global_index):
            raise RuntimeError("trace-commit pair-cache identity drift")
        partition = archive["partition_code"].astype(np.uint8)
        association_valid = archive["association_valid"].astype(np.bool_)
        xyz = archive["sensor_xyz_m"].astype(np.float64)
    if int(np.sum(partition == 0)) != 142_184 or int(np.sum(partition == 1)) != 45_942:
        raise RuntimeError("trace-commit split drift")
    probability = _infer_action_ensemble(args.action_cache.resolve(), args.action_model_run.resolve())
    uncertainty = -np.sum(probability * np.log(np.clip(probability, 1e-8, 1.0)), axis=1) / np.log(5.0)
    graph_xyz = xyz
    center_projection = None
    position_uncertainty_m = np.zeros(EXPECTED, dtype=np.float32)
    if args.center_offsets is not None:
        with np.load(args.center_offsets.resolve(), allow_pickle=False) as archive:
            if not np.array_equal(archive["global_sequence_index"], global_index):
                raise RuntimeError("event-center projection identity drift")
            center_schema = str(archive["schema_version"].item())
            if center_schema not in (
                "gse_event_center_projection_v1",
                "gse_spatial_longitudinal_center_ensemble_all_v1",
            ):
                raise RuntimeError("event-center projection schema drift")
            graph_xyz = archive["projected_center_xyz_m"].astype(np.float64)
            position_uncertainty_m = archive["offset_std_m"].astype(np.float32)
            center_projection = {
                "schema_version": center_schema,
                "ensemble_offset_std_mean_m": float(np.mean(archive["offset_std_m"])),
                "ensemble_offset_std_p95_m": float(np.percentile(archive["offset_std_m"], 95)),
            }
            if graph_xyz.shape != xyz.shape or not np.all(np.isfinite(graph_xyz)):
                raise RuntimeError("event-center projected positions are invalid")
    token_paths = tuple(getattr(args, f"tokens{seed}").resolve() for seed in range(3))
    methods: dict[tuple[str, int | float], tuple[list[int], dict[int, int]]] = {}
    all_trigger_rows: set[int] = set()
    for code in (0, 1):
        subset = np.flatnonzero(partition == code)
        for threshold in LEARNED_THRESHOLDS:
            proposal = _learned_trigger_rows(probability, subset, traversal, sequence, uncertainty, threshold)
            methods[("learned", code, threshold)] = proposal
            all_trigger_rows.update(proposal[0])
        structured = _structured_trigger_rows(token_paths, subset, traversal, sequence)
        methods[("structured", code, 0.28)] = structured
        all_trigger_rows.update(structured[0])
    left, right, distance = _candidate_pairs(all_trigger_rows, parent, graph_xyz, association_valid)
    observation_paths = tuple(getattr(args, f"observation{seed}").resolve() for seed in range(3))
    references = np.load(args.action_cache / "history_references.npy", mmap_mode="r")
    history_mask = np.load(args.action_cache / "history_mask.npy", mmap_mode="r")
    scores, accepted = _score_association_pairs(
        left, right, args.association_capacity_run.resolve(), observation_paths,
        token_paths, references, history_mask,
    )
    accepted_pairs = {
        (int(l), int(r)): bool(value)
        for l, r, value in zip(left, right, accepted, strict=True)
    }
    valid_rows = set(np.flatnonzero(association_valid).tolist())

    truths = {code: _truth(np.flatnonzero(partition == code), parent, traversal, sequence, target, identity) for code in (0, 1)}
    fit_grid = []
    for threshold in LEARNED_THRESHOLDS:
        proposal_rows, events = methods[("learned", 0, threshold)]
        _, metrics = _run_method(
            proposal_rows, events, parent=parent, order=order, traversal=traversal,
            sequence=sequence, xyz=graph_xyz, probability=probability, uncertainty=uncertainty,
            identity=identity, accepted_pairs=accepted_pairs, valid_rows=valid_rows,
            truth_nodes=truths[0][0], truth_edges=truths[0][1], commit_immediately=False,
            position_uncertainty_m=position_uncertainty_m,
        )
        metrics["proposal_threshold"] = threshold
        metrics["qualifies"] = bool(
            metrics["node_precision"] >= .98 and metrics["false_loop_merge_fraction"] <= .01
            and metrics["node_recall"] >= .25 and metrics["edge_precision"] >= .98
            and metrics["edge_recall"] >= .25
        )
        fit_grid.append(metrics)
    if args.fixed_proposal_threshold is not None:
        matching = [
            value for value in fit_grid
            if abs(float(value["proposal_threshold"]) - args.fixed_proposal_threshold) <= 1e-12
        ]
        if len(matching) != 1:
            raise RuntimeError("fixed proposal threshold is absent from the frozen grid")
        selected = matching[0]
    else:
        eligible = [value for value in fit_grid if value["qualifies"]]
        selected = max(
            eligible if eligible else fit_grid,
            key=lambda value: (value["qualifies"], value["node_edge_macro_f1"], value["node_recall"], -value["proposal_threshold"]),
        )
    threshold = float(selected["proposal_threshold"])
    selection_rows, selection_events = methods[("learned", 1, threshold)]
    selected_replay, selection_metrics = _run_method(
        selection_rows, selection_events, parent=parent, order=order, traversal=traversal,
        sequence=sequence, xyz=graph_xyz, probability=probability, uncertainty=uncertainty,
        identity=identity, accepted_pairs=accepted_pairs, valid_rows=valid_rows,
        truth_nodes=truths[1][0], truth_edges=truths[1][1], commit_immediately=False,
        position_uncertainty_m=position_uncertainty_m,
    )
    _, ghost_metrics = _run_method(
        selection_rows, selection_events, parent=parent, order=order, traversal=traversal,
        sequence=sequence, xyz=graph_xyz, probability=probability, uncertainty=uncertainty,
        identity=identity, accepted_pairs=accepted_pairs, valid_rows=valid_rows,
        truth_nodes=truths[1][0], truth_edges=truths[1][1], commit_immediately=True,
        position_uncertainty_m=position_uncertainty_m,
    )
    rule_rows, rule_events = methods[("structured", 1, 0.28)]
    _, rule_metrics = _run_method(
        rule_rows, rule_events, parent=parent, order=order, traversal=traversal,
        sequence=sequence, xyz=graph_xyz, probability=probability, uncertainty=uncertainty,
        identity=identity, accepted_pairs=accepted_pairs, valid_rows=valid_rows,
        truth_nodes=truths[1][0], truth_edges=truths[1][1], commit_immediately=False,
        position_uncertainty_m=position_uncertainty_m,
    )
    proposal_metric = evaluate_decision_mass_triggers(
        probability[partition == 1], target[partition == 1],
        np.load(args.action_cache / "decision_episode_id.npy")[partition == 1],
        traversal[partition == 1], sequence[partition == 1], uncertainty[partition == 1],
        decision_threshold=threshold,
    )
    improvement = selection_metrics["node_edge_macro_f1"] - max(
        ghost_metrics["node_edge_macro_f1"], rule_metrics["node_edge_macro_f1"]
    )
    gates = {
        "fit_configuration_satisfies_safety_and_recall": bool(selected["qualifies"]),
        "selection_node_precision_at_least_0p98": selection_metrics["node_precision"] >= .98,
        "selection_false_loop_merge_at_most_0p01": selection_metrics["false_loop_merge_fraction"] <= .01,
        "selection_node_recall_at_least_0p25": selection_metrics["node_recall"] >= .25,
        "selection_edge_precision_at_least_0p98": selection_metrics["edge_precision"] >= .98,
        "selection_edge_recall_at_least_0p25": selection_metrics["edge_recall"] >= .25,
        "node_edge_f1_improves_baselines_by_0p05": improvement >= .05,
    }
    passed = all(gates.values())
    summary = {
        "schema_version": "gse_trace_commit_capacity_v1",
        "status": "PASS_GSE_TRACE_COMMIT_CAPACITY_V1" if passed else "FAIL_GSE_TRACE_COMMIT_CAPACITY_V1",
        "question": "Do learned geometry-semantic hypotheses improve the execution-verified graph?",
        "selected_proposal_threshold": threshold,
        "proposal_threshold_policy": (
            "fixed_predecessor_threshold" if args.fixed_proposal_threshold is not None
            else "fit_selected"
        ),
        "fit_selection": selected, "fit_grid": fit_grid,
        "selection": {
            "gse_trace_commit": selection_metrics,
            "learned_ghost_immediate_commit": ghost_metrics,
            "structured_rule_trace_commit": rule_metrics,
            "improvement_over_stronger_baseline": improvement,
            "raw_proposal": proposal_metric,
        },
        "association": {
            "candidate_pairs": len(left), "accepted_pairs": int(np.sum(accepted)),
            "votes_required": 2, "distance_cap_m": 4.0,
            "seed_thresholds": list(SEED_THRESHOLDS),
        },
        "center_projection": center_projection,
        "gates": gates, "fit_observations": 142_184, "selection_observations": 45_942,
        "optimizer_steps": 0, "model_updates": 0,
        "c09_worlds_read": 0, "c10_worlds_read": 0, "mtare_worlds_read": 0,
        "duration_seconds": time.monotonic() - started,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    np.savez_compressed(
        args.output_dir / "association_pairs.npz", left=left, right=right,
        distance_m=distance, scores=scores, accepted=accepted,
    )
    np.savez_compressed(
        args.output_dir / "action_ensemble.npz", global_sequence_index=global_index,
        probability=probability, uncertainty=uncertainty.astype(np.float32),
    )
    _write_replay(args.output_dir, selected_replay)
    print(json.dumps(summary, sort_keys=True))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
