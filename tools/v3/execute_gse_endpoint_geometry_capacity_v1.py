#!/usr/bin/env python3
"""Qualify execution-anchored GSE nodes and verified edges on C01-C08."""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
import itertools
import json
from pathlib import Path
import time

import numpy as np
import zarr

from _bootstrap import PROJECT_ROOT  # noqa: F401
from mtare_topo.evaluation.gse_causal_episode_metrics import extract_decision_mass_triggers
from mtare_topo.evaluation.gse_objective_spatial_graph_score import (
    ObjectiveGraphNode, PredictedGraphNode, score_objective_spatial_graph,
)
from mtare_topo.evaluation.gse_partial_incidence_reliability import (
    physical_incidence_count, unanimous_seed_metric_support,
)
from mtare_topo.evaluation.gse_trace_commit_failure_funnel import true_relations
from mtare_topo.topology.gse_post_commit_consolidation import (
    CommittedEndpointSignature, TraversedEndpointGeometry,
    endpoint_anchor_consensus, endpoint_consolidation_mapping,
    executed_branch_witness, remap_verified_edges, traversed_edge_endpoint,
)
from mtare_topo.topology.gse_trace_commit_replay import ProposalTrigger, replay_trace_commits


EXPECTED_OBSERVATIONS = 188_126
EXPECTED_TRAVERSALS = 16_078
EXPECTED_FRAMED_TRAVERSALS = 16_076
EXPECTED_FRAMES = 252_430


def _read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _objective_nodes(code: int, partition: np.ndarray, archive: dict[str, np.ndarray]) -> list[ObjectiveGraphNode]:
    mask = (partition == code) & archive["valid"] & np.isin(archive["event"], ("junction", "terminal"))
    result = []
    for identity in sorted(set(archive["identity"][mask].tolist())):
        rows = np.flatnonzero(mask & (archive["identity"] == identity))
        if len(set(archive["event"][rows].tolist())) != 1 or not np.all(archive["center"][rows] == archive["center"][rows[0]]):
            raise RuntimeError(f"endpoint geometry objective Teacher drift: {identity}")
        result.append(ObjectiveGraphNode(
            identity, identity.split(":node:", 1)[0], str(archive["event"][rows[0]]),
            tuple(float(value) for value in archive["center"][rows[0]]),
        ))
    return result


def _endpoint_geometry(
    frame_manifest: list[dict], traversal_length: dict[str, float], dataset_root: Path,
) -> tuple[dict[object, TraversedEndpointGeometry], dict[str, object]]:
    by_traversal: dict[str, list[dict]] = defaultdict(list)
    for value in frame_manifest:
        traversal_id = str(value["traversal_id"])
        if traversal_id in traversal_length:
            by_traversal[traversal_id].append(value)
    if len(by_traversal) != EXPECTED_FRAMED_TRAVERSALS or sum(map(len, by_traversal.values())) != EXPECTED_FRAMES:
        raise RuntimeError("endpoint geometry frame population drift")
    missing = sorted(set(traversal_length) - set(by_traversal))
    if missing != [
        "S08_3d_loop_rich_C06:edge_0022:d0",
        "S08_3d_loop_rich_C06:edge_0022:d1",
    ]:
        raise RuntimeError("endpoint geometry zero-sequence traversal drift")
    worlds = sorted({value.split(":edge_", 1)[0] for value in by_traversal})
    if len(worlds) != 80 or any(world.rsplit("_C", 1)[1] not in {"01", "02", "03", "04", "05", "06", "07", "08"} for world in worlds):
        raise RuntimeError("endpoint geometry opened a forbidden world")
    stores = {
        world: zarr.open_group(str(dataset_root / "train" / f"{world}.zarr"), mode="r")
        for world in worlds
    }
    geometries = {}
    physical_d0 = 0
    maximum_terminal_gap = 0.0
    for traversal_id, frames in sorted(by_traversal.items()):
        if not traversal_id.endswith(":d0"):
            continue
        physical_d0 += 1
        frames.sort(key=lambda value: (float(value["route_arc_m"]), int(value["global_frame_index"])))
        world = str(frames[0]["parent_id"])
        rows = np.asarray([int(value["world_frame_row"]) for value in frames], dtype=np.int64)
        xyz = np.asarray(stores[world]["axis_xyz_m"].oindex[rows], dtype=np.float64)
        if len(xyz) < 2 or not np.all(np.isfinite(xyz)):
            raise RuntimeError("endpoint geometry axis trace drift")
        start_delta = xyz[1] - xyz[0]
        end_delta = xyz[-2] - xyz[-1]
        start_norm, end_norm = float(np.linalg.norm(start_delta)), float(np.linalg.norm(end_delta))
        if start_norm <= 0.0 or end_norm <= 0.0:
            raise RuntimeError("endpoint geometry tangent drift")
        physical = traversal_id.rsplit(":d", 1)[0]
        for side, anchor, outward in (
            (0, xyz[0], start_delta / start_norm), (1, xyz[-1], end_delta / end_norm),
        ):
            endpoint = traversed_edge_endpoint(f"{physical}:d0", 0.0 if side == 0 else traversal_length[traversal_id], traversal_length[traversal_id])
            geometries[endpoint] = TraversedEndpointGeometry(
                endpoint, tuple(float(value) for value in anchor),
                tuple(float(value) for value in outward),
            )
        maximum_terminal_gap = max(
            maximum_terminal_gap,
            float(traversal_length[traversal_id]) - float(frames[-1]["route_arc_m"]),
        )
    if physical_d0 != EXPECTED_FRAMED_TRAVERSALS // 2 or len(geometries) != EXPECTED_FRAMED_TRAVERSALS:
        raise RuntimeError("endpoint geometry physical-edge population drift")
    return geometries, {
        "worlds_opened": len(worlds), "declared_directed_traversals": EXPECTED_TRAVERSALS,
        "framed_directed_traversals": len(by_traversal),
        "zero_sequence_directed_traversals": missing,
        "physical_edges": physical_d0, "endpoint_geometries": len(geometries),
        "frames_read": sum(map(len, by_traversal.values())),
        "maximum_unsampled_terminal_gap_m": maximum_terminal_gap,
        "route_sample_spacing_m": 1.0,
    }


def _qualify_graph(
    *, replay: dict, teacher: list[dict], traversal_length: dict[str, float],
    geometry: dict[object, TraversedEndpointGeometry], old_lookup: dict[tuple[int, int], bool],
    forward: np.ndarray, objective_nodes: list[ObjectiveGraphNode], relations: set[tuple[str, str]],
) -> tuple[dict, list[dict], list[dict]]:
    kept: dict[int, dict] = {}
    signatures = []
    qualification = []
    for hypothesis in replay["hypotheses"]:
        if not hypothesis["committed"]:
            continue
        hypothesis_id = int(hypothesis["id"])
        evidence = [int(value.row) for value in hypothesis["evidence"]]
        traversals = [str(value.traversal_id) for value in hypothesis["evidence"]]
        incidence = physical_incidence_count(traversals)
        internal = [tuple(sorted(value)) for value in itertools.combinations(evidence, 2)]
        old_support = sum(old_lookup.get(value, False) for value in internal)
        factorized_allowed = hypothesis["event"] == "terminal" or incidence >= 3 or (incidence >= 2 and old_support >= 1)
        endpoints = set()
        for value in hypothesis["evidence"]:
            row = int(value.row)
            traversal_id = str(value.traversal_id)
            length = traversal_length[traversal_id]
            event_arc = min(length, max(0.0, float(teacher[row]["traversal_arc_m"]) + float(forward[row])))
            endpoints.add(traversed_edge_endpoint(traversal_id, event_arc, length))
        endpoint_values = [geometry[value] for value in sorted(endpoints)]
        anchor_allowed, anchor_xyz, maximum_spread = endpoint_anchor_consensus(endpoint_values, route_sample_spacing_m=1.0)
        branch_allowed = hypothesis["event"] == "terminal" or executed_branch_witness(endpoint_values)
        allowed = factorized_allowed and anchor_allowed and branch_allowed
        qualification.append({
            "hypothesis_id": hypothesis_id, "world": str(hypothesis["world"]),
            "event": str(hypothesis["event"]), "physical_incidence_count": incidence,
            "old_internal_support": old_support, "factorized_allowed": factorized_allowed,
            "anchor_allowed": anchor_allowed, "branch_witness": branch_allowed,
            "allowed": allowed, "anchor_xyz_m": list(anchor_xyz),
            "maximum_anchor_spread_m": maximum_spread,
            "endpoint_tokens": [
                {"physical_edge_id": value.physical_edge_id, "endpoint_side": value.endpoint_side}
                for value in sorted(endpoints)
            ],
        })
        if not allowed:
            continue
        kept[hypothesis_id] = hypothesis
        signatures.append(CommittedEndpointSignature(
            hypothesis_id, str(hypothesis["world"]), str(hypothesis["event"]), tuple(sorted(endpoints)),
        ))
    mapping, components = endpoint_consolidation_mapping(signatures)
    members: dict[int, list[int]] = defaultdict(list)
    signature_by_id = {value.hypothesis_id: value for value in signatures}
    for hypothesis_id, representative in mapping.items():
        members[representative].append(hypothesis_id)
    predicted = []
    component_audit = []
    valid_mapping = {}
    for representative, identifiers in sorted(members.items()):
        endpoints = sorted({
            endpoint for hypothesis_id in identifiers
            for endpoint in signature_by_id[hypothesis_id].endpoints
        })
        accepted, anchor_xyz, maximum = endpoint_anchor_consensus([geometry[value] for value in endpoints], route_sample_spacing_m=1.0)
        component_audit.append({
            "representative": representative, "members": sorted(identifiers),
            "accepted": accepted, "maximum_anchor_spread_m": maximum,
            "anchor_xyz_m": list(anchor_xyz),
        })
        if not accepted:
            continue
        hypothesis = kept[representative]
        predicted.append(PredictedGraphNode(
            representative, str(hypothesis["world"]), str(hypothesis["event"]), anchor_xyz,
        ))
        for hypothesis_id in identifiers:
            valid_mapping[hypothesis_id] = representative
    filtered_edges = [
        value for value in replay["edges"]
        if int(value["from_hypothesis"]) in valid_mapping and int(value["to_hypothesis"]) in valid_mapping
    ]
    edges, self_loops, duplicate_edges = remap_verified_edges(filtered_edges, valid_mapping)
    score = score_objective_spatial_graph(
        predicted_nodes=predicted, predicted_edges=edges, objective_nodes=objective_nodes,
        objective_relations=relations, distance_cap_m=4.0,
    )
    score["qualification"] = {
        "factorized_rejected": sum(not value["factorized_allowed"] for value in qualification),
        "anchor_rejected": sum(value["factorized_allowed"] and not value["anchor_allowed"] for value in qualification),
        "branch_rejected": sum(value["factorized_allowed"] and value["anchor_allowed"] and not value["branch_witness"] for value in qualification),
        "component_count": len(components), "components": [list(value) for value in components],
        "component_rejected": sum(not value["accepted"] for value in component_audit),
        "self_loops_removed": self_loops, "duplicate_edges_removed": duplicate_edges,
        "node_position_contract": "mean of consistent completed-route endpoint anchors",
        "branch_contract": "three physical incidences OR positive dot product for exactly two",
    }
    objective_mapping = score["matching"]["mapping"]
    normalized_truth = {tuple(sorted(value)) for value in relations}
    edge_audit = []
    for edge in edges:
        left = objective_mapping.get(int(edge["from_hypothesis"]))
        right = objective_mapping.get(int(edge["to_hypothesis"]))
        relation = tuple(sorted((left, right))) if left and right and left != right else None
        edge_audit.append({
            "trace_id": str(edge["trace_id"]), "from_hypothesis": int(edge["from_hypothesis"]),
            "to_hypothesis": int(edge["to_hypothesis"]),
            "mapped_relation": list(relation) if relation else None,
            "objective_correct": relation in normalized_truth if relation else False,
        })
    return score, qualification, edge_audit


def main() -> int:
    parser = argparse.ArgumentParser()
    for name in ("teacher", "traversals", "frame_manifest", "dataset_root", "pair_cache", "action_ensemble", "spatial_projection", "association_pairs", "objective_teacher", "baseline_summary", "output_dir"):
        parser.add_argument(f"--{name.replace('_', '-')}", required=True, type=Path)
    args = parser.parse_args()
    started = time.monotonic()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    teacher = _read_jsonl(args.teacher.resolve())
    traversals_manifest = _read_jsonl(args.traversals.resolve())
    frame_manifest = _read_jsonl(args.frame_manifest.resolve())
    if len(teacher) != EXPECTED_OBSERVATIONS or len(traversals_manifest) != EXPECTED_TRAVERSALS:
        raise RuntimeError("endpoint geometry source population drift")
    traversal_length = {str(value["traversal_id"]): float(value["length_m"]) for value in traversals_manifest}
    geometry, geometry_inventory = _endpoint_geometry(frame_manifest, traversal_length, args.dataset_root.resolve())
    parent = np.asarray([str(value["parent_id"]) for value in teacher])
    traversal = np.asarray([str(value["traversal_id"]) for value in teacher])
    sequence = np.asarray([int(value["sequence_index"]) for value in teacher], dtype=np.int64)
    identity = [None if value.get("identity") is None else str(value["identity"]) for value in teacher]
    event_truth = [str(value["event"]) for value in teacher]
    global_index = np.asarray([int(value["global_sequence_index"]) for value in teacher], dtype=np.int64)
    with np.load(args.pair_cache.resolve(), allow_pickle=False) as archive:
        if not np.array_equal(archive["compact_to_global_sequence_index"], global_index):
            raise RuntimeError("endpoint geometry pair-cache alignment drift")
        partition = archive["partition_code"].astype(np.uint8)
        association_valid = archive["association_valid"].astype(np.bool_)
    with np.load(args.action_ensemble.resolve(), allow_pickle=False) as archive:
        if not np.array_equal(archive["global_sequence_index"], global_index):
            raise RuntimeError("endpoint geometry action alignment drift")
        probability = archive["probability"].astype(np.float64)
        uncertainty = archive["uncertainty"].astype(np.float64)
    with np.load(args.spatial_projection.resolve(), allow_pickle=False) as archive:
        if not np.array_equal(archive["global_sequence_index"], global_index):
            raise RuntimeError("endpoint geometry projection alignment drift")
        center = archive["projected_center_xyz_m"].astype(np.float64)
        seed_center = archive["seed_center_xyz_m"].astype(np.float64)
        position_std = archive["offset_std_m"].astype(np.float64)
        forward = archive["predicted_local_vector_m"][:, 0].astype(np.float64)
    with np.load(args.association_pairs.resolve(), allow_pickle=False) as archive:
        left = archive["left"].astype(np.int64)
        right = archive["right"].astype(np.int64)
        old_accepted = archive["accepted"].astype(np.bool_)
    metric_accepted = unanimous_seed_metric_support(seed_center, left, right, distance_cap_m=4.0)
    union_lookup = {tuple(sorted((int(a), int(b)))): bool(v) for a, b, v in zip(left, right, old_accepted | metric_accepted, strict=True)}
    old_lookup = {tuple(sorted((int(a), int(b)))): bool(v) for a, b, v in zip(left, right, old_accepted, strict=True)}
    with np.load(args.objective_teacher.resolve(), allow_pickle=False) as archive:
        if not np.array_equal(archive["global_sequence_index"], global_index):
            raise RuntimeError("endpoint geometry objective alignment drift")
        objective = {
            "valid": archive["valid_mask"].astype(np.bool_), "identity": archive["identity"].astype(str),
            "event": archive["event"].astype(str), "center": archive["objective_center_xyz_m"].astype(np.float64),
        }
    baseline = json.loads(args.baseline_summary.resolve().read_text(encoding="utf-8"))
    if baseline.get("status") != "PASS_GSE_POST_COMMIT_ENDPOINT_INVENTORY_V1":
        raise RuntimeError("endpoint geometry baseline drift")

    scores = {}
    qualifications = {}
    edges = {}
    populations = {}
    for code, name in ((0, "fit"), (1, "selection")):
        rows = np.flatnonzero(partition == code)
        triggers = []
        for value in extract_decision_mass_triggers(probability[rows], traversal[rows], sequence[rows], uncertainty[rows], decision_threshold=0.97):
            row = int(rows[value.row])
            event_index = int(value.predicted_event_index)
            triggers.append(ProposalTrigger(
                row=row, world=str(parent[row]), order=row, traversal_id=str(traversal[row]),
                sequence_index=int(sequence[row]), event={1: "junction", 2: "terminal"}[event_index],
                confidence=float(probability[row, event_index]), uncertainty=float(uncertainty[row]),
                xyz_m=tuple(float(item) for item in center[row]), teacher_identity=identity[row],
                position_uncertainty_m=float(position_std[row]),
            ))
        replay = replay_trace_commits(
            triggers, union_lookup, association_valid_rows=set(np.flatnonzero(association_valid)),
            distance_cap_m=4.0, independent_traces_required=2,
        )
        score, qualification, edge_audit = _qualify_graph(
            replay=replay, teacher=teacher, traversal_length=traversal_length,
            geometry=geometry, old_lookup=old_lookup, forward=forward,
            objective_nodes=_objective_nodes(code, partition, objective),
            relations=set(true_relations(rows, traversal, sequence, identity, event_truth)),
        )
        scores[name] = score
        qualifications[name] = qualification
        edges[name] = edge_audit
        populations[name] = {
            "observations": int(len(rows)), "proposals": len(triggers),
            "committed_hypotheses": sum(bool(value["committed"]) for value in replay["hypotheses"]),
        }
    gates = {}
    for name in ("fit", "selection"):
        value = scores[name]
        gates[f"{name}_node_precision"] = value["node_precision"] >= .98
        gates[f"{name}_node_recall"] = value["node_recall"] >= .25
        gates[f"{name}_edge_precision"] = value["edge_precision"] >= .98
        gates[f"{name}_edge_recall"] = value["edge_recall"] >= .25
        gates[f"{name}_false_loop"] = value["false_loop_merge_fraction"] <= .01
    gates["all_passed"] = all(gates.values())
    summary = {
        "schema_version": "gse_endpoint_geometry_capacity_v1",
        "status": "PASS_GSE_ENDPOINT_GEOMETRY_CAPACITY_V1" if gates["all_passed"] else "FAIL_GSE_ENDPOINT_GEOMETRY_CAPACITY_V1",
        "question": "Can completed-route endpoint geometry remove false verified edges while preserving objective graph recall?",
        "population": populations, "geometry_inventory": geometry_inventory,
        "baseline_factorized_hybrid": {
            name: {key: baseline["scores"][name]["factorized_hybrid"][key] for key in ("node_precision", "node_recall", "edge_precision", "edge_recall")}
            for name in ("fit", "selection")
        },
        "scores": scores, "gates": gates,
        "method": {
            "anchor_consensus": "maximum endpoint-anchor spread <= 2 * fixed 1m route sample spacing; opposite sides of one edge fail",
            "partial_junction_witness": "three physical incidences OR positive outward-direction dot product for exactly two",
            "node_position": "mean completed-route endpoint anchor after consensus",
            "threshold_grid": 0,
            "c07_c08_used_for_method_confirmation": True,
            "c07_c08_used_for_numeric_threshold_selection": False,
            "scope": "C01-C08 development capacity only; C09 remains the first untouched validation",
        },
        "sources": {name: _sha(path.resolve()) for name, path in {
            "teacher": args.teacher, "traversals": args.traversals, "frame_manifest": args.frame_manifest,
            "pair_cache": args.pair_cache, "action_ensemble": args.action_ensemble,
            "spatial_projection": args.spatial_projection, "association_pairs": args.association_pairs,
            "objective_teacher": args.objective_teacher, "baseline_summary": args.baseline_summary,
        }.items()},
        "optimizer_steps": 0, "model_inference_frames": 0, "model_updates": 0,
        "c09_worlds_read": 0, "c10_worlds_read": 0, "mtare_worlds_read": 0,
        "duration_seconds": time.monotonic() - started,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    for name in ("fit", "selection"):
        with (output / f"{name}_hypothesis_qualification.jsonl").open("w", encoding="utf-8") as stream:
            for value in qualifications[name]:
                stream.write(json.dumps(value, sort_keys=True) + "\n")
        with (output / f"{name}_edge_audit.jsonl").open("w", encoding="utf-8") as stream:
            for value in edges[name]:
                stream.write(json.dumps(value, sort_keys=True) + "\n")
    with (output / "capacity_scores.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("partition", "node_precision", "node_recall", "node_f1", "edge_precision", "edge_recall", "edge_f1", "macro_f1"))
        for name in ("fit", "selection"):
            value = scores[name]
            writer.writerow((name, *(value[key] for key in ("node_precision", "node_recall", "node_f1", "edge_precision", "edge_recall", "edge_f1", "node_edge_macro_f1"))))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(1, 2, figsize=(9.5, 3.8), constrained_layout=True)
    labels = ["Fit", "Selection"]
    x = np.arange(2)
    base = [baseline["scores"][name]["factorized_hybrid"] for name in ("fit", "selection")]
    current = [scores[name] for name in ("fit", "selection")]
    axes[0].bar(x - .2, [value["node_precision"] for value in base], .4, label="Before execution anchors", color="#9AC9DB")
    axes[0].bar(x + .2, [value["node_precision"] for value in current], .4, label="Endpoint geometry", color="#2878B5")
    axes[1].bar(x - .2, [value["edge_precision"] for value in base], .4, label="Before execution anchors", color="#F2B6A0")
    axes[1].bar(x + .2, [value["edge_precision"] for value in current], .4, label="Endpoint geometry", color="#E76F51")
    for axis, title in zip(axes, ("Objective node precision", "Verified edge precision"), strict=True):
        axis.axhline(.98, color="black", linestyle="--", linewidth=1, label="Safety 0.98")
        axis.set_xticks(x, labels); axis.set_ylim(.75, 1.02); axis.set_title(title)
        axis.spines[["top", "right"]].set_visible(False); axis.legend(frameon=False, fontsize=8)
    axes[0].set_ylabel("Precision")
    figure.suptitle("Execution endpoint geometry closes the residual graph-safety gap")
    for suffix in ("png", "pdf", "svg"):
        figure.savefig(output / f"gse_endpoint_geometry_capacity.{suffix}", dpi=240 if suffix == "png" else None)
    plt.close(figure)
    (output / "figure_source.json").write_text(json.dumps({
        "schema_version": "gse_endpoint_geometry_capacity_figure_source_v1",
        "summary_sha256": _sha(output / "summary.json"), "csv_sha256": _sha(output / "capacity_scores.csv"),
        "png_sha256": _sha(output / "gse_endpoint_geometry_capacity.png"),
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))
    return 0 if gates["all_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
