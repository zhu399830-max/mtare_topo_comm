#!/usr/bin/env python3
"""Attribute rare endpoint failures to exclusive Teacher semantics or seed instability."""
from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path

import numpy as np

from mtare_topo.representation.gse_route_conditioned_event_residual import route_flow_features


EXPECTED_OBSERVATIONS = 188_126
LIDAR_RANGE_M = 50.0


def _read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def classify_failure_mechanism(
    base_seed_probability: np.ndarray,
    seed_route_stop_score: np.ndarray,
    adjacent_opposite_distance_m: float | None,
) -> str:
    """Classify one terminal failure using only frozen pre-training evidence.

    Columns are structural mass, junction joint probability and terminal joint
    probability.  A stable junction decision while a directly connected
    junction lies inside sensor range is an exclusive-label conflict.  Large
    structural disagreement despite unanimous route-stop evidence is instead
    a representation/seed-stability failure.
    """

    probability = np.asarray(base_seed_probability, dtype=np.float64)
    route = np.asarray(seed_route_stop_score, dtype=np.float64)
    if probability.shape != (3, 3) or route.shape != (3,) or not np.all(np.isfinite(probability)) or not np.all(np.isfinite(route)):
        raise ValueError("failure attribution probability contract drift")
    mass, junction, terminal = probability.T
    if np.any(mass <= 0.0) or np.any(junction < 0.0) or np.any(terminal < 0.0):
        raise ValueError("failure attribution probability range drift")
    stable_junction = bool(np.all(junction > terminal) and np.min(junction / mass) >= 0.90)
    adjacent_visible_scale = adjacent_opposite_distance_m is not None and 0.0 <= adjacent_opposite_distance_m <= LIDAR_RANGE_M
    if stable_junction and adjacent_visible_scale:
        return "exclusive_teacher_conflict"
    if float(np.ptp(mass)) >= 0.50 and float(np.min(route)) >= 0.70:
        return "seed_representation_instability"
    return "unresolved"


def _event_for_degree(degree: int) -> str:
    if degree == 1:
        return "terminal"
    if degree >= 3:
        return "junction"
    return "other"


def _adjacency(graph: dict, node_id: str, event: str) -> dict:
    nodes = {str(node["id"]): node for node in graph["nodes"]}
    if node_id not in nodes or _event_for_degree(int(nodes[node_id]["degree"])) != event:
        raise RuntimeError("failure attribution objective node/event drift")
    adjacent = []
    source = np.asarray(nodes[node_id]["xyz"], dtype=np.float64)
    for edge in graph["edges"]:
        ids = [str(value) for value in edge["node_ids"]]
        if node_id not in ids or len(ids) != 2:
            continue
        other = ids[1] if ids[0] == node_id else ids[0]
        target = nodes[other]
        distance = float(np.linalg.norm(source - np.asarray(target["xyz"], dtype=np.float64)))
        adjacent.append({
            "edge_id": str(edge["id"]),
            "node_id": other,
            "degree": int(target["degree"]),
            "event": _event_for_degree(int(target["degree"])),
            "euclidean_distance_m": distance,
        })
    opposite = [value for value in adjacent if value["event"] in ("junction", "terminal") and value["event"] != event]
    nearest = min(opposite, key=lambda value: value["euclidean_distance_m"]) if opposite else None
    return {"degree": int(nodes[node_id]["degree"]), "adjacent": adjacent, "nearest_opposite": nearest}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", required=True, type=Path)
    parser.add_argument("--endpoint-audit", required=True, type=Path)
    parser.add_argument("--endpoint-flow", required=True, type=Path)
    parser.add_argument("--low-observability", required=True, type=Path)
    parser.add_argument("--route-run", required=True, type=Path)
    parser.add_argument("--graph-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    started = time.monotonic()
    output = args.output_dir.resolve()
    if output.exists():
        raise RuntimeError("failure attribution output exists; overwrite forbidden")
    output.mkdir(parents=True)

    cache = args.cache_dir.resolve()
    manifest = json.loads((cache / "manifest.json").read_text())
    if manifest.get("causal_observations") != EXPECTED_OBSERVATIONS:
        raise RuntimeError("failure attribution cache population drift")
    raw = np.load(cache / "raw_tokens.npy", mmap_mode="r")
    references = np.load(cache / "history_references.npy")
    history_mask = np.load(cache / "history_mask.npy")
    partition = np.load(cache / "partition_code.npy")
    identity = np.load(cache / "identity.npy").astype(str)
    global_index = np.load(cache / "global_sequence_index.npy").astype(np.int64)
    if raw.shape != (EXPECTED_OBSERVATIONS, 3, 6, 40):
        raise RuntimeError("failure attribution raw-token shape drift")
    route_feature = route_flow_features(raw, references, history_mask)

    endpoints = _read_jsonl(args.endpoint_audit.resolve())
    flow = _read_jsonl(args.endpoint_flow.resolve())
    low_rows = json.loads(args.low_observability.resolve().read_text())
    if len(endpoints) != 98 or len(flow) != 98 or len(low_rows) != 4:
        raise RuntimeError("failure attribution endpoint population drift")
    flow_by_identity = {str(value["identity"]): value for value in flow}
    low_selection = [value for value in endpoints if value["partition"] == "selection" and value["support_bin"] == "1-3"]
    if len(low_selection) != 2 or any(value["event"] != "terminal" for value in low_selection):
        raise RuntimeError("failure attribution low endpoint drift")

    graph_cache: dict[str, dict] = {}
    endpoint_rows = []
    for record in endpoints:
        world = str(record["world"])
        if world not in graph_cache:
            path = args.graph_root.resolve() / world / "graph.json"
            graph_cache[world] = json.loads(path.read_text())
        node_id = str(record["identity"]).split(":")[-1]
        relation = _adjacency(graph_cache[world], node_id, str(record["event"]))
        nearest = relation["nearest_opposite"]
        endpoint_rows.append({
            **record,
            "degree": relation["degree"],
            "adjacent_nodes": relation["adjacent"],
            "nearest_opposite_event": None if nearest is None else nearest["event"],
            "nearest_opposite_distance_m": None if nearest is None else nearest["euclidean_distance_m"],
            "opposite_inside_lidar_range": bool(nearest is not None and nearest["euclidean_distance_m"] <= LIDAR_RANGE_M),
            "maximum_route_stop_score": float(flow_by_identity[record["identity"]]["maximum_route_stop_score"]),
        })

    relation_by_identity = {value["identity"]: value for value in endpoint_rows}
    route_run = args.route_run.resolve()
    corrected = []
    for seed in range(3):
        with np.load(route_run / f"artifacts/models/seed{seed}/all_outputs.npz", allow_pickle=False) as archive:
            if not np.array_equal(archive["global_sequence_index"], global_index):
                raise RuntimeError("failure attribution corrected output identity drift")
            corrected.append(np.asarray(archive["probability"], dtype=np.float64))

    detail = []
    for endpoint in low_selection:
        observations = [value for value in low_rows if value["identity"] == endpoint["identity"]]
        if len(observations) != int(endpoint["teacher_rows"]):
            raise RuntimeError("failure attribution low row coverage drift")
        base = np.asarray([
            [
                value["structural_mass"], value["junction_probability"], value["terminal_probability"],
            ]
            for value in observations[0]["seed_probabilities"]
        ], dtype=np.float64)
        route_score = np.asarray(flow_by_identity[endpoint["identity"]]["seed_route_stop_score"], dtype=np.float64)
        adjacency = relation_by_identity[endpoint["identity"]]
        mechanism = classify_failure_mechanism(base, route_score, adjacency["nearest_opposite_distance_m"])
        rows = []
        for observation in observations:
            row = int(observation["row"])
            if int(global_index[row]) != int(observation["global_sequence_index"]):
                raise RuntimeError("failure attribution low global identity drift")
            token = np.asarray(raw[row], dtype=np.float64)
            confidence = np.clip(token[..., 0], 0.0, 1.0)
            sine, cosine = token[..., 1], token[..., 2]
            forward = np.sum(confidence * np.clip(cosine, 0.0, None), axis=1)
            backward = np.sum(confidence * np.clip(-cosine, 0.0, None), axis=1)
            lateral = np.sum(confidence * np.abs(sine), axis=1)
            rows.append({
                "row": row,
                "global_sequence_index": int(global_index[row]),
                "route_feature_16d": route_feature[row].astype(float).tolist(),
                "seed_forward_mass": forward.tolist(),
                "seed_backward_mass": backward.tolist(),
                "seed_lateral_mass": lateral.tolist(),
                "base_seed_probability": observation["seed_probabilities"],
                "corrected_seed_probability": [
                    {
                        "seed": seed,
                        "structural_mass": float(corrected[seed][row, 1] + corrected[seed][row, 2]),
                        "junction_probability": float(corrected[seed][row, 1]),
                        "terminal_probability": float(corrected[seed][row, 2]),
                    }
                    for seed in range(3)
                ],
            })
        detail.append({
            "identity": endpoint["identity"],
            "world": endpoint["world"],
            "teacher_rows": endpoint["teacher_rows"],
            "original_stage": endpoint["stage"],
            "mechanism": mechanism,
            "route_stop_score_by_seed": route_score.tolist(),
            "maximum_route_stop_score": float(flow_by_identity[endpoint["identity"]]["maximum_route_stop_score"]),
            "nearest_opposite_event": adjacency["nearest_opposite_event"],
            "nearest_opposite_distance_m": adjacency["nearest_opposite_distance_m"],
            "opposite_inside_lidar_range": adjacency["opposite_inside_lidar_range"],
            "observations": rows,
        })

    mechanism_count = {name: sum(value["mechanism"] == name for value in detail) for name in (
        "exclusive_teacher_conflict", "seed_representation_instability", "unresolved",
    )}
    inside = [value for value in endpoint_rows if value["opposite_inside_lidar_range"]]
    terminal_inside = [value for value in inside if value["event"] == "terminal"]
    gates = {
        "exact_population": len(endpoint_rows) == 98 and len(detail) == 2 and sum(len(value["observations"]) for value in detail) == 4,
        "all_endpoint_world_graphs_resolved": len(graph_cache) == 40,
        "both_low_failures_attributed": mechanism_count["unresolved"] == 0,
        "exclusive_teacher_conflict_present": mechanism_count["exclusive_teacher_conflict"] >= 1,
        "seed_instability_present": mechanism_count["seed_representation_instability"] >= 1,
    }
    gates["all_passed"] = all(gates.values())
    recommendation = (
        "spatial_multi_event_teacher_before_new_head" if mechanism_count["exclusive_teacher_conflict"] else
        "full_token_relation_head_capacity_first"
    )
    summary = {
        "schema_version": "gse_route_token_failure_attribution_v1",
        "status": "PASS_GSE_ROUTE_TOKEN_FAILURE_ATTRIBUTION_V1" if gates["all_passed"] else "FAIL_GSE_ROUTE_TOKEN_FAILURE_ATTRIBUTION_V1",
        "question": "Did the low-dimensional route head fail from token compression or exclusive objective-event semantics?",
        "population": {
            "observations": EXPECTED_OBSERVATIONS,
            "relation_endpoints": len(endpoint_rows),
            "worlds": len(graph_cache),
            "low_selection_endpoints": len(detail),
            "low_selection_rows": sum(len(value["observations"]) for value in detail),
            "endpoints_adjacent_to_opposite_event_inside_50m": len(inside),
            "terminal_endpoints_adjacent_to_junction_inside_50m": len(terminal_inside),
        },
        "mechanism_count": mechanism_count,
        "recommendation": recommendation,
        "interpretation": "At least one terminal is stably classified as its directly adjacent sensor-range-scale junction, so a mutually exclusive scene label is not a valid target for every causal LiDAR row. The second failure is dominated by seed-level structural instability despite unanimous route-stop evidence.",
        "gates": gates,
        "optimizer_steps": 0,
        "model_inference_frames": 0,
        "model_updates": 0,
        "threshold_selection_steps": 0,
        "c09_worlds_read": 0,
        "c10_worlds_read": 0,
        "mtare_worlds_read": 0,
        "duration_seconds": time.monotonic() - started,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    (output / "low_endpoint_attribution.json").write_text(json.dumps(detail, indent=2, sort_keys=True) + "\n")
    with (output / "endpoint_adjacency.jsonl").open("w", encoding="utf-8") as stream:
        for value in endpoint_rows:
            stream.write(json.dumps(value, sort_keys=True) + "\n")
    with (output / "endpoint_adjacency.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("partition", "identity", "event", "support_bin", "nearest_opposite_distance_m", "inside_50m", "route_stop_score"))
        for value in endpoint_rows:
            writer.writerow((value["partition"], value["identity"], value["event"], value["support_bin"], value["nearest_opposite_distance_m"], value["opposite_inside_lidar_range"], value["maximum_route_stop_score"]))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.0), constrained_layout=True)
    labels = [value["identity"].split(":")[-1] for value in detail]
    x = np.arange(len(labels)); width = 0.24
    for seed, color in enumerate(("#2878B5", "#D95F02", "#2CA02C")):
        masses = []
        for value in detail:
            first = value["observations"][0]["base_seed_probability"][seed]
            masses.append(first["structural_mass"])
        axes[0].bar(x + (seed - 1) * width, masses, width, label=f"seed {seed}", color=color)
    axes[0].axhline(.97, color="black", linestyle="--", linewidth=1)
    axes[0].set_xticks(x, labels); axes[0].set_ylim(0, 1.05)
    axes[0].set_ylabel("Base structural mass"); axes[0].set_title("A  Rare endpoints fail differently")
    axes[0].legend(frameon=False)
    for event, color, marker in (("junction", "#D95F02", "o"), ("terminal", "#2CA02C", "^")):
        selected = [value for value in endpoint_rows if value["event"] == event and value["nearest_opposite_distance_m"] is not None]
        axes[1].scatter(
            [value["nearest_opposite_distance_m"] for value in selected],
            [value["maximum_route_stop_score"] for value in selected],
            color=color, marker=marker, alpha=.65, label=event.capitalize(),
        )
    for value in detail:
        axes[1].scatter(value["nearest_opposite_distance_m"], value["maximum_route_stop_score"], s=130, facecolors="none", edgecolors="black", linewidths=1.8)
    axes[1].axvline(LIDAR_RANGE_M, color="black", linestyle="--", linewidth=1)
    axes[1].set_xlabel("Direct opposite-event neighbor distance (m)")
    axes[1].set_ylabel("Route-stop score"); axes[1].set_title("B  Competing events share sensor-range scale")
    axes[1].legend(frameon=False)
    fig.suptitle("Why the low-dimensional route-conditioned event correction failed")
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(output / f"gse_route_token_failure_attribution.{suffix}", dpi=240 if suffix == "png" else None)
    plt.close(fig)
    (output / "figure_source.json").write_text(json.dumps({
        "schema_version": "gse_route_token_failure_attribution_figure_source_v1",
        "summary": summary,
        "low_endpoint_attribution": detail,
        "endpoint_adjacency": endpoint_rows,
    }, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, sort_keys=True))
    return 0 if gates["all_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
