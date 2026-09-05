#!/usr/bin/env python3
"""Teacher-free C09 inference, trace replay and execution-endpoint qualification."""

from __future__ import annotations

import argparse
from collections import defaultdict
import gc
import hashlib
import itertools
import json
from pathlib import Path
import shutil
import time

import numpy as np
import zarr

from _bootstrap import PROJECT_ROOT  # noqa: F401
from mtare_topo.data.cano_sensor_smoke import MAX_RANGE_M, NEAR_RANGE_M
from mtare_topo.evaluation.gse_causal_episode_metrics import extract_decision_mass_triggers
from mtare_topo.evaluation.gse_partial_incidence_reliability import (
    physical_incidence_count,
    unanimous_seed_metric_support,
)
from mtare_topo.evaluation.gse_spatial_center_projection import (
    combine_seed_projections,
    project_local_vectors,
)
from mtare_topo.representation.gse_action_set_node import (
    ActionSetNodeDetector,
    raw_action_tokens_one_seed,
)
from mtare_topo.representation.gse_causal_episode_detector import encode_spatial_scan_features
from mtare_topo.representation.gse_event_center_offset import EventCenterOffsetHead
from mtare_topo.representation.gse_graph import GeometrySemanticEventNet
from mtare_topo.representation.gse_spatial_event_center import (
    SpatialEventCenterDecoder,
    SpatialLongitudinalCorrector,
)
from mtare_topo.teacher.gse_event_center_teacher import route_local_basis
from mtare_topo.teacher.gse_factorized_association_teacher import (
    causal_history_row_references_unordered,
)
from mtare_topo.topology.gse_post_commit_consolidation import (
    CommittedEndpointSignature,
    TraversedEndpointGeometry,
    endpoint_anchor_consensus,
    endpoint_consolidation_mapping,
    executed_branch_witness,
    remap_verified_edges,
    traversed_edge_endpoint,
)
from mtare_topo.topology.gse_trace_commit_replay import ProposalTrigger, replay_trace_commits


EXPECTED_WORLDS = 10
EXPECTED_OBSERVATIONS = 24_462
EXPECTED_FRAMES = 32_678
EXPECTED_TRAVERSALS = 2_054
TOKEN_KEYS = (
    "exit_confidence", "exit_heading_unit", "exit_opening_width_m",
    "exit_vertical_profile", "exit_descriptor",
)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True) + "\n")


def _metadata(
    dataset_run: Path, sequence_manifest: Path, canonical_global_index: np.ndarray,
) -> dict[str, object]:
    source = [
        row for row in _read_jsonl(sequence_manifest)
        if str(row.get("split")) == "validation" and str(row.get("parent_id", "")).endswith("_C09")
    ]
    by_index = {int(row["global_sequence_index"]): row for row in source}
    canonical = np.asarray(canonical_global_index, dtype=np.int64)
    if (
        len(source) != EXPECTED_OBSERVATIONS or len(by_index) != EXPECTED_OBSERVATIONS
        or canonical.shape != (EXPECTED_OBSERVATIONS,)
        or len(np.unique(canonical)) != EXPECTED_OBSERVATIONS
        or set(canonical.tolist()) != set(by_index)
    ):
        raise RuntimeError("C09 sequence population drift")
    manifest = [by_index[int(value)] for value in canonical]
    worlds = sorted({str(row["parent_id"]) for row in manifest})
    if len(worlds) != EXPECTED_WORLDS:
        raise RuntimeError("C09 world population drift")
    groups = {
        world: zarr.open_group(
            str(dataset_run / "artifacts/dataset/validation" / f"{world}.zarr"), mode="r"
        )
        for world in worlds
    }
    parent = np.asarray([str(row["parent_id"]) for row in manifest])
    traversal = np.asarray([str(row["traversal_id"]) for row in manifest])
    observation = np.asarray([str(row["observation_id"]) for row in manifest])
    global_index = canonical.copy()
    world_order = np.asarray([int(row["world_sequence_row"]) for row in manifest], dtype=np.int64)
    anchor_row = np.asarray([int(row["world_anchor_frame_row"]) for row in manifest], dtype=np.int64)
    sequence = np.asarray([int(value.rsplit(":s", 1)[1]) for value in observation], dtype=np.int64)
    sensor = np.empty((EXPECTED_OBSERVATIONS, 3), dtype=np.float32)
    tangent = np.empty_like(sensor)
    route_arc = np.empty(EXPECTED_OBSERVATIONS, dtype=np.float64)
    association_valid = np.empty(EXPECTED_OBSERVATIONS, dtype=np.bool_)
    five_references = np.empty((EXPECTED_OBSERVATIONS, 5), dtype=np.int64)
    for world in worlds:
        rows = np.flatnonzero(parent == world)
        group = groups[world]
        local_sequence = world_order[rows]
        local_anchor = anchor_row[rows]
        if not np.array_equal(
            np.asarray(group["global_sequence_index"].oindex[local_sequence], dtype=np.int64),
            global_index[rows],
        ):
            raise RuntimeError(f"C09 sequence/shard alignment drift: {world}")
        sensor[rows] = np.asarray(group["sensor_xyz_m"].oindex[local_anchor], dtype=np.float32)
        tangent[rows] = np.asarray(group["tangent_world_xyz"].oindex[local_anchor], dtype=np.float32)
        route_arc[rows] = np.asarray(group["route_arc_m"].oindex[local_anchor], dtype=np.float64)
        association_valid[rows] = np.asarray(
            group["association_valid_mask"].oindex[local_sequence], dtype=np.bool_
        )
        five_references[rows] = np.asarray(
            group["global_frame_references"].oindex[local_sequence], dtype=np.int64
        )
    tangent_norm = np.linalg.norm(tangent.astype(np.float64), axis=1)
    tangent_valid = tangent_norm > 0.5
    if not np.all(tangent_valid):
        raise RuntimeError("C09 has a degenerate runtime route tangent")
    return {
        "manifest": manifest, "groups": groups, "worlds": worlds,
        "parent": parent, "traversal": traversal, "observation": observation,
        "global_index": global_index, "world_order": world_order,
        "sequence": sequence, "sensor": sensor, "tangent": tangent,
        "route_arc": route_arc, "association_valid": association_valid,
        "five_references": five_references,
        "basis": route_local_basis(tangent, tangent_valid),
    }


def _validation_tokens(paths: list[Path], global_index: np.ndarray) -> np.ndarray:
    tokens = np.empty((EXPECTED_OBSERVATIONS, 3, 6, 40), dtype=np.float16)
    for seed, path in enumerate(paths):
        with np.load(path, allow_pickle=False) as archive:
            if not np.array_equal(archive["global_sequence_index"], global_index):
                raise RuntimeError(f"C09 validation token identity drift: seed{seed}")
            tokens[:, seed] = raw_action_tokens_one_seed(archive).astype(np.float16)
    return tokens


def _action_inference(
    tokens: np.ndarray, traversal: np.ndarray, sequence: np.ndarray,
    checkpoints: list[Path], normalization_dir: Path, device,
) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
    import torch

    references, mask = causal_history_row_references_unordered(traversal, sequence, maximum=5)
    mean = np.load(normalization_dir / "normalization_mean.npy").astype(np.float32)
    scale = np.load(normalization_dir / "normalization_scale.npy").astype(np.float32)
    seed_probability: list[np.ndarray] = []
    seed_context: list[np.ndarray] = []
    batch_size = 256
    for seed, path in enumerate(checkpoints):
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        if checkpoint.get("schema_version") != "gse_action_set_node_checkpoint_v1" or checkpoint.get("seed") != seed:
            raise RuntimeError(f"C09 action checkpoint drift: seed{seed}")
        model = ActionSetNodeDetector().to(device)
        model.load_state_dict(checkpoint["model"], strict=True)
        model.eval()
        probability = np.empty((EXPECTED_OBSERVATIONS, 3), dtype=np.float32)
        context = np.empty((EXPECTED_OBSERVATIONS, 128), dtype=np.float32)
        with torch.inference_mode():
            for start in range(0, EXPECTED_OBSERVATIONS, batch_size):
                stop = min(start + batch_size, EXPECTED_OBSERVATIONS)
                rows = np.arange(start, stop, dtype=np.int64)
                local_mask = mask[rows]
                safe = references[rows].copy()
                fallback = np.broadcast_to(rows[:, None], safe.shape)
                safe[~local_mask] = fallback[~local_mask]
                values = (np.asarray(tokens[safe], dtype=np.float32) - mean) / scale
                values[~local_mask] = 0.0
                result = model(
                    torch.from_numpy(values).to(device),
                    torch.from_numpy(local_mask).to(device),
                )
                probability[start:stop] = result["decision_probability"].cpu().numpy()
                context[start:stop] = result["causal_context"].cpu().numpy()
        seed_probability.append(probability)
        seed_context.append(context)
        del model, checkpoint
        gc.collect(); torch.cuda.empty_cache()
    probability3 = np.mean(np.stack(seed_probability), axis=0)
    probability5 = np.zeros((EXPECTED_OBSERVATIONS, 5), dtype=np.float32)
    probability5[:, :3] = probability3
    if not np.allclose(probability5.sum(axis=1), 1.0, atol=1e-5):
        raise RuntimeError("C09 action ensemble simplex drift")
    uncertainty = -np.sum(
        probability5 * np.log(np.clip(probability5, 1e-8, 1.0)), axis=1
    ) / np.log(5.0)
    return probability5, uncertainty.astype(np.float32), seed_context


def _frame_inventory(metadata: dict[str, object]) -> tuple[np.ndarray, list[tuple[str, int, int]]]:
    frames = []
    shards = []
    offset = 0
    for world in metadata["worlds"]:
        group = metadata["groups"][world]
        current = np.asarray(group["global_frame_index"][:], dtype=np.int64)
        if len(current) == 0 or np.any(np.diff(current) <= 0):
            raise RuntimeError(f"C09 frame identity drift: {world}")
        frames.append(current)
        shards.append((world, offset, len(current)))
        offset += len(current)
    global_frames = np.concatenate(frames)
    if len(global_frames) != EXPECTED_FRAMES or np.any(np.diff(global_frames) <= 0):
        raise RuntimeError("C09 global frame population drift")
    references = np.asarray(metadata["five_references"], dtype=np.int64)
    compact = np.searchsorted(global_frames, references)
    if np.any(compact >= len(global_frames)) or not np.array_equal(global_frames[compact], references):
        raise RuntimeError("C09 five-frame reference drift")
    return compact.astype(np.int32), shards


def _spatial_inference(
    metadata: dict[str, object], base_checkpoints: list[Path], seed_context: list[np.ndarray],
    scalar_checkpoints: list[Path], spatial_checkpoints: list[Path],
    corrective_checkpoints: list[Path], scratch: Path, device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[dict]]:
    import torch

    compact, shards = _frame_inventory(metadata)
    seed_vectors = []
    provenance = []
    for seed in range(3):
        base_checkpoint = torch.load(base_checkpoints[seed], map_location="cpu", weights_only=False)
        if base_checkpoint.get("schema_version") != "gse_graph_checkpoint_v1" or base_checkpoint.get("seed") != seed:
            raise RuntimeError(f"C09 base checkpoint drift: seed{seed}")
        base = GeometrySemanticEventNet().to(device)
        base.load_state_dict(base_checkpoint["model"], strict=True)
        base.eval()
        for parameter in base.parameters(): parameter.requires_grad_(False)
        feature_dir = scratch / f"seed{seed}_features"
        feature_dir.mkdir()
        pooled = np.lib.format.open_memmap(
            feature_dir / "pooled.npy", mode="w+", dtype=np.float16,
            shape=(EXPECTED_FRAMES, 128),
        )
        azimuth = np.lib.format.open_memmap(
            feature_dir / "azimuth.npy", mode="w+", dtype=np.float16,
            shape=(EXPECTED_FRAMES, 128, 36),
        )
        elevation = np.lib.format.open_memmap(
            feature_dir / "elevation.npy", mode="w+", dtype=np.float16,
            shape=(EXPECTED_FRAMES, 128, 2),
        )
        with torch.inference_mode():
            for world, offset, count in shards:
                group = metadata["groups"][world]
                for start in range(0, count, 128):
                    stop = min(start + 128, count)
                    range_m = np.asarray(group["range_m"][start:stop], dtype=np.float32)
                    valid = np.asarray(group["valid_mask"][start:stop], dtype=np.float32)
                    if (
                        range_m.shape != (stop - start, 16, 720)
                        or valid.shape != range_m.shape or not np.all(np.isfinite(range_m))
                        or np.any(range_m < NEAR_RANGE_M) or np.any(range_m > MAX_RANGE_M)
                    ):
                        raise RuntimeError(f"C09 sensor payload drift: {world}")
                    scans = np.stack((range_m / MAX_RANGE_M, valid), axis=1)[:, None]
                    features = encode_spatial_scan_features(
                        base.encoder, torch.from_numpy(scans).to(device=device, dtype=torch.float32)
                    )
                    destination = slice(offset + start, offset + stop)
                    pooled[destination] = features["pooled"][:, 0].cpu().numpy().astype(np.float16)
                    azimuth[destination] = features["directional"][:, 0].cpu().numpy().astype(np.float16)
                    elevation[destination] = features["vertical"][:, 0].cpu().numpy().astype(np.float16)
        pooled.flush(); azimuth.flush(); elevation.flush()
        del base, base_checkpoint
        gc.collect(); torch.cuda.empty_cache()

        scalar_checkpoint = torch.load(scalar_checkpoints[seed], map_location=device, weights_only=False)
        spatial_checkpoint = torch.load(spatial_checkpoints[seed], map_location=device, weights_only=False)
        corrective_checkpoint = torch.load(corrective_checkpoints[seed], map_location=device, weights_only=False)
        if (
            scalar_checkpoint.get("schema_version") != "gse_event_center_dual_batch_corrective_checkpoint_v1"
            or spatial_checkpoint.get("schema_version") != "gse_spatial_event_center_checkpoint_v1"
            or corrective_checkpoint.get("schema_version") != "gse_spatial_longitudinal_corrective_checkpoint_v1"
            or any(value.get("seed") != seed for value in (
                scalar_checkpoint, spatial_checkpoint, corrective_checkpoint,
            ))
        ):
            raise RuntimeError(f"C09 spatial checkpoint drift: seed{seed}")
        if (
            corrective_checkpoint.get("scalar_checkpoint_sha256") != _sha(scalar_checkpoints[seed])
            or corrective_checkpoint.get("spatial_checkpoint_sha256") != _sha(spatial_checkpoints[seed])
        ):
            raise RuntimeError(f"C09 spatial corrective provenance drift: seed{seed}")
        scalar = EventCenterOffsetHead().to(device)
        scalar.load_state_dict(scalar_checkpoint["head"], strict=True)
        spatial = SpatialEventCenterDecoder().to(device)
        spatial.load_state_dict(spatial_checkpoint["decoder"], strict=True)
        corrector = SpatialLongitudinalCorrector(spatial).to(device)
        corrector.longitudinal_residual.load_state_dict(
            corrective_checkpoint["longitudinal_residual"], strict=True
        )
        corrector.freeze_spatial_decoder()
        scalar.eval(); corrector.eval()
        vector = np.empty((EXPECTED_OBSERVATIONS, 3), dtype=np.float32)
        with torch.inference_mode():
            for start in range(0, EXPECTED_OBSERVATIONS, 128):
                stop = min(start + 128, EXPECTED_OBSERVATIONS)
                refs = compact[start:stop]
                context = torch.from_numpy(seed_context[seed][start:stop]).to(device)
                longitudinal = scalar(context)
                vector[start:stop] = corrector(
                    torch.from_numpy(np.asarray(azimuth[refs], dtype=np.float32)).to(device),
                    torch.from_numpy(np.asarray(elevation[refs], dtype=np.float32)).to(device),
                    torch.from_numpy(np.asarray(pooled[refs], dtype=np.float32)).to(device),
                    longitudinal,
                ).cpu().numpy()
        if not np.all(np.isfinite(vector)):
            raise RuntimeError(f"C09 spatial output nonfinite: seed{seed}")
        seed_vectors.append(vector)
        provenance.append({
            "seed": seed, "base_checkpoint_sha256": _sha(base_checkpoints[seed]),
            "scalar_checkpoint_sha256": _sha(scalar_checkpoints[seed]),
            "spatial_checkpoint_sha256": _sha(spatial_checkpoints[seed]),
            "corrective_checkpoint_sha256": _sha(corrective_checkpoints[seed]),
        })
        del scalar, spatial, corrector, scalar_checkpoint, spatial_checkpoint, corrective_checkpoint
        del pooled, azimuth, elevation
        shutil.rmtree(feature_dir)
        gc.collect(); torch.cuda.empty_cache()
    combined = combine_seed_projections(
        np.stack(seed_vectors), np.asarray(metadata["sensor"]), np.asarray(metadata["basis"])
    )
    return (
        combined["predicted_local_vector_m"], combined["projected_center_xyz_m"],
        combined["seed_center_xyz_m"], combined["position_uncertainty_m"], provenance,
    )


def _endpoint_geometry(
    frame_manifest: Path, traversal_manifest: Path, dataset_run: Path,
) -> tuple[dict[object, TraversedEndpointGeometry], dict[str, object], dict[str, float]]:
    traversal_rows = [
        row for row in _read_jsonl(traversal_manifest)
        if str(row.get("split")) == "validation" and str(row.get("parent_id", "")).endswith("_C09")
    ]
    lengths = {str(row["traversal_id"]): float(row["length_m"]) for row in traversal_rows}
    if len(lengths) != EXPECTED_TRAVERSALS:
        raise RuntimeError("C09 traversal population drift")
    by_traversal: dict[str, list[dict]] = defaultdict(list)
    for row in _read_jsonl(frame_manifest):
        traversal = str(row["traversal_id"])
        if traversal in lengths:
            by_traversal[traversal].append(row)
    if len(by_traversal) != EXPECTED_TRAVERSALS or sum(map(len, by_traversal.values())) != EXPECTED_FRAMES:
        raise RuntimeError("C09 route-frame population drift")
    worlds = sorted({str(row["parent_id"]) for values in by_traversal.values() for row in values})
    groups = {
        world: zarr.open_group(
            str(dataset_run / "artifacts/dataset/validation" / f"{world}.zarr"), mode="r"
        ) for world in worlds
    }
    geometries = {}
    maximum_gap = 0.0
    physical = 0
    for traversal_id, frames in sorted(by_traversal.items()):
        if not traversal_id.endswith(":d0"):
            continue
        physical += 1
        frames.sort(key=lambda row: (float(row["route_arc_m"]), int(row["global_frame_index"])))
        world = str(frames[0]["parent_id"])
        rows = np.asarray([int(row["world_frame_row"]) for row in frames], dtype=np.int64)
        xyz = np.asarray(groups[world]["axis_xyz_m"].oindex[rows], dtype=np.float64)
        if len(xyz) < 2 or not np.all(np.isfinite(xyz)):
            raise RuntimeError(f"C09 endpoint geometry invalid: {traversal_id}")
        deltas = (xyz[1] - xyz[0], xyz[-2] - xyz[-1])
        physical_id = traversal_id.rsplit(":d", 1)[0]
        for side, anchor, delta in ((0, xyz[0], deltas[0]), (1, xyz[-1], deltas[1])):
            norm = float(np.linalg.norm(delta))
            if norm <= 0.0:
                raise RuntimeError(f"C09 endpoint tangent invalid: {traversal_id}")
            endpoint = traversed_edge_endpoint(
                f"{physical_id}:d0", 0.0 if side == 0 else lengths[traversal_id],
                lengths[traversal_id],
            )
            geometries[endpoint] = TraversedEndpointGeometry(
                endpoint, tuple(float(value) for value in anchor),
                tuple(float(value) for value in delta / norm),
            )
        maximum_gap = max(maximum_gap, lengths[traversal_id] - float(frames[-1]["route_arc_m"]))
    if physical != EXPECTED_TRAVERSALS // 2 or len(geometries) != EXPECTED_TRAVERSALS:
        raise RuntimeError("C09 endpoint geometry count drift")
    return geometries, {
        "worlds": len(worlds), "directed_traversals": len(lengths),
        "physical_edges": physical, "endpoint_geometries": len(geometries),
        "route_frames": sum(map(len, by_traversal.values())),
        "maximum_unsampled_terminal_gap_m": maximum_gap,
        "route_sample_spacing_m": 1.0,
    }, lengths


def _candidate_pairs(
    trigger_rows: list[int], parent: np.ndarray, center: np.ndarray,
    association_valid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    from scipy.spatial import cKDTree

    pairs = set()
    selected = set(trigger_rows)
    for world in sorted(set(parent[trigger_rows].tolist())):
        rows = np.asarray(sorted(
            row for row in selected if parent[row] == world and association_valid[row]
        ), dtype=np.int64)
        if len(rows) < 2:
            continue
        for left, right in cKDTree(center[rows]).query_pairs(4.0 + 1e-12):
            pairs.add(tuple(sorted((int(rows[left]), int(rows[right])))))
    ordered = sorted(pairs)
    return (
        np.asarray([row[0] for row in ordered], dtype=np.int64),
        np.asarray([row[1] for row in ordered], dtype=np.int64),
    )


def _qualify(
    replay: dict, metadata: dict[str, object], lengths: dict[str, float],
    geometry: dict[object, TraversedEndpointGeometry], old_lookup: dict[tuple[int, int], bool],
    forward: np.ndarray,
) -> tuple[list[dict], list[dict], list[dict], dict[str, int]]:
    kept = {}
    signatures = []
    audit = []
    for hypothesis in replay["hypotheses"]:
        if not hypothesis["committed"]:
            continue
        identifier = int(hypothesis["id"])
        evidence = [int(value.row) for value in hypothesis["evidence"]]
        traversals = [str(value.traversal_id) for value in hypothesis["evidence"]]
        incidence = physical_incidence_count(traversals)
        internal = [tuple(sorted(value)) for value in itertools.combinations(evidence, 2)]
        old_support = sum(old_lookup.get(value, False) for value in internal)
        factorized = hypothesis["event"] == "terminal" or incidence >= 3 or (incidence >= 2 and old_support >= 1)
        endpoints = set()
        for value in hypothesis["evidence"]:
            row = int(value.row)
            traversal = str(value.traversal_id)
            length = lengths[traversal]
            event_arc = min(length, max(0.0, float(metadata["route_arc"][row]) + float(forward[row])))
            endpoints.add(traversed_edge_endpoint(traversal, event_arc, length))
        endpoint_values = [geometry[value] for value in sorted(endpoints)]
        anchor_allowed, anchor, spread = endpoint_anchor_consensus(
            endpoint_values, route_sample_spacing_m=1.0
        )
        branch = hypothesis["event"] == "terminal" or executed_branch_witness(endpoint_values)
        allowed = factorized and anchor_allowed and branch
        audit.append({
            "hypothesis_id": identifier, "world": str(hypothesis["world"]),
            "event": str(hypothesis["event"]), "physical_incidence_count": incidence,
            "old_internal_support": old_support, "factorized_allowed": factorized,
            "anchor_allowed": anchor_allowed, "branch_witness": branch, "allowed": allowed,
            "anchor_xyz_m": list(anchor), "maximum_anchor_spread_m": spread,
            "endpoint_tokens": [
                {"physical_edge_id": value.physical_edge_id, "endpoint_side": value.endpoint_side}
                for value in sorted(endpoints)
            ],
        })
        if allowed:
            kept[identifier] = hypothesis
            signatures.append(CommittedEndpointSignature(
                identifier, str(hypothesis["world"]), str(hypothesis["event"]), tuple(sorted(endpoints))
            ))
    mapping, components = endpoint_consolidation_mapping(signatures)
    signature_by_id = {value.hypothesis_id: value for value in signatures}
    members: dict[int, list[int]] = defaultdict(list)
    for identifier, representative in mapping.items(): members[representative].append(identifier)
    nodes = []
    valid_mapping = {}
    component_rejected = 0
    for representative, identifiers in sorted(members.items()):
        endpoints = sorted({
            endpoint for identifier in identifiers for endpoint in signature_by_id[identifier].endpoints
        })
        accepted, anchor, spread = endpoint_anchor_consensus(
            [geometry[value] for value in endpoints], route_sample_spacing_m=1.0
        )
        if not accepted:
            component_rejected += 1
            continue
        hypothesis = kept[representative]
        nodes.append({
            "hypothesis_id": representative, "world": str(hypothesis["world"]),
            "event": str(hypothesis["event"]), "xyz_m": list(anchor),
            "member_hypothesis_ids": sorted(identifiers),
            "endpoint_tokens": [
                {"physical_edge_id": value.physical_edge_id, "endpoint_side": value.endpoint_side}
                for value in endpoints
            ], "maximum_anchor_spread_m": spread,
        })
        for identifier in identifiers: valid_mapping[identifier] = representative
    filtered = [
        edge for edge in replay["edges"]
        if int(edge["from_hypothesis"]) in valid_mapping and int(edge["to_hypothesis"]) in valid_mapping
    ]
    edges, self_loops, duplicates = remap_verified_edges(filtered, valid_mapping)
    stats = {
        "raw_hypotheses": len(replay["hypotheses"]),
        "raw_committed_hypotheses": sum(bool(value["committed"]) for value in replay["hypotheses"]),
        "qualified_hypotheses": len(signatures), "consolidated_nodes": len(nodes),
        "verified_edges": len(edges), "endpoint_components": len(components),
        "component_rejected": component_rejected, "self_loops_removed": self_loops,
        "duplicate_edges_removed": duplicates,
    }
    return nodes, edges, audit, stats


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-run", required=True, type=Path)
    parser.add_argument("--sequence-manifest", required=True, type=Path)
    parser.add_argument("--frame-manifest", required=True, type=Path)
    parser.add_argument("--traversal-manifest", required=True, type=Path)
    parser.add_argument("--action-model-run", required=True, type=Path)
    parser.add_argument("--training-run", required=True, type=Path)
    parser.add_argument("--scalar-run", required=True, type=Path)
    parser.add_argument("--spatial-run", required=True, type=Path)
    parser.add_argument("--corrective-run", required=True, type=Path)
    parser.add_argument("--association-pairs", required=True, type=Path)
    parser.add_argument("--association-decisions", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    started = time.monotonic()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    scratch = output / "scratch"
    scratch.mkdir()

    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("formal C09 endpoint-geometry inference requires CUDA")
    torch.use_deterministic_algorithms(True); torch.backends.cudnn.benchmark = False
    torch.manual_seed(0); torch.cuda.manual_seed_all(0)
    device = torch.device("cuda")

    dataset_run = args.dataset_run.resolve()
    validation_paths = [
        args.training_run.resolve() / f"artifacts/models/seed{seed}/validation_outputs.npz"
        for seed in range(3)
    ]
    with np.load(validation_paths[0], allow_pickle=False) as archive:
        canonical_global_index = archive["global_sequence_index"].astype(np.int64)
    metadata = _metadata(dataset_run, args.sequence_manifest.resolve(), canonical_global_index)
    tokens = _validation_tokens(validation_paths, metadata["global_index"])
    action_paths = [
        args.action_model_run.resolve() / f"artifacts/models/seed{seed}/best.pt"
        for seed in range(3)
    ]
    probability, uncertainty, contexts = _action_inference(
        tokens, metadata["traversal"], metadata["sequence"], action_paths,
        args.action_model_run.resolve() / "artifacts", device,
    )
    np.savez_compressed(
        output / "action_ensemble.npz", global_sequence_index=metadata["global_index"],
        probability=probability, uncertainty=uncertainty,
    )
    base_paths = [
        args.training_run.resolve() / f"artifacts/models/seed{seed}/best.pt" for seed in range(3)
    ]
    scalar_paths = [args.scalar_run.resolve() / f"artifacts/models/seed{seed}/best.pt" for seed in range(3)]
    spatial_paths = [args.spatial_run.resolve() / f"artifacts/models/seed{seed}/best.pt" for seed in range(3)]
    corrective_paths = [args.corrective_run.resolve() / f"artifacts/models/seed{seed}/best.pt" for seed in range(3)]
    vector, center, seed_center, position_std, checkpoint_provenance = _spatial_inference(
        metadata, base_paths, contexts, scalar_paths, spatial_paths, corrective_paths,
        scratch, device,
    )
    shutil.rmtree(scratch)
    position_std = position_std.astype(np.float32)
    np.savez_compressed(
        output / "spatial_center_ensemble.npz",
        schema_version=np.asarray("gse_spatial_longitudinal_center_ensemble_c09_v1"),
        global_sequence_index=metadata["global_index"], seed_center_xyz_m=seed_center,
        predicted_local_vector_m=vector, projected_center_xyz_m=center,
        offset_std_m=position_std, sensor_xyz_m=metadata["sensor"],
        route_local_basis=metadata["basis"],
    )

    triggers_raw = extract_decision_mass_triggers(
        probability, metadata["traversal"], metadata["sequence"], uncertainty,
        decision_threshold=.97,
    )
    trigger_rows = [int(value.row) for value in triggers_raw]
    left, right = _candidate_pairs(
        trigger_rows, metadata["parent"], center, metadata["association_valid"]
    )
    metric = unanimous_seed_metric_support(seed_center, left, right, distance_cap_m=4.0)
    with np.load(args.association_pairs.resolve(), allow_pickle=False) as archive:
        old_left = archive["left"].astype(np.int64); old_right = archive["right"].astype(np.int64)
    with np.load(args.association_decisions.resolve(), allow_pickle=False) as archive:
        old_accept = archive["runtime_accept"].astype(np.bool_)
        if not (
            np.array_equal(archive["runtime_label"].shape, old_accept.shape)
            and len(old_left) == len(old_accept)
        ):
            raise RuntimeError("C09 archived association decision population drift")
    old_lookup = {
        tuple(sorted((int(a), int(b)))): bool(value)
        for a, b, value in zip(old_left, old_right, old_accept, strict=True)
    }
    union_lookup = {
        tuple(sorted((int(a), int(b)))): bool(value or old_lookup.get(tuple(sorted((int(a), int(b)))), False))
        for a, b, value in zip(left, right, metric, strict=True)
    }
    triggers = []
    for value in triggers_raw:
        row = int(value.row); event_index = int(value.predicted_event_index)
        triggers.append(ProposalTrigger(
            row=row, world=str(metadata["parent"][row]), order=int(metadata["world_order"][row]),
            traversal_id=str(metadata["traversal"][row]), sequence_index=int(metadata["sequence"][row]),
            event={1: "junction", 2: "terminal"}[event_index],
            confidence=float(probability[row, event_index]), uncertainty=float(uncertainty[row]),
            xyz_m=tuple(float(item) for item in center[row]), teacher_identity=None,
            position_uncertainty_m=float(position_std[row]),
        ))
    replay = replay_trace_commits(
        triggers, union_lookup,
        association_valid_rows=set(np.flatnonzero(metadata["association_valid"]).tolist()),
        distance_cap_m=4.0, independent_traces_required=2,
    )
    geometry, geometry_inventory, lengths = _endpoint_geometry(
        args.frame_manifest.resolve(), args.traversal_manifest.resolve(), dataset_run
    )
    nodes, edges, qualification, graph_stats = _qualify(
        replay, metadata, lengths, geometry, old_lookup, vector[:, 0]
    )
    _write_jsonl(output / "verified_nodes.jsonl", nodes)
    _write_jsonl(output / "verified_edges.jsonl", edges)
    _write_jsonl(output / "hypothesis_qualification.jsonl", qualification)
    _write_jsonl(output / "decision_trace.jsonl", replay["decision_trace"])
    manifest = {
        "schema_version": "gse_endpoint_geometry_c09_teacher_free_inference_v1",
        "status": "PASS_TEACHER_FREE_C09_GRAPH_FREEZE_V1",
        "worlds": EXPECTED_WORLDS, "causal_observations": EXPECTED_OBSERVATIONS,
        "unique_lidar_frames": EXPECTED_FRAMES, "directed_traversals": EXPECTED_TRAVERSALS,
        "action_proposal_threshold": .97, "proposal_triggers": len(triggers),
        "association_candidate_pairs": len(left), "metric_accepted_pairs": int(np.sum(metric)),
        "archived_old_accepted_pairs": int(np.sum(old_accept)),
        "graph": graph_stats, "geometry_inventory": geometry_inventory,
        "checkpoint_provenance": checkpoint_provenance,
        "teacher_inputs_read": 0, "teacher_identity_inputs_read": 0,
        "optimizer_steps": 0, "model_updates": 0, "checkpoint_selection_steps": 0,
        "threshold_selection_steps": 0, "c10_worlds_read": 0, "mtare_worlds_read": 0,
        "gpu_used": True, "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()),
        "duration_seconds": time.monotonic() - started,
    }
    (output / "inference_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
