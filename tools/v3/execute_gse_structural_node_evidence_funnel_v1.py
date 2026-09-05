#!/usr/bin/env python3
"""Training-free structural-node separability audit on sealed P1b geometry."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
import time

import numpy as np
import zarr

from mtare_topo.representation.gse_structural_node_evidence import (
    PrimitivePortEvidence,
    pair_distance,
    safest_nonempty_threshold,
    structural_node_evidence_descriptor,
)


MODES = ("oracle_full_composition", "causal_five_frame", "causal_single_frame")


def _curvature(points: np.ndarray, endpoint: int) -> float:
    values = np.asarray(points, dtype=np.float64)
    if values.shape[0] < 3 or values.shape[1] != 3:
        raise ValueError("primitive axis requires at least three 3D points")
    local = values[:3] if endpoint == 0 else values[-1:-4:-1]
    left, right = local[1] - local[0], local[2] - local[1]
    nl, nr = np.linalg.norm(left), np.linalg.norm(right)
    if nl <= 1e-8 or nr <= 1e-8:
        return 0.0
    angle = np.arccos(np.clip(np.dot(left, right) / (nl * nr), -1.0, 1.0))
    return float(angle / max((nl + nr) * .5, 1e-8))


def _port(points: np.ndarray, axes: np.ndarray, exponent: float, endpoint: int, support: int) -> PrimitivePortEvidence:
    values = np.asarray(points, dtype=np.float64)
    if endpoint == 0:
        direction = values[1] - values[0]
    elif endpoint == 1:
        direction = values[-2] - values[-1]
    else:
        raise ValueError("endpoint index must be zero or one")
    return PrimitivePortEvidence(
        away_direction_xyz=tuple(float(value) for value in direction),
        half_axes_m=tuple(float(value) for value in axes),
        shape_exponent=float(exponent),
        curvature_per_m=_curvature(values, endpoint),
        support_rays=int(support),
    )


def _traversal_targets(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            traversal = str(row["traversal_id"])
            if traversal in result:
                raise ValueError("duplicate traversal identity")
            result[traversal] = str(row["to_node_id"])
    return result


def _full_ports(construction: dict, composition: dict) -> tuple[PrimitivePortEvidence, ...]:
    realized = {str(row["primitive_id"]): row for row in construction["realized_primitives"]}
    ports = []
    for member in composition["member_endpoints"]:
        primitive = realized[str(member["primitive_id"])]
        endpoint = int(member["endpoint_index"])
        ports.append(_port(
            np.asarray(primitive["centerline_xyz_m"], dtype=np.float64),
            np.asarray(primitive["endpoint_half_axes_m"][endpoint], dtype=np.float64),
            float(primitive["endpoint_shape_exponent"][endpoint]), endpoint, 1_000_000,
        ))
    return tuple(ports)


def _causal_ports(group: zarr.Group, row: int, composition: dict, primitive_lookup: dict[str, int], *, single: bool) -> tuple[PrimitivePortEvidence, ...]:
    primitive_index = np.asarray(group["primitive_index"][row], dtype=np.int64)
    primitive_mask = np.asarray(group["primitive_mask"][row], dtype=np.bool_)
    visibility = np.asarray(group["temporal_visibility"][row, -1], dtype=np.bool_)
    axes = np.asarray(group["endpoint_half_axes_m"][row], dtype=np.float64)
    exponent = np.asarray(group["endpoint_shape_exponent"][row], dtype=np.float64)
    controls = np.asarray(group["axis_control_current_sensor_m"][row], dtype=np.float64)
    support = np.asarray(group["support_ray_count"][row], dtype=np.int64)
    ports = []
    for member in composition["member_endpoints"]:
        index = primitive_lookup[str(member["primitive_id"])]
        candidates = np.flatnonzero(primitive_mask & (primitive_index == index))
        if len(candidates) > 1:
            raise RuntimeError("one primitive occupies multiple causal slots")
        if len(candidates) == 0:
            continue
        slot = int(candidates[0])
        if single and not visibility[slot]:
            continue
        endpoint = int(member["endpoint_index"])
        ports.append(_port(
            controls[slot], axes[slot, endpoint], exponent[slot, endpoint], endpoint,
            1 if single else int(support[slot]),
        ))
    return tuple(ports)


def _task_observations(construction_path: Path, teacher_path: Path, targets: dict[str, str]) -> list[dict]:
    construction = json.loads(construction_path.read_text(encoding="utf-8"))
    group = zarr.open_group(str(teacher_path), mode="r")
    if construction["parent_id"] != group.attrs["parent_id"] or construction["geometry_realization"] != group.attrs["geometry_realization"]:
        raise RuntimeError("construction and P1b task identity drift")
    primitive_lookup = {
        str(row["primitive_id"]): index for index, row in enumerate(construction["realized_primitives"])
    }
    compositions = {
        str(row["node_id"]): row for row in construction["base_construction"]["composition_operations"]
    }
    traversal = np.asarray(group.attrs["traversal_ids"], dtype=str)
    if traversal.shape != (group["primitive_index"].shape[0],):
        raise RuntimeError("P1b traversal population drift")
    observations = []
    start = 0
    while start < len(traversal):
        end = start + 1
        while end < len(traversal) and traversal[end] == traversal[start]:
            end += 1
        traversal_id = str(traversal[start])
        if traversal_id not in targets:
            raise RuntimeError("P1b traversal is absent from the sealed manifest")
        node = targets[traversal_id]
        composition = compositions[node]
        row = end - 1
        mode_ports = {
            "oracle_full_composition": _full_ports(construction, composition),
            "causal_five_frame": _causal_ports(group, row, composition, primitive_lookup, single=False),
            "causal_single_frame": _causal_ports(group, row, composition, primitive_lookup, single=True),
        }
        descriptor = {
            mode: structural_node_evidence_descriptor(ports) if ports else None
            for mode, ports in mode_ports.items()
        }
        observations.append({
            "node": node,
            "degree": int(composition["degree"]),
            "traversal": traversal_id,
            "descriptor": descriptor,
            "observed_ports": {mode: len(ports) for mode, ports in mode_ports.items()},
        })
        start = end
    return observations


def _pairs(observations: list[dict], mode: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    retained = [row for row in observations if row["descriptor"][mode] is not None]
    if len(retained) < 2:
        return np.empty(0), np.empty(0, dtype=np.bool_), np.empty(0, dtype=np.int8)
    descriptor = np.stack([row["descriptor"][mode] for row in retained])
    left, right = np.triu_indices(len(retained), 1)
    distance = pair_distance(descriptor[left], descriptor[right])
    same = np.asarray([retained[a]["node"] == retained[b]["node"] for a, b in zip(left, right, strict=True)], dtype=np.bool_)
    degree = np.asarray([retained[a]["degree"] if same[index] else 0 for index, a in enumerate(left)], dtype=np.int8)
    return distance, same, degree


def _score(distance: np.ndarray, target: np.ndarray, threshold: float) -> dict:
    predicted = distance <= threshold
    tp = int(np.sum(predicted & target)); fp = int(np.sum(predicted & ~target))
    positives = int(np.sum(target)); predicted_count = tp + fp
    precision = tp / predicted_count if predicted_count else 0.0
    recall = tp / positives if positives else 0.0
    return {
        "pairs": int(len(target)), "positive_pairs": positives,
        "predicted_positive": predicted_count, "true_positive": tp, "false_positive": fp,
        "precision": precision, "recall": recall,
        "false_accept_fraction": fp / predicted_count if predicted_count else 1.0,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--construction-root", required=True, type=Path)
    parser.add_argument("--teacher-root", required=True, type=Path)
    parser.add_argument("--traversal-manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    started = time.monotonic()
    if args.output_dir.exists():
        raise RuntimeError("output directory exists; overwrite is forbidden")
    args.output_dir.mkdir(parents=True)
    targets = _traversal_targets(args.traversal_manifest.resolve())
    population = defaultdict(int)
    pair_bank: dict[str, dict[str, list[np.ndarray]]] = {
        partition: {mode: [] for mode in MODES} for partition in ("fit", "c07")
    }
    target_bank = {partition: {mode: [] for mode in MODES} for partition in ("fit", "c07")}
    degree_bank = {partition: {mode: [] for mode in MODES} for partition in ("fit", "c07")}
    family_bank: dict[str, dict[str, list[tuple[np.ndarray, np.ndarray]]]] = defaultdict(lambda: defaultdict(list))
    digest = hashlib.sha256()
    for partition in ("fit", "c07"):
        construction_paths = sorted((args.construction_root / partition).glob("*.json"))
        if len(construction_paths) != (180 if partition == "fit" else 30):
            raise RuntimeError(f"{partition} task population drift")
        for construction_path in construction_paths:
            teacher_path = args.teacher_root / partition / f"{construction_path.stem}.zarr"
            if not teacher_path.is_dir():
                raise RuntimeError("matching P1b Teacher shard is missing")
            observations = _task_observations(construction_path, teacher_path, targets)
            population[f"{partition}_tasks"] += 1
            population[f"{partition}_observations"] += len(observations)
            family = construction_path.stem[:3]
            for mode in MODES:
                distance, target, degree = _pairs(observations, mode)
                pair_bank[partition][mode].append(distance)
                target_bank[partition][mode].append(target)
                degree_bank[partition][mode].append(degree)
                if partition == "c07":
                    family_bank[family][mode].append((distance, target))
                digest.update(np.asarray(distance, dtype="<f8").tobytes())
                digest.update(np.asarray(target, dtype=np.uint8).tobytes())
    results = {}
    for mode in MODES:
        fit_distance = np.concatenate(pair_bank["fit"][mode])
        fit_target = np.concatenate(target_bank["fit"][mode])
        selected = safest_nonempty_threshold(fit_distance, fit_target, minimum_precision=.98)
        mode_result = {"fit_selection": selected}
        if selected["found"]:
            threshold = float(selected["threshold"])
            for partition in ("fit", "c07"):
                distance = np.concatenate(pair_bank[partition][mode])
                target = np.concatenate(target_bank[partition][mode])
                mode_result[partition] = _score(distance, target, threshold)
            mode_result["c07_per_family"] = {
                family: _score(
                    np.concatenate([value[0] for value in values[mode]]),
                    np.concatenate([value[1] for value in values[mode]]), threshold,
                )
                for family, values in sorted(family_bank.items())
            }
        results[mode] = mode_result
    five = results["causal_five_frame"]
    oracle = results["oracle_full_composition"]
    gates = {
        "oracle_has_safe_nonempty_region": bool(oracle["fit_selection"]["found"]),
        "five_frame_has_safe_nonempty_region": bool(five["fit_selection"]["found"]),
        "five_frame_fit_precision_ge_0_98": bool(five.get("fit", {}).get("precision", 0.0) >= .98),
        "five_frame_fit_recall_ge_0_25": bool(five.get("fit", {}).get("recall", 0.0) >= .25),
        "five_frame_c07_precision_ge_0_98": bool(five.get("c07", {}).get("precision", 0.0) >= .98),
        "five_frame_c07_recall_ge_0_25": bool(five.get("c07", {}).get("recall", 0.0) >= .25),
        "all_ten_c07_families_have_true_positive": bool(
            len(five.get("c07_per_family", {})) == 10
            and all(value["true_positive"] > 0 for value in five.get("c07_per_family", {}).values())
        ),
    }
    passed = all(gates.values())
    summary = {
        "schema_version": "gse_structural_node_evidence_funnel_v1",
        "status": "PASS_GSE_STRUCTURAL_NODE_EVIDENCE_FUNNEL_V1" if passed else "FAIL_GSE_STRUCTURAL_NODE_EVIDENCE_FUNNEL_V1",
        "scientific_pass": passed,
        "question": "Can causal primitive composition support high-precision structural-node association before training?",
        "population": dict(population),
        "modes": results,
        "gates": gates,
        "deterministic_pair_digest_sha256": digest.hexdigest(),
        "optimizer_steps": 0,
        "model_inference_frames": 0,
        "c08_c09_c10_worlds_read": 0,
        "graph_replays": 0,
        "mtare_runs": 0,
        "duration_seconds": time.monotonic() - started,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
