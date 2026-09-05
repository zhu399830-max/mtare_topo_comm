"""Train-only feature cache for the GSE physics-guided slope corrective."""

from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any, Iterable

import numpy as np

from mtare_topo.representation.gse_slope_corrective import (
    HISTORY_FRAMES,
    RAW_FEATURES_PER_FRAME,
    slope_frame_features,
)
from mtare_topo.semantics.range_geometry_baseline import RangeGeometryBaseline


FIT_CODES = frozenset(range(1, 7))
SELECTION_CODES = frozenset((7, 8))
EXPECTED_FIT_WORLDS = 60
EXPECTED_SELECTION_WORLDS = 20
EXPECTED_FIT_SEQUENCES = 142_184
EXPECTED_SELECTION_SEQUENCES = 45_942
EXPECTED_TOTAL_FRAMES = 252_430
GEOMETRY_FIELD_ORDER = ("width_m", "height_m", "slope_deg", "curvature_per_m")
TOPOLOGY_FAMILIES = (
    "S01_flat_tree_small",
    "S02_3d_tree_small",
    "S03_flat_unicyclic_small",
    "S04_3d_unicyclic_small",
    "S05_flat_branch_medium",
    "S06_3d_branch_medium",
    "S07_flat_loop_rich",
    "S08_3d_loop_rich",
    "S09_flat_complex",
    "S10_3d_complex",
)
APPROVED_CORRECTIVE_PARENTS = frozenset(
    f"{family}_C{code:02d}"
    for family in TOPOLOGY_FAMILIES
    for code in sorted(FIT_CODES | SELECTION_CODES)
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def corrective_partition(parent_id: str) -> str:
    if parent_id not in APPROVED_CORRECTIVE_PARENTS:
        raise ValueError(f"parent is not in the approved corrective world set: {parent_id}")
    match = re.search(r"_C(\d+)$", str(parent_id))
    if match is None:
        raise ValueError(f"parent id has no geometry code: {parent_id}")
    code = int(match.group(1))
    if code in FIT_CODES:
        return "fit"
    if code in SELECTION_CODES:
        return "selection"
    raise ValueError(f"parent is outside the train-only corrective split: {parent_id}")


def fit_normalization(feature_blocks: Iterable[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    blocks = [np.asarray(block, dtype=np.float64) for block in feature_blocks]
    if not blocks or any(block.ndim != 3 or block.shape[1:] != (HISTORY_FRAMES, RAW_FEATURES_PER_FRAME) for block in blocks):
        raise ValueError("normalization requires nonempty [N,5,6] feature blocks")
    if any(not np.all(np.isfinite(block)) for block in blocks):
        raise ValueError("normalization features must be finite")
    values = np.concatenate(blocks, axis=0).reshape(-1, RAW_FEATURES_PER_FRAME)
    mean = values.mean(axis=0)
    scale = values.std(axis=0)
    scale = np.where(scale < 1e-6, 1.0, scale)
    if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(scale)) or np.any(scale <= 0.0):
        raise RuntimeError("fit-only corrective normalization is invalid")
    return mean.astype(np.float32), scale.astype(np.float32)


def normalize_features(features: np.ndarray, mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
    values = np.asarray(features, dtype=np.float32)
    center = np.asarray(mean, dtype=np.float32)
    divisor = np.asarray(scale, dtype=np.float32)
    if values.ndim != 3 or values.shape[1:] != (HISTORY_FRAMES, RAW_FEATURES_PER_FRAME):
        raise ValueError("features must be [N,5,6]")
    if center.shape != (RAW_FEATURES_PER_FRAME,) or divisor.shape != center.shape or np.any(divisor <= 0.0):
        raise ValueError("normalization vectors must be positive [6]")
    result = (values - center[None, None, :]) / divisor[None, None, :]
    if not np.all(np.isfinite(result)):
        raise RuntimeError("normalized corrective features are non-finite")
    return result.astype(np.float32, copy=False)


def validate_causal_references(
    references: np.ndarray,
    global_references: np.ndarray,
    global_frame_index: np.ndarray,
    local_frame_index: np.ndarray,
    *,
    frame_count: int,
) -> None:
    """Validate five past-to-current rows while allowing traversal-local resets.

    ``local_frame_index`` intentionally restarts at zero for every directed
    traversal.  Causality is therefore a property of each referenced sequence,
    not a requirement that the full world column equal ``arange(frame_count)``.
    """

    refs = np.asarray(references, dtype=np.int64)
    global_refs = np.asarray(global_references, dtype=np.int64)
    global_index = np.asarray(global_frame_index, dtype=np.int64)
    local_index = np.asarray(local_frame_index, dtype=np.int64)
    sequence_count = len(refs)
    if (
        refs.shape != (sequence_count, HISTORY_FRAMES)
        or global_refs.shape != refs.shape
        or global_index.shape != (frame_count,)
        or local_index.shape != (frame_count,)
        or np.any(local_index < 0)
        or len(np.unique(global_index)) != frame_count
        or np.any(refs < 0)
        or np.any(refs >= frame_count)
        or np.any(np.diff(refs, axis=1) != 1)
        or np.any(np.diff(local_index[refs], axis=1) != 1)
        or np.any(refs[:, -1] != refs.max(axis=1))
        or not np.array_equal(global_refs, global_index[refs])
    ):
        raise RuntimeError("five-frame causal reference contract drift")


def load_slope_corrective_partition(cache_dir: Path, partition: str) -> dict[str, np.ndarray]:
    """Load one frozen cache partition and recheck its complete array contract."""

    cache_dir = Path(cache_dir).resolve()
    if partition not in {"fit", "selection"}:
        raise ValueError("corrective partition must be fit or selection")
    manifest = json.loads((cache_dir / "manifest.json").read_text(encoding="utf-8"))
    normalization = json.loads((cache_dir / "normalization.json").read_text(encoding="utf-8"))
    if (
        manifest.get("schema_version") != "gse_slope_corrective_cache_v1"
        or manifest.get("geometry_field_order") != list(GEOMETRY_FIELD_ORDER)
        or normalization.get("fit_worlds_only") is not True
        or normalization.get("feature_order")
        != [
            "slope_deg",
            "width_m",
            "height_m",
            "curvature_per_m",
            "log1p_local_return_count",
            "populated_sections",
        ]
    ):
        raise RuntimeError("corrective cache or normalization schema drift")
    items = manifest.get("partitions", {}).get(partition)
    if not isinstance(items, list) or not items:
        raise RuntimeError(f"corrective cache has no {partition} worlds")
    expected_worlds = EXPECTED_FIT_WORLDS if partition == "fit" else EXPECTED_SELECTION_WORLDS
    expected_sequences = EXPECTED_FIT_SEQUENCES if partition == "fit" else EXPECTED_SELECTION_SEQUENCES
    if len(items) != expected_worlds:
        raise RuntimeError(f"corrective {partition} world count drift")
    blocks: dict[str, list[np.ndarray]] = {
        "features": [],
        "prior_slope_deg": [],
        "current_slope_deg": [],
        "target_slope_deg": [],
        "global_sequence_index": [],
    }
    parent_rows: list[np.ndarray] = []
    observed_parents: set[str] = set()
    for item in items:
        parent_id = str(item.get("parent_id"))
        if parent_id in observed_parents or corrective_partition(parent_id) != partition:
            raise RuntimeError(f"corrective cache parent duplication or split drift: {parent_id}")
        observed_parents.add(parent_id)
        path = cache_dir / str(item.get("path"))
        with np.load(path, allow_pickle=False) as archive:
            arrays = {key: np.asarray(archive[key]) for key in archive.files}
        count = int(item.get("sequences", -1))
        if (
            arrays.get("features", np.empty(0)).shape
            != (count, HISTORY_FRAMES, RAW_FEATURES_PER_FRAME)
            or arrays.get("prior_slope_deg", np.empty(0)).shape != (count,)
            or arrays.get("current_slope_deg", np.empty(0)).shape != (count,)
            or arrays.get("target_slope_deg", np.empty(0)).shape != (count,)
            or arrays.get("global_sequence_index", np.empty(0)).shape != (count,)
            or arrays.get("local_frame_references", np.empty(0)).shape != (count, HISTORY_FRAMES)
            or arrays.get("global_frame_references", np.empty(0)).shape != (count, HISTORY_FRAMES)
            or not all(np.all(np.isfinite(arrays[key])) for key in blocks if key != "global_sequence_index")
        ):
            raise RuntimeError(f"corrective cache world array drift: {parent_id}")
        for key in blocks:
            blocks[key].append(arrays[key])
        parent_rows.append(np.full(count, parent_id, dtype="U64"))
    packed = {key: np.concatenate(values) for key, values in blocks.items()}
    packed["parent_id"] = np.concatenate(parent_rows)
    if len(packed["features"]) != expected_sequences:
        raise RuntimeError(f"corrective {partition} sequence count drift")
    indices = packed["global_sequence_index"].astype(np.int64, copy=False)
    if len(np.unique(indices)) != len(indices):
        raise RuntimeError(f"corrective {partition} global sequence indices are not unique")
    mean = np.asarray(normalization.get("mean"), dtype=np.float32)
    scale = np.asarray(normalization.get("scale"), dtype=np.float32)
    packed["features"] = normalize_features(packed["features"], mean, scale)
    return packed


def build_slope_corrective_cache(dataset_run: Path, output_dir: Path) -> dict[str, Any]:
    """Derive cache only from C01--C08 train worlds; C09/C10 are rejected."""

    import zarr

    dataset_run = Path(dataset_run).resolve()
    output_dir = Path(output_dir).resolve()
    artifacts = dataset_run / "artifacts"
    train_root = artifacts / "dataset" / "train"
    shard_paths = sorted(train_root.glob("*.zarr"), key=lambda path: path.name)
    observed_parents = [path.stem for path in shard_paths]
    if (
        len(observed_parents) != len(APPROVED_CORRECTIVE_PARENTS)
        or len(set(observed_parents)) != len(observed_parents)
        or set(observed_parents) != set(APPROVED_CORRECTIVE_PARENTS)
    ):
        raise RuntimeError("corrective cache train shards do not exactly match the approved 80 worlds")

    predictor = RangeGeometryBaseline()
    per_partition: dict[str, list[dict[str, Any]]] = {"fit": [], "selection": []}
    output_dir.mkdir(parents=True, exist_ok=False)
    world_dir = output_dir / "worlds"
    world_dir.mkdir()
    all_global_sequence_indices: list[np.ndarray] = []
    total_frames = 0
    read_paths: list[str] = []
    for shard_path in shard_paths:
        parent_id = shard_path.stem
        partition = corrective_partition(parent_id)
        group = zarr.open_group(str(shard_path), mode="r")
        if (
            group.attrs.get("schema_version") != "gse_deduplicated_world_v1"
            or group.attrs.get("split") != "train"
            or group.attrs.get("parent_id") != parent_id
            or group.attrs.get("student_input")
            != "five referenced range_m/valid_mask frames only"
        ):
            raise RuntimeError(f"corrective source shard identity drift: {parent_id}")
        frame_count = int(group["range_m"].shape[0])
        sequence_count = int(group["local_frame_references"].shape[0])
        if group["range_m"].shape != (frame_count, 16, 720) or group["valid_mask"].shape != (
            frame_count,
            16,
            720,
        ):
            raise RuntimeError(f"corrective sensor schema drift: {parent_id}")
        references = np.asarray(group["local_frame_references"][:], dtype=np.int64)
        global_references = np.asarray(group["global_frame_references"][:], dtype=np.int64)
        global_frame_index = np.asarray(group["global_frame_index"][:], dtype=np.int64)
        local_frame_index = np.asarray(group["local_frame_index"][:], dtype=np.int64)
        global_indices = np.asarray(group["global_sequence_index"][:], dtype=np.int64)
        geometry = np.asarray(group["geometry"][:], dtype=np.float32)
        geometry_mask = np.asarray(group["geometry_valid_mask"][:], dtype=np.uint8)
        validate_causal_references(
            references,
            global_references,
            global_frame_index,
            local_frame_index,
            frame_count=frame_count,
        )
        if (
            global_indices.shape != (sequence_count,)
            or len(np.unique(global_indices)) != sequence_count
            or geometry.shape != (sequence_count, len(GEOMETRY_FIELD_ORDER))
            or geometry_mask.shape != geometry.shape
            or np.any(geometry_mask[:, 2] != 1)
            or not np.all(np.isfinite(geometry))
            or np.any(np.abs(geometry[:, 2]) > 45.0)
        ):
            raise RuntimeError(f"corrective causal index or geometry schema drift: {parent_id}")
        total_frames += frame_count
        all_global_sequence_indices.append(global_indices)
        read_paths.append(str(shard_path.relative_to(dataset_run)))
        frame_features = np.empty((frame_count, RAW_FEATURES_PER_FRAME), dtype=np.float32)
        for frame_index in range(frame_count):
            result = predictor.predict(
                np.asarray(group["range_m"][frame_index]),
                np.asarray(group["valid_mask"][frame_index]),
            )
            frame_features[frame_index] = slope_frame_features(result)
        features = frame_features[references]
        target = geometry[:, 2]
        prior = features[:, :, 0].mean(axis=1, dtype=np.float64).astype(np.float32)
        current = features[:, -1, 0].astype(np.float32)
        if (
            features.shape != (sequence_count, HISTORY_FRAMES, RAW_FEATURES_PER_FRAME)
            or target.shape != (sequence_count,)
            or not np.all(np.isfinite(features))
            or not np.all(np.isfinite(target))
        ):
            raise RuntimeError(f"corrective world arrays invalid: {parent_id}")
        destination = world_dir / f"{parent_id}.npz"
        np.savez_compressed(
            destination,
            features=features,
            prior_slope_deg=prior,
            current_slope_deg=current,
            target_slope_deg=target,
            global_sequence_index=global_indices,
            local_frame_references=references.astype(np.int32),
            global_frame_references=global_references,
        )
        per_partition[partition].append(
            {
                "parent_id": parent_id,
                "sequences": sequence_count,
                "unique_frames": frame_count,
                "path": str(destination.relative_to(output_dir)),
                "source_shard": str(shard_path.relative_to(dataset_run)),
            }
        )

    concatenated_indices = np.concatenate(all_global_sequence_indices)
    if len(np.unique(concatenated_indices)) != len(concatenated_indices):
        raise RuntimeError("global sequence indices are not unique across corrective worlds")
    if total_frames != EXPECTED_TOTAL_FRAMES:
        raise RuntimeError(f"corrective unique-frame count drift: {total_frames}")

    counts = {name: sum(item["sequences"] for item in items) for name, items in per_partition.items()}
    if counts != {"fit": EXPECTED_FIT_SEQUENCES, "selection": EXPECTED_SELECTION_SEQUENCES}:
        raise RuntimeError(f"corrective split sequence counts drift: {counts}")
    fit_blocks: list[np.ndarray] = []
    for item in per_partition["fit"]:
        with np.load(output_dir / item["path"], allow_pickle=False) as archive:
            fit_blocks.append(np.asarray(archive["features"], dtype=np.float32))
    mean, scale = fit_normalization(fit_blocks)
    normalization = {
        "feature_order": [
            "slope_deg",
            "width_m",
            "height_m",
            "curvature_per_m",
            "log1p_local_return_count",
            "populated_sections",
        ],
        "fit_worlds_only": True,
        "fit_sequences": EXPECTED_FIT_SEQUENCES,
        "fit_worlds": [item["parent_id"] for item in per_partition["fit"]],
        "mean": mean.tolist(),
        "scale": scale.tolist(),
    }
    (output_dir / "normalization.json").write_text(
        json.dumps(normalization, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    manifest = {
        "schema_version": "gse_slope_corrective_cache_v1",
        "dataset_run": str(dataset_run),
        "partitions": per_partition,
        "counts": {
            "fit_worlds": len(per_partition["fit"]),
            "selection_worlds": len(per_partition["selection"]),
            "fit_sequences": counts["fit"],
            "selection_sequences": counts["selection"],
            "unique_frames": total_frames,
            "strict_test_worlds_read": 0,
            "c09_worlds_read": 0,
            "mtare_worlds_read": 0,
        },
        "prior": "arithmetic mean of five causal deterministic slope estimates",
        "geometry_field_order": list(GEOMETRY_FIELD_ORDER),
        "causal_reference_contract": "five strictly increasing local references; global references exactly match the frozen global frame index; fifth reference is current",
        "actual_source_paths_read": read_paths,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


__all__ = [
    "EXPECTED_FIT_SEQUENCES",
    "EXPECTED_FIT_WORLDS",
    "EXPECTED_SELECTION_SEQUENCES",
    "EXPECTED_SELECTION_WORLDS",
    "APPROVED_CORRECTIVE_PARENTS",
    "EXPECTED_TOTAL_FRAMES",
    "GEOMETRY_FIELD_ORDER",
    "build_slope_corrective_cache",
    "corrective_partition",
    "fit_normalization",
    "normalize_features",
    "load_slope_corrective_partition",
    "validate_causal_references",
]
