"""Strict join of sealed C01 caches, with identities separate from model inputs.

The coordinate producer stored only task/row/count in its sample manifest.
Its original local-teacher audit is therefore a required, independently hashed
identity bridge. Never infer source identities or frame rows from list order.
This reader grants no training authority and never loads a model or raw scan.
"""
from dataclasses import dataclass
import hashlib
import io
import json
from pathlib import Path

import numpy as np

from mtare_topo.governance_field_recovery import TASK, _digest, _relative_source


SOURCE_NAMES = {
    "predictions": "all_predictions.npz",
    "scoring_targets": "existing_scoring_targets.npz",
    "sample_manifest": "sample_manifest.json",
    "identity_audit": "observation_audit.json",
    "partial_targets": "observation_targets.json",
}
PREDICTION_KEYS = frozenset(("legacy_frozen", "raw_coordinates_initial", "raw_coordinates",
                             "mean_broadcast_coordinates_initial", "mean_broadcast_coordinates"))


@dataclass(frozen=True)
class PartialStructureCache:
    prediction_axes: np.ndarray
    gt_axes: np.ndarray
    loss_only_gt_mask: np.ndarray
    partial_targets: tuple
    manifest: tuple
    read_hashes: dict
    provenance: dict


def _integer(value):
    return type(value) is int and value >= 0


def _rows(data, *, name):
    rows = json.loads(data, parse_constant=lambda x: (_ for _ in ()).throw(ValueError("nonfinite JSON")))
    if not isinstance(rows, list) or len(rows) != 180 or any(not isinstance(r, dict) for r in rows):
        raise ValueError(f"{name}: exactly 180 object rows required")
    indexed = {}
    sources = set()
    for row in rows:
        task, index = row.get("task"), row.get("row_index")
        if not isinstance(task, str) or not TASK.fullmatch(task) or not _integer(index):
            raise ValueError(f"{name}: invalid C01 task/row identity")
        key = (task, index)
        if key in indexed:
            raise ValueError(f"{name}: duplicate task/row identity")
        indexed[key] = row
        if name != "sample_manifest":
            source, frames = row.get("source_global_sequence_index"), row.get("frame_rows")
            if (not _integer(source) or (task, source) in sources or not isinstance(frames, list)
                    or len(frames) != 5 or any(not _integer(f) for f in frames)
                    or any(b <= a for a, b in zip(frames, frames[1:]))):
                raise ValueError(f"{name}: missing, duplicate or noncausal source/frame identity")
            sources.add((task, source))
    tasks = sorted({task for task, _ in indexed})
    if (len(tasks) != 10 or {task[:3] for task in tasks} != {f"S{i:02d}" for i in range(1, 11)}
            or any(sum(t == task for t, _ in indexed) != 18 for task in tasks)):
        raise ValueError(f"{name}: exact ten parents / eighteen rows required")
    if name != "sample_manifest" and len({(r["task"], f) for r in rows for f in r["frame_rows"]}) != 900:
        raise ValueError(f"{name}: exact 900 unique source frames required")
    return rows, indexed


def load_partial_structure_cache(*, project_root, sources, prediction_branch="raw_coordinates"):
    """Read five explicit path/SHA objects; return all 32 unfiltered predictions.

    `sources` keys are SOURCE_NAMES. Paths are repository-relative and SHA-256
    is verified on the exact bytes subsequently decoded (no hash/read race).
    Consumers must keep manifest/teacher/masks outside predicted-mode forward.
    """
    if prediction_branch != "raw_coordinates":
        raise ValueError("only the frozen final raw_coordinates branch is authorized")
    if not isinstance(sources, dict) or set(sources) != set(SOURCE_NAMES):
        raise ValueError("five explicit sealed sources including identity_audit required")
    root = Path(project_root).resolve()
    paths, seen = {}, set()
    for key, record in sources.items():
        if (not isinstance(record, dict) or set(record) != {"path", "sha256"}
                or not _relative_source(record["path"]) or not _digest(record["sha256"])
                or Path(record["path"]).name != SOURCE_NAMES[key]):
            raise ValueError(f"invalid explicit sealed source: {key}")
        path = (root / record["path"]).resolve()
        if not path.is_relative_to(root) or path in seen:
            raise ValueError("source escape or duplicate resolved path")
        paths[key] = path
        seen.add(path)
    payloads, reads = {}, {}
    for key, path in paths.items():
        data = path.read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        if digest != sources[key]["sha256"]:
            raise ValueError(f"sealed source drift: {key}")
        payloads[key] = data
        reads[sources[key]["path"]] = digest
    manifest, manifest_index = _rows(payloads["sample_manifest"], name="sample_manifest")
    _, identities = _rows(payloads["identity_audit"], name="identity_audit")
    _, partial = _rows(payloads["partial_targets"], name="partial_targets")
    if set(manifest_index) != set(identities) or set(identities) != set(partial):
        raise ValueError("cache/audit/teacher task-row populations differ")
    with np.load(io.BytesIO(payloads["predictions"]), allow_pickle=False) as archive:
        if len(archive.files) != len(PREDICTION_KEYS) or set(archive.files) != PREDICTION_KEYS:
            raise ValueError("coordinate producer prediction schema drift")
        predictions = archive["raw_coordinates"]
    with np.load(io.BytesIO(payloads["scoring_targets"]), allow_pickle=False) as archive:
        if len(archive.files) != 2 or set(archive.files) != {"axis_control_m", "mask"}:
            raise ValueError("coordinate producer target schema drift")
        axes, masks = archive["axis_control_m"], archive["mask"]
    if (predictions.shape != (180, 32, 3, 3) or predictions.dtype != np.float32
            or not np.isfinite(predictions).all() or axes.shape != predictions.shape
            or axes.dtype != np.float32 or masks.shape != (180, 32) or masks.dtype != np.bool_
            or not masks.any(axis=1).all() or int(masks.sum()) != 1452
            or not np.isfinite(axes).all()):
        raise ValueError("full 32-slot geometry or exact 1452 target population drift")
    # Original visible_primitive_targets_from_frame_support initializes all
    # axes to zero and only fills active slots. GT-mode forwards all 32 slots,
    # so invalid/nonzero padding must fail rather than be hidden by a mask.
    if np.any(axes[~masks] != 0):
        raise ValueError("original inactive GT axis padding must remain zero")
    aligned, joined = [], []
    for i, item in enumerate(manifest):
        key = (item["task"], item["row_index"])
        identity, target = identities[key], partial[key]
        if (identity["source_global_sequence_index"] != target["source_global_sequence_index"]
                or identity["frame_rows"] != target["frame_rows"]):
            raise ValueError("explicit source sequence / frame rows disagree")
        count = int(masks[i].sum())
        if (type(item.get("visible_fragments")) is not int or item["visible_fragments"] != count
                or type(identity.get("visible_primitive_fragments")) is not int
                or identity["visible_primitive_fragments"] != count
                or type(target.get("visible_fragments")) is not int or target["visible_fragments"] != count):
            raise ValueError("per-observation fragment count differs across sealed sources")
        if not isinstance(target.get("regions"), list):
            raise ValueError("partial teacher regions must be an explicit list (possibly empty)")
        for field in ("source_global_sequence_index", "frame_rows"):
            if field in item and item[field] != identity[field]:
                raise ValueError("optional manifest identity contradicts the sealed audit")
        aligned.append(target)
        joined.append({"task": key[0], "row_index": key[1], "source_global_sequence_index": identity["source_global_sequence_index"],
                       "frame_rows": list(identity["frame_rows"]), "visible_fragments": count})
    for array in (predictions, axes, masks):
        array.setflags(write=False)
    provenance = {"prediction_branch": "raw_coordinates", "cache_embeds_frame_identity": False,
        "identity_binding": "task_row_via_sealed_original_local_teacher_audit_then_exact_source_and_frame_join",
        "producer_frame_parity_is_inherited_not_independent_cache_id_check": True,
        "required_execution_provenance": "new_card_binds_coordinate_control_spec_card_and_producer_source_version",
        "gt_masks_are_loss_only": True, "all_prediction_slots_retained": 32,
        "new_sensor_frames_read": 0, "model_or_checkpoint_reads": 0,
        "scientific_gate_pass": False}
    return PartialStructureCache(predictions, axes, masks, tuple(aligned), tuple(joined), reads, provenance)
