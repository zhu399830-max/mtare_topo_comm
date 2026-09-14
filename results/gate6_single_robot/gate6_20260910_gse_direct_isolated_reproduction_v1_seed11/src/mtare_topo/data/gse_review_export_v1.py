"""In-memory review export using the SAME causal projection as the model.

No file reads, selection, labels or authorization are performed here. A formal
runner must verify the supplied source/selection digests against its frozen run
spec BEFORE loading these arrays. Metadata assertions cannot authenticate data.
Only one segment/variant is materialized at once; no scan downsampling occurs.
"""
from dataclasses import asdict, dataclass
import hashlib
import json
import re

import numpy as np
import torch

from .gse_review_sampling_v1 import SelectedSegment, sample_review_segments
from .gse_structure_review_v1 import canonical_sha, validate_blind_bundle
from ..representation.primitive_relation_model import register_causal_lidar_points


@dataclass(frozen=True)
class ReviewSensorWindowV1:
    task: str
    source_sequence_id: int
    sequence_row: int
    frame_rows: tuple[int, ...]
    frame_traversal_ids: tuple[str, ...]
    range_valid: np.ndarray
    relative_translation_current_sensor_m: np.ndarray
    relative_yaw_current_sensor_deg: np.ndarray


@dataclass(frozen=True)
class BlindExportV1:
    bundle_bytes: bytes
    # Private provenance is NOT shipped in the blind view: contains selection
    # stratum/structure identity. It belongs in the executor evidence directory.
    private_provenance: dict


def _digest(value):
    if type(value) is not str or not re.fullmatch(r"[a-f0-9]{64}", value):
        raise ValueError("explicit SHA-256 binding required")
    return value


def _array(value, shape):
    if type(value) is not np.ndarray or value.shape != shape or value.dtype != np.dtype("float32"):
        raise ValueError("source arrays must have exact float32 shape; no implicit coercion")
    if not np.isfinite(value).all():
        raise ValueError("nonfinite source array")
    # Copy removes aliases to the loader's mutable buffers while this decision
    # is hashed and projected; no modification of the sealed source is allowed.
    return np.array(value, dtype="<f4", order="C", copy=True)


def _array_sha(value):
    descriptor = json.dumps({"dtype": "<f4", "shape": list(value.shape)},
                            sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(descriptor + b"\0" + value.tobytes(order="C")).hexdigest()


def export_blind_segment_v1(population, selected, windows, *, variant_names,
                            variant_name, selection_sha256, sensor_source_sha256,
                            sequence_source_sha256):
    """Project exact selected rows, retaining all valid past/current returns.

    Binding digests are caller assertions until a run-spec-verified reader is
    attached. Population/candidate metadata and split are revalidated, but this
    function does NOT certify that these are the optimal hash-selected48.
    It cannot generate a reference annotation or mark human review complete.
    """
    bindings = {"selection_sha256": _digest(selection_sha256),
                "sensor_source_sha256": _digest(sensor_source_sha256),
                "sequence_source_sha256": _digest(sequence_source_sha256)}
    if type(selected) is not SelectedSegment:
        raise ValueError("strict selected metadata required")
    validation = sample_review_segments(population, (selected.candidate,), variant_names=variant_names)
    if len(validation.selected) != 1 or validation.selected[0].split != selected.split:
        raise ValueError("selected split differs from complete parent partition")
    matches = [v for v in selected.candidate.variants if v.variant == variant_name]
    if len(matches) != 1:
        raise ValueError("variant absent from selected segment")
    variant = matches[0]
    if type(windows) is not tuple or len(windows) != 21:
        raise ValueError("exact21 source windows required")
    segment_sha = canonical_sha(asdict(selected))
    bundle_id = canonical_sha({**bindings, "selected_segment_sha256": segment_sha,
                               "variant": variant_name})
    frame_hashes, decisions, provenance = {}, [], []
    for i, window in enumerate(windows):
        if type(window) is not ReviewSensorWindowV1:
            raise ValueError("strict sensor-only window required")
        if (type(window.source_sequence_id) is not int or type(window.sequence_row) is not int
                or type(window.frame_rows) is not tuple or type(window.frame_traversal_ids) is not tuple
                or any(type(f) is not int for f in window.frame_rows)):
            raise ValueError("strict source row identity required")
        if (window.task != variant.task or window.source_sequence_id != variant.source_sequence_ids[i]
                or window.sequence_row != variant.sequence_rows[i] or window.frame_rows != variant.frame_rows[i]
                or window.frame_traversal_ids != variant.frame_traversal_ids[i]):
            raise ValueError("sensor window does not match selected source identity")
        ranges = _array(window.range_valid, (5, 2, 16, 720))
        translation = _array(window.relative_translation_current_sensor_m, (5, 3))
        yaw = _array(window.relative_yaw_current_sensor_deg, (5,))
        for history, frame in enumerate(window.frame_rows):
            digest = _array_sha(ranges[history])
            if frame in frame_hashes and frame_hashes[frame] != digest:
                raise ValueError("shared raw frame changed between source windows")
            frame_hashes[frame] = digest
        with torch.no_grad():
            xyz, valid = register_causal_lidar_points(torch.from_numpy(ranges)[None],
                torch.from_numpy(translation)[None], torch.from_numpy(yaw)[None])
        # Time, elevation, azimuth order; invalid/non-return points are NOT
        # rendered as surfaces or interpreted as confirmed free space.
        points = xyz[0][valid[0]].tolist()
        frame_keys = [canonical_sha({"sensor_source_sha256": sensor_source_sha256,
                                    "task": variant.task, "frame_row": frame})
                      for frame in window.frame_rows]
        decision = {"decision_index": i, "source_frame_keys": frame_keys,
                    "source_order_indices": list(window.frame_rows), "points_xyz_m": points}
        decisions.append(decision)
        provenance.append({"decision_index": i, "source_sequence_id": window.source_sequence_id,
            "sequence_row": window.sequence_row, "frame_rows": list(window.frame_rows),
            "frame_traversal_ids": list(window.frame_traversal_ids),
            "route_arc_m": variant.route_arc_m[i], "range_valid_sha256": _array_sha(ranges),
            "relative_translation_sha256": _array_sha(translation), "relative_yaw_sha256": _array_sha(yaw),
            "valid_returns_by_frame": valid[0].sum(dim=(1, 2)).tolist(),
            "observation_sha256": canonical_sha(decision)})
    bundle = {"schema": "gse_structure_blind_bundle_v1", "bundle_id": bundle_id,
              "coordinate_frame": "current_sensor_m", "decisions": decisions}
    validate_blind_bundle(bundle)
    data = json.dumps(bundle, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode()
    # Same input-file budget as the offline review UI. Fail before writing;
    # do not downsample an oversized segment to make it pass.
    if len(data) > 128 * 1024 * 1024:
        raise ValueError("blind bundle exceeds128MiB; no truncation or downsampling")
    private = {"schema": "gse_structure_blind_export_provenance_v1", **bindings,
        "selected_segment_sha256": segment_sha, "selected_segment": asdict(selected),
        "variant": variant_name, "bundle_id": bundle_id, "bundle_file_sha256": hashlib.sha256(data).hexdigest(),
        "bundle_canonical_sha256": canonical_sha(bundle), "bundle_bytes": len(data),
        "decisions": provenance, "unique_raw_frames": len(frame_hashes),
        "source_raw_frames_sha256": {str(k): v for k, v in sorted(frame_hashes.items())},
        "projection": "primitive_relation_model.register_causal_lidar_points",
        "projection_device": "cpu", "projection_dtype": "float32", "downsampled": False,
        "source_authentication": "EXTERNAL_FROZEN_READER_REQUIRED",
        "pose_consistency": "RELATIVE_SOURCE_VALUES_BOUND_NOT_INDEPENDENTLY_VERIFIED",
        "reference_annotation_generated": False, "human_review_complete": False,
        "automatic_training_eligibility": False}
    return BlindExportV1(data, private)
