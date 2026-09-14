"""Sealed six-field input bytes to one encoder window; no teacher/model I/O.

Expected digest, row and identities must come from the independently frozen
input manifest. This does not authorize reads or prove label validity.
"""
from dataclasses import dataclass
import hashlib
import io
import json
import re
import zipfile

import numpy as np


@dataclass(frozen=True)
class SurfaceFeatureInput:
    range_valid: np.ndarray
    translation_m: np.ndarray
    yaw_deg: np.ndarray
    provenance: dict
    input_binding_sha256: str


def bind_feature_input(payload: bytes, *, expected_sha256: str, task: str,
                       row: int, source_sequence_id: int, frame_rows: list[int]):
    return _bind_feature_input(payload,expected_sha256=expected_sha256,task=task,row=row,
        source_sequence_id=source_sequence_id,frame_rows=frame_rows,observation_count=16)


def _bind_feature_input(payload: bytes, *, expected_sha256: str, task: str,
                       row: int, source_sequence_id: int, frame_rows: list[int], observation_count: int):
    """Verify before decompressing; keep IDs in provenance, never tensor input."""
    if (type(payload) is not bytes or not payload or len(payload) > 32 * 1024**2 or
            type(expected_sha256) is not str or re.fullmatch(r"[a-f0-9]{64}", expected_sha256) is None):
        raise ValueError("bounded immutable input and frozen SHA-256 required")
    if hashlib.sha256(payload).hexdigest() != expected_sha256:
        raise ValueError("input byte SHA-256 drift")
    if (type(task) is not str or re.fullmatch(
            r"S(?:0[1-9]|10)_[a-z0-9_]+_C0[1-7]__(?:ellipse|rounded_rectangle|c1_mixed)", task) is None or
            type(observation_count) is not int or not 1 <= observation_count <= 64 or
            type(row) is not int or not 0 <= row < observation_count or
            type(source_sequence_id) is not int or source_sequence_id < 0 or
            type(frame_rows) is not list or len(frame_rows) != 5 or
            any(type(i) is not int or i < 0 for i in frame_rows) or
            any(b <= a for a, b in zip(frame_rows, frame_rows[1:]))):
        raise ValueError("exact C01-C07 task and causal row identity required")
    contract = {
        "ranges_m": ((observation_count, 5, 16, 720), np.dtype("float32")),
        "valid_mask": ((observation_count, 5, 16, 720), np.dtype("uint8")),
        "relative_translation_current_sensor_m": ((observation_count, 5, 3), np.dtype("float32")),
        "relative_yaw_current_sensor_deg": ((observation_count, 5), np.dtype("float32")),
        "frame_rows": ((observation_count, 5), np.dtype("int32")),
        "source_sequence_ids": ((observation_count,), np.dtype("int64")),
    }
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        entries = archive.infolist()
        if (len(entries) != 6 or {e.filename for e in entries} != {k + ".npy" for k in contract} or
                sum(e.file_size for e in entries) > 32 * 1024**2):
            raise ValueError("six-field bounded NPZ required; no teacher extras")
    with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
        selected = {}
        for key, (shape, dtype) in contract.items():
            value = archive[key]
            if value.shape != shape or value.dtype != dtype or not np.isfinite(value).all():
                raise ValueError(f"input array contract drift: {key}")
            selected[key] = value[row].copy()
    if (selected["source_sequence_ids"].item() != source_sequence_id or
            selected["frame_rows"].tolist() != frame_rows):
        raise ValueError("input manifest/source row mismatch")
    ranges, valid = selected["ranges_m"], selected["valid_mask"]
    translation = selected["relative_translation_current_sensor_m"]
    yaw = selected["relative_yaw_current_sensor_deg"]
    if (np.any((ranges < 0) | (ranges > 50)) or np.any((valid != 0) & (valid != 1)) or
            np.any(translation[-1] != 0) or yaw[-1] != 0):
        raise ValueError("range/mask/current-frame motion contract drift")
    # Exactly the legacy 50 m normalization; raw valid returns are not ROI-cropped.
    range_valid = np.stack((ranges / np.float32(50), valid.astype(np.float32)), axis=1)
    provenance = {"schema_version": "gse_surface_feature_source_v1", "task": task,
                  "input_file_sha256": expected_sha256, "input_row": row,
                  "source_sequence_id": source_sequence_id, "frame_rows": frame_rows.copy(),
                  "coordinate_frame": "current_sensor"}
    digest = hashlib.sha256(json.dumps(provenance, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return SurfaceFeatureInput(range_valid, translation, yaw, provenance, digest)
