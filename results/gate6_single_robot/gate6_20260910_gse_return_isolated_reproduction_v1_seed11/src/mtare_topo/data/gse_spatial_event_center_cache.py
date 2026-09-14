"""Compact frozen spatial-feature cache for five-frame event-center learning."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from mtare_topo.representation.gse_spatial_event_center import (
    DIRECTIONAL_BINS,
    ELEVATION_BINS,
    ENCODER_DIM,
    HISTORY_FRAMES,
)


def compact_five_frame_references(
    global_references: np.ndarray,
    global_frame_index: np.ndarray,
) -> np.ndarray:
    """Map exact global frame ids to a strictly increasing compact cache."""

    references = np.asarray(global_references, dtype=np.int64)
    frames = np.asarray(global_frame_index, dtype=np.int64)
    if references.ndim != 2 or references.shape[1] != HISTORY_FRAMES:
        raise ValueError("global references must have shape [N,5]")
    if len(frames) == 0 or frames.ndim != 1 or np.any(np.diff(frames) <= 0):
        raise ValueError("global frame ids must be a nonempty increasing vector")
    if np.any(references < 0):
        raise ValueError("five-frame references cannot contain padding")
    locations = np.searchsorted(frames, references)
    if np.any(locations >= len(frames)) or not np.array_equal(frames[locations], references):
        raise RuntimeError("one or more five-frame ids are absent from the cache")
    if len(frames) > np.iinfo(np.int32).max:
        raise ValueError("spatial cache exceeds int32 indexing")
    return locations.astype(np.int32)


class SpatialEventCenterCache:
    """Memory-map one seed's frozen separable spatial feature cache."""

    def __init__(self, cache_dir: str | Path) -> None:
        self.cache_dir = Path(cache_dir).resolve()
        manifest = json.loads((self.cache_dir / "manifest.json").read_text(encoding="utf-8"))
        if (
            manifest.get("schema_version") != "gse_spatial_event_center_cache_v1"
            or manifest.get("history_frames") != HISTORY_FRAMES
            or manifest.get("encoder_dim") != ENCODER_DIM
            or manifest.get("directional_bins") != DIRECTIONAL_BINS
            or manifest.get("elevation_bins") != ELEVATION_BINS
        ):
            raise RuntimeError("spatial event-center cache schema drift")
        frames = int(manifest["unique_frames"])
        observations = int(manifest["causal_observations"])
        self.pooled = np.load(self.cache_dir / "pooled.npy", mmap_mode="r")
        self.azimuth = np.load(self.cache_dir / "azimuth.npy", mmap_mode="r")
        self.elevation = np.load(self.cache_dir / "elevation.npy", mmap_mode="r")
        self.references = np.load(self.cache_dir / "compact_reference_index.npy", mmap_mode="r")
        if (
            self.pooled.shape != (frames, ENCODER_DIM)
            or self.azimuth.shape != (frames, ENCODER_DIM, DIRECTIONAL_BINS)
            or self.elevation.shape != (frames, ENCODER_DIM, ELEVATION_BINS)
            or self.references.shape != (observations, HISTORY_FRAMES)
            or any(value.dtype != np.float16 for value in (self.pooled, self.azimuth, self.elevation))
            or self.references.dtype != np.int32
            or np.any(self.references < 0)
            or np.any(self.references >= frames)
        ):
            raise RuntimeError("spatial event-center cache array contract drift")

    def gather(self, observation_rows: np.ndarray) -> dict[str, np.ndarray]:
        rows = np.asarray(observation_rows, dtype=np.int64)
        if rows.ndim != 1 or np.any(rows < 0) or np.any(rows >= len(self.references)):
            raise ValueError("spatial cache observation rows are invalid")
        references = np.asarray(self.references[rows], dtype=np.int64)
        return {
            "pooled": np.asarray(self.pooled[references], dtype=np.float32),
            "azimuth": np.asarray(self.azimuth[references], dtype=np.float32),
            "elevation": np.asarray(self.elevation[references], dtype=np.float32),
        }


__all__ = ["SpatialEventCenterCache", "compact_five_frame_references"]
