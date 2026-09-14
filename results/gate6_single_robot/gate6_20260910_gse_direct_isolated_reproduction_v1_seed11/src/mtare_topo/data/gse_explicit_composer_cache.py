"""Schema and validation for frozen geometry-only Composer inputs."""

from __future__ import annotations

from typing import Mapping

import numpy as np


HISTORY_FRAMES = 5
MAX_EXIT_TOKENS = 6
EXPLICIT_CACHE_FIELDS = (
    "global_sequence_index",
    "token_bearing_deg",
    "token_existence_logits",
    "token_opening_width_m",
    "token_vertical_profile_m",
    "token_geometry_uncertainty",
    "token_count_probability",
    "transport_row_probability",
    "transport_reveal_probability",
    "geometry",
    "observation_uncertainty",
)
FORBIDDEN_CACHE_FIELDS = (
    "event_logits", "event_probability", "context", "hidden",
    "place_descriptor", "token_descriptor", "exit_descriptor",
    "pose", "world", "tng", "identity", "teacher_identity",
)


def validate_explicit_composer_world_cache(
    arrays: Mapping[str, np.ndarray], *, expected_rows: int | None = None,
) -> dict[str, object]:
    """Fail closed if a cache is incomplete, malformed, or exposes a bypass."""

    names = tuple(sorted(arrays))
    if set(names) != set(EXPLICIT_CACHE_FIELDS):
        missing = sorted(set(EXPLICIT_CACHE_FIELDS) - set(names))
        unexpected = sorted(set(names) - set(EXPLICIT_CACHE_FIELDS))
        raise ValueError(f"explicit Composer cache field drift: missing={missing}, unexpected={unexpected}")
    if any(name in arrays for name in FORBIDDEN_CACHE_FIELDS):
        raise ValueError("explicit Composer cache contains a forbidden bypass field")
    global_index = np.asarray(arrays["global_sequence_index"])
    if global_index.ndim != 1 or global_index.dtype != np.int64 or len(global_index) == 0:
        raise ValueError("global_sequence_index must be nonempty int64 [N]")
    rows = len(global_index)
    if expected_rows is not None and rows != expected_rows:
        raise ValueError("explicit Composer cache row-count drift")
    if len(np.unique(global_index)) != rows or np.any(np.diff(global_index) <= 0):
        raise ValueError("cache global indices must be unique and strictly increasing")
    shapes = {
        "token_bearing_deg": (rows, HISTORY_FRAMES, MAX_EXIT_TOKENS),
        "token_existence_logits": (rows, HISTORY_FRAMES, MAX_EXIT_TOKENS),
        "token_opening_width_m": (rows, HISTORY_FRAMES, MAX_EXIT_TOKENS),
        "token_vertical_profile_m": (rows, HISTORY_FRAMES, MAX_EXIT_TOKENS, 4),
        "token_geometry_uncertainty": (rows, HISTORY_FRAMES, MAX_EXIT_TOKENS, 5),
        "token_count_probability": (rows, HISTORY_FRAMES, MAX_EXIT_TOKENS + 1),
        "transport_row_probability": (rows, HISTORY_FRAMES - 1, MAX_EXIT_TOKENS, MAX_EXIT_TOKENS + 1),
        "transport_reveal_probability": (rows, HISTORY_FRAMES - 1, MAX_EXIT_TOKENS),
        "geometry": (rows, 4),
        "observation_uncertainty": (rows,),
    }
    for name, shape in shapes.items():
        value = np.asarray(arrays[name])
        if value.shape != shape or value.dtype not in (np.float16, np.float32):
            raise ValueError(f"{name} shape/dtype drift")
        if not np.all(np.isfinite(value)):
            raise ValueError(f"{name} contains nonfinite values")
    if np.any(np.asarray(arrays["token_opening_width_m"]) <= 0.0):
        raise ValueError("cached opening width must be positive")
    if np.any(np.asarray(arrays["geometry"])[:, :2] <= 0.0):
        raise ValueError("cached global width/height must be positive")
    if np.any(np.asarray(arrays["token_geometry_uncertainty"]) <= 0.0):
        raise ValueError("cached token uncertainty must be positive")
    for name in (
        "token_count_probability", "transport_row_probability",
        "transport_reveal_probability", "observation_uncertainty",
    ):
        value = np.asarray(arrays[name], dtype=np.float32)
        if np.any((value < 0.0) | (value > 1.0)):
            raise ValueError(f"{name} is outside probability bounds")
    for name in ("token_count_probability", "transport_row_probability"):
        value = np.asarray(arrays[name], dtype=np.float32)
        if not np.allclose(value.sum(axis=-1), 1.0, atol=3e-3, rtol=0.0):
            raise ValueError(f"{name} does not sum to one")
    return {
        "rows": rows,
        "global_index_min": int(global_index[0]),
        "global_index_max": int(global_index[-1]),
        "fields": EXPLICIT_CACHE_FIELDS,
        "forbidden_fields_present": [],
    }


__all__ = [
    "EXPLICIT_CACHE_FIELDS", "FORBIDDEN_CACHE_FIELDS",
    "validate_explicit_composer_world_cache",
]
