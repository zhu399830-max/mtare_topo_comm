"""Corrected causal supervision for the two typed GSE Composers."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np


HISTORY_FRAMES = 5
ACTION_TARGET = {"corridor": 0, "junction": 1, "terminal": 2, "turn": 0, "geometry_transition": 0}
METRIC_TARGET = {"corridor": 0, "junction": 0, "terminal": 0, "turn": 1, "geometry_transition": 2}


@dataclass(frozen=True)
class ComposerWorldSupervision:
    global_sequence_index: np.ndarray
    action_target: np.ndarray
    metric_target: np.ndarray
    event_name: np.ndarray
    identity: np.ndarray
    traversal_id: np.ndarray
    sequence_index: np.ndarray
    history_row_index: np.ndarray
    valid_history_mask: np.ndarray
    backprojection_steps_ago: np.ndarray
    backprojection_valid: np.ndarray
    backprojection_source: np.ndarray


def _turn_anchor_by_traversal_identity(
    rows: Sequence[Mapping[str, object]],
) -> dict[tuple[str, str], float]:
    candidates: dict[tuple[str, str], list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        if row["event"] == "turn":
            candidates[(str(row["traversal_id"]), str(row["identity"]))].append(row)
    anchors = {}
    for key, values in candidates.items():
        # Curvature is objective spline geometry.  The earliest global row is
        # a deterministic tie break and never moves the anchor into the future.
        selected = max(
            values,
            key=lambda row: (float(row["curvature_per_m"]), -int(row["global_sequence_index"])),
        )
        anchors[key] = float(selected["traversal_arc_m"])
    return anchors


def materialize_composer_world_supervision(
    global_sequence_index: np.ndarray,
    teacher_rows: Sequence[Mapping[str, object]],
    transition_timing_rows: Sequence[Mapping[str, object]],
) -> ComposerWorldSupervision:
    """Align labels and strictly past/current histories to one cached world."""

    global_index = np.asarray(global_sequence_index, dtype=np.int64)
    rows = sorted(teacher_rows, key=lambda row: int(row["global_sequence_index"]))
    if (
        global_index.ndim != 1 or len(global_index) == 0
        or not np.array_equal(global_index, np.asarray([
            int(row["global_sequence_index"]) for row in rows
        ], dtype=np.int64))
    ):
        raise RuntimeError("Composer supervision/cache global identity drift")
    if len({(str(row["traversal_id"]), int(row["sequence_index"])) for row in rows}) != len(rows):
        raise RuntimeError("Composer supervision traversal/sequence identity is not unique")
    timing = {
        (str(row["traversal_id"]), int(row["sequence_index"])): row
        for row in transition_timing_rows
    }
    if len(timing) != len(transition_timing_rows):
        raise RuntimeError("transition timing key is duplicated")
    lookup = {
        (str(row["traversal_id"]), int(row["sequence_index"])): index
        for index, row in enumerate(rows)
    }
    turn_anchor = _turn_anchor_by_traversal_identity(rows)
    count = len(rows)
    history = np.full((count, HISTORY_FRAMES), -1, dtype=np.int32)
    mask = np.zeros((count, HISTORY_FRAMES), dtype=np.bool_)
    backprojection = np.zeros(count, dtype=np.float32)
    backprojection_valid = np.zeros(count, dtype=np.bool_)
    source = np.full(count, "none", dtype="<U24")
    for index, row in enumerate(rows):
        traversal = str(row["traversal_id"]); sequence = int(row["sequence_index"])
        positions = range(max(0, sequence - (HISTORY_FRAMES - 1)), sequence + 1)
        indices = [lookup[(traversal, position)] for position in positions]
        offset = HISTORY_FRAMES - len(indices)
        history[index, offset:] = indices; mask[index, offset:] = True
        if any(value > index for value in indices):
            raise RuntimeError("Composer history contains a future row")
        event = str(row["event"])
        if event == "geometry_transition":
            key = (traversal, sequence)
            if key not in timing:
                raise RuntimeError("geometry-transition row lacks timing supervision")
            timing_row = timing[key]
            steps = float(timing_row["metres_after_boundary"])
            inside = str(timing_row["boundary_inside_five_frame_history"]).lower() == "true" \
                if isinstance(timing_row["boundary_inside_five_frame_history"], str) \
                else bool(timing_row["boundary_inside_five_frame_history"])
            if inside != (0.0 <= steps <= HISTORY_FRAMES - 1):
                raise RuntimeError("transition boundary/history flag drift")
            if inside:
                backprojection[index] = np.float32(steps)
                backprojection_valid[index] = True; source[index] = "transition_boundary"
        elif event == "turn":
            key = (traversal, str(row["identity"]))
            if key not in turn_anchor:
                raise RuntimeError("turn row lacks an objective curvature anchor")
            steps = float(row["traversal_arc_m"]) - turn_anchor[key]
            if 0.0 <= steps <= HISTORY_FRAMES - 1:
                backprojection[index] = np.float32(steps)
                backprojection_valid[index] = True; source[index] = "turn_max_curvature"
    event = np.asarray([str(row["event"]) for row in rows], dtype="<U24")
    identity = np.asarray([
        "" if row.get("identity") is None else str(row["identity"]) for row in rows
    ], dtype="<U160")
    if np.any((event == "corridor") != (identity == "")):
        raise RuntimeError("corridor/structural identity contract drift")
    return ComposerWorldSupervision(
        global_sequence_index=global_index,
        action_target=np.asarray([ACTION_TARGET[value] for value in event], dtype=np.int8),
        metric_target=np.asarray([METRIC_TARGET[value] for value in event], dtype=np.int8),
        event_name=event,
        identity=identity,
        traversal_id=np.asarray([str(row["traversal_id"]) for row in rows], dtype="<U180"),
        sequence_index=np.asarray([int(row["sequence_index"]) for row in rows], dtype=np.int32),
        history_row_index=history,
        valid_history_mask=mask,
        backprojection_steps_ago=backprojection,
        backprojection_valid=backprojection_valid,
        backprojection_source=source,
    )


def summarize_composer_supervision(supervision: ComposerWorldSupervision) -> dict[str, object]:
    valid = supervision.backprojection_valid
    return {
        "rows": len(supervision.global_sequence_index),
        "event_frames": {
            name: int(np.sum(supervision.event_name == name))
            for name in ACTION_TARGET
        },
        "backprojection_frames": {
            "turn": int(np.sum(valid & (supervision.event_name == "turn"))),
            "geometry_transition": int(np.sum(valid & (supervision.event_name == "geometry_transition"))),
        },
        "backprojection_identities": {
            name: len(set(supervision.identity[valid & (supervision.event_name == name)].tolist()))
            for name in ("turn", "geometry_transition")
        },
        "history_valid_counts": {
            str(value): int(np.sum(supervision.valid_history_mask.sum(axis=1) == value))
            for value in range(1, HISTORY_FRAMES + 1)
        },
        "minimum_steps_ago": float(supervision.backprojection_steps_ago[valid].min()) if np.any(valid) else None,
        "maximum_steps_ago": float(supervision.backprojection_steps_ago[valid].max()) if np.any(valid) else None,
    }


__all__ = [
    "ACTION_TARGET", "METRIC_TARGET", "ComposerWorldSupervision",
    "materialize_composer_world_supervision", "summarize_composer_supervision",
]
