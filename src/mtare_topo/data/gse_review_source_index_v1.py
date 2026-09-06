"""Validate already-loaded full identity indices, without any source I/O.

This is not a structure sampler or physical-continuity proof. Inputs must be
the complete identity arrays for one permitted parent's three source shards.
"""
from dataclasses import dataclass
import re

import numpy as np

VARIANTS = ("ellipse", "rounded_rectangle", "c1_mixed")
SOURCE_SEQUENCE_POPULATION = 188126
SENSOR_FIELDS = frozenset(("global_frame_index", "local_frame_index", "traversal_index", "route_arc_m"))
TEACHER_FIELDS = frozenset(("source_global_sequence_index", "variant_global_sequence_index", "frame_row"))


@dataclass(frozen=True)
class VariantTraversalIndices:
    variant: str
    task: str
    sequence_rows: tuple[int, ...]
    source_sequence_ids: tuple[int, ...]
    frame_rows: tuple[tuple[int, ...], ...]
    decision_arc_m: tuple[float, ...]


@dataclass(frozen=True)
class SourceWindowIndices:
    traversal_id: str
    variants: tuple[VariantTraversalIndices, ...]


@dataclass(frozen=True)
class TraversalIndexInterval:
    traversal_id: str
    variants: tuple[VariantTraversalIndices, ...]

    @property
    def sequence_count(self):
        return len(self.variants[0].sequence_rows)

    @property
    def window_count(self):
        return max(0, self.sequence_count - 20)

    def iter_windows(self):
        """Enumerate all21-decision index windows; does not nominate a sample."""
        for start in range(self.window_count):
            stop = start + 21
            yield SourceWindowIndices(self.traversal_id, tuple(
                VariantTraversalIndices(v.variant, v.task, v.sequence_rows[start:stop],
                    v.source_sequence_ids[start:stop], v.frame_rows[start:stop], v.decision_arc_m[start:stop])
                for v in self.variants))


@dataclass(frozen=True)
class SourceIndexReport:
    parent_id: str
    partition: str
    intervals: tuple[TraversalIndexInterval, ...]
    short_intervals: tuple[dict, ...]
    counts: dict
    continuity_requires_source_audit: bool = True
    metadata_truth_requires_source_reader: bool = True
    duration_s: None = None


def _variant_dict(value, label):
    if type(value) is not dict or set(value) != set(VARIANTS):
        raise ValueError(f"{label} requires exactly the three declared variants")


def _identities(value, *, parent_id, label, unique):
    if type(value) is not list or any(type(x) is not str or not x.startswith(parent_id + ":")
                                    or not re.fullmatch(r"[A-Za-z0-9_.:-]+:d[01]", x) for x in value):
        raise ValueError(f"{label} requires source traversal identity strings")
    if unique and len(set(value)) != len(value):
        raise ValueError("P1a traversal table must be unique")
    return tuple(value)


def _int_array(value, shape, label):
    if type(value) is not np.ndarray or value.shape != shape or value.dtype.kind not in "iu":
        raise ValueError(f"{label} must be an integer ndarray of shape{shape}")
    if np.any(value < 0):
        raise ValueError(f"{label} contains negative identities")


def _attrs(sensor, teacher, parent_id, variant, partition):
    if type(sensor) is not dict or type(teacher) is not dict:
        raise ValueError("explicit source attrs dictionaries required")
    if sensor.get("schema_version") != "primitive_relation_p1a_sensor_shard_v1" or teacher.get("schema_version") != "primitive_relation_p1b_teacher_shard_v1":
        raise ValueError("P1a/P1b schema mismatch")
    for attrs in (sensor, teacher):
        if any(attrs.get(k) != v for k, v in (("parent_id", parent_id), ("partition", partition), ("geometry_realization", variant))):
            raise ValueError("parent/cohort/partition/variant metadata drift")
    if (type(sensor.get("sensor_shape")) is not list or sensor["sensor_shape"] != [16, 720]
            or any(type(i) is not int for i in sensor["sensor_shape"])
            or type(sensor.get("maximum_range_m")) is not float or sensor["maximum_range_m"] != 50.
            or sensor.get("student_pose_input_forbidden") is not True):
        raise ValueError("P1a sensor/pose boundary drift")
    if (type(teacher.get("window_frames")) is not int or teacher["window_frames"] != 5
            or type(teacher.get("maximum_slots")) is not int or teacher["maximum_slots"] != 32
            or teacher.get("student_identity_input_forbidden") is not True):
        raise ValueError("P1b window/identity boundary drift")
    return (_identities(sensor.get("traversal_ids"), parent_id=parent_id, label="P1a", unique=True),
            _identities(teacher.get("traversal_ids"), parent_id=parent_id, label="P1b", unique=False))


def _one_variant(parent_id, variant, partition, sensor_attrs, teacher_attrs, sensor, teacher):
    traversals, per_sequence = _attrs(sensor_attrs, teacher_attrs, parent_id, variant, partition)
    if type(sensor) is not dict or set(sensor) != SENSOR_FIELDS or type(teacher) is not dict or set(teacher) != TEACHER_FIELDS:
        raise ValueError("only exact identity index arrays allowed; no scans/geometry/model fields")
    global_frames, sources = sensor["global_frame_index"], teacher["source_global_sequence_index"]
    if type(global_frames) is not np.ndarray or global_frames.ndim != 1 or not len(global_frames):
        raise ValueError("nonempty full sensor frame index required")
    if type(sources) is not np.ndarray or sources.ndim != 1:
        raise ValueError("full teacher source sequence vector required")
    f, q = len(global_frames), len(sources)
    for field in SENSOR_FIELDS - {"route_arc_m"}:
        _int_array(sensor[field], (f,), field)
    for field in TEACHER_FIELDS:
        _int_array(teacher[field], (q, 5) if field == "frame_row" else (q,), field)
    arc = sensor["route_arc_m"]
    if type(arc) is not np.ndarray or arc.shape != (f,) or arc.dtype.kind != "f" or not np.isfinite(arc).all() or np.any(arc < 0):
        raise ValueError("finite nonnegative floating route arc vector required")
    # Convert identities to Python integers before differences/additions: uint
    # subtraction or fixed-width overflow must not conceal ordering faults.
    globals_ = tuple(map(int, global_frames))
    if any(b != a + 1 for a, b in zip(globals_, globals_[1:])):
        raise ValueError("full P1a global frames must be unique and contiguous")
    source_ids = tuple(map(int, sources))
    if any(i >= SOURCE_SEQUENCE_POPULATION for i in source_ids) or any(b <= a for a, b in zip(source_ids, source_ids[1:])):
        raise ValueError("source sequence IDs outside frozen population or not ordered unique")
    offset = VARIANTS.index(variant) * SOURCE_SEQUENCE_POPULATION
    if tuple(map(int, teacher["variant_global_sequence_index"])) != tuple(i + offset for i in source_ids):
        raise ValueError("paired variant sequence offset identity drift")
    if len(per_sequence) != q or not set(per_sequence) <= set(traversals):
        raise ValueError("per-sequence traversal inventory mismatch")
    traversal_index = tuple(map(int, sensor["traversal_index"]))
    if any(i >= len(traversals) for i in traversal_index) or set(traversal_index) != set(range(len(traversals))):
        raise ValueError("sensor traversal lookup out of bounds or unused entries")
    frames_by_traversal, rows_by_traversal = {}, {t: [] for t in traversals}
    for i, traversal in enumerate(traversals):
        rows = tuple(j for j, idx in enumerate(traversal_index) if idx == i)
        if rows != tuple(range(rows[0], rows[-1] + 1)):
            raise ValueError("traversal frames cannot interleave or resume after another segment")
        if tuple(int(sensor["local_frame_index"][r]) for r in rows) != tuple(range(len(rows))):
            raise ValueError("full traversal local frame index must start0 and be contiguous")
        distances = tuple(float(arc[r]) for r in rows)
        if any(b <= a for a, b in zip(distances, distances[1:])):
            raise ValueError("route arc reverses or repeats within traversal")
        frames_by_traversal[traversal] = rows
    for row, traversal in enumerate(per_sequence):
        frames = tuple(map(int, teacher["frame_row"][row]))
        if any(i >= f for i in frames) or any(b != a + 1 for a, b in zip(frames, frames[1:])):
            raise ValueError("five strictly causal consecutive source frame rows required")
        if any(traversals[traversal_index[i]] != traversal for i in frames):
            raise ValueError("five frames cross a traversal boundary")
        rows_by_traversal[traversal].append(row)
    result = {}
    for traversal in traversals:
        frames = frames_by_traversal[traversal]
        rows = tuple(rows_by_traversal[traversal])
        expected = max(0, len(frames) - 4)
        if len(rows) != expected or (rows and rows != tuple(range(rows[0], rows[-1] + 1))):
            raise ValueError("full teacher rolling-window population missing/duplicated/interleaved")
        for j, row in enumerate(rows):
            if tuple(map(int, teacher["frame_row"][row])) != frames[j:j + 5]:
                raise ValueError("teacher history differs from exhaustive chronological frame windows")
        ids = tuple(source_ids[row] for row in rows)
        if any(b != a + 1 for a, b in zip(ids, ids[1:])):
            raise ValueError("source sequence identities gap within traversal")
        result[traversal] = VariantTraversalIndices(variant, parent_id + "__" + variant, rows, ids,
            tuple(tuple(map(int, teacher["frame_row"][row])) for row in rows),
            tuple(float(arc[int(teacher["frame_row"][row, -1])]) for row in rows))
    return result, (globals_, tuple(map(int, sensor["local_frame_index"])),
                    tuple(traversals[i] for i in traversal_index)), f, q


def validate_review_source_indices(*, parent_id, sensor_attrs_by_variant, teacher_attrs_by_variant,
                                  sensor_arrays_by_variant, teacher_arrays_by_variant):
    """Pure strict full-shard index validation; neither sampling nor labeling."""
    if type(parent_id) is not str or not re.fullmatch(r"[A-Za-z0-9_]+_C0[1-7]", parent_id):
        raise ValueError("explicit C01-C07 parent required; C08+ forbidden")
    partition = "c07" if parent_id.endswith("_C07") else "fit"
    for label, value in (("sensor attrs", sensor_attrs_by_variant), ("teacher attrs", teacher_attrs_by_variant),
                         ("sensor indices", sensor_arrays_by_variant), ("teacher indices", teacher_arrays_by_variant)):
        _variant_dict(value, label)
    outputs, first_identity, counts = {}, None, {}
    for variant in VARIANTS:
        output, frame_identity, f, q = _one_variant(parent_id, variant, partition,
            sensor_attrs_by_variant[variant], teacher_attrs_by_variant[variant],
            sensor_arrays_by_variant[variant], teacher_arrays_by_variant[variant])
        if first_identity is None:
            first_identity = frame_identity
        elif frame_identity != first_identity:
            raise ValueError("paired variants have different original frame/traversal identity")
        outputs[variant] = output
        counts[variant] = {"frames": f, "sequences": q}
    first = outputs[VARIANTS[0]]
    for variant in VARIANTS[1:]:
        other = outputs[variant]
        if set(other) != set(first):
            raise ValueError("paired traversal population drift")
        for traversal, record in first.items():
            if record.source_sequence_ids != other[traversal].source_sequence_ids:
                raise ValueError("cross-variant source sequence/traversal correspondence drift")
    intervals = tuple(TraversalIndexInterval(t, tuple(outputs[v][t] for v in VARIANTS)) for t in first)
    short = tuple({"traversal_id": i.traversal_id, "sequence_count": i.sequence_count,
                   "required": 21, "missing": 21 - i.sequence_count} for i in intervals if i.sequence_count < 21)
    return SourceIndexReport(parent_id, partition, intervals, short, {
        "variants": counts, "traversals": len(intervals), "logical_sequences": sum(i.sequence_count for i in intervals),
        "eligible_21_decision_windows": sum(i.window_count for i in intervals),
        "variant_observations": sum(x["sequences"] for x in counts.values()),
        "unique_variant_raw_frames": sum(x["frames"] for x in counts.values()),
        "structure_candidates_nominated": 0, "structure_labels_created": 0,
    })
