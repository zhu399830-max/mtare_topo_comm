"""Selected six-field, lossless surface input reader; no teacher targets.

Only range/first-return validity and previously stored relative motion become
student input. Frame rows/sequence identities are provenance, not model input.
The caller validates a new approved data-export card before real-source use.
This module neither loads weights nor generates geometry/physical labels.
"""
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re

import numpy as np
import zarr


SENSOR_FIELDS = ("range_m", "valid_mask")
WINDOW_FIELDS = ("frame_row", "source_global_sequence_index",
                 "relative_translation_current_sensor_m", "relative_yaw_current_sensor_deg")
_CONTRACT = {
    ("sensor", "range_m"): ((16, 720), "<f4", 16),
    ("sensor", "valid_mask"): ((16, 720), "|u1", 32),
    ("teacher", "frame_row"): ((5,), "<i4", 256),
    ("teacher", "source_global_sequence_index"): ((), "<i8", 256),
    ("teacher", "relative_translation_current_sensor_m"): ((5, 3), "<f4", 256),
    ("teacher", "relative_yaw_current_sensor_deg"): ((5,), "<f4", 256),
}
_TASK = re.compile(r"(S(?:0[1-9]|10)_[a-z0-9_]+_C0[1-7])__(ellipse|rounded_rectangle|c1_mixed)\Z")
_HASH = re.compile(r"[a-f0-9]{64}\Z")
MAX_DECODED_CHUNK_BYTES = 32 * 1024 ** 2


def plan_array_access(header: dict, field: str, role: str, selected_rows: list[int]) -> dict:
    """Plan exact first-axis chunks from an already hash-verified .zarray.

    decoded_actual_rows/bytes count in-bounds rows; decoded_padded_bytes also
    counts tail-chunk padding a codec may materialize. Neither count is the
    effective sample population. All tail dimensions must be one full chunk.
    """
    if (role, field) not in _CONTRACT or type(header) is not dict:
        raise ValueError("only the six declared surface input fields are allowed")
    tail, dtype_string, first_chunk = _CONTRACT[role, field]
    shape, chunks = header.get("shape"), header.get("chunks")
    if (type(shape) is not list or type(chunks) is not list or len(shape) != 1 + len(tail)
            or len(chunks) != len(shape) or any(type(v) is not int or v < 1 for v in shape + chunks)
            or tuple(shape[1:]) != tail or tuple(chunks[1:]) != tail):
        raise ValueError("stored shape/tail chunk layout differs from the six-field contract")
    expected_chunk = first_chunk if role == "sensor" else min(first_chunk, shape[0])
    if (header.get("zarr_format") != 2 or type(header.get("zarr_format")) is not int
            or header.get("dtype") != dtype_string or chunks[0] != expected_chunk
            or header.get("order") != "C" or header.get("dimension_separator", ".") != "."):
        raise ValueError("stored Zarr version/dtype/chunk/order/separator drift")
    if (type(selected_rows) is not list or not selected_rows
            or any(type(r) is not int or r < 0 or r >= shape[0] for r in selected_rows)):
        raise ValueError("nonempty exact in-bounds integer source rows required")
    itemsize = np.dtype(dtype_string).itemsize
    padded_chunk_bytes = math.prod(chunks) * itemsize
    if padded_chunk_bytes > MAX_DECODED_CHUNK_BYTES:
        raise ValueError("decoded chunk exceeds32MiB before codec access")
    indices = sorted({r // chunks[0] for r in selected_rows})
    intervals = [[i * chunks[0], min((i + 1) * chunks[0], shape[0])] for i in indices]
    actual_rows = sum(stop - start for start, stop in intervals)
    return {"role": role, "field": field, "shape": shape.copy(), "chunks": chunks.copy(),
            "dtype": dtype_string, "chunk_keys": [".".join(map(str, (i,) + (0,) * len(tail))) for i in indices],
            "chunk_first_axis_indices": indices, "decoded_row_intervals": intervals,
            "selected_unique_rows": len(set(selected_rows)), "decoded_actual_rows": actual_rows,
            "decoded_actual_bytes": actual_rows * math.prod(tail) * itemsize,
            "decoded_padded_bytes": len(indices) * padded_chunk_bytes,
            "decoded_chunk_count": len(indices)}


def _relative(value):
    if (type(value) is not str or not value or Path(value).is_absolute()
            or any(p in ("", ".", "..") for p in value.split("/"))):
        raise ValueError("explicit clean project-relative source path required")
    return value


def _json_object(raw):
    def reject(value):
        raise ValueError("nonfinite JSON metadata")
    value = json.loads(raw.decode("utf8"), parse_constant=reject)
    if type(value) is not dict:
        raise ValueError("Zarr metadata must be an object")
    return value


def _interval_union_count(intervals):
    end, count = -1, 0
    for start, stop in sorted(intervals):
        count += max(0, stop - max(start, end))
        end = max(end, stop)
    return count


class _ExactStore(zarr.storage.DirectoryStore):
    """Only verified headers initially; computed chunk keys enabled afterwards."""
    def __init__(self, root, relative, sealed, opened):
        self.project_root, self.relative, self.sealed, self.opened = root, relative, sealed, opened
        self.allowed = set(); self.cache = {}; self.read_counts = {}
        super().__init__(str(root / relative))

    def _path(self, key):
        if key not in self.allowed:
            raise PermissionError(f"key outside selected header/chunk scope: {key}")
        relative = self.relative + "/" + key
        if relative not in self.sealed:
            raise ValueError("selected key missing from exact source seal")
        target = self.project_root / relative
        if target.resolve() != target or not target.is_relative_to(self.project_root):
            raise PermissionError("source file or ancestor is a symlink/path escape")
        return relative, target

    def __getitem__(self, key):
        relative, target = self._path(key)
        if key not in self.cache:
            try:
                raw = target.read_bytes()
            except FileNotFoundError:
                raise ValueError("sealed source key missing; no Zarr fill fallback") from None
            actual = hashlib.sha256(raw).hexdigest()
            if self.sealed[relative] != actual:
                raise ValueError("selected source byte SHA-256 drift")
            self.opened[relative] = actual
            self.read_counts[key] = self.read_counts.get(key, 0) + 1
            self.cache[key] = raw
        return self.cache[key]

    def __contains__(self, key):
        if key not in self.allowed:
            return False
        self._path(key)
        # Force sealed-byte presence checks instead of silent missing-chunk fill.
        self[key]
        return True

    def __setitem__(self, key, value):
        raise PermissionError("surface input source is read-only")

    def __delitem__(self, key):
        raise PermissionError("surface input source is read-only")


@dataclass(frozen=True)
class SurfaceTaskInputs:
    task: str
    ranges_m: np.ndarray
    valid_mask: np.ndarray
    relative_translation_current_sensor_m: np.ndarray
    relative_yaw_current_sensor_deg: np.ndarray
    frame_rows: np.ndarray
    source_sequence_ids: np.ndarray
    read_report: dict


class SurfaceInputReader:
    def __init__(self, project_root, *, task_sources, selection, sealed_keys, variable_population=False):
        if type(variable_population) is not bool:
            raise ValueError('explicit boolean population mode required')
        self.root = Path(project_root).absolute()
        if self.root.resolve() != self.root or not self.root.is_dir():
            raise ValueError("real project root without symlink required")
        if type(task_sources) is not dict or not task_sources or len(task_sources) > 210:
            raise ValueError("explicit source mapping for at most210 authorized tasks required")
        if type(selection) is not list or not selection or type(sealed_keys) is not dict:
            raise ValueError("explicit selection and exact source-key SHA mapping required")
        self.task_sources = {}
        for task, source in task_sources.items():
            match = _TASK.fullmatch(task) if type(task) is str else None
            if not match or type(source) is not dict or set(source) != {"sensor", "teacher", "parent_id", "variant", "partition"}:
                raise ValueError("exact C01-C07 three-variant task source mapping required")
            parent, variant = match.groups(); partition = "c07" if parent.endswith("C07") else "fit"
            if (source["parent_id"], source["variant"], source["partition"]) != (parent, variant, partition):
                raise ValueError("source parent/variant/physical partition mismatch")
            for role in ("sensor", "teacher"):
                path = _relative(source[role])
                if Path(path).name != task + ".zarr" or Path(path).parent.name != partition:
                    raise ValueError("explicit task shard under its fit/c07 partition required")
            self.task_sources[task] = dict(source)
        prefixes = {s[r] + "/" for s in self.task_sources.values() for r in ("sensor", "teacher")}
        self.sealed = {}
        for relative, digest in sealed_keys.items():
            _relative(relative)
            if type(digest) is not str or not _HASH.fullmatch(digest) or not any(relative.startswith(p) for p in prefixes):
                raise ValueError("sealed key outside authorized task shards or invalid SHA")
            self.sealed[relative] = digest
        self.selection = {task: [] for task in self.task_sources}
        for row in selection:
            if type(row) is not dict or row.get("task") not in self.selection:
                raise ValueError("selection outside exact source task mapping")
            task = row["task"]; source = self.task_sources[task]
            for name in ("sequence_row", "source_sequence_id", "source_frame_count", "source_sequence_count"):
                if type(row.get(name)) is not int or row[name] < (1 if name.endswith("count") else 0):
                    raise ValueError("manifest source row/count must be explicit integers")
            frames = row.get("frame_rows")
            if (type(frames) is not list or len(frames) != 5 or any(type(f) is not int or f < 0 for f in frames)
                    or frames != list(range(frames[0], frames[0] + 5))
                    or frames[-1] >= row["source_frame_count"] or row["sequence_row"] >= row["source_sequence_count"]):
                raise ValueError("manifest must bind exact five consecutive in-bounds source frames")
            if (row.get("parent_id"), row.get("variant")) != (source["parent_id"], source["variant"]):
                raise ValueError("manifest parent/variant identity drift")
            if row.get("split") not in (("calibration", "development") if source["partition"] == "c07" else ("fit",)):
                raise ValueError("cal/dev both read c07 source; fit must read fit source")
            self.selection[task].append(dict(row, frame_rows=frames.copy()))
        for rows in self.selection.values():
            n = len(rows)
            if (not n or (not variable_population and n != 16)
                    or len({r["sequence_row"] for r in rows}) != n
                    or len({r["source_sequence_id"] for r in rows}) != n
                    or (not variable_population and len({f for r in rows for f in r["frame_rows"]}) != 80)
                    or len({(r["source_frame_count"], r["source_sequence_count"], r["split"]) for r in rows}) != 1):
                raise ValueError("distinct observations and consistent bounds required; default exact16/80 contract")
        self.opened = {}; self._completed = set()

    def _prepare(self, task, role, rows):
        source = self.task_sources[task]
        names = SENSOR_FIELDS if role == "sensor" else WINDOW_FIELDS
        store = _ExactStore(self.root, source[role], self.sealed, self.opened)
        headers = {".zgroup", ".zattrs", *(f + "/.zarray" for f in names)}
        store.allowed = headers.copy()
        if _json_object(store[".zgroup"]) != {"zarr_format": 2}:
            raise ValueError("source group must be exactly ZarrV2")
        attrs = _json_object(store[".zattrs"])
        expected_schema = "primitive_relation_p1a_sensor_shard_v1" if role == "sensor" else "primitive_relation_p1b_teacher_shard_v1"
        if (attrs.get("schema_version") != expected_schema or attrs.get("parent_id") != source["parent_id"]
                or attrs.get("geometry_realization") != source["variant"] or attrs.get("partition") != source["partition"]):
            raise ValueError("source shard schema/identity/variant/split drift")
        if role == "sensor":
            if attrs.get("sensor_shape") != [16, 720] or attrs.get("maximum_range_m") != 50. or attrs.get("student_pose_input_forbidden") is not True:
                raise ValueError("source sensor geometry/identity boundary drift")
        elif attrs.get("window_frames") != 5 or attrs.get("student_identity_input_forbidden") is not True:
            raise ValueError("source causal/identity boundary drift")
        selected = [f for r in rows for f in r["frame_rows"]] if role == "sensor" else [r["sequence_row"] for r in rows]
        expected_n = rows[0]["source_frame_count" if role == "sensor" else "source_sequence_count"]
        plans = {}
        for name in names:
            header = _json_object(store[name + "/.zarray"])
            plans[name] = plan_array_access(header, name, role, selected)
            if header["shape"][0] != expected_n:
                raise ValueError("stored source population differs from sealed manifest")
        allowed = headers | {p["field"] + "/" + key for p in plans.values() for key in p["chunk_keys"]}
        actual_sealed = {path[len(source[role]) + 1:] for path in self.sealed if path.startswith(source[role] + "/")}
        if actual_sealed != allowed:
            raise ValueError("source seal must contain exactly headers plus computed selected chunks")
        store.allowed = allowed
        return store, plans, selected

    @staticmethod
    def _read_field(group, name, indices, plan):
        array = group[name]
        result = np.empty((len(indices), *plan["shape"][1:]), dtype=np.dtype(plan["dtype"]))
        index = np.asarray(indices, dtype=np.int64)
        for start, stop in plan["decoded_row_intervals"]:
            # Each selected physical chunk is decoded once, then only exact
            # requested row positions are copied; all neighboring rows expire.
            decoded = np.asarray(array[start:stop])
            expected_shape = (stop - start, *plan["shape"][1:])
            if decoded.shape != expected_shape or decoded.dtype != result.dtype:
                raise ValueError("decoded chunk shape/dtype drift")
            locations = np.flatnonzero((index >= start) & (index < stop))
            result[locations] = decoded[index[locations] - start]
        return result

    def read_task(self, task):
        if task not in self.selection or task in self._completed:
            raise ValueError("task outside selection or already exported by this reader")
        rows = self.selection[task]
        # BOTH roles' actual headers and complete key plans are validated before
        # the first array payload chunk is opened.
        prepared = {role: self._prepare(task, role, rows) for role in ("sensor", "teacher")}
        arrays, report = {}, {}
        for role, (store, plans, indices) in prepared.items():
            group = zarr.open_group(store=store, mode="r")
            for name, plan in plans.items():
                arrays[name] = self._read_field(group, name, indices, plan)
            if set(store.cache) != store.allowed or any(n != 1 for n in store.read_counts.values()):
                raise ValueError("actual byte reads differ from exact once-per-chunk plan")
            report[role] = plans
        frame_rows, sequence = arrays["frame_row"], arrays["source_global_sequence_index"]
        if not np.array_equal(frame_rows, [r["frame_rows"] for r in rows]) or not np.array_equal(sequence, [r["source_sequence_id"] for r in rows]):
            raise ValueError("selected history/source sequence identity drift")
        n = len(rows)
        unique_frames = len({f for row in rows for f in row['frame_rows']})
        ranges, valid = arrays["range_m"].reshape(n, 5, 16, 720), arrays["valid_mask"].reshape(n, 5, 16, 720)
        translation, yaw = arrays["relative_translation_current_sensor_m"], arrays["relative_yaw_current_sensor_deg"]
        if (not np.isfinite(ranges).all() or np.any(ranges < 0) or np.any(ranges > 50.)
                or np.any(valid > 1) or not np.isfinite(translation).all() or not np.isfinite(yaw).all()
                or np.any(translation[:, -1] != 0) or np.any(yaw[:, -1] != 0)):
            raise ValueError("range/first-return mask/finite relative motion/current-origin contract drift")
        for value in (ranges, valid, translation, yaw, frame_rows, sequence):
            value.setflags(write=False)
        role_coverage = {}
        for role, plans in report.items():
            actual = _interval_union_count([interval for p in plans.values() for interval in p["decoded_row_intervals"]])
            selected = unique_frames if role == "sensor" else n
            role_coverage[role] = {"decoded_actual_unique_rows": actual,
                "selected_unique_rows": selected, "collateral_actual_unique_rows": actual - selected}
        self._completed.add(task)
        return SurfaceTaskInputs(task, ranges, valid, translation, yaw, frame_rows, sequence,
            {"arrays": report, "selected_observations": n, "selected_unique_sensor_frames": unique_frames,
             "role_row_coverage": role_coverage,
             "selected_sequence_rows": [r["sequence_row"] for r in rows],
             "array_chunk_count": sum(p["decoded_chunk_count"] for plans in report.values() for p in plans.values()),
             "decoded_padded_bytes": sum(p["decoded_padded_bytes"] for plans in report.values() for p in plans.values()),
             "neighbor_rows_are_training_samples": False, "new_labels": 0, "model_windows": 0})
