"""Read selected primitive metadata/evidence without opening other tasks.

Not a training loader: raw range/valid arrays and checkpoint access are absent.
Recorded chunk reads may include neighboring C01 rows in the same physical
chunk; logical row selection remains exact and is reported separately.
"""
from collections import defaultdict
import hashlib
from pathlib import Path
import re

import numpy as np
import zarr

from mtare_topo.data.primitive_attachment_observability_sidecar import unpack_endpoint_observed


TASK = re.compile(r"S(?:0[1-9]|10)_[a-z0-9_]+_C0[1-6]__c1_mixed$")
TEACHER_ARRAYS = frozenset(("frame_row", "primitive_index", "primitive_mask", "source_global_sequence_index",
                          "support_ray_count", "temporal_visibility", "relative_translation_current_sensor_m"))
SIDECAR_ARRAYS = frozenset(("endpoint_observed_packed", "source_global_sequence_index"))


def validate_selection(records):
    if not records:
        raise ValueError("empty selection")
    tasks, seen = defaultdict(list), set()
    for row in records:
        task = row.get("task", "")
        index, source = row.get("row_index"), row.get("source_global_sequence_index")
        if (not isinstance(task, str) or not TASK.fullmatch(task)
                or type(index) is not int or index < 0 or type(source) is not int or source < 0):
            raise ValueError("selection must name an exact fit task and nonnegative integer rows")
        if (task, index) in seen:
            raise ValueError("duplicate selected row")
        seen.add((task, index))
        tasks[task].append(dict(row))
    return dict(sorted(tasks.items()))


class EvidenceStore(zarr.storage.DirectoryStore):
    """Read-only key allowlist and optional sealed-byte integrity check."""
    def __init__(self, path, arrays, opened, expected_sha256=None):
        self.root = Path(path).resolve(strict=True)
        self.arrays = frozenset(arrays)
        self.opened = opened
        self.expected_sha256 = expected_sha256
        super().__init__(str(self.root))

    def __getitem__(self, key):
        parts = key.split("/")
        if (key not in (".zgroup", ".zattrs") and parts[0] not in self.arrays
                or any(p in ("", ".", "..") for p in parts)):
            raise PermissionError(f"inventory field not authorized: {key}")
        path = self.root.joinpath(*parts)
        if not path.resolve().is_relative_to(self.root):
            raise PermissionError("inventory chunk escapes selected shard")
        try:
            data = super().__getitem__(key)
        except KeyError:
            if self.expected_sha256 is not None and str(path) in self.expected_sha256:
                raise ValueError(f"sealed chunk missing: {path}") from None
            raise
        digest = hashlib.sha256(data).hexdigest()
        if self.expected_sha256 is not None and self.expected_sha256.get(str(path)) != digest:
            raise ValueError(f"sealed input drift or unsealed chunk: {path}")
        self.opened[str(path)] = digest
        return data

    def __setitem__(self, key, value):
        raise PermissionError("inventory store is read-only")

    def __contains__(self, key):
        present = super().__contains__(key)
        path = self.root / key
        if not present and self.expected_sha256 is not None and str(path) in self.expected_sha256:
            # Zarr checks membership before __getitem__ and otherwise treats
            # a deleted chunk as fill-value data. A seal forbids that fallback.
            raise ValueError(f"sealed chunk missing: {path}")
        return present

    def __delitem__(self, key):
        raise PermissionError("inventory store is read-only")


class ScopedCompositionInventory:
    def __init__(self, teacher_root, sidecar_root, records, *, expected_sha256=None):
        # Validate every requested identity before any filesystem access.
        self.selection = validate_selection(records)
        self.teacher_root = Path(teacher_root).resolve(strict=True)
        self.sidecar_root = Path(sidecar_root).resolve(strict=True)
        self.opened = {}
        self.expected_sha256 = expected_sha256

    def _open(self, root, task, arrays):
        path = root / (task + ".zarr")
        if path.resolve().parent != root:
            raise PermissionError("selected task is a symlink outside its root")
        store = EvidenceStore(path, arrays, self.opened, self.expected_sha256)
        return zarr.open_group(store=store, mode="r")

    def read_task(self, task):
        if task not in self.selection:
            raise PermissionError("task not selected")
        records = self.selection[task]
        indices = np.asarray([r["row_index"] for r in records], dtype=np.int64)
        expected_source = np.asarray([r["source_global_sequence_index"] for r in records], dtype=np.int64)
        teacher = self._open(self.teacher_root, task, TEACHER_ARRAYS)
        sidecar = self._open(self.sidecar_root, task, SIDECAR_ARRAYS)
        if (teacher.attrs.get("parent_id") != task.split("__")[0]
                or teacher.attrs.get("partition") != "fit"
                or teacher.attrs.get("geometry_realization") != "c1_mixed"
                or teacher.attrs.get("student_identity_input_forbidden") is not True):
            raise ValueError("teacher identity/split contract drift")
        if (sidecar.attrs.get("schema_version") != "primitive_attachment_observability_sidecar_v1"
                or sidecar.attrs.get("hidden_pair_semantics") != "unknown_never_negative"
                or sidecar.attrs.get("support_band_m") != .25):
            raise ValueError("observability sidecar contract drift")
        count = teacher["frame_row"].shape[0]
        if (indices >= count).any():
            raise IndexError("selected row outside teacher shard")

        def read(group, name):
            return np.asarray(group[name].oindex[indices])

        source = read(teacher, "source_global_sequence_index")
        if not np.array_equal(source, expected_source) or not np.array_equal(read(sidecar, "source_global_sequence_index"), expected_source):
            raise ValueError("selected row source identity drift")
        output = {name: read(teacher, name) for name in TEACHER_ARRAYS if name != "source_global_sequence_index"}
        output["endpoint_observed"] = unpack_endpoint_observed(read(sidecar, "endpoint_observed_packed"))
        b = len(indices)
        shapes = {"frame_row": (b, 5), "primitive_index": (b, 32), "primitive_mask": (b, 32),
                  "support_ray_count": (b, 32), "temporal_visibility": (b, 5, 32),
                  "relative_translation_current_sensor_m": (b, 5, 3), "endpoint_observed": (b, 32, 2)}
        for name, shape in shapes.items():
            if output[name].shape != shape or not np.isfinite(output[name]).all():
                raise ValueError(f"inventory shape/nonfinite: {name}")
        if not np.array_equal(output["primitive_mask"], output["primitive_index"] >= 0):
            raise ValueError("inventory primitive mask drift")
        if np.any(output["endpoint_observed"] & ~output["primitive_mask"].astype(bool)[..., None]):
            raise ValueError("inactive primitive endpoint observed")
        rows = output["frame_row"]
        if np.any(rows < 0) or np.any(np.diff(rows, axis=1) <= 0):
            raise ValueError("noncausal source frame rows")
        if np.any(output["relative_translation_current_sensor_m"][:, -1] != 0):
            raise ValueError("current-frame translation is not zero")
        return output
