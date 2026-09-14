"""Read only sealed, explicitly authorized P1a/P1b identity index fields.

No directory traversal over datasets, scan decoding, construction labels or
model access. The executor validates the approved card before calling this.
"""
import hashlib
import math
from pathlib import Path
import re

import numpy as np
import zarr

from .gse_scoped_inventory import EvidenceStore
from .gse_review_source_index_v1 import (
    SENSOR_FIELDS, TEACHER_FIELDS, VARIANTS, validate_review_source_indices,
)


def sha_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for part in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(part)
    return digest.hexdigest()


def project_path(root, relative):
    if type(relative) is not str:
        raise ValueError("explicit project-relative source path required")
    path = Path(relative)
    if path.is_absolute() or any(part in ("..", ".") for part in path.parts):
        raise ValueError("source path must be project-relative without traversal")
    target = root / path
    if target.resolve() != target or not target.is_relative_to(root):
        raise ValueError("source symlink/path escape")
    return target


def allowed_key(key, fields):
    if key in (".zgroup", ".zattrs"):
        return True
    parts = key.split("/")
    return (len(parts) == 2 and parts[0] in fields and
            (parts[1] == ".zarray" or bool(re.fullmatch(r"[0-9]+(?:\.[0-9]+)*", parts[1]))))


class IdentityStore(EvidenceStore):
    def __getitem__(self, key):
        if not allowed_key(key, self.arrays):
            raise PermissionError(f"not an authorized identity key: {key}")
        path = self.root / key
        if path.resolve() != path:
            raise PermissionError("identity chunk symlink")
        try:
            data = zarr.storage.DirectoryStore.__getitem__(self, key)
        except KeyError:
            if str(path) in self.expected_sha256:
                raise ValueError(f"sealed identity chunk missing: {path}") from None
            raise
        actual = hashlib.sha256(data).hexdigest()
        self.opened[str(path)] = actual
        if self.expected_sha256.get(str(path)) != actual:
            raise ValueError(f"identity chunk changed or unsealed: {path}")
        return data

    def __contains__(self, key):
        # Zarr probes whether the explicitly requested group is an array. This
        # is not an authorization to open a root array or unlisted field.
        if key == ".zarray":
            return False
        if not allowed_key(key, self.arrays):
            return False
        return super().__contains__(key)


def collect_identity_seals(root, scope, index_reads=None):
    """Filter shared seal INDEX text before resolving any payload path.

    The shared index contains historical task names. No excluded task payload
    is opened, resolved or hashed. Return only allowed input-file digests.
    """
    root = Path(root).resolve(strict=True)
    expected = {}
    if index_reads is None:
        index_reads = {}
    for role, fields in (("sensor", SENSOR_FIELDS), ("teacher", TEACHER_FIELDS)):
        prefixes = {scope["source_roots"][role][task["partition"]] + "/" + task["task"] + ".zarr/"
                    for task in scope["tasks"]}
        source = scope["source_seals"][role]
        seal = project_path(root, source["path"])
        raw = seal.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        index_reads[str(seal)] = digest
        if digest != source["sha256"]:
            raise ValueError("source seal index drift")
        for line in raw.decode("utf-8").splitlines():
            digest, path = line.split(None, 1)
            # The slash preceding `.zarr` is fixed by the exact authorized
            # task list; never resolve the unmatched entries of a shared seal.
            marker = path.find(".zarr/")
            if marker < 0:
                continue
            prefix, key = path[:marker + 6], path[marker + 6:]
            if prefix not in prefixes or not allowed_key(key, fields):
                continue
            if not re.fullmatch(r"[a-f0-9]{64}", digest):
                raise ValueError("invalid selected seal digest")
            target = str(project_path(root, path))
            if target in expected:
                raise ValueError("duplicate selected seal entry")
            expected[target] = digest
        for prefix in prefixes:
            for key in (".zgroup", ".zattrs", *(f"{field}/.zarray" for field in sorted(fields))):
                if str(project_path(root, prefix + key)) not in expected:
                    raise ValueError(f"missing authorized index header in source seal: {prefix}{key}")
    return expected, index_reads


class ReviewIdentityInventoryReader:
    def __init__(self, root, scope, expected):
        self.root = Path(root).resolve(strict=True)
        self.scope = scope
        self.expected = expected
        self.opened = {}
        self.tasks = {(x["parent_id"], x["variant"]): x for x in scope["tasks"]}
        if len(self.tasks) != len(scope["tasks"]):
            raise ValueError("duplicate authorized task")

    def _read(self, task, role, fields):
        prefix = self.scope["source_roots"][role][task["partition"]]
        path = project_path(self.root, prefix + "/" + task["task"] + ".zarr")
        store = IdentityStore(path, fields, self.opened, self.expected)
        group = zarr.open_group(store=store, mode="r")
        attrs = group.attrs.asdict()
        if (attrs.get("parent_id") != task["parent_id"] or attrs.get("partition") != task["partition"]
                or attrs.get("geometry_realization") != task["variant"]):
            raise ValueError("authorized task and stored attributes differ")
        arrays = {field: group[field] for field in sorted(fields)}
        nbytes = sum(math.prod(a.shape) * a.dtype.itemsize for a in arrays.values())
        # Per-shard decoded index budget, not a scientific sample threshold.
        if nbytes > 256 * 1024 ** 2:
            raise ValueError("identity index shard exceeds256MiB before decoding")
        for field, array in arrays.items():
            expected_ndim = 2 if field == "frame_row" else 1
            if (array.ndim != expected_ndim or (expected_ndim == 2 and array.shape[1] != 5)
                    or array.dtype.kind not in ("f" if field == "route_arc_m" else "iu")):
                raise ValueError("identity header shape/dtype drift before decoding")
            if math.prod(array.chunks) * array.dtype.itemsize > 256 * 1024 ** 2:
                raise ValueError("identity compressed chunk exceeds256MiB decoded limit")
        return attrs, {name: np.asarray(array[:]) for name, array in arrays.items()}

    def read_parent(self, parent):
        if parent not in self.scope["parent_ids"]:
            raise PermissionError("parent not authorized")
        sensor_attrs, teacher_attrs, sensor, teacher = {}, {}, {}, {}
        for variant in VARIANTS:
            task = self.tasks[parent, variant]
            sensor_attrs[variant], sensor[variant] = self._read(task, "sensor", SENSOR_FIELDS)
            teacher_attrs[variant], teacher[variant] = self._read(task, "teacher", TEACHER_FIELDS)
        return validate_review_source_indices(parent_id=parent, sensor_attrs_by_variant=sensor_attrs,
            teacher_attrs_by_variant=teacher_attrs, sensor_arrays_by_variant=sensor,
            teacher_arrays_by_variant=teacher)
