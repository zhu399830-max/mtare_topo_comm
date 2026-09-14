"""Hash-bound nomination inputs only; not a card validator or authorization.

The executor must bind the inventory and explicitly added codebook files in a
new card. Codebook source_sets bytes are read but never used; ordered primitive
IDs are checked directly. No scans, geometry arrays or labels are made.
"""
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re

import numpy as np
from numcodecs import Blosc, blosc

from .gse_review_inventory_reader_v1 import project_path
from .gse_review_nomination_v1 import OverlapRows, validate_paired_constructions
from .gse_review_source_index_v1 import VARIANTS, TraversalIndexInterval, VariantTraversalIndices

FIELDS = {"primitive_index": ((32,), np.dtype("<i4")),
          "primitive_mask": ((32,), np.dtype("u1")),
          "disconnected_overlap_packed": ((32, 4), np.dtype("u1"))}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
        ensure_ascii=True, allow_nan=False).encode()).hexdigest()


def strict_json(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("duplicate JSON key")
            result[key] = value
        return result
    def constant(value):
        raise ValueError("nonfinite JSON constant")
    return json.loads(raw, object_pairs_hook=pairs, parse_constant=constant)


def integer(value, low=0):
    return type(value) is int and value >= low


@dataclass(frozen=True)
class NominationParentData:
    parent_id: str
    split: str
    construction: object
    intervals: tuple
    overlap_by_task: dict
    continuity_requires_source_audit: bool = True
    codebook_order_verified: bool = True

    def overlaps_for_interval(self, interval):
        if interval not in self.intervals:
            raise ValueError("interval is not in this authenticated parent")
        result = {}
        for variant in interval.variants:
            source = self.overlap_by_task[variant.task]
            lookup = {row: i for i, row in enumerate(source.sequence_rows)}
            indices = [lookup[row] for row in variant.sequence_rows]
            result[variant.variant] = OverlapRows(variant.sequence_rows,
                source.primitive_index[indices], source.primitive_mask[indices],
                source.disconnected_overlap_packed[indices])
        return result


class ReviewNominationReader:
    """Read exact scoped files with fail-closed missing chunks and byte ledger.

    Scope is the new card's scope: source_scope retains the draft verbatim,
    actual_read_scope.codebook_paths explicitly adds codebook authorization.
    Synthetic
    fixtures can have fewer parents; the future card, not this I/O layer,
    enforces the real 55-parent/165-task population.
    """
    def __init__(self, root, scope, source_seals):
        self.root = Path(root).resolve(strict=True)
        # Snapshot caller dictionaries; never allow later scope widening.
        outer = strict_json(json.dumps(scope, allow_nan=False))
        self.scope = outer["source_scope"]
        self.scope["codebook_paths"] = outer["actual_read_scope"]["codebook_paths"]
        self.seals = strict_json(json.dumps(source_seals, allow_nan=False))
        if set(self.seals) != {"construction", "teacher"}:
            raise ValueError("exact construction/teacher seals required")
        self.opened, self.chunk_reads, self._seal_bytes = {}, [], {}
        self.parents = self.scope["eligible_parents"]
        if (not self.parents or len(set(self.parents)) != len(self.parents)
                or any(type(p) is not str or not re.fullmatch(r"[A-Za-z0-9_]+_C0[1-7]", p) for p in self.parents)):
            raise ValueError("unique explicit C01-C07 parents required")
        self.tasks = {row["task"]: row for row in self.scope["task_row_summary"]}
        if len(self.tasks) != len(self.scope["task_row_summary"]) or set(self.tasks) != {p + "__" + v for p in self.parents for v in VARIANTS}:
            raise ValueError("exact three tasks per scoped parent required")
        paths = self.scope.get("codebook_paths")
        if type(paths) is not dict or set(paths) != set(self.tasks) or len(set(paths.values())) != len(paths):
            raise ValueError("explicit distinct codebook path per exact task required")
        for parent in self.parents:
            for variant in VARIANTS:
                row = self.tasks[parent + "__" + variant]
                part = "c07" if parent.endswith("C07") else "fit"
                if row["source_partition"] != part or row["split"] not in (("calibration", "development") if part == "c07" else ("fit",)):
                    raise ValueError("parent partition/split drift")
                if not integer(row["source_sequence_count"], 21):
                    raise ValueError("integer full source sequence count required")
            if len({self.tasks[parent + "__" + v]["split"] for v in VARIANTS}) != 1:
                raise ValueError("paired split drift")
        if digest(self.scope["task_row_summary"]) != self.scope["task_row_summary_sha256"]:
            raise ValueError("task summary hash drift")

    def _read(self, relative, expected):
        if relative not in expected or not re.fullmatch(r"[a-f0-9]{64}", expected[relative]):
            raise PermissionError("unsealed or unauthorized source")
        path = project_path(self.root, relative)
        try:
            with path.open("rb") as stream:
                raw = stream.read(64 * 1024 ** 2 + 1)
        except FileNotFoundError:
            raise ValueError("sealed source missing; never fill absent chunks") from None
        self.opened[relative] = hashlib.sha256(raw).hexdigest()
        if len(raw) > 64 * 1024 ** 2:
            raise ValueError("source file exceeds bounded reader size")
        if self.opened[relative] != expected[relative]:
            raise ValueError("source hash drift")
        return raw

    def _collect(self, role, allowed):
        source = self.seals[role]
        if role not in self._seal_bytes:
            self._seal_bytes[role] = self._read(source["path"], {source["path"]: source["sha256"]})
        result = {}
        for line in self._seal_bytes[role].decode().splitlines():
            parts = line.split(None, 1)
            if len(parts) != 2 or parts[1] not in allowed:
                continue  # Filter BEFORE resolving any excluded source path.
            value, path = parts
            if path in result or not re.fullmatch(r"[a-f0-9]{64}", value):
                raise ValueError("duplicate or invalid selected source seal")
            result[path] = value
        if set(result) != allowed:
            raise ValueError("selected source missing from seal")
        return result

    def _intervals(self, parent):
        expected = self.scope["inventory_source_files"]
        paths = [p for p in expected if p.endswith("/" + parent + "_identity_intervals.json")]
        if len(paths) != 1:
            raise ValueError("one bound parent interval source required")
        document = strict_json(self._read(paths[0], expected))
        part = self.tasks[parent + "__" + VARIANTS[0]]["source_partition"]
        if document["parent_id"] != parent or document["partition"] != part or document.get("duration_s") is not None:
            raise ValueError("inventory parent/partition/duration drift")
        intervals, identities, seen_traversals = [], {v: [] for v in VARIANTS}, set()
        for row in document["intervals"]:
            traversal = row["traversal_id"]
            if not isinstance(traversal, str) or not re.fullmatch(re.escape(parent) + r":[A-Za-z0-9_.:-]+:d[01]", traversal) or traversal in seen_traversals:
                raise ValueError("unique parent traversal identity required")
            seen_traversals.add(traversal)
            variants = row["variants"]
            if len(variants) != 3 or tuple(v["variant"] for v in variants) != VARIANTS:
                raise ValueError("ordered three inventory variants required")
            if len(variants[0]["sequence_rows"]) < 21:
                continue
            typed = []
            for v in variants:
                task = parent + "__" + v["variant"]
                rows, ids, frames, arcs = (v[k] for k in ("sequence_rows", "source_sequence_ids", "frame_rows", "decision_arc_m"))
                q = len(rows)
                if v["task"] != task or not (q == len(ids) == len(frames) == len(arcs)):
                    raise ValueError("interval task/field length drift")
                if any(not integer(n) or n >= self.tasks[task]["source_sequence_count"] for n in rows) or any(b != a + 1 for a, b in zip(rows, rows[1:])):
                    raise ValueError("exact causal consecutive source row indices required")
                if any(not integer(n) for n in ids) or len(set(ids)) != q:
                    raise ValueError("source IDs must be unique integers")
                if any(type(fs) is not list or len(fs) != 5 or any(not integer(n) for n in fs) or any(b != a + 1 for a, b in zip(fs, fs[1:])) for fs in frames):
                    raise ValueError("five causal consecutive frame identities required")
                if any(b != [n + 1 for n in a] for a, b in zip(frames, frames[1:])):
                    raise ValueError("rolling history drift")
                if any(type(a) not in (float, int) or not math.isfinite(a) for a in arcs) or any(b <= a for a, b in zip(arcs, arcs[1:])):
                    raise ValueError("finite increasing source arc required")
                if ids != variants[0]["source_sequence_ids"] or frames != variants[0]["frame_rows"]:
                    raise ValueError("cross-variant source/frame identity drift")
                identities[v["variant"]].extend(dict(row=r, source_sequence_id=s, frame_rows=f) for r, s, f in zip(rows, ids, frames))
                typed.append(VariantTraversalIndices(v["variant"], task, tuple(rows), tuple(ids), tuple(map(tuple, frames)), tuple(arcs)))
            intervals.append(TraversalIndexInterval(traversal, tuple(typed)))
        selected = {}
        for variant, entries in identities.items():
            entries.sort(key=lambda x: x["row"])
            rows = [e["row"] for e in entries]; task = parent + "__" + variant; summary = self.tasks[task]
            if (not rows or len(set(rows)) != len(rows) or summary["row_count"] != len(rows)
                    or digest(rows) != summary["unique_rows_sha256"] or digest(entries) != summary["row_identity_sha256"]
                    or len({f for e in entries for f in e["frame_rows"]}) != summary["unique_frame_count"]):
                raise ValueError("eligible exact row/identity/count hash drift")
            selected[task] = tuple(rows)
        return tuple(intervals), selected

    def read_parent(self, parent):
        if parent not in self.parents:
            raise PermissionError("parent not authorized")
        intervals, selected = self._intervals(parent)
        documents, headers, prefixes, chunks = {}, {}, {}, {}
        construct_paths = {v: self.scope["source_roots"]["construction"] + "/" + self.tasks[parent + "__" + v]["source_partition"] + "/" + parent + "__" + v + ".json" for v in VARIANTS}
        codebook_paths = {v: self.scope["codebook_paths"][parent + "__" + v] for v in VARIANTS}
        expected = self._collect("construction", set(construct_paths.values()) | set(codebook_paths.values()))
        for variant, path in construct_paths.items():
            doc = strict_json(self._read(path, expected))
            if doc.get("parent_id") != parent or doc.get("geometry_realization") != variant:
                raise ValueError("construction root identity drift")
            base = doc.get("base_construction", {})
            if base.get("endpoint_attachment_mode") != "free_space_overlap" or base.get("node_degree_source") != "edge_incidence" or any(r.get("operation") != "endpoint_union" for r in base.get("composition_operations", [])):
                raise ValueError("construction incidence source contract drift")
            documents[variant] = doc
        construction = validate_paired_constructions(documents)
        for variant, path in codebook_paths.items():
            book = strict_json(self._read(path, expected))
            if (book.get("schema_version") != "primitive_membership_codebook_v1"
                    or book.get("parent_id") != parent or book.get("geometry_realization") != variant
                    or book.get("primitive_ids") != [p.primitive_id for p in construction.primitives]):
                raise ValueError("codebook/construction ordered primitive identity drift")
        allowed = set()
        for task, rows in selected.items():
            summary = self.tasks[task]; prefix = self.scope["source_roots"]["teacher"] + "/" + summary["source_partition"] + "/" + task + ".zarr/"
            prefixes[task] = prefix; chunk = min(256, summary["source_sequence_count"])
            chunks[task] = sorted({i // chunk for i in rows})
            allowed.update(prefix + k for k in (".zgroup", ".zattrs"))
            for field, (tail, _) in FIELDS.items():
                allowed.add(prefix + field + "/.zarray")
                allowed.update(prefix + field + "/" + str(i) + ".0" * len(tail) for i in chunks[task])
        expected = self._collect("teacher", allowed)
        # All three tasks' headers pass before ANY new teacher chunk is read.
        for task, prefix in prefixes.items():
            group_header = strict_json(self._read(prefix + ".zgroup", expected))
            if group_header != {"zarr_format": 2} or type(group_header["zarr_format"]) is not int:
                raise ValueError("Zarr group schema drift")
            attrs = strict_json(self._read(prefix + ".zattrs", expected)); variant = task.split("__", 1)[1]
            if any(attrs.get(k) != value for k, value in dict(schema_version="primitive_relation_p1b_teacher_shard_v1", parent_id=parent, partition=self.tasks[task]["source_partition"], geometry_realization=variant, maximum_slots=32, window_frames=5, student_identity_input_forbidden=True).items()):
                raise ValueError("teacher task/header drift")
            if type(attrs["maximum_slots"]) is not int or type(attrs["window_frames"]) is not int or attrs["student_identity_input_forbidden"] is not True:
                raise ValueError("strict integer/boolean teacher header required")
            q = self.tasks[task]["source_sequence_count"]
            for field, (tail, dtype) in FIELDS.items():
                h = strict_json(self._read(prefix + field + "/.zarray", expected))
                if (type(h.get("zarr_format")) is not int or h.get("zarr_format") != 2 or h.get("shape") != [q, *tail] or h.get("chunks") != [min(q, 256), *tail]
                        or any(type(n) is not int for n in h["shape"] + h["chunks"]) or h.get("dtype") != dtype.str
                        or h.get("order") != "C" or h.get("filters") is not None or h.get("dimension_separator", ".") != "."
                        or h.get("compressor") != {"id": "blosc", "cname": "zstd", "clevel": 5, "shuffle": 2, "blocksize": 0}):
                    raise ValueError("teacher array schema/chunk/codec drift before payload")
                headers[task, field] = h
        outputs = {}
        for task, rows in selected.items():
            data = {}; row_array = np.asarray(rows, dtype=np.int64)
            for field, (tail, dtype) in FIELDS.items():
                h = headers[task, field]; chunk = h["chunks"][0]; out = np.empty((len(rows), *tail), dtype=dtype)
                for index in chunks[task]:
                    path = prefixes[task] + field + "/" + str(index) + ".0" * len(tail)
                    raw = self._read(path, expected)
                    if len(raw) < 16 or blosc.cbuffer_sizes(raw)[:2] != (math.prod(h["chunks"]) * dtype.itemsize, len(raw)):
                        raise ValueError("compressed chunk size header drift before allocation")
                    decoded = Blosc(cname="zstd", clevel=5, shuffle=2).decode(raw)
                    if len(decoded) != math.prod(h["chunks"]) * dtype.itemsize:
                        raise ValueError("decoded chunk length drift")
                    values = np.frombuffer(decoded, dtype=dtype).reshape(h["chunks"])
                    positions = np.flatnonzero(row_array // chunk == index)
                    out[positions] = values[row_array[positions] - index * chunk]
                    logical = min(chunk, h["shape"][0] - index * chunk)
                    self.chunk_reads.append(dict(task=task, field=field, chunk=index, compressed_bytes=len(raw),
                        decoded_storage_rows=chunk, decoded_logical_rows=logical, effective_rows=len(positions),
                        collateral_logical_rows=logical - len(positions)))
                out.setflags(write=False); data[field] = out
            outputs[task] = OverlapRows(rows, **data)
        return NominationParentData(parent, self.tasks[parent + "__" + VARIANTS[0]]["split"], construction, intervals, outputs)
