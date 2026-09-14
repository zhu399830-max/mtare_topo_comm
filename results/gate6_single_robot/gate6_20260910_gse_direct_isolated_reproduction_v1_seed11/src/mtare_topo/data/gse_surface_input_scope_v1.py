"""Compile exact chunk permissions from pinned selection and Zarr headers only."""
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re

from mtare_topo.governance_surface_input import (
    SELECTION, SELECTION_SEAL, SEALS, FIELDS, RESOURCES, ENVIRONMENT, FORBIDDEN, expected_tasks,
)
from .gse_surface_input_export_v1 import plan_array_access


def safe_path(root, relative):
    if (type(relative) is not str or Path(relative).is_absolute()
            or any(p in ("", ".", "..") for p in relative.split("/"))):
        raise ValueError("clean explicit relative path required")
    path = root / relative
    if path.resolve(strict=True) != path:
        raise ValueError("source symlink or alias forbidden")
    return path


def checked_read(root, relative, expected, reads):
    raw = safe_path(root, relative).read_bytes()
    reads[relative] = hashlib.sha256(raw).hexdigest()
    if reads[relative] != expected:
        raise ValueError("pinned source drift: " + relative)
    return raw


def _object(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("duplicate JSON key")
            result[key] = value
        return result
    def nonfinite(value):
        raise ValueError("nonfinite JSON")
    return json.loads(raw, object_pairs_hook=pairs, parse_constant=nonfinite)


def selected_rows(manifest, tasks):
    if manifest.get("schema_version") != "gse_surface_identity_selection_v1":
        raise ValueError("completed exact surface selection required")
    rows = defaultdict(list)
    for row in manifest["observations"]:
        if row["task"] not in tasks:
            raise ValueError("selected task outside C01-C07 source scope")
        source = tasks[row["task"]]
        if row["parent_id"] != source["parent_id"] or row["variant"] != source["variant"]:
            raise ValueError("selected identity mismatch")
        rows[row["task"]].append(row)
    if set(rows) != set(tasks) or any(len(r) != 16 for r in rows.values()):
        raise ValueError("exact sixteen observations per task required")
    if Counter(r["split"] for r in manifest["observations"]) != {"fit": 2880, "calibration": 240, "development": 240}:
        raise ValueError("original selected split population drift")
    for task, values in rows.items():
        frames = [f for r in values for f in r["frame_rows"]]
        if len(frames) != 80 or len(set(frames)) != 80 or len({r["sequence_row"] for r in values}) != 16:
            raise ValueError("source frame independence or row uniqueness drift")
    return rows


def compile_input_scope(root):
    """Read only exact selection, shared hash INDEX text and2100 headers.

    Never open or hash scan/motion chunks here. Their expected hashes come from
    the pinned historical seal and are rechecked at the actual read operation.
    Shared seal metadata can name excluded tasks, but those paths are neither
    resolved nor opened. The permitted payload set is derived, not caller-made.
    """
    root = Path(root).resolve(strict=True)
    reads = {}
    sel_seal = SELECTION + "/artifacts/evidence_sha256.txt"
    sel_path = SELECTION + "/artifacts/selection_manifest.json"
    raw = checked_read(root, sel_seal, SELECTION_SEAL, reads)
    matches = [line.split("  ", 1)[0] for line in raw.decode().splitlines() if line.endswith("  " + sel_path)]
    if len(matches) != 1:
        raise ValueError("selection seal must name manifest exactly once")
    manifest_raw = checked_read(root, sel_path, matches[0], reads)
    manifest = _object(manifest_raw)
    tasks = expected_tasks()
    rows = selected_rows(manifest, tasks)
    files, plans, header_reads = {}, {}, {}
    for role, fields in FIELDS.items():
        seal = SEALS[role]
        index = checked_read(root, seal["path"], seal["sha256"], reads)
        prefixes = {source[role] + "/" for source in tasks.values()}
        candidates = {}
        for line in index.decode().splitlines():
            h, path = line.split(None, 1)
            marker = path.find(".zarr/")
            if marker < 0 or path[:marker + 6] not in prefixes:
                continue
            key = path[marker + 6:]
            parts = key.split("/")
            if key not in (".zgroup", ".zattrs") and not (len(parts) == 2 and parts[0] in fields and (parts[1] == ".zarray" or re.fullmatch(r"[0-9]+(?:\.[0-9]+)*", parts[1]))):
                continue
            if path in candidates or re.fullmatch(r"[a-f0-9]{64}", h) is None:
                raise ValueError("selected source seal duplicate or bad digest")
            candidates[path] = h
        for task, sources in tasks.items():
            prefix = sources[role] + "/"
            indices = ([f for row in rows[task] for f in row["frame_rows"]]
                       if role == "sensor" else [row["sequence_row"] for row in rows[task]])
            for key in (".zgroup", ".zattrs", *(f + "/.zarray" for f in fields)):
                path = prefix + key
                if path not in candidates:
                    raise ValueError("required header absent from original seal: " + path)
                data = checked_read(root, path, candidates[path], header_reads)
                files[path] = candidates[path]
                value = _object(data)
                if key == ".zgroup" and value.get("zarr_format") != 2:
                    raise ValueError("group format drift")
                if key == ".zattrs":
                    if (value.get("parent_id") != sources["parent_id"] or value.get("partition") != sources["partition"]
                            or value.get("geometry_realization") != sources["variant"]):
                        raise ValueError("stored group identity differs from selection")
                if key.endswith("/.zarray"):
                    field = key.split("/")[0]
                    plan = plan_array_access(value, field, role, indices)
                    dimension = "source_frame_count" if role == "sensor" else "source_sequence_count"
                    if any(row[dimension] != plan["shape"][0] for row in rows[task]):
                        raise ValueError("actual header shape differs from sealed identity inventory")
                    plans[prefix + field] = plan
                    for chunk in plan["chunk_keys"]:
                        path = prefix + field + "/" + chunk
                        if path not in candidates:
                            raise ValueError("required selected chunk absent from source seal: " + path)
                        files[path] = candidates[path]
    scope = {"selection": {"path": sel_path, "sha256": matches[0], "seal_path": sel_seal, "seal_sha256": SELECTION_SEAL},
        "source_seals": SEALS, "task_sources": tasks, "fields": FIELDS, "file_sha256": files, "array_access": plans,
        "population": {"parents": 70, "physical_edge_units": 1120, "observations": 3360, "unique_variant_frames": 16800,
            "split_observations": {"fit": 2880, "calibration": 240, "development": 240}, "new_labels": 0,
            "effective_label_count": None, "independent_structure_count": None, "duration_s": None},
        "spatial_temporal_basis": "Five consecutive causal source rows; exact per-decision route_arc and source history in pinned selection. Full history arc spacing and acquisition duration remain unknown; stored relative translation/yaw is exported without claiming a sampling clock or continuous route.",
        "teacher_source": "No teacher targets exported. P1b supplies stored causal relative motion and identity checks only; no physical/reference/construction field becomes student input. Physical-root labels paused under registered missing-robot-contract contingency.",
        "leakage_audit": "Fixed C01-C06 fit/C07 original five/five calibration-development parents, three variants within parent. C07 historically exposed. Source indices sealed independently before any model score. Adjacent compressed rows are counted as collateral per field, never exported samples. No C08-C10 or benchmark payload path resolution.",
        "resources": RESOURCES, "environment": ENVIRONMENT, "forbidden_payloads": FORBIDDEN}
    return scope, manifest, {**reads, **header_reads}


def access_summary(scope):
    result = {}
    for role, fields in FIELDS.items():
        for field in fields:
            values = [p for p in scope["array_access"].values() if p["role"] == role and p["field"] == field]
            result[role + "/" + field] = {k: sum(p[k] for p in values) for k in
                ("selected_unique_rows", "decoded_actual_rows", "decoded_actual_bytes", "decoded_padded_bytes", "decoded_chunk_count")}
    return result
