"""Exact scope for the approved surface experiment's sensor-input export only."""
import hashlib
import json
import math
import re

from mtare_topo.governance_identity_inventory import P1A, P1B, VARIANTS
from mtare_topo.governance_surface_selection import PARENTS, digest

SCHEMA = "v3_surface_input_export_card_v1"
CARD_ID = "gse_surface_input_export_v1"
SELECTION = "results/gate3_semantics/gate3_20260907_gse_surface_identity_selection_v1_seed20260906"
SELECTION_SEAL = "0284a664605b0c9fa3e5d8a3a56c4bc90e4b1a78861c16aedc24a4f1941602e2"
FIELDS = {
    "sensor": ["range_m", "valid_mask"],
    "teacher": ["frame_row", "source_global_sequence_index",
                "relative_translation_current_sensor_m", "relative_yaw_current_sensor_deg"],
}
SEALS = {
    "sensor": {"path": P1A + "/artifacts/evidence_sha256.txt", "sha256": "79fd988ac8c205d74c93e4858b7b579571a06502e778f31634046b48791a0668"},
    "teacher": {"path": P1B + "/artifacts/evidence_sha256.txt", "sha256": "f629b511e9a945bbe15249e3bc212be57a01b1aa0118d0d7d228824aa25f7d47"},
}
RESOURCES = {"wall_time_s": 3600, "host_ram_bytes": 4294967296, "gpu_bytes": 0, "output_bytes": 2147483648}
ENVIRONMENT = {"python": "3.13.5", "numpy": "2.1.3", "zarr": "2.18.7", "numcodecs": "0.15.1"}
FORBIDDEN = ["construction", "membership_codes", "geometry_targets", "absolute_poses", "mesh", "models", "new_labels", "training", "C08-C10", "benchmark"]


def _exact(actual, expected):
    """JSON equality preserving bool/int/float distinctions; malformed is false."""
    try:
        return json.dumps(actual, sort_keys=True, allow_nan=False) == json.dumps(expected, sort_keys=True, allow_nan=False)
    except (TypeError, ValueError):
        return False


def _plan_files(plan, role, field, array_path):
    """Pure-standard-library validation so governance needs no Zarr import."""
    fields = {"role", "field", "shape", "chunks", "dtype", "chunk_keys", "chunk_first_axis_indices",
              "decoded_row_intervals", "selected_unique_rows", "decoded_actual_rows", "decoded_actual_bytes",
              "decoded_padded_bytes", "decoded_chunk_count"}
    contract = {"range_m": ([16, 720], "<f4", 16, 4), "valid_mask": ([16, 720], "|u1", 32, 1),
        "frame_row": ([5], "<i4", 256, 4), "source_global_sequence_index": ([], "<i8", 256, 8),
        "relative_translation_current_sensor_m": ([5, 3], "<f4", 256, 4),
        "relative_yaw_current_sensor_deg": ([5], "<f4", 256, 4)}
    if type(plan) is not dict or set(plan) != fields:
        raise ValueError("closed exact array plan required")
    tail, dtype, first_chunk, itemsize = contract[field]
    shape, chunks, indices = (plan[k] for k in ("shape", "chunks", "chunk_first_axis_indices"))
    if (type(shape) is not list or type(chunks) is not list or len(shape) != 1 + len(tail)
            or len(chunks) != len(shape) or any(type(v) is not int or v < 1 for v in shape + chunks)
            or shape[1:] != tail or chunks[1:] != tail):
        raise ValueError("array shape/chunks differ from input contract")
    if chunks[0] != (first_chunk if role == "sensor" else min(first_chunk, shape[0])):
        raise ValueError("array first-axis chunk drift")
    if (type(indices) is not list or not indices or any(type(v) is not int or v < 0 for v in indices)
            or indices != sorted(set(indices)) or indices[-1] * chunks[0] >= shape[0]):
        raise ValueError("exact sorted unique in-bounds chunk indices required")
    intervals = [[i * chunks[0], min((i + 1) * chunks[0], shape[0])] for i in indices]
    actual_rows = sum(b - a for a, b in intervals)
    selected = 80 if role == "sensor" else 16
    if actual_rows < selected:
        raise ValueError("decoded rows cannot cover selected population")
    keys = [".".join(map(str, (i,) + (0,) * len(tail))) for i in indices]
    expected = {"role": role, "field": field, "shape": shape, "chunks": chunks, "dtype": dtype,
        "chunk_keys": keys, "chunk_first_axis_indices": indices, "decoded_row_intervals": intervals,
        "selected_unique_rows": selected, "decoded_actual_rows": actual_rows,
        "decoded_actual_bytes": actual_rows * math.prod(tail) * itemsize,
        "decoded_padded_bytes": len(indices) * math.prod(chunks) * itemsize,
        "decoded_chunk_count": len(indices)}
    if not _exact(plan, expected):
        raise ValueError("array role/field/dtype or decoded collateral accounting drift")
    return {array_path + "/" + key for key in keys}


def expected_tasks():
    result = {}
    for parent in PARENTS:
        partition = "c07" if parent.endswith("_C07") else "fit"
        for variant in VARIANTS:
            task = parent + "__" + variant
            result[task] = {"parent_id": parent, "variant": variant, "partition": partition,
                "sensor": P1A + "/artifacts/dataset/" + partition + "/" + task + ".zarr",
                "teacher": P1B + "/artifacts/teacher/" + partition + "/" + task + ".zarr"}
    return result


def validate_surface_input_card(card):
    from mtare_topo.governance import ValidationReport
    errors = []
    if type(card) is not dict:
        return ValidationReport(False, ("surface input card object required",))
    required = {"schema_version", "card_id", "operation", "purpose", "limitations", "scope", "scope_sha256", "approval", "scientific_gate_pass", "training_eligibility"}
    if set(card) != required or (card.get("schema_version"), card.get("card_id"), card.get("operation")) != (SCHEMA, CARD_ID, "data_export"):
        errors.append("closed sensor-only export schema required")
    scope = card.get("scope")
    if type(scope) is not dict:
        return ValidationReport(False, tuple(errors + ["exact scope object required"]))
    required_scope = {"selection", "source_seals", "task_sources", "fields", "file_sha256", "array_access", "population", "spatial_temporal_basis", "teacher_source", "leakage_audit", "resources", "environment", "forbidden_payloads"}
    if set(scope) != required_scope:
        errors.append("closed input scope required")
    selection = scope.get("selection", {})
    if (type(selection) is not dict or set(selection) != {"path", "sha256", "seal_path", "seal_sha256"}
            or selection.get("path") != SELECTION + "/artifacts/selection_manifest.json"
            or selection.get("seal_path") != SELECTION + "/artifacts/evidence_sha256.txt"
            or selection.get("seal_sha256") != SELECTION_SEAL
            or type(selection.get("sha256")) is not str
            or not re.fullmatch(r"[a-f0-9]{64}", selection.get("sha256", ""))):
        errors.append("independent completed selection binding required")
    tasks = expected_tasks()
    if scope.get("task_sources") != tasks or scope.get("source_seals") != SEALS or scope.get("fields") != FIELDS:
        errors.append("exact210 C01-C07 tasks and six input fields required")
    if not _exact(scope.get("population"), {"parents": 70, "physical_edge_units": 1120, "observations": 3360, "unique_variant_frames": 16800, "split_observations": {"fit": 2880, "calibration": 240, "development": 240}, "new_labels": 0, "effective_label_count": None, "independent_structure_count": None, "duration_s": None}):
        errors.append("measured input counts cannot imply labels or independent structures")
    if not _exact(scope.get("resources"), RESOURCES) or scope.get("environment") != ENVIRONMENT or scope.get("forbidden_payloads") != FORBIDDEN:
        errors.append("fixed resource/environment/isolation policy required")
    files, plans = scope.get("file_sha256"), scope.get("array_access")
    if type(files) is not dict or type(plans) is not dict:
        errors.append("exact headers/chunks and decoded collateral plans required")
    else:
        prefixes = {sources[role] + "/": role for sources in tasks.values() for role in FIELDS}
        allowed_headers = {prefix + key for prefix, role in prefixes.items()
                           for key in (".zgroup", ".zattrs", *(f + "/.zarray" for f in FIELDS[role]))}
        if not allowed_headers.issubset(files):
            errors.append("all210 task headers required")
        for path, h in files.items():
            if type(path) is not str:
                errors.append("source path must be a string"); break
            marker = path.find(".zarr/")
            prefix, key = path[:marker + 6], path[marker + 6:]
            role = prefixes.get(prefix)
            if (role is None or type(h) is not str or re.fullmatch(r"[a-f0-9]{64}", h) is None):
                errors.append("source path/hash outside exact tasks"); break
            parts = key.split("/")
            if key not in (".zgroup", ".zattrs") and not (len(parts) == 2 and parts[0] in FIELDS[role] and (parts[1] == ".zarray" or re.fullmatch(r"[0-9]+(?:\.[0-9]+)*", parts[1]))):
                errors.append("source field outside sensor inputs"); break
        expected_plans = {s[role] + "/" + f for s in tasks.values() for role in FIELDS for f in FIELDS[role]}
        if set(plans) != expected_plans:
            errors.append("all1260 array access plans required")
        exact_files = set(allowed_headers)
        for source in tasks.values():
            for role, field_names in FIELDS.items():
                for field in field_names:
                    array_path = source[role] + "/" + field
                    try:
                        exact_files.update(_plan_files(plans.get(array_path), role, field, array_path))
                    except (ValueError, TypeError, KeyError, OverflowError):
                        errors.append("invalid exact array access plan: " + array_path)
        if set(files) != exact_files:
            errors.append("file permissions must equal all headers plus precisely planned chunks")
        # The executor independently regenerates these plans from pinned headers
        # and selected rows, before reading any chunk; no caller-malleable grant.
    for key in ("spatial_temporal_basis", "teacher_source", "leakage_audit"):
        if type(scope.get(key)) is not str or not scope[key].strip():
            errors.append(key + " required")
    for key in ("purpose", "limitations"):
        if type(card.get(key)) is not str or not card[key].strip():
            errors.append(key + " required")
    try:
        h = digest(scope)
    except (TypeError, ValueError):
        return ValidationReport(False, tuple(errors + ["finite JSON scope required"]))
    approval = card.get("approval")
    if type(approval) is not dict:
        approval = {}
    if (card.get("scope_sha256") != h or approval.get("scope_sha256") != h
            or approval.get("status") != "APPROVED" or approval.get("authorized_operations") != ["data_export"]
            or not _exact(approval.get("authorized_gates"), [3])):
        errors.append("standing authorization must bind this exact input export, not training")
    for key in ("approved_by", "approved_at", "scope", "confirmation_reference"):
        if type(approval.get(key)) is not str or not approval[key].strip():
            errors.append("approval." + key + " required")
    if card.get("scientific_gate_pass") is not False or card.get("training_eligibility") is not False:
        errors.append("input export does not produce qualified labels or science PASS")
    return ValidationReport(not errors, tuple(errors))
