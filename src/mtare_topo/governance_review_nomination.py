"""Closed-scope review nomination audit; no file reads or training authority.

The scope digest below is an independently frozen source constant. Validation
never derives its expected scope from a mutable document or the submitted card.
The embedded source draft is provenance; the actual card explicitly adds
same-task codebooks before any corresponding real input is opened.
"""
import hashlib
import json
from pathlib import PurePosixPath
import re


SCHEMA = "v3_review_nomination_audit_card_v1"
CARD_ID = "gse_review_nomination_v1"
EXPECTED_SCOPE_SHA256 = "523c5522cb1f63164ad81c71fc311d14779d1729f85b3491779ffdd96b5aa4fa"
VARIANTS = ("ellipse", "rounded_rectangle", "c1_mixed")
ROW_KEYS = {"task", "source_partition", "split", "row_count", "unique_frame_count",
            "unique_rows_sha256", "row_identity_sha256", "source_sequence_count",
            "projected_chunks_per_field", "projected_decoded_rows"}
CARD_KEYS = {"schema_version", "card_id", "operation", "gate", "purpose", "source_provenance",
             "limitations", "scope", "scope_sha256", "approval", "scientific_gate_pass",
             "real_nomination_completed", "training_eligibility"}
APPROVAL_KEYS = {"status", "approved_by", "approved_at", "authorized_operations", "authorized_gates",
                "scope", "confirmation_reference", "scope_sha256"}


def scope_sha256(scope):
    return hashlib.sha256(json.dumps(scope, sort_keys=True, ensure_ascii=True,
                                    separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _json_types(value, depth=0):
    if depth > 30:
        return False
    if value is None or type(value) in (str, int, bool):
        return True
    if type(value) is list:
        return all(_json_types(v, depth + 1) for v in value)
    if type(value) is dict:
        return all(type(k) is str and _json_types(v, depth + 1) for k, v in value.items())
    return False  # No floats, NaN, tuples or bool-as-int conversion in this exact metadata card.


def _digest(value):
    return type(value) is str and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _relative(value):
    return (type(value) is str and bool(value) and not value.startswith("/")
            and "\\" not in value and all(p not in ("", ".", "..") for p in value.split("/"))
            and str(PurePosixPath(value)) == value)


def _manifest_errors(source, actual):
    """Structural checks remain explicit even though the full digest is fixed."""
    errors = []
    if type(source) is not dict:
        return ["source_scope object required"]
    parents, rows = source.get("eligible_parents"), source.get("task_row_summary")
    if (type(parents) is not list or len(parents) != 55
            or any(type(p) is not str or re.fullmatch(r"S(?:0[1-9]|10)_[a-z0-9_]+_C0[1-7]", p) is None for p in parents)
            or parents != sorted(set(parents))):
        return ["exact unique sorted55 C01-C07 eligible parents required"]
    if type(rows) is not list or len(rows) != 165:
        return ["exact165 task rows required"]
    tasks, row_total, frame_total, chunks, decoded = [], 0, 0, 0, 0
    for row in rows:
        if type(row) is not dict or set(row) != ROW_KEYS:
            errors.append("task row keys differ from closed schema")
            continue
        task = row["task"]
        if type(task) is not str or "__" not in task:
            errors.append("typed task identity required")
            continue
        parent, variant = task.rsplit("__", 1)
        if parent not in parents or variant not in VARIANTS:
            errors.append("task outside exact parent/variant inventory")
        expected_partition = "c07" if parent.endswith("_C07") else "fit"
        if row["source_partition"] != expected_partition or row["split"] not in (
                ("calibration", "development") if expected_partition == "c07" else ("fit",)):
            errors.append("task partition/split mismatch")
        for key in ("row_count", "unique_frame_count", "source_sequence_count", "projected_chunks_per_field", "projected_decoded_rows"):
            if type(row[key]) is not int or row[key] <= 0:
                errors.append("positive non-bool integer task counts required")
        if errors:
            continue
        if not (row["row_count"] <= row["projected_decoded_rows"] <= row["source_sequence_count"]):
            errors.append("effective/decoded/source row count ordering invalid")
        if not all(_digest(row[k]) for k in ("unique_rows_sha256", "row_identity_sha256")):
            errors.append("row-set and identity digests required")
        tasks.append(task)
        row_total += row["row_count"]; frame_total += row["unique_frame_count"]
        chunks += row["projected_chunks_per_field"]; decoded += row["projected_decoded_rows"]
    expected_tasks = sorted(p + "__" + v for p in parents for v in VARIANTS)
    if tasks != expected_tasks or (row_total, frame_total, chunks, decoded) != (62718, 73734, 960, 239838):
        errors.append("task population/order or exact inherited counts drift")
    if source.get("task_row_summary_sha256") != scope_sha256(rows):
        errors.append("task row summary hash mismatch")
    if type(actual) is not dict:
        return errors + ["explicit actual_read_scope object required"]
    codebooks = actual.get("codebook_paths")
    if type(codebooks) is not dict or set(codebooks) != set(expected_tasks):
        errors.append("exact165 explicit same-task codebook paths required")
    else:
        prefix = ("results/gate3_semantics/"
                  "gate3_20260830_primitive_relation_p1a_lossless_storage_corrective_v1_seed0/artifacts/codebooks/")
        for task, path in codebooks.items():
            partition = "c07" if task.split("__")[0].endswith("_C07") else "fit"
            if not _relative(path) or path != prefix + partition + "/" + task + ".json":
                errors.append("codebook path scope drift")
    if type(actual.get("codebook_files_required")) is not int or actual["codebook_files_required"] != 165:
        errors.append("new card must disclose165 codebooks, superseding draft zero-file suggestion")
    return errors


def validate_review_nomination_card(card):
    from mtare_topo.governance import ValidationReport
    errors = []
    if type(card) is not dict or not _json_types(card):
        return ValidationReport(False, ("strict finite JSON metadata card required",))
    if set(card) != CARD_KEYS:
        errors.append("closed nomination audit card keys required")
    if (card.get("schema_version") != SCHEMA or card.get("card_id") != CARD_ID
            or card.get("operation") != "audit" or type(card.get("gate")) is not int or card["gate"] != 3):
        errors.append("independent Gate3 review nomination audit only")
    for key in ("purpose", "source_provenance", "limitations"):
        if type(card.get(key)) is not str or not card[key].strip():
            errors.append(f"{key} must document scope and limitations")
    scope = card.get("scope")
    if type(scope) is not dict:
        errors.append("scope object required")
    else:
        errors.extend(_manifest_errors(scope.get("source_scope"), scope.get("actual_read_scope")))
        if scope_sha256(scope) != EXPECTED_SCOPE_SHA256:
            errors.append("scope differs from independently frozen expected content")
    if card.get("scope_sha256") != EXPECTED_SCOPE_SHA256:
        errors.append("card scope digest drift")
    approval = card.get("approval")
    if type(approval) is not dict or set(approval) != APPROVAL_KEYS:
        errors.append("closed standing approval object required")
        approval = {}
    if (approval.get("status") != "APPROVED" or approval.get("approved_by") != "user-standing-scope-authorization"
            or approval.get("authorized_operations") != ["audit"]
            or type(approval.get("authorized_gates")) is not list
            or len(approval["authorized_gates"]) != 1
            or type(approval["authorized_gates"][0]) is not int or approval["authorized_gates"] != [3]):
        errors.append("standing authorization must be audit-only Gate3, without invented new reply")
    for key in ("approved_at", "scope", "confirmation_reference"):
        if type(approval.get(key)) is not str or not approval[key].strip():
            errors.append(f"approval.{key} required")
    if approval.get("scope_sha256") != EXPECTED_SCOPE_SHA256:
        errors.append("approval must bind the independently fixed exact scope")
    if any(card.get(k) is not False for k in ("scientific_gate_pass", "real_nomination_completed", "training_eligibility")):
        errors.append("preparation grants no science PASS, completed selection or training eligibility")
    return ValidationReport(not errors, tuple(errors))
