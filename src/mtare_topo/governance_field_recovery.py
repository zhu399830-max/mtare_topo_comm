"""Exact-population frozen inference authorization, never a training card.

This validator performs no file I/O. The executor must independently verify
source seals, the declared inventory counts, and the actual inference ledger.
The 100 inventory nodes are provenance counts, not model labels or inputs.
"""
import hashlib
import json
import re


SCHEMA = "v3_scoped_field_recovery_card_v1"
TASK = re.compile(r"S(?:0[1-9]|10)_[a-z0-9_]+_C01__c1_mixed$")
SHA = re.compile(r"[a-f0-9]{64}$")
INPUT_FIELDS = ("range_m", "valid_mask", "relative_translation_current_sensor_m",
                "relative_yaw_current_sensor_deg")
OUTPUT_FIELDS = ("axis_control_current_sensor_m", "endpoint_half_axes_m",
                 "endpoint_shape_exponent", "existence_logits",
                 "geometry_uncertainty", "endpoint_evidence_logits")
RESTRICTIONS = ("read_only_sources", "explicit_task_rows_only", "no_optimizer",
                "no_teacher_generation", "no_new_labels", "no_calibration",
                "no_test_worlds", "no_graph", "no_checkpoint_selection",
                "no_training", "no_identity_model_inputs", "causal_five_frames_only")


def selection_sha256(rows):
    """Hash the full ordered selection, including any provenance-only fields."""
    return hashlib.sha256(json.dumps(rows, sort_keys=True, separators=(",", ":"),
                                    ensure_ascii=True, allow_nan=False).encode()).hexdigest()


def _relative_source(path):
    return (isinstance(path, str) and bool(path) and not path.startswith("/")
            and "\\" not in path and ":" not in path and "\x00" not in path
            and all(part not in ("", ".", "..") for part in path.split("/")))


def _digest(value):
    return isinstance(value, str) and SHA.fullmatch(value) is not None


def validate_scoped_field_recovery_card(card):
    from mtare_topo.governance import ValidationReport
    errors = []
    if not isinstance(card, dict):
        return ValidationReport(False, ("field recovery card must be an object",))
    if card.get("schema_version") != SCHEMA:
        errors.append("invalid scoped field recovery schema")
    for key in ("card_id", "purpose", "teacher_source", "sampling_rule", "license_or_allowed_use"):
        if not isinstance(card.get(key), str) or not card[key].strip():
            errors.append(f"field recovery {key} is required")
    if ("duration_s" not in card or card["duration_s"] is not None
            or card.get("time_basis") != "distance_sampled_no_acquisition_clock"):
        errors.append("distance-sampled field recovery requires explicit null duration and no invented clock")
    if card.get("independent_sampling_unit") != "topology_parent":
        errors.append("independent sampling unit must be topology_parent")
    if card.get("partition") != "fit":
        errors.append("field recovery is fit-only")
    counts = {"observation_count": 180, "parent_count": 10, "node_count": 100,
              "unique_source_frame_count": 900, "frames_per_observation": 5}
    for key, expected in counts.items():
        if type(card.get(key)) is not int or card[key] != expected:
            errors.append(f"field recovery {key} must be exactly {expected}")

    rows = card.get("selected_rows")
    if not isinstance(rows, list) or len(rows) != 180:
        errors.append("field recovery must bind exactly 180 selected rows")
        rows = []
    seen, sources, task_counts = set(), set(), {}
    for row in rows:
        if not isinstance(row, dict):
            errors.append("selected row must be an object")
            continue
        task, index, source = row.get("task"), row.get("row_index"), row.get("source_global_sequence_index")
        if (not isinstance(task, str) or not TASK.fullmatch(task)
                or type(index) is not int or index < 0
                or type(source) is not int or source < 0):
            errors.append("selected row outside exact C01 mixed fit allowlist")
            continue
        if (task, index) in seen or (task, source) in sources:
            errors.append("duplicate selected row or source sequence")
        seen.add((task, index)); sources.add((task, source))
        task_counts[task] = task_counts.get(task, 0) + 1
    tasks = sorted(task_counts)
    if (len(tasks) != 10 or any(count != 18 for count in task_counts.values())
            or {task[:3] for task in tasks} != {f"S{i:02d}" for i in range(1, 11)}):
        errors.append("exactly ten S01--S10 C01 parents with eighteen rows each are required")
    if card.get("worlds") != sorted(task.split("__")[0] for task in tasks):
        errors.append("worlds must equal exactly the selected topology parents")
    try:
        selection_digest = selection_sha256(rows)
    except (TypeError, ValueError):
        selection_digest = None
        errors.append("selected rows must be finite JSON data")
    if not _digest(card.get("selection_sha256")) or card.get("selection_sha256") != selection_digest:
        errors.append("selected row hash mismatch")

    if card.get("input_fields") != list(INPUT_FIELDS):
        errors.append("only range/valid and causal relative motion may enter the model")
    if card.get("output_fields") != list(OUTPUT_FIELDS):
        errors.append("field recovery output fields must match the six frozen geometry fields")
    restrictions = card.get("restrictions", {})
    for key in RESTRICTIONS:
        if not isinstance(restrictions, dict) or restrictions.get(key) is not True:
            errors.append(f"field recovery restriction missing: {key}")

    sealed_sources = card.get("sealed_sources", {})
    if not isinstance(sealed_sources, dict) or not sealed_sources:
        errors.append("field recovery requires exact sealed source hashes")
        sealed_sources = {}
    for path, digest in sealed_sources.items():
        if not _relative_source(path) or not _digest(digest):
            errors.append("invalid field recovery source path/hash")
    checkpoint = card.get("checkpoint", {})
    if not isinstance(checkpoint, dict):
        checkpoint = {}
    checkpoint_path, checkpoint_digest = checkpoint.get("path"), checkpoint.get("sha256")
    if (not _relative_source(checkpoint_path) or not _digest(checkpoint_digest)
            or (isinstance(checkpoint_path, str) and sealed_sources.get(checkpoint_path) != checkpoint_digest)):
        errors.append("checkpoint must be an exact sealed relative source")
    if (type(checkpoint.get("seed")) is not int or checkpoint.get("seed") != 0
            or checkpoint.get("frozen") is not True or checkpoint.get("selection") != "none"):
        errors.append("only the specified frozen seed0 checkpoint is allowed; no selection")
    inference = card.get("inference", {})
    if not isinstance(inference, dict):
        inference = {}
    for key, expected in {"primary_observations": 180, "repeat_observations": 18,
                          "total_observations": 198}.items():
        if type(inference.get(key)) is not int or inference[key] != expected:
            errors.append(f"inference {key} must be exactly {expected}")
    if not tasks or inference.get("repeat_task") != tasks[0]:
        errors.append("determinism repeat must use only all eighteen rows of the first sorted task")

    approval = card.get("approval", {})
    if not isinstance(approval, dict):
        approval = {}
    if (approval.get("status") != "APPROVED" or approval.get("authorized_operations") != ["data_export"]
            or approval.get("authorized_gates") != [3]
            or any(type(gate) is not int for gate in approval.get("authorized_gates", []))):
        errors.append("field recovery authorization must be export-only in Gate 3")
    for key in ("approved_by", "approved_at", "scope", "confirmation_reference"):
        if not isinstance(approval.get(key), str) or not approval[key].strip():
            errors.append(f"field recovery approval.{key} is required")
    if approval.get("selection_sha256") != selection_digest or not selection_digest:
        errors.append("approval is not bound to the exact ordered selection")
    if not _digest(checkpoint_digest) or approval.get("checkpoint_sha256") != checkpoint_digest:
        errors.append("approval is not bound to the frozen checkpoint")
    return ValidationReport(not errors, tuple(errors))
