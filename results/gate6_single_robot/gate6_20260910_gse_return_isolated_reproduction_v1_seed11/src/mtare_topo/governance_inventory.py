"""Narrow approval for reading an exact existing sample inventory, not training.

Unlike a materialized training Data Card, this audit answers unknown counts
and records an untimed distance-sampled sensor source honestly. It cannot
authorize sensor decoding, export, teacher generation, fitting or calibration.
"""
import re


SCHEMA = "v3_scoped_inventory_card_v1"
TASK = re.compile(r"S(?:0[1-9]|10)_[a-z0-9_]+_C0[1-6]__c1_mixed$")
SHA = re.compile(r"[a-f0-9]{64}$")


def validate_scoped_inventory_card(card):
    return _validate_inventory(card, sensor_coordinates=False)


def validate_scoped_coordinate_audit_card(card):
    """Separate audit schema: exact existing scans allowed, no model allowed."""
    from mtare_topo.governance import ValidationReport
    from mtare_topo.governance_field_recovery import selection_sha256, INPUT_FIELDS
    if not isinstance(card, dict):
        return ValidationReport(False, ("coordinate audit card must be an object",))
    report = _validate_inventory(card, sensor_coordinates=True)
    if not report.passed:
        return report
    errors = list(report.errors)
    if card.get("input_fields") != list(INPUT_FIELDS):
        errors.append("coordinate audit only permits range/valid and causal relative motion")
    rows = card.get("selected_rows", [])
    if (not isinstance(rows, list) or len(rows) != 180
            or any(not isinstance(r, dict) or not isinstance(r.get("task"), str)
                   or "_C01__c1_mixed" not in r["task"] for r in rows)):
        errors.append("coordinate audit requires exact180 C01 rows")
    else:
        tasks = {r["task"] for r in rows}
        if (len(tasks) != 10 or {t[:3] for t in tasks} != {f"S{i:02d}" for i in range(1, 11)}
                or any(sum(r["task"] == t for r in rows) != 18 for t in tasks)):
            errors.append("coordinate audit requires ten parents/eighteen rows each")
        if len({(r["task"], r.get("source_global_sequence_index")) for r in rows}) != len(rows):
            errors.append("coordinate audit duplicate source sequence")
        approval = card.get("approval")
        try:
            digest = selection_sha256(rows)
        except (ValueError, TypeError):
            digest = None
        if (digest is None or card.get("selection_sha256") != digest
                or not isinstance(approval, dict) or approval.get("selection_sha256") != digest):
            errors.append("coordinate audit selection must be hash-bound to approval")
    for key, number in {"unique_source_frame_count": 900, "frames_per_observation": 5,
            "parent_count": 10, "node_count": 100, "observation_count": 180}.items():
        if type(card.get(key)) is not int or card[key] != number:
            errors.append("coordinate audit inventory counts drift")
    if card.get("sensor_usage") != "existing_coordinate_support_only_no_model_or_new_labels":
        errors.append("coordinate audit usage must not permit inference or labels")
    return ValidationReport(not errors, tuple(errors))


def _validate_inventory(card, *, sensor_coordinates):
    from mtare_topo.governance import ValidationReport
    if not isinstance(card, dict):
        return ValidationReport(False, ("inventory card must be an object",))
    errors = []
    if card.get("schema_version") != ("v3_scoped_coordinate_audit_card_v1" if sensor_coordinates else SCHEMA):
        errors.append("invalid scoped inventory schema")
    for key in ("card_id", "purpose", "teacher_source", "sampling_rule", "time_basis", "license_or_allowed_use"):
        if not isinstance(card.get(key), str) or not card[key].strip():
            errors.append(f"inventory {key} is required")
    approval = card.get("approval", {})
    if not isinstance(approval, dict):
        approval = {}
    if (approval.get("status") != "APPROVED" or approval.get("authorized_operations") != ["audit"]
            or approval.get("authorized_gates") != [3]):
        errors.append("inventory authorization must be audit-only in Gate 3")
    for key in ("approved_by", "approved_at", "scope", "confirmation_reference"):
        if not isinstance(approval.get(key), str) or not approval[key].strip():
            errors.append(f"inventory approval.{key} is required")
    restrictions = card.get("restrictions", {})
    sensor_restriction = "existing_selected_sensor_only" if sensor_coordinates else "no_sensor_decoding"
    for key in ("read_only_sources", sensor_restriction, "no_model_or_checkpoint", "no_optimizer",
                "no_teacher_generation", "no_calibration", "no_test_worlds", "explicit_task_rows_only"):
        if not isinstance(restrictions, dict) or restrictions.get(key) is not True:
            errors.append(f"inventory restriction missing: {key}")
    rows = card.get("selected_rows", [])
    if not isinstance(rows, list) or not 1 <= len(rows) <= 180:
        errors.append("inventory must bind 1..180 exact existing observations")
        rows = []
    seen, worlds = set(), set()
    for row in rows:
        if not isinstance(row, dict):
            errors.append("inventory row must be an object")
            continue
        task, index, source = row.get("task"), row.get("row_index"), row.get("source_global_sequence_index")
        if (not isinstance(task, str) or not TASK.fullmatch(task)
                or type(index) is not int or index < 0 or type(source) is not int or source < 0):
            errors.append("inventory row outside exact fit allowlist")
            continue
        worlds.add(task.split("__")[0])
        if (task, index) in seen:
            errors.append("duplicate inventory row")
        seen.add((task, index))
    if card.get("worlds") != sorted(worlds):
        errors.append("inventory worlds must equal exact selected parents")
    if card.get("observation_count") != len(rows) or card.get("independent_sampling_unit") != "topology_parent":
        errors.append("inventory count/unit mismatch")
    if card.get("duration_s") is not None or card.get("time_basis") != "distance_sampled_no_acquisition_clock":
        errors.append("scoped static inventory must not invent acquisition duration")
    sources = card.get("sealed_sources", {})
    if not isinstance(sources, dict) or not sources:
        errors.append("inventory sealed source hashes required")
    else:
        for path, digest in sources.items():
            if (not isinstance(path, str) or path.startswith("/") or ".." in path.split("/")
                    or not isinstance(digest, str) or not SHA.fullmatch(digest)):
                errors.append("invalid inventory source path/hash")
    return ValidationReport(not errors, tuple(errors))
