"""Narrow two-variant geometry-only training authorization.

No I/O is performed. The executor must check source seals, realized counts,
frozen-backbone invariance and the actual optimizer ledger. Inventory node
counts do not authorize identity input or new event labels. This card cannot
be obtained simply by relabelling the old frozen field-export approval.
"""
from mtare_topo.governance_field_recovery import (
    INPUT_FIELDS, TASK, _digest, _relative_source, selection_sha256,
)


SCHEMA = "v3_scoped_point_axis_training_card_v1"
GEOMTEACHER_FIELDS = ("frame_row", "source_global_sequence_index", "primitive_mask",
                      "axis_control_current_sensor_m")
RESTRICTIONS = ("read_only_sources", "explicit_task_rows_only", "no_teacher_generation",
                "no_new_labels", "no_calibration", "no_test_worlds", "no_graph",
                "no_identity_model_inputs", "causal_five_frames_only", "no_event_training",
                "frozen_backbone", "no_checkpoint_selection")
TRAINING = {
    "variants": ["raw_no_offset", "raw_slot_offset"],
    "epochs": 3, "batch_size": 1, "steps_per_variant": 540, "total_steps": 1080,
    "seed": 0, "optimizer": "Adam", "learning_rate": 0.001, "weight_decay": 0.0,
    "sample_schedule": "three_seed0_permutations_of_exact180",
    "checkpoint_selection": "final_only",
    "loss": "geometry_only_reversal_invariant_hungarian_coordinate_l1_div50",
}


def validate_point_axis_training_card(card):
    from mtare_topo.governance import ValidationReport
    errors = []
    if not isinstance(card, dict):
        return ValidationReport(False, ("point axis training card must be an object",))
    if card.get("schema_version") != SCHEMA:
        errors.append("invalid scoped point axis training schema")
    for key in ("card_id", "purpose", "teacher_source", "sampling_rule", "license_or_allowed_use"):
        if not isinstance(card.get(key), str) or not card[key].strip():
            errors.append(f"point axis {key} is required")
    if ("duration_s" not in card or card["duration_s"] is not None
            or card.get("time_basis") != "distance_sampled_no_acquisition_clock"):
        errors.append("distance-sampled source requires explicit null duration and no invented clock")
    if card.get("independent_sampling_unit") != "topology_parent":
        errors.append("independent sampling unit must be topology_parent")
    if card.get("partition") != "fit":
        errors.append("point axis training is fit-only")
    for key, expected in {"observation_count": 180, "parent_count": 10, "node_count": 100,
                          "unique_source_frame_count": 900, "frames_per_observation": 5}.items():
        if type(card.get(key)) is not int or card[key] != expected:
            errors.append(f"point axis {key} must be exactly {expected}")
    rows = card.get("selected_rows")
    if not isinstance(rows, list) or len(rows) != 180:
        errors.append("point axis training must bind exactly 180 selected rows")
        rows = []
    seen, source_seen, task_counts = set(), set(), {}
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
        if (task, index) in seen or (task, source) in source_seen:
            errors.append("duplicate selected row or source sequence")
        seen.add((task, index)); source_seen.add((task, source))
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
    if card.get("geomteacher_fields") != list(GEOMTEACHER_FIELDS):
        errors.append("geometry teacher must be exactly the existing four alignment/axis fields")
    for obsolete in ("inference", "output_fields", "identity_reference"):
        if obsolete in card:
            errors.append(f"point axis training must not carry obsolete export authorization field: {obsolete}")
    restrictions = card.get("restrictions", {})
    for key in RESTRICTIONS:
        if not isinstance(restrictions, dict) or restrictions.get(key) is not True:
            errors.append(f"point axis training restriction missing: {key}")
    if isinstance(restrictions, dict) and any(key in restrictions for key in ("no_training", "no_optimizer")):
        errors.append("remove obsolete export no_training/no_optimizer fields; contradictory cards cannot authorize training")
    sealed_sources = card.get("sealed_sources", {})
    if not isinstance(sealed_sources, dict) or not sealed_sources:
        errors.append("point axis training requires exact sealed source hashes")
        sealed_sources = {}
    for path, digest in sealed_sources.items():
        if not _relative_source(path) or not _digest(digest):
            errors.append("invalid point axis source path/hash")
    checkpoint = card.get("checkpoint", {})
    if not isinstance(checkpoint, dict):
        checkpoint = {}
    checkpoint_path, checkpoint_digest = checkpoint.get("path"), checkpoint.get("sha256")
    if (not _relative_source(checkpoint_path) or not _digest(checkpoint_digest)
            or (isinstance(checkpoint_path, str) and sealed_sources.get(checkpoint_path) != checkpoint_digest)):
        errors.append("checkpoint must be an exact sealed relative source")
    if (type(checkpoint.get("seed")) is not int or checkpoint.get("seed") != 0
            or checkpoint.get("frozen") is not True or checkpoint.get("selection") != "none"):
        errors.append("only the specified frozen seed0 backbone checkpoint is allowed")
    training = card.get("training")
    if not isinstance(training, dict):
        training = {}
    if set(training) != set(TRAINING):
        errors.append("training fields must exactly match the fixed two-variant contract")
    for key, expected in TRAINING.items():
        value = training.get(key)
        if type(value) is not type(expected) or value != expected:
            errors.append(f"point axis training.{key} must equal the fixed value {expected!r}")
    approval = card.get("approval", {})
    if not isinstance(approval, dict):
        approval = {}
    if (approval.get("status") != "APPROVED" or approval.get("authorized_operations") != ["training"]
            or approval.get("authorized_gates") != [3]
            or any(type(gate) is not int for gate in approval.get("authorized_gates", []))):
        errors.append("point axis authorization must be training-only in Gate 3")
    for key in ("approved_by", "approved_at", "scope", "confirmation_reference"):
        if not isinstance(approval.get(key), str) or not approval[key].strip():
            errors.append(f"point axis approval.{key} is required")
    if approval.get("selection_sha256") != selection_digest or not selection_digest:
        errors.append("training approval is not bound to the exact ordered selection")
    if not _digest(checkpoint_digest) or approval.get("checkpoint_sha256") != checkpoint_digest:
        errors.append("training approval is not bound to the frozen checkpoint")
    return ValidationReport(not errors, tuple(errors))
