"""Frozen C02 head development inference, never strict unseen evaluation.

This validator has no I/O. Executors must compare the exact selected records
to the hashed metadata selection, verify every source read and checkpoint,
and check their realized inference ledger. The original failed training run
and offset-branch research decision are not changed by this new export.
"""
import re

from mtare_topo.governance_field_recovery import INPUT_FIELDS, _digest, _relative_source, selection_sha256
from mtare_topo.governance_head_development import HEAD_EXPOSURE, TASK


SCHEMA = "v3_head_development_inference_card_v1"
GEOMTEACHER_FIELDS = ("frame_row", "source_global_sequence_index", "primitive_mask",
                      "axis_control_current_sensor_m")
OUTPUT_FIELDS = ("raw_no_offset_axis_control_m", "legacy_frozen_axis_control_m")
RESTRICTIONS = ("read_only_sources", "explicit_task_rows_only", "no_optimizer",
                "no_teacher_generation", "no_new_labels", "no_calibration", "no_test_worlds",
                "no_C07_C10", "no_graph", "no_checkpoint_selection", "no_training",
                "no_identity_model_inputs", "causal_five_frames_only", "no_offset_checkpoint",
                "preserve_original_training_failure_and_offset_stop")
INFERENCE_COUNTS = {"raw_primary_observations": 180, "raw_repeat_observations": 18,
                    "legacy_primary_observations": 180, "legacy_repeat_observations": 18,
                    "cached_backbone_windows": 180, "optimizer_steps": 0}


def validate_head_development_inference_card(card):
    from mtare_topo.governance import ValidationReport
    errors = []
    if not isinstance(card, dict):
        return ValidationReport(False, ("head inference card must be an object",))
    if card.get("schema_version") != SCHEMA:
        errors.append("invalid head development inference schema")
    for key in ("card_id", "purpose", "teacher_source", "sampling_rule", "license_or_allowed_use"):
        if not isinstance(card.get(key), str) or not card[key].strip():
            errors.append(f"head inference {key} is required")
    if card.get("head_exposure") != HEAD_EXPOSURE:
        errors.append("head-only C02 holdout must disclose backbone C01--C06 exposure, not strict test")
    if card.get("partition") != "fit" or card.get("independent_sampling_unit") != "topology_parent":
        errors.append("head inference must retain fit partition and topology_parent sampling unit")
    if ("duration_s" not in card or card["duration_s"] is not None
            or card.get("time_basis") != "distance_sampled_no_acquisition_clock"):
        errors.append("distance-sampled source needs explicit null duration and no invented clock")
    for key, expected in {"observation_count": 180, "parent_count": 10, "rows_per_parent": 18,
                          "unique_source_frame_count": 900, "visible_fragment_count": 1489,
                          "frames_per_observation": 5}.items():
        if type(card.get(key)) is not int or card[key] != expected:
            errors.append(f"head inference {key} must be exactly {expected}")
    rows = card.get("selected_rows")
    if not isinstance(rows, list) or len(rows) != 180:
        errors.append("head inference requires exactly 180 selected rows")
        rows = []
    seen, source_seen, unique_frames, task_counts = set(), set(), set(), {}
    fragments = 0
    for row in rows:
        if not isinstance(row, dict):
            errors.append("selected row must be an object")
            continue
        task, index, source = row.get("task"), row.get("row_index"), row.get("source_global_sequence_index")
        if (not isinstance(task, str) or not TASK.fullmatch(task)
                or type(index) is not int or index < 0
                or type(source) is not int or source < 0):
            errors.append("selected row outside C02 exact mixed task/index allowlist")
            continue
        if (task, index) in seen or (task, source) in source_seen:
            errors.append("duplicate selected row or source identity")
        seen.add((task, index)); source_seen.add((task, source))
        task_counts[task] = task_counts.get(task, 0) + 1
        frames = row.get("frame_rows")
        if (not isinstance(frames, list) or len(frames) != 5
                or any(type(frame) is not int or frame < 0 for frame in frames)
                or frames != list(range(frames[0], frames[0] + 5))):
            errors.append("each selected row must contain exactly five consecutive causal frame rows")
        else:
            unique_frames.update((task, frame) for frame in frames)
        count = row.get("visible_fragments")
        if type(count) is not int or not 1 <= count <= 32:
            errors.append("visible_fragments must be an integer in 1..32 per row")
        else:
            fragments += count
    tasks = sorted(task_counts)
    if (len(tasks) != 10 or any(count != 18 for count in task_counts.values())
            or {task[:3] for task in tasks} != {f"S{i:02d}" for i in range(1, 11)}):
        errors.append("exactly S01--S10 C02 parents with eighteen rows each are required")
    if card.get("tasks") != tasks or card.get("worlds") != [task.split("__")[0] for task in tasks]:
        errors.append("tasks/worlds must exactly match the sorted selected C02 parents")
    if len(unique_frames) != 900 or fragments != 1489:
        errors.append("selection must realize exactly 900 unique task frames and 1489 visible fragments")
    try:
        selection_digest = selection_sha256(rows)
    except (TypeError, ValueError):
        selection_digest = None
        errors.append("selection must be finite JSON records")
    if not _digest(card.get("selection_sha256")) or card.get("selection_sha256") != selection_digest:
        errors.append("selection hash mismatch")
    if card.get("input_fields") != list(INPUT_FIELDS):
        errors.append("model inputs limited to range/valid and causal relative translation/yaw")
    if card.get("geomteacher_fields") != list(GEOMTEACHER_FIELDS):
        errors.append("existing teacher geometry restricted to the four alignment/mask/axis fields")
    if card.get("output_fields") != list(OUTPUT_FIELDS):
        errors.append("only raw_no_offset and legacy_frozen axis outputs are authorized")
    restrictions = card.get("restrictions")
    for key in RESTRICTIONS:
        if not isinstance(restrictions, dict) or restrictions.get(key) is not True:
            errors.append(f"head inference restriction missing: {key}")
    roots = card.get("source_roots")
    if not isinstance(roots, dict) or set(roots) != {"sensor", "teacher"}:
        errors.append("source_roots must contain only sensor and teacher")
    else:
        for root in roots.values():
            if (not _relative_source(root) or root.split("/")[-1] != "fit"
                    or re.search(r"(?:^|[/_])C(?:0[7-9]|10)(?:[/_]|$)", root)):
                errors.append("source roots must be repository-relative fit roots, excluding C07--C10")
    sources = card.get("sealed_sources")
    if not isinstance(sources, dict) or not sources:
        errors.append("sealed_sources path/hash map is required")
        sources = {}
    for path, digest in sources.items():
        if not _relative_source(path) or not _digest(digest):
            errors.append("invalid source relative path or SHA256")
    seals = card.get("source_seals")
    if not isinstance(seals, dict) or set(seals) != {"sensor", "teacher"}:
        errors.append("source_seals must contain exactly sensor and teacher")
    else:
        for seal in seals.values():
            if not _relative_source(seal) or not _digest(sources.get(seal) if isinstance(seal, str) else None):
                errors.append("every source seal must have a relative path and bound SHA256")
    selection_path = card.get("metadata_selection_path")
    if (not _relative_source(selection_path)
            or not _digest(sources.get(selection_path) if isinstance(selection_path, str) else None)):
        errors.append("exact completed metadata selection file must be bound in sealed_sources")
    checkpoints = card.get("checkpoints")
    if not isinstance(checkpoints, dict) or set(checkpoints) != {"backbone", "head"}:
        errors.append("only the exact backbone and raw head checkpoints are allowed")
        checkpoints = {}
    checkpoint_hashes, checkpoint_paths = {}, []
    for role, checkpoint in checkpoints.items():
        if not isinstance(checkpoint, dict):
            errors.append(f"checkpoint {role} must be an object")
            continue
        path, digest = checkpoint.get("path"), checkpoint.get("sha256")
        if (not _relative_source(path) or not _digest(digest)
                or (isinstance(path, str) and sources.get(path) != digest)):
            errors.append(f"checkpoint {role} path/hash must match sealed_sources")
        if (type(checkpoint.get("seed")) is not int or checkpoint.get("seed") != 0
                or checkpoint.get("frozen") is not True or checkpoint.get("selection") != "none"):
            errors.append(f"checkpoint {role} must be frozen seed0 without selection")
        if isinstance(path, str):
            checkpoint_paths.append(path)
            if "raw_slot_offset" in path:
                errors.append("the stopped raw_slot_offset checkpoint is forbidden")
        checkpoint_hashes[role] = digest
    if len(checkpoint_paths) == 2 and checkpoint_paths[0] == checkpoint_paths[1]:
        errors.append("backbone and head must be distinct checkpoint files")
    inference = card.get("inference")
    if not isinstance(inference, dict):
        inference = {}
    if set(inference) != set(INFERENCE_COUNTS) | {"repeat_task"}:
        errors.append("inference must contain exactly the frozen two-output/cached-backbone ledger")
    for key, expected in INFERENCE_COUNTS.items():
        if type(inference.get(key)) is not int or inference[key] != expected:
            errors.append(f"inference {key} must be exactly {expected}")
    if not tasks or inference.get("repeat_task") != tasks[0]:
        errors.append("only the first sorted task's eighteen rows may be repeated")
    approval = card.get("approval")
    if not isinstance(approval, dict):
        approval = {}
    if (approval.get("status") != "APPROVED" or approval.get("authorized_operations") != ["data_export"]
            or approval.get("authorized_gates") != [3]
            or any(type(gate) is not int for gate in approval.get("authorized_gates", []))):
        errors.append("head inference requires data_export-only approval in Gate 3")
    for key in ("approved_by", "approved_at", "scope", "confirmation_reference"):
        if not isinstance(approval.get(key), str) or not approval[key].strip():
            errors.append(f"head inference approval.{key} is required")
    if not selection_digest or approval.get("selection_sha256") != selection_digest:
        errors.append("approval must bind the exact ordered selection hash")
    if (set(checkpoint_hashes) != {"backbone", "head"}
            or approval.get("checkpoint_sha256") != checkpoint_hashes):
        errors.append("approval must bind both specified backbone and head checkpoint hashes")
    return ValidationReport(not errors, tuple(errors))
