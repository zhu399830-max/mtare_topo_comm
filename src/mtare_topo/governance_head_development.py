"""Metadata-only C02 sampling authorization, not model evaluation permission.

Counts of actual source frames and visible fragments are deliberately unknown
before this audit. Source seals cover the immutable metadata; the executor
must enforce the exact task/key allowlist and independently verify each read.
This validator performs no file I/O and does not claim whole-model holdout.
"""
import re

from mtare_topo.governance_field_recovery import _digest, _relative_source


SCHEMA = "v3_head_development_metadata_card_v1"
TASK = re.compile(r"S(?:0[1-9]|10)_[a-z0-9_]+_C02__c1_mixed$")
FIELDS = ("frame_row", "source_global_sequence_index", "primitive_mask")
SAMPLING_RULE = "floor((2*k+1)*N/36),k=0..17,per_parent"
HEAD_EXPOSURE = "C02_not_used_for_current_head_C01_fit_backbone_exposed_C01_C06_not_strict_test"
RESTRICTIONS = ("read_only_sources", "explicit_tasks_only", "metadata_fields_only",
                "only_task_zgroup_zattrs_and_field_zarray_and_chunks", "no_model", "no_checkpoint",
                "no_optimizer", "no_teacher_generation", "no_new_labels", "no_raw_decode",
                "no_C07_C10", "no_score_based_selection", "no_calibration", "no_graph")
UNKNOWN_FIELDS = ("source_global_sequence_indices", "unique_source_frame_count", "visible_fragment_count")


def validate_head_development_metadata_card(card):
    from mtare_topo.governance import ValidationReport
    errors = []
    if not isinstance(card, dict):
        return ValidationReport(False, ("head development metadata card must be an object",))
    if card.get("schema_version") != SCHEMA:
        errors.append("invalid head development metadata schema")
    for key in ("card_id", "purpose", "teacher_source", "license_or_allowed_use"):
        if not isinstance(card.get(key), str) or not card[key].strip():
            errors.append(f"head development {key} is required")
    if card.get("head_exposure") != HEAD_EXPOSURE:
        errors.append("C02 must disclose head-only holdout and C01--C06 backbone exposure; not strict test")
    if card.get("partition") != "fit":
        errors.append("metadata sources must remain in the existing fit partition")
    if card.get("independent_sampling_unit") != "topology_parent":
        errors.append("independent sampling unit must be topology_parent")
    if ("duration_s" not in card or card["duration_s"] is not None
            or card.get("time_basis") != "distance_sampled_no_acquisition_clock"):
        errors.append("distance-sampled source requires explicit null duration, not an invented clock")
    for key, expected in {"target_observations": 180, "parent_count": 10, "rows_per_parent": 18}.items():
        if type(card.get(key)) is not int or card[key] != expected:
            errors.append(f"head development {key} must be exactly {expected}")
    for key in UNKNOWN_FIELDS:
        if key not in card or card[key] is not None:
            errors.append(f"{key} must be explicitly unknown/null until the metadata audit completes")
    if card.get("fields") != list(FIELDS):
        errors.append("only frame_row, source_global_sequence_index and primitive_mask metadata are authorized")
    if card.get("sampling_rule") != SAMPLING_RULE:
        errors.append("metadata selection must use the fixed eighteen midpoint quantiles per parent")
    tasks = card.get("tasks")
    if (not isinstance(tasks, list) or len(tasks) != 10
            or any(not isinstance(task, str) or not TASK.fullmatch(task) for task in tasks)):
        errors.append("exactly ten C02 c1_mixed tasks are required")
        tasks = []
    if (tasks != sorted(set(tasks))
            or {task[:3] for task in tasks} != {f"S{i:02d}" for i in range(1, 11)}):
        errors.append("tasks must be unique, sorted, and cover exactly S01--S10")
    if card.get("worlds") != sorted(task.split("__")[0] for task in tasks):
        errors.append("worlds must equal exactly the ten selected C02 parents")
    roots = card.get("source_roots")
    if (not isinstance(roots, dict) or set(roots) != {"teacher"}
            or not _relative_source(roots.get("teacher"))
            or roots["teacher"].split("/")[-1] != "fit"):
        errors.append("source_roots must bind exactly one repository-relative teacher fit root")
    elif re.search(r"(?:^|[/_])C(?:0[7-9]|10)(?:[/_]|$)", roots["teacher"]):
        errors.append("C07--C10 roots are forbidden even if named fit")
    sources = card.get("sealed_sources")
    if not isinstance(sources, dict) or not sources:
        errors.append("sealed source path/hash map is required")
        sources = {}
    for path, digest in sources.items():
        if not _relative_source(path) or not _digest(digest):
            errors.append("invalid sealed source relative path or SHA256")
    seals = card.get("source_seals")
    if not isinstance(seals, dict) or set(seals) != {"teacher"}:
        errors.append("source_seals must bind exactly the teacher source seal")
    else:
        seal = seals["teacher"]
        if not _relative_source(seal) or not _digest(sources.get(seal) if isinstance(seal, str) else None):
            errors.append("teacher source seal must be a relative path present in sealed_sources with SHA256")
    restrictions = card.get("restrictions")
    for key in RESTRICTIONS:
        if not isinstance(restrictions, dict) or restrictions.get(key) is not True:
            errors.append(f"head development restriction missing: {key}")
    for forbidden in ("checkpoint", "inference", "training", "selected_rows", "selection_sha256",
                      "observation_count", "model_inputs", "geomteacher_fields"):
        if forbidden in card:
            errors.append(f"metadata preparation must not carry model/count/selection authority: {forbidden}")
    approval = card.get("approval")
    if not isinstance(approval, dict):
        approval = {}
    if (approval.get("status") != "APPROVED" or approval.get("authorized_operations") != ["audit"]
            or approval.get("authorized_gates") != [3]
            or any(type(gate) is not int for gate in approval.get("authorized_gates", []))):
        errors.append("head development metadata authorization must be audit-only in Gate 3")
    for key in ("approved_by", "approved_at", "scope", "confirmation_reference"):
        if not isinstance(approval.get(key), str) or not approval[key].strip():
            errors.append(f"head development approval.{key} is required")
    return ValidationReport(not errors, tuple(errors))
