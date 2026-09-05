"""Independent fixed-budget class-balance factorial, never old run authority."""
from copy import deepcopy
from pathlib import PurePosixPath
import re

from mtare_topo.governance_field_recovery import _digest, _relative_source
from mtare_topo.governance_geometry_bound import validate_geometry_bound_training_card
from mtare_topo.governance_geometry_bound_rank import CORRECTIVE_SEAL_SHA256, RANK_POLICY
from mtare_topo.governance_partial_structure_training import (
    IDENTITY_FIELDS, EXPORT_SEAL_SHA256, TRAINING as OLD_TRAINING,
    EVALUATION as OLD_EVALUATION, RESOURCES, SOURCE_NAMES as OLD_SOURCES, _exact,
)

SCHEMA = "gse_class_balance_factorial_v1"
RANK_SEAL_SHA256 = "dc3803e6265302f7c26d306fffcd4c36b48866706723847699043d1ecbc0f16a"
SOURCE_NAMES = {**OLD_SOURCES, "baseline_geometry_summary": "summary.json", "baseline_rank_summary": "summary.json"}
TRAINING = {**deepcopy(OLD_TRAINING), "total_steps": 2700, "factor_order": ["10", "01", "11"]}
EVALUATION = {**deepcopy(OLD_EVALUATION), "total_head_inference_windows": 3240}
BALANCE_POLICY = {
    "factor_order": ["10", "01", "11"], "baseline_factor": "00_reuse_sealed_results_no_training",
    "member_class_counts": [13489, 2152], "event_class_counts": [872, 14, 0],
    "weight_formula": "N/(K*n_class),K=nonzero_classes", "zero_count_weight": 0.,
    "population": "fixed_global_sealed_training_targets", "denominator": "original_fixed_batch_label_count",
    "loss_binding": "geometry_only_unique_center_binding_v1", "other_losses_and_scales": "unchanged",
    "event_output": "original_three_class_softmax_argmax", "factor_selection": "none_report_all",
    "baseline_weights_or_predictions_read": False,
}
RESTRICTIONS = (
    "read_only_sources", "only_four_training_payloads_and_two_comparison_summaries", "exact_C01_population",
    "no_history", "no_old_weights_or_predictions", "no_backbone", "no_new_scan", "no_new_teacher",
    "no_threshold_search", "no_calibration", "no_C02_C10", "no_graph", "no_old_source_or_result_modification",
    "no_scientific_gate_pass", "no_detection_or_generalization_claim", "no_complete_three_class_claim",
    "all_factors_reported", "same_initialization_order_budget", "all_32_prediction_slots", "gt_masks_loss_only",
)


def validate_class_balance_factorial_card(card):
    from mtare_topo.governance import ValidationReport
    errors = []
    if not isinstance(card, dict):
        return ValidationReport(False, ("factorial card must be object",))
    if card.get("schema_version") != SCHEMA or card.get("operation") != "training":
        errors.append("independent factorial training schema required")
    base = card.get("base_card")
    if not validate_geometry_bound_training_card(base).passed:
        errors.append("valid sealed corrective base_card required")
    if not isinstance(base, dict):
        base = {}
    for key in (*IDENTITY_FIELDS, "effective_counts", "export_reference"):
        if key not in card or key not in base or not _exact(card[key], base[key]):
            errors.append(f"original population drift: {key}")
    for key in ("card_id", "purpose", "teacher_source", "scope_limitations"):
        if not isinstance(card.get(key), str) or not card[key].strip():
            errors.append(f"{key} required")
    for key, expected in (("training", TRAINING), ("evaluation", EVALUATION), ("balance_policy", BALANCE_POLICY),
                          ("rank_policy", RANK_POLICY), ("resources", RESOURCES),
                          ("restrictions", dict.fromkeys(RESTRICTIONS, True))):
        if not _exact(card.get(key), expected):
            errors.append(f"fixed factorial contract drift: {key}")
    if card.get("capacity_ready") is not False or card.get("scientific_gate_pass") is not False:
        errors.append("partial factorial cannot claim scientific PASS")
    if not _digest(card.get("baseline_initial_state_sha256")):
        errors.append("sealed baseline initial state digest required as metadata only")
    sealed = card.get("sealed_sources")
    if not isinstance(sealed, dict) or not sealed:
        errors.append("sealed sources required")
        sealed = {}
    sources = card.get("sources")
    if not isinstance(sources, dict) or set(sources) != set(SOURCE_NAMES):
        errors.append("exact six payloads required")
        sources = {}
    old_sources = base.get("sources", {})
    if not isinstance(old_sources, dict):
        old_sources = {}
    paths = []
    for key, value in sources.items():
        if not isinstance(value, dict) or set(value) != {"path", "sha256"}:
            errors.append("explicit source path/SHA required")
            continue
        path, digest = value["path"], value["sha256"]
        if (not _relative_source(path) or not _digest(digest)
                or PurePosixPath(path).name != SOURCE_NAMES[key] or sealed.get(path) != digest):
            errors.append("source path/hash binding drift")
        if isinstance(path, str):
            paths.append(path)
        if key in OLD_SOURCES and not _exact(value, old_sources.get(key)):
            errors.append("original four inputs changed")
    if len(paths) != 6 or len(set(paths)) != 6:
        errors.append("six distinct payloads required")
    for path, digest in sealed.items():
        if not _relative_source(path) or not _digest(digest):
            errors.append("invalid source path/hash")
        elif (re.search(r"(?:^|[/_])C(?:0[2-9]|10)(?:[/_]|$)", path)
              or path.endswith((".pt", ".pth", ".ckpt", "training_history.json"))):
            errors.append("old weights/history/non-C01 source forbidden")
        elif PurePosixPath(path).name in ("summary.json", "manifest.json", "target_transport.json") or path.endswith(".npz"):
            if path not in paths:
                errors.append("extra payload forbidden")
    for name, digest, sourcekey in (
        ("corrective_reference", CORRECTIVE_SEAL_SHA256, "baseline_geometry_summary"),
        ("rank_reference", RANK_SEAL_SHA256, "baseline_rank_summary"),
    ):
        reference = card.get(name)
        if not isinstance(reference, dict) or set(reference) != {"spec", "card", "seal"}:
            errors.append(f"{name} metadata required")
            continue
        for path in reference.values():
            if not _relative_source(path) or not _digest(sealed.get(path) if isinstance(path, str) else None):
                errors.append(f"unbound {name}")
        seal = reference.get("seal")
        if not isinstance(seal, str) or sealed.get(seal) != digest:
            errors.append(f"{name} seal drift")
        source = sources.get(sourcekey, {})
        if isinstance(seal, str) and isinstance(source, dict):
            root = str(PurePosixPath(seal).parent.parent)
            if source.get("path") != root + "/metrics/summary.json":
                errors.append("comparison summary must belong to its sealed run")
    export = card.get("export_reference", {})
    if not isinstance(export, dict) or not isinstance(export.get("seal"), str) or sealed.get(export["seal"]) != EXPORT_SEAL_SHA256:
        errors.append("original export seal required")
    for key in ("checkpoint", "checkpoints", "history", "sensor_root", "teacher_root", "optimizer", "loss_weights", "loss_scales"):
        if key in card:
            errors.append(f"unauthorized field: {key}")
    approval = card.get("approval", {})
    if not isinstance(approval, dict):
        approval = {}
    if (approval.get("status") != "APPROVED" or not _exact(approval.get("authorized_operations"), ["training"])
            or not _exact(approval.get("authorized_gates"), [3])):
        errors.append("independent training-only Gate3 approval required")
    for key in ("approved_by", "approved_at", "scope", "confirmation_reference"):
        if not isinstance(approval.get(key), str) or not approval[key].strip():
            errors.append(f"approval.{key} required")
    if (not _digest(card.get("selection_sha256")) or approval.get("selection_sha256") != card.get("selection_sha256")
            or not _exact(approval.get("sources"), sources) or not _exact(approval.get("balance_policy"), BALANCE_POLICY)
            or not _exact(approval.get("training"), TRAINING)
            or approval.get("baseline_initial_state_sha256") != card.get("baseline_initial_state_sha256")):
        errors.append("approval must bind six sources, selection, factorial policy and budget")
    return ValidationReport(not errors, tuple(errors))
