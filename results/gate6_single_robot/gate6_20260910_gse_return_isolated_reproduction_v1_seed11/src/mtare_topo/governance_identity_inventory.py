"""Narrow identity-only inventory audit with honestly unknown populations."""
import hashlib
import json

from mtare_topo.governance_partial_structure_training import _exact

SCHEMA = "v3_identity_inventory_card_v1"
CARD_ID = "gse_review_source_inventory_v1"
P1A = "results/gate3_semantics/gate3_20260830_primitive_relation_p1a_lossless_storage_corrective_v1_seed0"
P1B = "results/gate3_semantics/gate3_20260830_primitive_relation_p1b_teacher_materialization_v1r_seed0"
FAMILIES = ("S01_flat_tree_small", "S02_3d_tree_small", "S03_flat_unicyclic_small", "S04_3d_unicyclic_small",
            "S05_flat_branch_medium", "S06_3d_branch_medium", "S07_flat_loop_rich", "S08_3d_loop_rich",
            "S09_flat_complex", "S10_3d_complex")
VARIANTS = ("ellipse", "rounded_rectangle", "c1_mixed")
RESTRICTIONS = ("read_only_sources", "identity_indices_only", "no_scan", "no_pose", "no_geometry",
                "no_construction", "no_teacher_labels", "no_model", "no_checkpoint", "no_optimizer",
                "no_training", "no_new_teacher", "no_selection_of_48", "no_strata_nomination", "no_annotation",
                "no_images", "no_C08_C10", "no_graph", "no_calibration", "no_threshold_search",
                "no_old_source_or_run_changes", "no_scientific_gate_pass", "filter_seal_before_path_resolution")


def expected_scope():
    parents = sorted(f"{family}_C{i:02}" for family in FAMILIES for i in range(1, 8))
    return {
        "parent_ids": parents, "variants": list(VARIANTS),
        "tasks": [{"parent_id": p, "partition": "c07" if p.endswith("_C07") else "fit", "variant": v,
                   "task": p + "__" + v} for p in parents for v in VARIANTS],
        "source_roots": {"sensor": {part: P1A + "/artifacts/dataset/" + part for part in ("fit", "c07")},
                         "teacher": {part: P1B + "/artifacts/teacher/" + part for part in ("fit", "c07")}},
        "source_seals": {
            "sensor": {"path": P1A + "/artifacts/evidence_sha256.txt", "sha256": "79fd988ac8c205d74c93e4858b7b579571a06502e778f31634046b48791a0668"},
            "teacher": {"path": P1B + "/artifacts/evidence_sha256.txt", "sha256": "f629b511e9a945bbe15249e3bc212be57a01b1aa0118d0d7d228824aa25f7d47"}},
        "allowed_arrays": {"sensor": ["global_frame_index", "local_frame_index", "traversal_index", "route_arc_m"],
                           "teacher": ["source_global_sequence_index", "variant_global_sequence_index", "frame_row"]},
        "allowed_metadata": [".zgroup", ".zattrs", "array/.zarray"],
        "expected_counts": {"parents": 70, "tasks": 210, "logical_sequences": 163732, "variant_sequences": 491196},
        "unknown_counts": {"unique_raw_frames": None, "traversals": None, "eligible_21_decision_windows": None,
                           "structure_events": None},
        "time_basis": "source sequence order and measured route_arc_m; no acquisition clock",
        "duration_s": None, "independent_sampling_unit": "parent_world_not_variant_or_sequence",
        "restrictions": dict.fromkeys(RESTRICTIONS, True),
        "resources": {"wall_time_cap_s": 600, "host_ram_bytes": 4294967296, "gpu_bytes": 0, "output_bytes": 250000000},
    }


def scope_sha256(scope):
    return hashlib.sha256(json.dumps(scope, sort_keys=True, ensure_ascii=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def validate_identity_inventory_card(card):
    from mtare_topo.governance import ValidationReport
    errors = []
    if type(card) is not dict:
        return ValidationReport(False, ("identity inventory card must be object",))
    if (card.get("schema_version") != SCHEMA or card.get("card_id") != CARD_ID or card.get("operation") != "audit"):
        errors.append("independent identity-only audit card required")
    for key in ("purpose", "source_provenance", "limitations"):
        if type(card.get(key)) is not str or not card[key].strip():
            errors.append(f"{key} must document this metadata-only audit")
    scope = expected_scope()
    if not _exact(card.get("scope"), scope):
        errors.append("exact70parents/210tasks/two seals/seven index fields/unknown counts/restrictions required")
    digest = scope_sha256(scope)
    if card.get("scope_sha256") != digest:
        errors.append("scope digest drift")
    approval = card.get("approval")
    if type(approval) is not dict:
        errors.append("approval object required")
        approval = {}
    if (approval.get("status") != "APPROVED" or not _exact(approval.get("authorized_operations"), ["audit"])
            or not _exact(approval.get("authorized_gates"), [3])):
        errors.append("standing authority must be scoped to audit-only Gate3")
    for key in ("approved_by", "approved_at", "scope", "confirmation_reference"):
        if type(approval.get(key)) is not str or not approval[key].strip():
            errors.append(f"approval.{key} required")
    if approval.get("scope_sha256") != digest:
        errors.append("approval must bind exact inventory scope")
    if card.get("scientific_gate_pass") is not False or card.get("real_inventory_completed") is not False:
        errors.append("inventory prep cannot assert actual completion or science PASS")
    allowed = {"schema_version", "card_id", "operation", "purpose", "source_provenance", "limitations", "scope",
               "scope_sha256", "approval", "scientific_gate_pass", "real_inventory_completed"}
    if set(card) - allowed:
        errors.append("extra top-level authority/payload/count fields forbidden")
    return ValidationReport(not errors, tuple(errors))
