"""Shared frozen encoder inference only; no label or training authority."""
from mtare_topo.governance_surface_selection import digest

SCHEMA = "v3_surface_features_card_v1"
SLUG = "gse_surface_features_v1"
POLICY = {"microbatch": 1, "wall_time_s": 1800, "host_ram_bytes": 32 * 1024**3,
          "gpu_bytes": 28 * 1024**3, "output_bytes": 8 * 1024**3,
          "labels": 0, "optimizer_steps": 0, "tf32": False, "no_retry": True}


def validate_card(card):
    from mtare_topo.governance import ValidationReport
    errors = []
    if type(card) is not dict or set(card) != {"schema_version", "card_id", "operation", "scope", "scope_sha256", "policy", "approval"}:
        return ValidationReport(False, ("closed feature export card required",))
    scope, approval = card["scope"], card["approval"]
    if (card["schema_version"], card["card_id"], card["operation"]) != (SCHEMA, SLUG, "data_export"):
        errors.append("feature-only export required")
    if type(scope) is not dict or scope.get("schema_version") != "gse_surface_feature_source_plan_v1":
        return ValidationReport(False, ("compiled feature source scope required",))
    if scope.get("counts") != {"parents": 70, "tasks": 210, "physical_edges": 1120, "observations": 3360,
            "source_frames": 16800, "splits": {"fit": 2880, "calibration": 240, "development": 240}}:
        errors.append("fixed feature population required")
    if card["policy"] != POLICY or card["scope_sha256"] != digest(scope):
        errors.append("policy or scope digest drift")
    if (type(approval) is not dict or approval.get("status") != "APPROVED" or
            approval.get("scope_sha256") != digest(scope) or approval.get("authorized_operations") != ["data_export"] or
            approval.get("authorized_gates") != [3] or not approval.get("confirmation_reference")):
        errors.append("exact standing authorization required")
    return ValidationReport(not errors, tuple(errors))
