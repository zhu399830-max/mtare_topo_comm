"""Narrow authority: select identity rows from one already sealed C01-C07 run."""
import hashlib
import json
import re

from mtare_topo.governance_identity_inventory import FAMILIES

SCHEMA = "v3_surface_identity_selection_card_v1"
CARD_ID = "gse_surface_identity_selection_v1"
INVENTORY = "results/gate3_semantics/gate3_20260906_gse_review_source_inventory_v1_seed20260906"
SEAL_SHA256 = "d5cb060439bfeef3cd85f5a58ae244e8abf9b5237451a6a59b9311f9225e52e6"
PARENTS = sorted(f"{f}_C{i:02}" for f in FAMILIES for i in range(1, 8))
INPUT_PATHS = sorted([INVENTORY + "/artifacts/" + p + "_identity_intervals.json" for p in PARENTS]
                     + [INVENTORY + "/artifacts/parent_split.json", INVENTORY + "/artifacts/parent_population.json"])
POLICY = {"seed": 20260906, "edges_per_parent": 16, "directions_per_edge": 1,
          "five_frame_observations_per_edge": 1, "variants": ["ellipse", "rounded_rectangle", "c1_mixed"],
          "decision_positions": ["first", "middle", "last"], "physical_edge_not_tunnel": True,
          "preserve_original_parent_split": True, "requires_21_decisions": False}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=True,
                                    separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def validate_surface_selection_card(card):
    from mtare_topo.governance import ValidationReport
    errors = []
    if type(card) is not dict:
        return ValidationReport(False, ("surface identity selection card object required",))
    required = {"schema_version", "card_id", "operation", "purpose", "source_provenance", "limitations",
                "scope", "scope_sha256", "approval", "scientific_gate_pass", "training_eligibility"}
    if set(card) != required or (card.get("schema_version"), card.get("card_id"), card.get("operation")) != (SCHEMA, CARD_ID, "audit"):
        errors.append("closed identity-only audit card schema required")
    for k in ("purpose", "source_provenance", "limitations"):
        if type(card.get(k)) is not str or not card[k].strip():
            errors.append(k + " required")
    scope = card.get("scope")
    if type(scope) is not dict:
        return ValidationReport(False, tuple(errors + ["scope required"]))
    if set(scope) != {"parent_ids", "input_files_sha256", "source_seal", "policy", "expected_observations",
                      "unknown_counts", "time_basis", "resources", "forbidden_payloads"}:
        errors.append("closed source scope required")
    if scope.get("parent_ids") != PARENTS or scope.get("policy") != POLICY:
        errors.append("fixed70 parents/single-observation16edge policy drift")
    if scope.get("source_seal") != {"path": INVENTORY + "/artifacts/evidence_sha256.txt", "sha256": SEAL_SHA256}:
        errors.append("independently pinned source seal required")
    files = scope.get("input_files_sha256")
    if (type(files) is not dict or sorted(files) != INPUT_PATHS
            or any(type(h) is not str or re.fullmatch(r"[a-f0-9]{64}", h) is None for h in files.values())):
        errors.append("exact72 sealed identity-only input files required")
    if scope.get("expected_observations") != {"fit": 2880, "calibration": 240, "development": 240}:
        errors.append("fixed proposed observation population required")
    if scope.get("unknown_counts") != {"unique_source_frames": None, "structure_entities": None, "labels": None, "duration_s": None}:
        errors.append("unknown frame/structure/label/duration counts cannot be fabricated")
    if scope.get("resources") != {"wall_time_s": 120, "host_ram_bytes": 1073741824, "gpu_bytes": 0, "output_bytes": 33554432}:
        errors.append("fixed identity-selection resource bounds required")
    if scope.get("forbidden_payloads") != ["scans", "poses", "mesh", "construction", "teacher", "models", "C08-C10", "benchmark"]:
        errors.append("no authority to open original data or protected worlds")
    if scope.get("time_basis") != "sealed decision order and route arc; acquisition clock/history spacing unknown":
        errors.append("honest temporal/spatial basis required")
    try:
        scope_hash = digest(scope)
    except (TypeError, ValueError):
        return ValidationReport(False, tuple(errors + ["finite JSON scope required"]))
    approval = card.get("approval", {})
    if type(approval) is not dict:
        approval = {}
    if (approval.get("status") != "APPROVED" or approval.get("authorized_operations") != ["audit"]
            or approval.get("authorized_gates") != [3] or approval.get("scope_sha256") != scope_hash
            or card.get("scope_sha256") != scope_hash):
        errors.append("standing user authorization must bind this exact audit-only Gate3 scope")
    for k in ("approved_by", "approved_at", "scope", "confirmation_reference"):
        if type(approval.get(k)) is not str or not approval[k].strip():
            errors.append("approval." + k + " required")
    if card.get("scientific_gate_pass") is not False or card.get("training_eligibility") is not False:
        errors.append("identity selection grants no science PASS or training permission")
    return ValidationReport(not errors, tuple(errors))
