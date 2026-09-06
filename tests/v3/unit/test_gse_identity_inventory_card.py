from copy import deepcopy

import pytest

from mtare_topo.governance_identity_inventory import (
    SCHEMA, CARD_ID, expected_scope, scope_sha256, validate_identity_inventory_card,
)


def card():
    scope = expected_scope()
    digest = scope_sha256(scope)
    return {"schema_version": SCHEMA, "card_id": CARD_ID, "operation": "audit",
        "purpose": "Synthetic exact identity population audit", "source_provenance": "Sealed identity indices only",
        "limitations": "Unknown frames/traversals/time are not fabricated; no candidate or labels",
        "scope": scope, "scope_sha256": digest, "scientific_gate_pass": False, "real_inventory_completed": False,
        "approval": {"status": "APPROVED", "authorized_operations": ["audit"], "authorized_gates": [3],
            "approved_by": "synthetic_fixture", "approved_at": "2026-09-06", "scope": "Exact scoped audit only",
            "confirmation_reference": "Synthetic fixture, not a new real user dialogue", "scope_sha256": digest}}


def test_exact_scoped_metadata_card_preserves_unknowns_and_does_not_mutate():
    c = card(); before = deepcopy(c)
    assert validate_identity_inventory_card(c).passed and c == before
    assert len(c["scope"]["parent_ids"]) == 70 and len(c["scope"]["tasks"]) == 210
    assert all(v is None for v in c["scope"]["unknown_counts"].values())
    assert c["scope"]["duration_s"] is None
    assert sum(len(v) for v in c["scope"]["allowed_arrays"].values()) == 7


@pytest.mark.parametrize("fault", ["training", "gate_bool", "schema", "parent", "missing_task", "duplicate_task",
    "scan", "pose", "geometry", "construction", "array_attrs", "source_root", "seal", "expected_float",
    "fabricated_actual", "fabricated_duration", "memory", "restriction", "scope_hash", "approval_hash",
    "annotation", "scientific_pass", "completed"])
def test_narrow_authority_rejects_drift_even_with_recomputed_scope_hash(fault):
    c = card(); s = c["scope"]
    if fault == "training": c["approval"]["authorized_operations"] = ["audit", "training"]
    elif fault == "gate_bool": c["approval"]["authorized_gates"] = [True]
    elif fault == "schema": c["schema_version"] = "v3_data_card_v1"
    elif fault == "parent": s["parent_ids"][0] = "S01_flat_tree_small_C08"
    elif fault == "missing_task": s["tasks"].pop()
    elif fault == "duplicate_task": s["tasks"][-1] = deepcopy(s["tasks"][0])
    elif fault in ("scan", "pose", "geometry"):
        s["allowed_arrays"]["sensor"].append({"scan": "range_m", "pose": "sensor_xyz_m", "geometry": "axis_xyz_m"}[fault])
    elif fault == "construction": s["source_roots"]["construction"] = {"fit": "results/constructions/fit"}
    elif fault == "array_attrs": s["allowed_metadata"].append("array/.zattrs")
    elif fault == "source_root": s["source_roots"]["teacher"]["c07"] = s["source_roots"]["teacher"]["c07"].replace("c07", "c08")
    elif fault == "seal": s["source_seals"]["sensor"]["sha256"] = "0" * 64
    elif fault == "expected_float": s["expected_counts"]["parents"] = 70.
    elif fault == "fabricated_actual": s["unknown_counts"]["unique_raw_frames"] = 1
    elif fault == "fabricated_duration": s["duration_s"] = 1.
    elif fault == "memory": s["resources"]["host_ram_bytes"] *= 2
    elif fault == "restriction": s["restrictions"]["no_annotation"] = False
    elif fault == "scope_hash": c["scope_sha256"] = "0" * 64
    elif fault == "approval_hash": c["approval"]["scope_sha256"] = "0" * 64
    elif fault == "annotation": c["annotation"] = {"enabled": True}
    elif fault == "scientific_pass": c["scientific_gate_pass"] = True
    else: c["real_inventory_completed"] = True
    if fault not in ("scope_hash", "approval_hash"):
        c["scope_sha256"] = c["approval"]["scope_sha256"] = scope_sha256(s)
    assert not validate_identity_inventory_card(c).passed


@pytest.mark.parametrize("field,value", [("scope", None), ("scope", []), ("approval", None), ("purpose", "")])
def test_malformed_card_reports_failure(field, value):
    c = card(); c[field] = value
    assert not validate_identity_inventory_card(c).passed
