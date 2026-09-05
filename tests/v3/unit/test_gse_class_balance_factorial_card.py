from copy import deepcopy

import pytest

from mtare_topo.governance_class_balance_factorial import (
    SCHEMA, TRAINING, EVALUATION, BALANCE_POLICY, RANK_POLICY, RESOURCES, RESTRICTIONS,
    CORRECTIVE_SEAL_SHA256, RANK_SEAL_SHA256, validate_class_balance_factorial_card,
)
from mtare_topo.governance_partial_structure_training import IDENTITY_FIELDS, EXPORT_SEAL_SHA256
from tests.v3.unit.test_gse_geometry_bound_card import card as old_card


def card():
    old = old_card()
    c = {key: deepcopy(old[key]) for key in (*IDENTITY_FIELDS, "effective_counts", "export_reference")}
    sources = deepcopy(old["sources"])
    sealed = {source["path"]: source["sha256"] for source in sources.values()}
    sealed[old["export_reference"]["seal"]] = EXPORT_SEAL_SHA256
    references = {}
    for name, digest, key in (("corrective_reference", CORRECTIVE_SEAL_SHA256, "baseline_geometry_summary"),
                              ("rank_reference", RANK_SEAL_SHA256, "baseline_rank_summary")):
        root = "fixtures/" + name
        reference = {"spec": "configs/" + name + ".json", "card": "configs/" + name + "_card.json",
                     "seal": root + "/artifacts/evidence_sha256.txt"}
        references[name] = reference
        sealed.update(dict.fromkeys(reference.values(), "f" * 64))
        sealed[reference["seal"]] = digest
        sources[key] = {"path": root + "/metrics/summary.json", "sha256": "e" * 64}
        sealed[sources[key]["path"]] = "e" * 64
    c.update({"schema_version": SCHEMA, "operation": "training", "card_id": "synthetic_factorial",
        "purpose": "Separate two loss balance effects", "teacher_source": "Same original four inputs",
        "scope_limitations": "Same-sample partial diagnostic only", "base_card": old, "sources": sources,
        "sealed_sources": sealed, **references, "baseline_initial_state_sha256": "d" * 64,
        "training": deepcopy(TRAINING), "evaluation": deepcopy(EVALUATION), "balance_policy": deepcopy(BALANCE_POLICY),
        "rank_policy": deepcopy(RANK_POLICY), "resources": dict(RESOURCES), "restrictions": dict.fromkeys(RESTRICTIONS, True),
        "capacity_ready": False, "scientific_gate_pass": False,
        "approval": {**deepcopy(old["approval"]), "sources": deepcopy(sources), "balance_policy": deepcopy(BALANCE_POLICY),
                     "training": deepcopy(TRAINING), "baseline_initial_state_sha256": "d" * 64}})
    c["approval"].pop("loss_contract", None)
    return c


def test_valid_card_is_nonmutating_and_old_authority_is_insufficient():
    value = card()
    original = deepcopy(value)
    assert validate_class_balance_factorial_card(value).passed
    assert value == original
    assert not validate_class_balance_factorial_card(old_card()).passed
    value["approval"] = deepcopy(value["base_card"]["approval"])
    assert not validate_class_balance_factorial_card(value).passed


@pytest.mark.parametrize("change", ["steps_float", "steps_bool", "factors", "member_counts", "event_counts",
    "normalization", "threshold", "rank_policy", "selection", "world", "source_hash", "duplicate_source",
    "wrong_summary_run", "extra_npz", "history", "oldweight", "seal", "initialhash", "initialapproval",
    "approval", "science", "extra_source"])
def test_exact_factorial_boundary(change):
    c = card()
    if change == "steps_float": c["training"]["total_steps"] = 2700.
    elif change == "steps_bool": c["training"]["seed"] = False
    elif change == "factors": c["training"]["factor_order"] = ["11"]
    elif change == "member_counts": c["balance_policy"]["member_class_counts"] = [13489, 2155]
    elif change == "event_counts": c["balance_policy"]["event_class_counts"] = [872, 14, 1]
    elif change == "normalization": c["balance_policy"]["denominator"] = "sum_weights"
    elif change == "threshold": c["evaluation"]["member_threshold"] = .3
    elif change == "rank_policy": c["rank_policy"]["threshold_search"] = True
    elif change == "selection": c["selection_sha256"] = "e" * 64
    elif change == "world": c["worlds"][0] = c["worlds"][0].replace("C01", "C02")
    elif change == "source_hash": c["sources"]["inputs"]["sha256"] = "0" * 64
    elif change == "duplicate_source": c["sources"]["baseline_rank_summary"] = deepcopy(c["sources"]["baseline_geometry_summary"])
    elif change == "wrong_summary_run":
        p = "fixtures/other/metrics/summary.json"
        c["sources"]["baseline_rank_summary"]["path"] = p
        c["sealed_sources"][p] = "e" * 64
    elif change == "extra_npz": c["sealed_sources"]["fixtures/final.npz"] = "e" * 64
    elif change == "history": c["sealed_sources"]["fixtures/training_history.json"] = "e" * 64
    elif change == "oldweight": c["sealed_sources"]["fixtures/shared_initial_state.pt"] = "d" * 64
    elif change == "seal": c["sealed_sources"][c["rank_reference"]["seal"]] = "e" * 64
    elif change == "initialhash": c["baseline_initial_state_sha256"] = None
    elif change == "initialapproval": c["approval"]["baseline_initial_state_sha256"] = "e" * 64
    elif change == "approval": c["approval"]["authorized_operations"] = ["data_export"]
    elif change == "science": c["scientific_gate_pass"] = True
    else: c["sources"]["other"] = {"path": "fixtures/other.npz", "sha256": "e" * 64}
    assert not validate_class_balance_factorial_card(c).passed


@pytest.mark.parametrize("key,value", [("base_card", None), ("sources", []), ("sealed_sources", None),
    ("rank_reference", {"spec": None, "card": [], "seal": {}}), ("approval", None)])
def test_malformed_nested_objects_fail_without_io(key, value):
    c = card()
    c[key] = value
    assert not validate_class_balance_factorial_card(c).passed
