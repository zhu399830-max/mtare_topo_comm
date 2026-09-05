import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]


def _json(relative):
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_calibration_card_freezes_online_domain_and_no_training():
    card = _json("configs/v3/gate3/data_cards/gse_distance_aware_ensemble_calibration_v1.json")
    assert card["status"] == "APPROVED_FOR_ONE_IMMUTABLE_GSE_DISTANCE_AWARE_ENSEMBLE_CALIBRATION_V1"
    assert card["approval"]["authorized_operations"] == ["threshold_calibration"]
    assert card["sampling"]["selection_pairs_all"] == 45372
    assert card["sampling"]["online_eligible_selection_pairs"] == 31469
    assert card["sampling"]["online_excluded_selection_pairs"] == 13903
    assert card["sampling"]["online_eligible_positive_pairs"] == 12813
    assert card["sampling"]["online_eligible_negative_pairs"] == 18656
    assert card["training_contract"]["ensemble_weights"] == "exact fixed [1/3,1/3,1/3]; no score calibration, stacking, learned weighting or seed selection."
    assert card["training_contract"]["optimizer_steps"] == 0
    assert card["training_contract"]["backbone_optimizer_steps"] == 0


def test_calibration_card_binds_both_failed_sources_without_rewriting_them():
    card = _json("configs/v3/gate3/data_cards/gse_distance_aware_ensemble_calibration_v1.json")
    v2 = ROOT / card["source"]["calibration_source_v2_run"]
    v1r = ROOT / card["source"]["baseline_v1r_run"]
    assert _sha256(v2 / "artifacts/evidence_sha256.txt") == card["source"]["calibration_source_v2_seal_sha256"]
    assert _sha256(v1r / "artifacts/evidence_sha256.txt") == card["source"]["baseline_v1r_seal_sha256"]
    assert _json(v2.relative_to(ROOT) / "RUN_STATE.json")["state"] == "FAILED"
    assert _json(v1r.relative_to(ROOT) / "RUN_STATE.json")["state"] == "FAILED"


def test_calibration_evaluator_uses_distance_aware_selector_and_arithmetic_mean():
    source = (ROOT / "tools/v3/evaluate_gse_distance_aware_ensemble_v1.py").read_text(encoding="utf-8")
    assert "select_online_candidate_threshold" in source
    assert "stacked.mean(axis=0, dtype=np.float64)" in source
    assert "expected_domain" in source
    assert '"optimizer_steps": 0' in source
    assert '"c09_worlds_read": 0' in source
