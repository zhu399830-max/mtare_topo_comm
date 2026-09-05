import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]


def _json(relative):
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_v2_card_preserves_v1r_population_and_split():
    v1r = _json("configs/v3/gate3/data_cards/gse_open_set_association_corrective_v1r.json")
    v2 = _json("configs/v3/gate3/data_cards/gse_exit_token_association_corrective_v2.json")
    assert v2["status"] == "APPROVED_FOR_ONE_IMMUTABLE_GSE_EXIT_TOKEN_ASSOCIATION_CORRECTIVE_V2"
    assert v2["worlds"] == v1r["worlds"]
    assert v2["trajectories"] == v1r["trajectories"]
    assert v2["sampling"] == v1r["sampling"]
    assert v2["split"] == v1r["split"]
    assert v2["acceptance"]["aggregate_precision_min"] == 0.98
    assert v2["acceptance"]["aggregate_false_accept_rate_max"] == 0.01
    assert v2["acceptance"]["aggregate_recall_min"] == 0.25
    assert v2["source"]["predecessor_partial_reuse"].startswith("NONE_")


def test_v2_card_binds_full_token_interface_and_sealed_sources():
    card = _json("configs/v3/gate3/data_cards/gse_exit_token_association_corrective_v2.json")
    assert "32D exit descriptors" in card["teacher"]["student_input"]
    assert "473->192->64->1" in card["training_contract"]["model"]
    assert "global-yaw invariance" in card["training_contract"]["pair_symmetry"]
    assert "highest-scoring negatives" in card["training_contract"]["tail_risk_loss"]
    dataset = ROOT / card["source"]["dataset_run"]
    training = ROOT / card["source"]["frozen_gse_training_run"]
    predecessor = ROOT / card["source"]["predecessor_failed_run"]
    assert _sha256(dataset / "artifacts/evidence_sha256.txt") == card["source"]["dataset_seal_sha256"]
    assert _sha256(training / "artifacts/evidence_sha256.txt") == card["source"]["frozen_gse_training_seal_sha256"]
    assert _sha256(predecessor / "artifacts/evidence_sha256.txt") == card["source"]["predecessor_failed_seal_sha256"]


def test_v2_trainer_extracts_exit_descriptor_and_uses_tail_loss():
    source = (ROOT / "tools/v3/train_gse_exit_token_association_v2.py").read_text(encoding="utf-8")
    assert '"exit_descriptor"' in source
    assert "frozen_exit_token_outputs.npz" in source
    assert "exit_token_pair_features" in source
    assert "hard_negative_tail_loss" in source
    assert "GSEExitTokenAssociationVerifier" in source
