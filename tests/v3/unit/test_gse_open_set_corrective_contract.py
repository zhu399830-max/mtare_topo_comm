import hashlib
import json
from pathlib import Path

import torch

from mtare_topo.representation.gse_open_set_association import (
    GSEOpenSetAssociationVerifier,
    OBSERVATION_FEATURE_DIM,
    PAIR_FEATURE_DIM,
)


ROOT = Path(__file__).resolve().parents[3]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_data_card_freezes_corrected_reproducible_population():
    card = json.loads(
        (ROOT / "configs/v3/gate3/data_cards/gse_open_set_association_corrective_v1.json")
        .read_text(encoding="utf-8")
    )
    assert card["status"] == "APPROVED_FOR_ONE_IMMUTABLE_GSE_OPEN_SET_ASSOCIATION_CORRECTIVE_V1"
    assert len(card["worlds"]["train"]) == 60
    assert len(card["worlds"]["validation"]) == 20
    assert set(card["worlds"]["train"]).isdisjoint(card["worlds"]["validation"])
    assert card["sampling"]["open_set_pairs"] == {"fit": 57066, "selection": 18111}
    assert card["sampling"]["total_pairs"] == {"fit": 135232, "selection": 45372}
    assert len(card["worlds"]["reserved_c09"]) == 10
    assert len(card["worlds"]["strict_test"]) == 10
    assert card["worlds"]["mtare_benchmark"] == []


def test_data_card_source_hashes_match_sealed_files():
    card = json.loads(
        (ROOT / "configs/v3/gate3/data_cards/gse_open_set_association_corrective_v1.json")
        .read_text(encoding="utf-8")
    )
    dataset = ROOT / card["source"]["dataset_run"]
    training = ROOT / card["source"]["frozen_gse_training_run"]
    assert _sha256(dataset / "artifacts/evidence_sha256.txt") == card["source"]["dataset_seal_sha256"]
    assert _sha256(dataset / "artifacts/sequence_manifest.jsonl") == card["source"]["sequence_manifest_sha256"]
    assert _sha256(dataset / "artifacts/association_pairs_numeric.jsonl") == card["source"]["association_pair_manifest_sha256"]
    assert _sha256(training / "artifacts/evidence_sha256.txt") == card["source"]["frozen_gse_training_seal_sha256"]
    for seed in (0, 1, 2):
        assert _sha256(training / f"artifacts/models/seed{seed}/best.pt") == card["source"]["checkpoint_sha256"][f"seed{seed}"]


def test_v1r_card_changes_only_system_mapping_scope_and_preserves_science():
    v1 = json.loads(
        (ROOT / "configs/v3/gate3/data_cards/gse_open_set_association_corrective_v1.json")
        .read_text(encoding="utf-8")
    )
    v1r = json.loads(
        (ROOT / "configs/v3/gate3/data_cards/gse_open_set_association_corrective_v1r.json")
        .read_text(encoding="utf-8")
    )
    assert v1r["status"] == "APPROVED_FOR_ONE_IMMUTABLE_GSE_OPEN_SET_ASSOCIATION_CORRECTIVE_V1R"
    assert v1r["worlds"] == v1["worlds"]
    assert v1r["trajectories"] == v1["trajectories"]
    assert v1r["sampling"]["open_set_pairs"] == v1["sampling"]["open_set_pairs"]
    assert v1r["sampling"]["total_pairs"] == v1["sampling"]["total_pairs"]
    assert v1r["teacher"] == v1["teacher"]
    assert v1r["split"] == v1["split"]
    assert v1r["source"]["predecessor_partial_reuse"] == "NONE_ZERO_INFERENCE_ZERO_OPTIMIZER_STEPS"
    assert "compact_to_global_sequence_index" in v1r["sampling"]["global_identity_mapping"]


def test_verifier_architecture_matches_card_and_has_no_backbone_parameters():
    model = GSEOpenSetAssociationVerifier()
    assert model.matchability[0].in_features == OBSERVATION_FEATURE_DIM == 146
    assert model.pair[0].in_features == PAIR_FEATURE_DIM == 293
    names = tuple(name for name, _ in model.named_parameters())
    assert names
    assert not any("encoder" in name or "backbone" in name for name in names)
    assert sum(parameter.numel() for parameter in model.parameters()) == 51266


def test_frozen_source_checkpoint_remains_unmodified_by_verifier_construction():
    before = torch.random.get_rng_state().clone()
    torch.manual_seed(1)
    _ = GSEOpenSetAssociationVerifier()
    torch.random.set_rng_state(before)
