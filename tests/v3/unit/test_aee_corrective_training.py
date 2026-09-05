"""Contracts for the V1R4 corrective training adapters."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from mtare_topo.data.aee_corrective_training import (
    AllSamplesBatchSampler,
    CorrectiveAEEMultitaskDataset,
    CorrectiveCanoMultitaskDataset,
    MaskMatchedCanoValidationDataset,
    _mask_index,
    mask_matched_cano_sample,
)


ROOT = Path(__file__).resolve().parents[3]
RUN = ROOT / "results/gate2_representation/gate2_20260822_aee_corrective_sensor_teacher_dataset_v1r4_seed20260822"


def _sample(frame_id: str, period: int | None = None) -> dict:
    student = np.full((2, 16, 720), 0.2, dtype=np.float32)
    student[1] = 1.0
    if period is not None:
        student[0, :, ::period] = 1.0
        student[1, :, ::period] = 0.0
    return {"student": student, "frame_id": frame_id}


def test_all_samples_sampler_covers_every_index_once_and_is_deterministic() -> None:
    sampler = AllSamplesBatchSampler(6000, 128, 7)
    sampler.set_epoch(1)
    first = [index for batch in sampler for index in batch]
    sampler.set_epoch(1)
    second = [index for batch in sampler for index in batch]
    assert first == second
    assert len(first) == len(set(first)) == 6000
    assert set(first) == set(range(6000))


def test_stable_mask_mapping_does_not_use_python_process_hash() -> None:
    assert _mask_index("cano:17", 20260822, 1000) == _mask_index("cano:17", 20260822, 1000)
    assert _mask_index("cano:17", 20260823, 1000) != _mask_index("cano:17", 20260822, 1000)


def test_mask_matching_never_invents_a_return_or_mutates_inputs() -> None:
    cano = _sample("cano:0", 5)
    aee = _sample("aee:0", 3)
    before = cano["student"].copy()
    result = mask_matched_cano_sample(cano, aee)
    expected = (before[1] > 0.5) & (aee["student"][1] > 0.5)
    assert np.array_equal(result["student"][1] > 0.5, expected)
    assert np.all(result["student"][0, ~expected] == 1.0)
    assert np.array_equal(cano["student"], before)


def test_sealed_v1r4_counts_splits_and_identity_contract() -> None:
    train = CorrectiveCanoMultitaskDataset(RUN, "train")
    validation = CorrectiveCanoMultitaskDataset(RUN, "validation")
    aee = CorrectiveAEEMultitaskDataset(RUN)
    assert (len(train), len(validation), len(aee)) == (5000, 5000, 1000)
    train_parents = {item["parent_id"] for item in train.records}
    validation_parents = {item["parent_id"] for item in validation.records}
    assert train_parents.isdisjoint(validation_parents)
    assert all(item["split"] == "corrective_train" for item in aee.records)
    sample = aee[0]
    assert sample["student"].shape == (2, 16, 720)
    assert sample["direction_target"].shape == (720,)


def test_validation_wrapper_uses_only_cano_labels() -> None:
    validation = CorrectiveCanoMultitaskDataset(RUN, "validation")
    aee = CorrectiveAEEMultitaskDataset(RUN)
    sparse = MaskMatchedCanoValidationDataset(validation, aee, 20260822)
    dense_item = validation[0]
    sparse_item = sparse[0]
    assert sparse_item["frame_id"] == dense_item["frame_id"]
    assert np.array_equal(sparse_item["direction_target"], dense_item["direction_target"])
    assert sparse_item["domain"] == "cano_mask_matched"
    assert sparse_item["mask_source_frame_id"].startswith(("00_tunnel", "01_tunnel", "02_tunnel", "03_tunnel", "04_tunnel", "05_garage", "06_garage", "07_garage", "08_garage", "09_garage"))
