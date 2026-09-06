"""Synthetic source rows; never read maps, shards, checkpoints or labels."""
from dataclasses import replace
import hashlib
import json

import numpy as np
import pytest
import torch

from mtare_topo.data.gse_review_export_v1 import ReviewSensorWindowV1, export_blind_segment_v1
from mtare_topo.data.gse_review_sampling_v1 import (
    CandidateSegment, ParentMetadata, ParentPopulation, SelectedSegment, VariantSegmentMetadata,
)
from mtare_topo.data.gse_structure_review_v1 import BlindReviewSession, validate_blind_bundle
from mtare_topo.representation.primitive_relation_model import register_causal_lidar_points


VARIANTS = ("synthetic_a", "synthetic_b", "synthetic_c")


def fixture():
    parent = "S01_synthetic_C01"
    population = ParentPopulation((ParentMetadata(parent, "C01"),
        ParentMetadata("S01_synthetic_C07", "C07"), ParentMetadata("S02_synthetic_C07", "C07")), "a" * 64, True)
    frames = tuple(tuple(range(i, i + 5)) for i in range(21))
    records = tuple(VariantSegmentMetadata(v, parent + "__" + v, tuple(range(21)), tuple(range(21)),
        frames, (("traversal0",) * 5,) * 21, tuple(i + 4 for i in range(21)),
        tuple(float(i + 4) for i in range(21)), 21, 25) for v in VARIANTS)
    selected = SelectedSegment("fit", CandidateSegment(parent, "junction0", "junction", "traversal0", records))
    windows = []
    for i in range(21):
        scan = np.zeros((5, 2, 16, 720), np.float32)
        for h, f in enumerate(frames[i]):
            # Same raw frame has the same scan in each overlapping window.
            scan[h, 0, 8, 0] = (f + 1) / 50
            scan[h, 1, 8, 0] = 1
        translation = np.zeros((5, 3), np.float32)
        translation[:, 0] = np.arange(-4, 1, dtype=np.float32)
        yaw = np.array([90, 45, 30, 10, 0], np.float32)
        windows.append(ReviewSensorWindowV1(records[0].task, i, i, frames[i], ("traversal0",) * 5,
                                          scan, translation, yaw))
    return population, selected, tuple(windows)


def export(pop=None, selected=None, windows=None, **kw):
    p, s, w = fixture()
    args = dict(variant_names=VARIANTS, variant_name=VARIANTS[0], selection_sha256="b" * 64,
                sensor_source_sha256="c" * 64, sequence_source_sha256="d" * 64)
    args.update(kw)
    return export_blind_segment_v1(p if pop is None else pop, s if selected is None else selected,
                                   w if windows is None else windows, **args)


def test_projection_exactly_matches_student_and_preserves_all_points():
    p, s, windows = fixture()
    result = export(p, s, windows)
    bundle = json.loads(result.bundle_bytes)
    validate_blind_bundle(bundle)
    for row, source in zip(bundle["decisions"], windows):
        xyz, valid = register_causal_lidar_points(torch.from_numpy(source.range_valid)[None],
            torch.from_numpy(source.relative_translation_current_sensor_m)[None],
            torch.from_numpy(source.relative_yaw_current_sensor_deg)[None])
        assert np.array_equal(row["points_xyz_m"], xyz[0][valid[0]].numpy())
        assert len(row["points_xyz_m"]) == 5
        assert row["source_order_indices"] == list(source.frame_rows)
    assert bundle["decisions"][0]["source_frame_keys"][1:] == bundle["decisions"][1]["source_frame_keys"][:4]
    ledger = result.private_provenance
    assert ledger["unique_raw_frames"] == 25
    assert ledger["bundle_file_sha256"] == hashlib.sha256(result.bundle_bytes).hexdigest()
    assert not ledger["automatic_training_eligibility"] and not ledger["human_review_complete"]
    assert ledger["source_authentication"] == "EXTERNAL_FROZEN_READER_REQUIRED"


def test_blind_payload_does_not_expose_sampling_or_teacher_identity():
    result = export()
    for private_value in ("S01_synthetic_C01", "junction0", "traversal0", "synthetic_a", '"fit"', '"junction"'):
        assert private_value.encode() not in result.bundle_bytes
    session = BlindReviewSession(json.loads(result.bundle_bytes), "synthetic reviewer")
    assert session.next_observation()["decision_index"] == 0
    assert not session.export()["blind_complete"]


def test_determinism_readonly_input_and_content_digests():
    p, s, w = fixture()
    before = [x.range_valid.copy() for x in w]
    for x in w:
        x.range_valid.flags.writeable = False
    first, second = export(p, s, w), export(p, s, w)
    assert first == second
    assert all(np.array_equal(a, b.range_valid) for a, b in zip(before, w))
    changed = export(p, s, w, sensor_source_sha256="e" * 64)
    assert first.bundle_bytes != changed.bundle_bytes
    assert first.private_provenance["decisions"][0]["range_valid_sha256"] == changed.private_provenance["decisions"][0]["range_valid_sha256"]


def test_empty_observation_is_preserved_not_filled_with_background():
    p, s, w = fixture()
    w = tuple(replace(x, range_valid=np.zeros_like(x.range_valid)) for x in w)
    result = export(p, s, w)
    assert all(row["points_xyz_m"] == [] for row in json.loads(result.bundle_bytes)["decisions"])
    assert not result.private_provenance["reference_annotation_generated"]


@pytest.mark.parametrize("fault", ["task", "sequence", "sequence_bool", "stored_row", "frame", "traversal",
    "dict", "dtype", "shape", "nan", "range", "valid", "translation", "yaw", "shared_frame"])
def test_source_drift_fails_before_bundle_delivery(fault):
    p, s, windows = fixture()
    w = windows[1] if fault == "shared_frame" else windows[0]
    if fault == "task": w = replace(w, task="different")
    elif fault == "sequence": w = replace(w, source_sequence_id=90)
    elif fault == "sequence_bool": w = replace(w, source_sequence_id=False)
    elif fault == "stored_row": w = replace(w, sequence_row=90)
    elif fault == "frame": w = replace(w, frame_rows=(1, 2, 3, 4, 5))
    elif fault == "traversal": w = replace(w, frame_traversal_ids=("other",) * 5)
    elif fault == "dict": w = w.__dict__
    elif fault == "dtype": w = replace(w, range_valid=w.range_valid.astype(np.float64))
    elif fault == "shape": w = replace(w, range_valid=w.range_valid[:4])
    elif fault == "nan": w.range_valid[0, 0, 0, 0] = np.nan
    elif fault == "range": w.range_valid[0, 0, 0, 0] = 1.01
    elif fault == "valid": w.range_valid[0, 1, 0, 0] = .5
    elif fault == "translation": w.relative_translation_current_sensor_m[-1, 0] = .1
    elif fault == "yaw": w.relative_yaw_current_sensor_deg[-1] = 1
    else: w.range_valid[0, 0, 8, 0] = .33
    windows = (windows[0], w) + windows[2:] if fault == "shared_frame" else (w,) + windows[1:]
    with pytest.raises(ValueError): export(p, s, windows)


@pytest.mark.parametrize("fault", ["split", "variant", "digest", "windows", "test_cohort"])
def test_scope_validation(fault):
    p, s, w = fixture()
    kwargs = {}
    if fault == "split": s = replace(s, split="development")
    elif fault == "variant": kwargs["variant_name"] = "unlisted"
    elif fault == "digest": kwargs["selection_sha256"] = "unknown"
    elif fault == "windows": w = w[:20]
    else: p = replace(p, parents=p.parents + (ParentMetadata("S01_synthetic_C08", "C08"),))
    with pytest.raises(ValueError): export(p, s, w, **kwargs)


def test_elevation_layers_are_not_averaged_before_human_review():
    p, s, w = fixture()
    w = tuple(replace(x, range_valid=x.range_valid.copy()) for x in w)
    for x in w:
        x.range_valid[:, :, 0, 0] = x.range_valid[:, :, 8, 0]
    bundle = json.loads(export(p, s, w).bundle_bytes)
    for row in bundle["decisions"]:
        points = np.array(row["points_xyz_m"])
        assert len(points) == 10
        assert all(points[2 * h, 2] != points[2 * h + 1, 2] for h in range(5))
