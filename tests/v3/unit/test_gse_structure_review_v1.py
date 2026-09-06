from copy import deepcopy
import hashlib
import json
import subprocess
from pathlib import Path
import pytest
from mtare_topo.data.gse_structure_review_v1 import (
    BlindReviewSession, canonical_sha, validate_blind_bundle, validate_annotation, import_browser_review,
)


def bundle():
    return {"schema": "gse_structure_blind_bundle_v1", "bundle_id": "opaque-001",
        "coordinate_frame": "current_sensor_m", "decisions": [
            {"decision_index": i, "source_frame_keys": [f"f-{j}" for j in range(i, i+5)],
             "source_order_indices": list(range(i, i+5)), "points_xyz_m": [[1., 2., 3.]]} for i in range(21)]}


def annotation():
    return {"structures": [], "complete_regions": [], "unknown_regions": [], "notes": "未确认完整背景"}


def reference(b):
    return {"schema": "gse_structure_reference_v1", "bundle_id": b["bundle_id"],
        "blind_bundle_sha256": canonical_sha(b), "decisions": [
            {"decision_index": i, "reference_annotation": annotation()} for i in range(21)]}


def test_future_and_teacher_reveal_require_blind_commits():
    b = bundle(); s = BlindReviewSession(b, "human-assertion")
    with pytest.raises(ValueError): s.reveal_reference(reference(b))
    with pytest.raises(ValueError): s.commit_blind(1, annotation())
    for i in range(21):
        assert s.next_observation()["decision_index"] == i
        s.commit_blind(i, annotation())
    before = s.export()["blind_records"]
    s.reveal_reference(reference(b))
    s.record_reference_notes("观察先于参考；不回填隐藏结构")
    assert s.export()["blind_records"] == before
    assert s.export()["review_complete"]
    assert not s.export()["automatic_training_eligibility"]
    with pytest.raises(ValueError): s.commit_blind(0, annotation())
    with pytest.raises(ValueError): s.record_reference_notes("edit")


@pytest.mark.parametrize("key", ["teacher", "world", "split", "node_id", "predictions"])
def test_blind_bundle_has_no_teacher_or_world_metadata(key):
    b = bundle(); b[key] = "hidden"
    with pytest.raises(ValueError): validate_blind_bundle(b)


def test_copies_prevent_backfill():
    b = bundle(); s = BlindReviewSession(b, "person")
    a = annotation(); s.commit_blind(0, a)
    a["notes"] = "future"; b["decisions"][0]["points_xyz_m"][0][0] = 999
    result = s.export(); result["blind_records"][0]["annotation"]["notes"] = "overwrite"
    assert s.export()["blind_records"][0]["annotation"]["notes"] == "未确认完整背景"


@pytest.mark.parametrize("mutation", [
    lambda b: b["decisions"].pop(),
    lambda b: b["decisions"][1].update(decision_index=3),
    lambda b: b["decisions"][0].update(source_order_indices=[0, 2, 1, 3, 4]),
    lambda b: b["decisions"][0].update(points_xyz_m=[[float("nan"), 0, 0]]),
    lambda b: b["decisions"][0].update(teacher=[]),
])
def test_invalid_sources_fail(mutation):
    b = bundle(); mutation(b)
    with pytest.raises(ValueError): validate_blind_bundle(b)


def test_unknown_not_background_and_overlapping_complete_boxes_rejected():
    a = annotation()
    box = {"min_xyz_m": [0, 0, 0], "max_xyz_m": [1, 1, 1], "reason": "reviewed"}
    a["complete_regions"] = [box]; a["unknown_regions"] = [deepcopy(box)]
    with pytest.raises(ValueError, match="overlap"): validate_annotation(a)


def test_cross_bundle_reference_rejected():
    b = bundle(); s = BlindReviewSession(b, "person")
    for i in range(21): s.commit_blind(i, annotation())
    ref = reference(b); ref["blind_bundle_sha256"] = "0" * 64
    with pytest.raises(ValueError): s.reveal_reference(ref)


def test_no_fabricated_dimension_for_annotation():
    a = annotation()
    a["structures"] = [{"local_id": 0, "event": "junction", "center_xyz_m": None,
        "evidence_note": "可见分叉，中心未知", "openings": [
        {"position_xyz_m": None, "direction": [1, 0, 0], "width_m": None, "height_m": None, "evidence_note": "方向可见"}]}]
    assert validate_annotation(a)["structures"][0]["openings"][0]["width_m"] is None


def test_actual_browser_record_roundtrip_to_authoritative_validator():
    b = bundle(); bb = json.dumps(b, ensure_ascii=False).encode()
    rr = json.dumps(reference(b), ensure_ascii=False).encode()
    module = Path(__file__).resolve().parents[3] / "tools/v3/review/gse_structure_review.js"
    script = """
const {BrowserReview,emptyAnnotation}=require(process.argv[1]);
const input=JSON.parse(require('node:fs').readFileSync(0,'utf8'));
const s=new BrowserReview(JSON.parse(input.bundle),input.bsha,'person');
for(let i=0;i<21;i++)s.commit(i,emptyAnnotation());
s.reveal(JSON.parse(input.reference),input.rsha);s.finish('reviewed without backfill');
process.stdout.write(JSON.stringify(s.export()));
"""
    result = subprocess.run(["node", "-e", script, str(module)], input=json.dumps({
        "bundle": bb.decode(), "reference": rr.decode(), "bsha": hashlib.sha256(bb).hexdigest(),
        "rsha": hashlib.sha256(rr).hexdigest()}), text=True, capture_output=True, check=True)
    browser = json.loads(result.stdout)
    checked = import_browser_review(bb, browser, reference_bytes=rr)
    assert checked["review_complete"] and len(checked["blind_records"]) == 21
    assert checked["automatic_training_eligibility"] is False
    with pytest.raises(ValueError, match="bytes drift"):
        import_browser_review(bb + b" ", browser, reference_bytes=rr)
    browser["blind_records"][0]["decision_index"] = 1
    with pytest.raises(ValueError, match="next unseen"):
        import_browser_review(bb, browser, reference_bytes=rr)
