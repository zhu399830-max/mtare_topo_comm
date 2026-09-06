"""Synthetic browser/Python import boundary; no source dataset or human labels."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess

import pytest

import mtare_topo.data.gse_structure_review_v1 as review
from tests.v3.unit.test_gse_structure_review_v1 import bundle, annotation, reference


JS = Path(__file__).resolve().parents[3] / "tools/v3/review/gse_structure_review.js"


def payload():
    b = bundle(); raw = json.dumps(b, ensure_ascii=False).encode()
    record = {"schema": "gse_structure_browser_review_v1", "bundle_id": b["bundle_id"],
        "blind_bundle_file_sha256": hashlib.sha256(raw).hexdigest(), "reviewer_assertion": "synthetic-person",
        "blind_records": [{"decision_index": i, "annotation": annotation()} for i in range(21)],
        "reference_file_sha256": None, "reference_notes": None, "automatic_training_eligibility": False}
    return b, raw, record


def browser_reveal(b, raw, ref):
    script = """
const {BrowserReview,emptyAnnotation}=require(process.argv[1]);
const x=JSON.parse(require('node:fs').readFileSync(0,'utf8'));
const s=new BrowserReview(x.b,x.hash,'synthetic-person');
for(let i=0;i<21;i++)s.commit(i,emptyAnnotation());
try{s.reveal(x.ref,'b'.repeat(64));process.stdout.write(JSON.stringify({rejected:false,exposed:s.reference!==null}));}
catch(e){process.stdout.write(JSON.stringify({rejected:true,exposed:s.reference!==null}));}
"""
    result = subprocess.run(["node", "-e", script, str(JS)], input=json.dumps({
        "b": b, "hash": hashlib.sha256(raw).hexdigest(), "ref": ref}), text=True,
        capture_output=True, check=True)
    return json.loads(result.stdout)


def test_browser_must_reject_reference_for_different_content_before_exposure():
    b, raw, _ = payload()
    other = deepcopy(b); other["decisions"][0]["points_xyz_m"][0][0] = 999
    ref = reference(other)
    assert browser_reveal(b, raw, ref) == {"rejected": True, "exposed": False}


@pytest.mark.parametrize("mutation", [
    lambda ref: ref.pop("blind_bundle_sha256"),
    lambda ref: ref.update(extra_teacher_metadata="unexpected"),
    lambda ref: ref["decisions"][0].update(extra_field="unexpected"),
    lambda ref: ref["decisions"][0]["reference_annotation"].update(structures="not-a-list"),
])
def test_reference_schema_mismatch_cannot_be_seen_and_locked(mutation):
    b, raw, _ = payload(); ref = reference(b); mutation(ref)
    assert browser_reveal(b, raw, ref) == {"rejected": True, "exposed": False}


def test_import_oversize_bundle_stops_before_decode_or_copy(monkeypatch):
    _, _, record = payload()
    raw = b" " * (128 * 1024 * 1024 + 1)
    record["blind_bundle_file_sha256"] = hashlib.sha256(raw).hexdigest()
    def forbidden(*args, **kwargs): raise AssertionError("oversize decoded before rejection")
    monkeypatch.setattr(review.json, "loads", forbidden)
    with pytest.raises(ValueError, match="(?i)(size|large|128|limit)"):
        review.import_browser_review(raw, record)


def test_import_oversize_reference_stops_before_any_decode(monkeypatch):
    _, raw, record = payload()
    ref = b" " * (128 * 1024 * 1024 + 1)
    record["reference_file_sha256"] = hashlib.sha256(ref).hexdigest()
    def forbidden(*args, **kwargs): raise AssertionError("oversize reference accepted until decode")
    monkeypatch.setattr(review.json, "loads", forbidden)
    with pytest.raises(ValueError, match="(?i)(size|large|128|limit)"):
        review.import_browser_review(raw, record, reference_bytes=ref)


def test_import_deep_copies_records_and_never_grants_automatic_training():
    _, raw, record = payload()
    result = review.import_browser_review(raw, record)
    record["blind_records"][0]["annotation"]["notes"] = "external rewrite"
    assert result["blind_records"][0]["annotation"]["notes"] != "external rewrite"
    assert result["automatic_training_eligibility"] is False and not result["review_complete"]


@pytest.mark.parametrize("field", ["reference_notes", "reference_file_sha256", "blind_records"])
def test_missing_browser_fields_refused(field):
    _, raw, record = payload(); record.pop(field)
    with pytest.raises(ValueError): review.import_browser_review(raw, record)


def test_notes_cannot_exist_without_exact_reference_bytes_and_prior_full_blind():
    b, raw, record = payload()
    record["reference_notes"] = "claimed checked"
    with pytest.raises(ValueError): review.import_browser_review(raw, record)
    ref = json.dumps(reference(b)).encode()
    record["reference_file_sha256"] = hashlib.sha256(ref).hexdigest()
    with pytest.raises(ValueError): review.import_browser_review(raw, record)
    record["blind_records"].pop()
    with pytest.raises(ValueError): review.import_browser_review(raw, record, reference_bytes=ref)


@pytest.mark.parametrize("number", [1, 1.0, -0.0, 1e-7, 1e20, 1.2345678901234567])
def test_real_browser_float_json_format_does_not_replace_exact_byte_identity(number):
    b, _, record = payload(); b["decisions"][0]["points_xyz_m"][0][0] = number
    raw = json.dumps(b, ensure_ascii=False).encode()
    ref = reference(b)
    assert browser_reveal(b, raw, ref) == {"rejected": False, "exposed": True}
    reference_raw = json.dumps(ref, ensure_ascii=False).encode()
    record["blind_bundle_file_sha256"] = hashlib.sha256(raw).hexdigest()
    record["reference_file_sha256"] = hashlib.sha256(reference_raw).hexdigest()
    record["reference_notes"] = "synthetic float encoding boundary"
    result = review.import_browser_review(raw, record, reference_bytes=reference_raw)
    assert result["review_complete"] and result["automatic_training_eligibility"] is False


def test_int_and_float_equal_browser_numbers_still_require_their_exact_files():
    b, raw, _ = payload()  # first coordinate 1.0
    integer = deepcopy(b); integer["decisions"][0]["points_xyz_m"][0][0] = 1
    assert integer == b  # numeric equality is NOT provenance equality
    assert review.canonical_sha(integer) != review.canonical_sha(b)
    ref = reference(integer)
    assert browser_reveal(b, raw, ref) == {"rejected": True, "exposed": False}


def test_import_requires_both_canonical_and_actual_file_digests():
    b, raw, record = payload()
    for field in ("blind_bundle_sha256", "blind_bundle_file_sha256"):
        ref = reference(b); ref[field] = "0" * 64
        reference_raw = json.dumps(ref).encode()
        record["reference_file_sha256"] = hashlib.sha256(reference_raw).hexdigest()
        with pytest.raises(ValueError): review.import_browser_review(raw, record, reference_bytes=reference_raw)


def test_python_and_browser_expose_same_per_file_byte_limit():
    result = subprocess.run(["node", "-e", "process.stdout.write(String(require(process.argv[1]).MAX_REVIEW_FILE_BYTES))", str(JS)],
                            text=True, capture_output=True, check=True)
    assert review.MAX_REVIEW_FILE_BYTES == int(result.stdout) == 128 * 1024 * 1024
