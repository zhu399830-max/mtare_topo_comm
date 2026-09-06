"""Strict blind-review interchange. No source reads or automatic annotation.

The reviewer's identity is an assertion, not authentication. Hashes make
accidental changes detectable; they do not prove a human made the judgments.
"""
from copy import deepcopy
import hashlib
import json
import math


SCHEMA = "gse_structure_blind_bundle_v1"


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
        separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _keys(value, names):
    if type(value) is not dict or set(value) != set(names):
        raise ValueError("unexpected or missing fields; teacher metadata is not a blind input")


def _text(value):
    if type(value) is not str or not value.strip():
        raise ValueError("nonempty text required")


def _integer(value):
    if type(value) is not int or value < 0:
        raise ValueError("nonnegative integer required")


def _xyz(value):
    if type(value) is not list or len(value) != 3 or any(
        isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) for x in value
    ):
        raise ValueError("finite XYZ list required")


def validate_blind_bundle(bundle):
    _keys(bundle, ("schema", "bundle_id", "coordinate_frame", "decisions"))
    if bundle["schema"] != SCHEMA or bundle["coordinate_frame"] != "current_sensor_m":
        raise ValueError("unsupported blind bundle schema/frame")
    _text(bundle["bundle_id"])
    if type(bundle["decisions"]) is not list or len(bundle["decisions"]) != 21:
        raise ValueError("exactly21 decisions required; no padding")
    previous = None
    previous_source = None
    frame_to_order, order_to_frame = {}, {}
    for row in bundle["decisions"]:
        _keys(row, ("decision_index", "source_frame_keys", "source_order_indices", "points_xyz_m"))
        _integer(row["decision_index"])
        if previous is not None and row["decision_index"] != previous + 1:
            raise ValueError("decisions must be consecutive within the segment")
        previous = row["decision_index"]
        frames, order = row["source_frame_keys"], row["source_order_indices"]
        if type(frames) is not list or len(frames) != 5 or type(order) is not list or len(order) != 5:
            raise ValueError("five-frame source provenance required")
        for key in frames:
            _text(key)
        for i in order:
            _integer(i)
        if len(set(frames)) != 5 or any(a >= b for a, b in zip(order, order[1:])):
            raise ValueError("unique causal source frames required")
        if previous_source is not None and order[-1] <= previous_source:
            raise ValueError("current source order must increase across decisions")
        previous_source = order[-1]
        for key, index in zip(frames, order):
            if key in frame_to_order and frame_to_order[key] != index:
                raise ValueError("frame identity changes source order")
            if index in order_to_frame and order_to_frame[index] != key:
                raise ValueError("source order changes frame identity")
            frame_to_order[key], order_to_frame[index] = index, key
        if type(row["points_xyz_m"]) is not list or len(row["points_xyz_m"]) > 57600:
            raise ValueError("point population exceeds five16x720 scans")
        for point in row["points_xyz_m"]:
            _xyz(point)
    return deepcopy(bundle)


def validate_annotation(annotation):
    _keys(annotation, ("structures", "complete_regions", "unknown_regions", "notes"))
    if type(annotation["notes"]) is not str:
        raise ValueError("notes must be text")
    structures = annotation["structures"]
    if type(structures) is not list or len(structures) > 32:
        raise ValueError("structure capacity exceeded; do not truncate labels")
    ids = []
    portal_count = 0
    for item in structures:
        _keys(item, ("local_id", "event", "center_xyz_m", "openings", "evidence_note"))
        _integer(item["local_id"])
        ids.append(item["local_id"])
        if item["event"] not in ("corridor", "junction", "terminal", "unknown"):
            raise ValueError("unknown is not no-object")
        if item["center_xyz_m"] is not None:
            _xyz(item["center_xyz_m"])
        _text(item["evidence_note"])
        if type(item["openings"]) is not list:
            raise ValueError("openings must be explicit list")
        portal_count += len(item["openings"])
        for portal in item["openings"]:
            _keys(portal, ("position_xyz_m", "direction", "width_m", "height_m", "evidence_note"))
            for k in ("position_xyz_m", "direction"):
                if portal[k] is not None:
                    _xyz(portal[k])
            if portal["direction"] is not None and abs(math.hypot(*portal["direction"]) - 1) > 1e-5:
                raise ValueError("known direction must be unit vector")
            for k in ("width_m", "height_m"):
                v = portal[k]
                if v is not None and (isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v <= 0):
                    raise ValueError("dimensions positive or null")
            _text(portal["evidence_note"])
    if len(set(ids)) != len(ids) or portal_count > 64:
        raise ValueError("duplicate structure identity or portal capacity exceeded")
    boxes = {}
    for kind in ("complete_regions", "unknown_regions"):
        if type(annotation[kind]) is not list:
            raise ValueError("regions must be explicit lists")
        boxes[kind] = annotation[kind]
        for box in boxes[kind]:
            _keys(box, ("min_xyz_m", "max_xyz_m", "reason"))
            _xyz(box["min_xyz_m"]); _xyz(box["max_xyz_m"]); _text(box["reason"])
            if any(a >= b for a, b in zip(box["min_xyz_m"], box["max_xyz_m"])):
                raise ValueError("region must have positive extent")
    # Conflicting completeness/unknown declarations cannot generate background.
    for a in boxes["complete_regions"]:
        for b in boxes["unknown_regions"]:
            if all(max(a["min_xyz_m"][i], b["min_xyz_m"][i]) < min(a["max_xyz_m"][i], b["max_xyz_m"][i]) for i in range(3)):
                raise ValueError("complete and unknown regions overlap")
    return deepcopy(annotation)


class BlindReviewSession:
    """Append-only causal judgments, followed by separate reference notes.

    Interface core only; callers must separately validate a source manifest.
    No inference that reviewed regions cover the complete sensor volume.
    """
    def __init__(self, bundle, reviewer):
        self._bundle = validate_blind_bundle(bundle)
        _text(reviewer)
        self._reviewer = reviewer
        self._blind = []
        self._reference = None
        self._reference_notes = None

    def next_observation(self):
        if len(self._blind) == 21:
            return None
        return deepcopy(self._bundle["decisions"][len(self._blind)])

    def commit_blind(self, decision_index, annotation):
        observation = self.next_observation()
        if observation is None or type(decision_index) is not int or decision_index != observation["decision_index"]:
            raise ValueError("only next unseen decision may be committed")
        if self._reference is not None:
            raise ValueError("blind annotations locked before reference reveal")
        annotation = validate_annotation(annotation)
        self._blind.append({"decision_index": decision_index, "annotation": annotation,
            "observation_sha256": canonical_sha(observation),
            "previous_record_sha256": canonical_sha(self._blind[-1]) if self._blind else None})

    def reveal_reference(self, reference):
        if len(self._blind) != 21 or self._reference is not None:
            raise ValueError("complete blind review once before reference reveal")
        _keys(reference, ("schema", "bundle_id", "blind_bundle_sha256", "decisions"))
        if reference["schema"] != "gse_structure_reference_v1" or reference["bundle_id"] != self._bundle["bundle_id"] or reference["blind_bundle_sha256"] != canonical_sha(self._bundle):
            raise ValueError("reference belongs to another observation bundle")
        if type(reference["decisions"]) is not list or len(reference["decisions"]) != 21:
            raise ValueError("reference must cover exact decisions")
        for row, original in zip(reference["decisions"], self._bundle["decisions"]):
            _keys(row, ("decision_index", "reference_annotation"))
            if type(row["decision_index"]) is not int or row["decision_index"] != original["decision_index"]:
                raise ValueError("reference decision order mismatch")
            validate_annotation(row["reference_annotation"])
        self._reference = deepcopy(reference)
        return deepcopy(reference)

    def record_reference_notes(self, notes):
        if self._reference is None or self._reference_notes is not None:
            raise ValueError("reference notes may be committed once after reveal")
        _text(notes)
        self._reference_notes = notes

    def export(self):
        return deepcopy({"schema": "gse_structure_review_record_v1", "reviewer_assertion": self._reviewer,
            "bundle_id": self._bundle["bundle_id"], "bundle_sha256": canonical_sha(self._bundle),
            "blind_records": self._blind, "reference_sha256": canonical_sha(self._reference) if self._reference else None,
            "reference_notes": self._reference_notes, "blind_complete": len(self._blind) == 21,
            "review_complete": self._reference_notes is not None,
            "automatic_training_eligibility": False})


def import_browser_review(bundle_bytes, browser_record, *, reference_bytes=None):
    """Revalidate a downloaded record against exact separately supplied bytes.

    Does not open files, accept automatic training eligibility, or authenticate
    a reviewer. Source-manifest authorization remains a separate requirement.
    """
    _keys(browser_record, ("schema", "bundle_id", "blind_bundle_file_sha256",
        "reviewer_assertion", "blind_records", "reference_file_sha256",
        "reference_notes", "automatic_training_eligibility"))
    if browser_record["schema"] != "gse_structure_browser_review_v1" or browser_record["automatic_training_eligibility"] is not False:
        raise ValueError("browser output is not a training authorization")
    if type(bundle_bytes) is not bytes or hashlib.sha256(bundle_bytes).hexdigest() != browser_record["blind_bundle_file_sha256"]:
        raise ValueError("blind bundle bytes drift")

    def strict_json(data):
        def unique(pairs):
            result = {}
            for key, value in pairs:
                if key in result:
                    raise ValueError("duplicate JSON key")
                result[key] = value
            return result
        return json.loads(data, object_pairs_hook=unique)

    bundle = strict_json(bundle_bytes)
    session = BlindReviewSession(bundle, browser_record["reviewer_assertion"])
    if browser_record["bundle_id"] != bundle["bundle_id"] or type(browser_record["blind_records"]) is not list:
        raise ValueError("browser bundle identity/records mismatch")
    for row in browser_record["blind_records"]:
        _keys(row, ("decision_index", "annotation"))
        session.commit_blind(row["decision_index"], row["annotation"])
    if browser_record["reference_file_sha256"] is not None:
        if type(reference_bytes) is not bytes or hashlib.sha256(reference_bytes).hexdigest() != browser_record["reference_file_sha256"]:
            raise ValueError("reference bytes drift or missing")
        session.reveal_reference(strict_json(reference_bytes))
        if browser_record["reference_notes"] is not None:
            session.record_reference_notes(browser_record["reference_notes"])
    elif reference_bytes is not None or browser_record["reference_notes"] is not None:
        raise ValueError("reference not revealed; no reference notes allowed")
    return session.export()
