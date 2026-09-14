"""One five-frame observation, blind judgment before construction reference.

No real source IO and no automatic label creation or training eligibility.
Reviewer identity is an assertion, not authentication of human participation.
"""
from copy import deepcopy
import math
import hashlib
import json

from mtare_topo.data.gse_structure_review_v1 import canonical_sha


def import_browser_review(bundle_bytes, reference_bytes, record):
    """Verify actual file-byte hashes before importing a completed browser record.

    File-byte binding avoids Python/JavaScript float serialization differences.
    This does not authenticate a human or authorize training.
    """
    def read(raw):
        if type(raw) is not bytes or len(raw)>128*1024*1024:
            raise ValueError("bounded source file bytes required")
        def pairs(items):
            result={}
            for k,v in items:
                if k in result:raise ValueError("duplicate JSON field")
                result[k]=v
            return result
        def invalid(v):raise ValueError("nonfinite JSON constant")
        return json.loads(raw,object_pairs_hook=pairs,parse_constant=invalid)
    if type(record) is not dict or set(record)!={"schema","bundle_file_sha256","reviewer_assertion",
            "blind_annotation","reference_file_sha256","comparison_notes","automatic_training_eligibility"}:
        raise ValueError("closed browser review record required")
    if record["schema"]!="gse_surface_browser_review_v1" or record["automatic_training_eligibility"] is not False:
        raise ValueError("browser schema or eligibility mismatch")
    b=read(bundle_bytes);r=read(reference_bytes)
    bh=hashlib.sha256(bundle_bytes).hexdigest();rh=hashlib.sha256(reference_bytes).hexdigest()
    if record["bundle_file_sha256"]!=bh or record["reference_file_sha256"]!=rh:
        raise ValueError("review source file hash mismatch")
    if type(r) is not dict or set(r)!={"bundle_file_sha256","construction_reference"} or r["bundle_file_sha256"]!=bh:
        raise ValueError("reference bound to different observation file")
    session=SurfaceReview(b,reviewer=record["reviewer_assertion"])
    session.commit_blind(record["blind_annotation"])
    session.reveal(dict(bundle_sha256=canonical_sha(b),construction_reference=r["construction_reference"]))
    session.finish(record["comparison_notes"])
    return {**session.export(),"bundle_file_sha256":bh,"reference_file_sha256":rh}


def validate_bundle(bundle):
    if type(bundle) is not dict or set(bundle)!={"schema","observation_id","coordinate_frame",
            "source_frame_indices","points_xyz_m","point_history_slots"}:
        raise ValueError("closed observation-only bundle required")
    if bundle["schema"]!="gse_surface_review_bundle_v1" or bundle["coordinate_frame"]!="current_sensor_m":
        raise ValueError("single-observation sensor schema required")
    if type(bundle["observation_id"]) is not str or not bundle["observation_id"].strip():
        raise ValueError("opaque observation identity required")
    frames=bundle["source_frame_indices"]
    if type(frames) is not list or len(frames)!=5 or any(type(x) is not int or x<0 for x in frames) or sorted(set(frames))!=frames:
        raise ValueError("exactly five causal source indices required")
    points=bundle["points_xyz_m"];slots=bundle["point_history_slots"]
    if type(points) is not list or len(points)>57600 or type(slots) is not list or len(slots)!=len(points):
        raise ValueError("aligned five-frame point population required")
    for p,s in zip(points,slots):
        if type(p) is not list or len(p)!=3 or any(type(x) not in (float,int) or not math.isfinite(x) for x in p):
            raise ValueError("finite XYZ required")
        if type(s) is not int or s not in range(5):raise ValueError("point history slot required")
    return deepcopy(bundle)


class SurfaceReview:
    def __init__(self,bundle,*,reviewer):
        if type(reviewer) is not str or not reviewer.strip():raise ValueError("reviewer assertion required")
        self._bundle=validate_bundle(bundle);self._reviewer=reviewer
        self._blind=None;self._reference=None;self._notes=None

    def commit_blind(self,annotation):
        if self._blind is not None:raise ValueError("blind judgment already locked")
        from mtare_topo.teacher.gse_surface_target_adapter_v1 import observed_targets
        observed_targets([annotation]) # independent anchors/openings, explicit UNKNOWN
        if annotation["source_frame_indices"]!=self._bundle["source_frame_indices"]:
            raise ValueError("annotation source differs from observation")
        self._blind=deepcopy(annotation)

    def reveal(self,reference):
        if self._blind is None or self._reference is not None:raise ValueError("lock blind judgment before one reference reveal")
        if type(reference) is not dict or set(reference)!={"bundle_sha256","construction_reference"}:
            raise ValueError("bound separate construction reference required")
        if reference["bundle_sha256"]!=canonical_sha(self._bundle):raise ValueError("wrong reference binding")
        canonical_sha(reference) # reject non-JSON/nonfinite references
        self._reference=deepcopy(reference)

    def finish(self,notes):
        if self._reference is None or self._notes is not None or type(notes) is not str or not notes.strip():
            raise ValueError("one explicit reference comparison required")
        self._notes=notes

    def export(self):
        return deepcopy(dict(schema="gse_surface_review_record_v1",bundle_sha256=canonical_sha(self._bundle),
            reviewer_assertion=self._reviewer,blind_annotation=self._blind,
            reference_sha256=None if self._reference is None else canonical_sha(self._reference),
            comparison_notes=self._notes,review_protocol_complete=self._notes is not None,
            human_identity_authenticated=False,automatic_training_eligibility=False))
