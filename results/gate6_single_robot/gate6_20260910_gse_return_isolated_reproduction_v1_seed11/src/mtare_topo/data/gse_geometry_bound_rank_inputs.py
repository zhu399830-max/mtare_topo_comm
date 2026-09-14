"""Eight-payload final-cache reader: no history, model, or checkpoint reads."""
from dataclasses import fields
import hashlib
import io
from pathlib import Path

import numpy as np
import torch

from mtare_topo.data.gse_partial_training_inputs import load_training_inputs, _json, _require, _same
from mtare_topo.governance_field_recovery import _digest, _relative_source
from mtare_topo.governance_geometry_bound_rank import SOURCE_NAMES, validate_geometry_bound_rank_card
from mtare_topo.governance_partial_structure_training import SOURCE_NAMES as EXPORT_SOURCES
from mtare_topo.representation.gse_partial_structure_training import BRANCHES
from mtare_topo.representation.gse_region_queries import RegionPrediction, tokens_from_axes


FINAL_SOURCES = dict(zip(BRANCHES, ("gt_final", "predicted_final", "no_relations_final")))


def load_rank_inputs(root, card):
    """Load three final snapshots and original targets under new read-only scope.

    The four new payloads are SHA-checked before decoding; the old reader then
    verifies all four export bytes before its own decoding. No path is read
    twice. NPZ decoding uses the exact already-hashed bytes, not reopened files.
    Saved assignments are checked structurally, not certified optimal: the
    caller must reproduce the independent scorer before ranking diagnostics.
    """
    report = validate_geometry_bound_rank_card(card)
    _require(report.passed, "geometry-bound rank card rejected: " + "; ".join(report.errors))
    sources = card["sources"]
    _require(isinstance(sources, dict) and set(sources) == set(SOURCE_NAMES), "exact eight cached sources required")
    embedded = card["corrective_card"]["base_training_card"]
    _same({key: sources[key] for key in EXPORT_SOURCES}, embedded["sources"], "original four export bindings differ")
    root, paths = Path(root).resolve(), {}
    for key, record in sources.items():
        _require(isinstance(record, dict) and set(record) == {"path", "sha256"}, "explicit source record required")
        relative, digest = record["path"], record["sha256"]
        _require(_relative_source(relative) and _digest(digest) and Path(relative).name == SOURCE_NAMES[key], "source path/hash invalid")
        _require(card["sealed_sources"].get(relative) == digest, "source binding drift")
        path = (root / relative).resolve()
        _require(path.is_relative_to(root) and path not in paths.values(), "source escape/duplicate")
        paths[key] = path
    payloads, reads = {}, {}
    for key in sources:
        if key in EXPORT_SOURCES:
            continue
        contents = paths[key].read_bytes()
        digest = hashlib.sha256(contents).hexdigest()
        _require(digest == sources[key]["sha256"], "source SHA drift: " + key)
        payloads[key], reads[sources[key]["path"]] = contents, digest
    data = load_training_inputs(root, embedded)
    reads.update(data["read_hashes"])
    _require(len(reads) == 8, "eight distinct payload reads required")
    _same(card["observation_count"], len(data["manifest"]), "rank population drift")
    _same(card["selected_rows"], embedded["selected_rows"], "rank selection drift")
    summary = _json(payloads["corrective_summary"])
    _require(isinstance(summary, dict) and summary.get("status") == "GEOMETRY_BOUND_FIXED_BUDGET_COMPLETE"
             and summary.get("error") is None and summary.get("scientific_gate_pass") is False
             and summary.get("full_three_class_ready") is False, "completed non-scientific geometry-bound summary required")
    result = summary["result"]
    for key, count in (("optimizer_steps", 900), ("head_inference_windows", 1080), ("backbone_windows", 0), ("new_sensor_frames", 0)):
        _same(result[key], count, "corrective training count drift")
    _require(result.get("same_initial_state_verified") is True and result.get("same_schedule_verified") is True,
             "paired corrective training verification missing")
    _require(isinstance(result.get("evaluation"), dict) and set(result["evaluation"]) == {"initial", "final"}
             and all(isinstance(result["evaluation"][stage], dict) and set(result["evaluation"][stage]) == set(BRANCHES)
                     for stage in ("initial", "final")), "all corrective branch evaluations required")
    predictions, saved = {}, {}
    shapes = {"centers_m": ((180, 64, 3), np.float32), "event_logits": ((180, 64, 3), np.float32),
              "presence_logits": ((180, 64), np.float32), "membership_logits": ((180, 64, 64), np.float32),
              "uncertainty": ((180, 64), np.float32), "query_supported": ((180, 64), bool),
              "member_supported": ((180, 64, 64), bool)}
    for branch, source in FINAL_SOURCES.items():
        key = "gt" if branch == "gt_axes" else "predicted"
        target = data["targets"][key]
        m = target.centers_m.shape[1]
        schema = {**shapes, "scored_member_mask": ((180, m, 64), bool), "unique_center_query": ((180, m), np.int64)}
        with np.load(io.BytesIO(payloads[source]), allow_pickle=False) as archive:
            _require(len(archive.files) == 9 and set(archive.files) == set(schema), "exact nine prediction/scoring fields required")
            arrays = {name: archive[name] for name in archive.files}
        for name, (shape, dtype) in schema.items():
            a = arrays[name]
            _require(a.shape == shape and a.dtype == np.dtype(dtype) and np.isfinite(a).all(),
                     "prediction array shape/dtype/finite drift: " + name)
        p = RegionPrediction(**{f.name: torch.from_numpy(arrays[f.name].copy()) for f in fields(RegionPrediction)})
        numerical = tokens_from_axes(data["axes"][key]).valid
        _require(torch.equal(p.query_supported, numerical)
                 and torch.equal(p.member_supported, numerical[:, :, None] & numerical[:, None]), "prediction numerical support drift")
        _require(bool(((p.uncertainty >= 0) & (p.uncertainty <= 1)).all()), "prediction uncertainty range drift")
        scoring = {name: torch.from_numpy(arrays[name].copy()) for name in ("scored_member_mask", "unique_center_query")}
        match, scored = scoring["unique_center_query"], scoring["scored_member_mask"]
        _require(bool(((match >= -1) & (match < 64)).all()), "saved center query out of range")
        _require(not bool(((match >= 0) & ~target.center_valid).any()), "saved match on unknown center")
        expected = torch.zeros_like(scored)
        for row in range(180):
            valid = match[row] >= 0
            ids = match[row, valid]
            _require(len(ids.unique()) == len(ids) and bool(p.query_supported[row, ids].all()),
                     "nonunique/unsupported saved center query")
            if valid.any(): expected[row, valid] = target.member_valid[row, valid] & p.member_supported[row, ids]
        _require(torch.equal(scored, expected), "saved member mask inconsistent with center/query support")
        predictions[branch], saved[branch] = p, scoring
    return dict(data=data, predictions=predictions, saved_scoring=saved,
                training_summary=summary, read_hashes=reads)
