"""Nine-payload cached attribution reader: no model/checkpoint/training IO."""
from dataclasses import fields
import hashlib
import io
import math
from pathlib import Path

import numpy as np
import torch

from mtare_topo.data.gse_partial_training_inputs import load_training_inputs, _json, _require, _same
from mtare_topo.governance_assignment_attribution import SOURCE_NAMES, validate_assignment_attribution_card
from mtare_topo.governance_field_recovery import _digest, _relative_source
from mtare_topo.governance_partial_structure_training import SOURCE_NAMES as EXPORT_SOURCES
from mtare_topo.representation.gse_partial_structure_training import BRANCHES, PartialTrainingConfig, paired_schedule
from mtare_topo.representation.gse_region_queries import RegionPrediction, tokens_from_axes


FINAL_SOURCES = dict(zip(BRANCHES, ("gt_final", "predicted_final", "no_relations_final")))


def load_assignment_inputs(root, card):
    """Load cached final predictions and logs after new read-only authorization.

Old four exported inputs use their unchanged validator/reader exactly once.
Five additional payloads are byte-hashed first, so all nine source hashes are
checked before decoding any payload. Saved scoring is validated structurally;
the caller must replay the original evaluator to verify its actual assignment.
"""
    report = validate_assignment_attribution_card(card)
    _require(report.passed, "assignment attribution card rejected: " + "; ".join(report.errors))
    sources, embedded = card["sources"], card["training_card"]
    _require(isinstance(sources, dict) and set(sources) == set(SOURCE_NAMES), "exact nine sources required")
    _same({key: sources[key] for key in EXPORT_SOURCES}, embedded["sources"], "embedded four input bindings differ")
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
        data = paths[key].read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        _require(digest == sources[key]["sha256"], f"source SHA drift: {key}")
        payloads[key], reads[sources[key]["path"]] = data, digest
    data = load_training_inputs(root, embedded)
    reads.update(data["read_hashes"])
    _require(len(reads) == 9, "nine distinct payload reads required")
    _same(card["observation_count"], len(data["manifest"]), "attribution population drift")
    _same(card["selected_rows"], embedded["selected_rows"], "attribution selection drift")
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
            _require(a.shape == shape and a.dtype == np.dtype(dtype) and np.isfinite(a).all(), f"prediction array shape/dtype/finite drift: {name}")
        p = RegionPrediction(**{f.name: torch.from_numpy(arrays[f.name].copy()) for f in fields(RegionPrediction)})
        numerical = tokens_from_axes(data["axes"][key]).valid
        _require(torch.equal(p.query_supported, numerical) and torch.equal(p.member_supported, numerical[:, :, None] & numerical[:, None]), "prediction numerical support drift")
        _require(bool(((p.uncertainty >= 0) & (p.uncertainty <= 1)).all()), "prediction uncertainty range drift")
        scoring = {name: torch.from_numpy(arrays[name].copy()) for name in ("scored_member_mask", "unique_center_query")}
        match, scored = scoring["unique_center_query"], scoring["scored_member_mask"]
        _require(bool(((match >= -1) & (match < 64)).all()), "saved center query out of range")
        _require(not bool(((match >= 0) & ~target.center_valid).any()), "saved match on unknown center")
        expected_mask = torch.zeros_like(scored)
        for row in range(180):
            valid = match[row] >= 0
            ids = match[row, valid]
            _require(len(ids.unique()) == len(ids) and bool(p.query_supported[row, ids].all()), "nonunique/unsupported saved center query")
            if valid.any():
                expected_mask[row, valid] = target.member_valid[row, valid] & p.member_supported[row, ids]
        _require(torch.equal(scored, expected_mask), "saved member mask inconsistent with center/query support")
        predictions[branch], saved[branch] = p, scoring
    history = _json(payloads["history"])
    _require(isinstance(history, dict) and set(history) == set(BRANCHES), "exact three history branches required")
    settings = embedded["training"]
    _same(settings["steps_per_branch"], 300, "exact300 steps required")
    _same(settings["batch_size"], 18, "exact18 batch required")
    config = PartialTrainingConfig(settings["seed"], 300, 18, settings["lr"], settings["device"])
    expected_schedule = paired_schedule(180, config)
    loss_names = {"total", "center", "event", "membership", "presence", "uncertainty"}
    count_names = {"targets", "matched", "unmatched_targets", "unsupported_member_targets",
                   "unconfirmed_presence_targets", "center", "event", "membership",
                   "presence_positive", "presence_negative"}
    for branch in BRANCHES:
        rows = history[branch]
        _require(isinstance(rows, list) and len(rows) == 300, "exact300 history records required")
        for index, row in enumerate(rows):
            _require(isinstance(row, dict), "history row must be object")
            _same(row["step"], index + 1, "history step drift")
            _same(row["optimizer_steps"], index + 1, "history update count drift")
            _same(row["sample_indices"], expected_schedule[index], "shared deterministic schedule drift")
            _require(isinstance(row["loss"], dict) and set(row["loss"]) == loss_names, "history task losses missing")
            for value in (*row["loss"].values(), row["gradient_l2"]):
                _require(type(value) in (int, float) and math.isfinite(value) and value >= 0, "invalid history loss/gradient")
            _require(type(row["gradient_tensor_count"]) is int and row["gradient_tensor_count"] > 0, "missing history gradients")
            _require(isinstance(row["counts"], dict) and set(row["counts"]) == count_names
                     and all(type(v) is int and v >= 0 for v in row["counts"].values()), "invalid history counts")
            _require(isinstance(row["matches"], list), "history matches must be list")
            for item in row["matches"]:
                _require(isinstance(item, list) and len(item) == 3 and all(type(v) is int for v in item)
                    and 0 <= item[0] < 18 and 0 <= item[1] < 64 and 0 <= item[2] < data["targets"]["gt" if branch == "gt_axes" else "predicted"].centers_m.shape[1], "invalid loss-only history match")
            counts = row["counts"]
            _require(counts["matched"] == len(row["matches"]) == counts["presence_positive"]
                     and counts["presence_negative"] == 0
                     and counts["targets"] == counts["matched"] + counts["unmatched_targets"], "history partial matching accounting drift")
            _require(len({(r, q) for r, q, _ in row["matches"]}) == len(row["matches"])
                     and len({(r, t) for r, _, t in row["matches"]}) == len(row["matches"]), "duplicate history matching")
    last_batches = expected_schedule[-10:]
    _require(len(last_batches) == 10 and all(len(b) == 18 for b in last_batches)
             and sorted(sum(last_batches, [])) == list(range(180)), "last epoch must partition all180 once")
    training_summary = _json(payloads["training_summary"])
    _require(isinstance(training_summary, dict) and training_summary.get("status") == "PARTIAL_STRUCTURE_FIXED_BUDGET_COMPLETE"
             and training_summary.get("error") is None and training_summary.get("scientific_gate_pass") is False
             and training_summary.get("full_three_class_ready") is False, "completed partial training summary required")
    result = training_summary["result"]
    for key, count in (("optimizer_steps", 900), ("head_inference_windows", 1080), ("backbone_windows", 0), ("new_sensor_frames", 0)):
        _same(result[key], count, "original training count drift")
    _require(result.get("same_initial_state_verified") is True and result.get("same_schedule_verified") is True, "paired training verification missing")
    _require(set(result["evaluation"]) == {"initial", "final"}
             and all(set(result["evaluation"][stage]) == set(BRANCHES) for stage in ("initial", "final")), "all original evaluations required")
    return {"data": data, "predictions": predictions, "saved_scoring": saved,
            "training_summary": training_summary, "history": history,
            "last_epoch_batches": last_batches, "read_hashes": reads}
