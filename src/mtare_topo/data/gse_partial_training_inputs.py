"""Read only four approved exported payloads; identities never become inputs."""
from dataclasses import fields
import hashlib
import io
import json
from pathlib import Path

import numpy as np
import torch

from mtare_topo.governance_field_recovery import TASK, _digest, _relative_source
from mtare_topo.governance_partial_structure_training import SOURCE_NAMES, validate_partial_structure_training_card
from mtare_topo.representation.gse_region_queries import RegionTargets, tokens_from_axes


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _same(actual, expected, message):
    # JSON counts must not silently accept float/bool aliases of integers.
    _require(type(actual) is type(expected), message)
    if isinstance(expected, dict):
        _require(set(actual) == set(expected), message)
        for key in expected:
            _same(actual[key], expected[key], message)
    elif isinstance(expected, list):
        _require(len(actual) == len(expected), message)
        for x, y in zip(actual, expected):
            _same(x, y, message)
    else:
        _require(actual == expected, message)


def _json(data):
    return json.loads(data, parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite JSON")))


def load_training_inputs(project_root, card):
    """Return detached CPU axes and loss-only targets, after complete validation.

Does not read the referenced historical data, checkpoints, export seal, or
teacher. Formal execution separately verifies those frozen metadata references.
"""
    report = validate_partial_structure_training_card(card)
    _require(report.passed, "training card rejected: " + "; ".join(report.errors))
    sources = card["sources"]
    _require(isinstance(sources, dict) and set(sources) == set(SOURCE_NAMES), "exact four sources required")
    root, paths = Path(project_root).resolve(), {}
    for key, record in sources.items():
        _require(isinstance(record, dict) and set(record) == {"path", "sha256"}, "explicit source record required")
        relative, digest = record["path"], record["sha256"]
        _require(_relative_source(relative) and _digest(digest) and Path(relative).name == SOURCE_NAMES[key], "invalid source path/hash")
        path = (root / relative).resolve()
        _require(path.is_relative_to(root) and path not in paths.values(), "source escape or duplicate")
        _require(card["sealed_sources"].get(relative) == digest, "source binding drift")
        paths[key] = path
    payloads, reads = {}, {}
    for key, path in paths.items():
        data = path.read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        _require(digest == sources[key]["sha256"], f"source SHA drift: {key}")
        payloads[key] = data
        reads[sources[key]["path"]] = digest
    manifest, transport, summary = (_json(payloads[key]) for key in ("manifest", "target_transport", "summary"))
    _require(isinstance(manifest, list) and len(manifest) == 180, "exact180 manifest required")
    _require(isinstance(transport, dict) and set(transport) == {"gt", "predicted"}, "two transport branches required")
    _require(isinstance(summary, dict) and summary.get("status") == "PARTIAL_STRUCTURE_INPUTS_EXPORTED"
             and summary.get("error") is None and summary.get("scientific_gate_pass") is False, "completed non-scientific export required")
    result = summary["result"]
    for field in ("new_sensor_frames", "model_inference", "optimizer_steps"):
        _same(result[field], 0, "unexpected export computation")
    _require(result.get("capacity_ready") is False and result.get("full_three_class_ready") is False, "partial-only export required")
    selected, seen, source_seen, frames, parent_counts = [], set(), set(), set(), {}
    for row in manifest:
        _require(isinstance(row, dict), "manifest row must be object")
        task, index, source = (row.get(k) for k in ("task", "row_index", "source_global_sequence_index"))
        _require(isinstance(task, str) and TASK.fullmatch(task) and type(index) is int and index >= 0
                 and type(source) is int and source >= 0, "invalid C01 row identity")
        _require((task, index) not in seen and (task, source) not in source_seen, "duplicate row/source identity")
        seen.add((task, index)); source_seen.add((task, source))
        ff = row.get("frame_rows")
        _require(isinstance(ff, list) and len(ff) == 5 and all(type(f) is int and f >= 0 for f in ff)
                 and all(a < b for a, b in zip(ff, ff[1:])), "invalid five causal frames")
        frames.update((task, f) for f in ff)
        parent_counts[task] = parent_counts.get(task, 0) + 1
        selected.append({k: row[k] for k in ("task", "row_index", "source_global_sequence_index")})
    _same(selected, card["selected_rows"], "selected rows/order drift")
    _same(sorted(parent_counts), card["tasks"], "parent task drift")
    _require(len(parent_counts) == 10 and set(parent_counts.values()) == {18} and len(frames) == 900, "exact180/10/900 population required")
    expected_keys = {"gt_axes", "predicted_axes", "loss_only_gt_mask"}
    for branch in ("gt", "predicted"):
        expected_keys.update(branch + "__" + f.name for f in fields(RegionTargets))
        expected_keys.update(branch + "__" + f for f in ("teacher_direction", "direction_valid", "input_direction_valid"))
    with np.load(io.BytesIO(payloads["inputs"]), allow_pickle=False) as archive:
        _require(len(archive.files) == 23 and set(archive.files) == expected_keys, "exact23 NPZ fields required")
        arrays = {key: archive[key] for key in archive.files}
    def array(key, shape, dtype):
        value = arrays[key]
        _require(value.shape == shape and value.dtype == np.dtype(dtype), f"array shape/dtype drift: {key}")
        _require(np.isfinite(value).all(), f"nonfinite array: {key}")
        return value
    axes = {branch: torch.from_numpy(array(branch + "_axes", (180, 32, 3, 3), np.float32).copy()) for branch in ("gt", "predicted")}
    mask = array("loss_only_gt_mask", (180, 32), bool)
    _require(int(mask.sum()) == 1452 and mask.any(1).all() and np.all(arrays["gt_axes"][~mask] == 0), "GT mask/padding drift")
    for i, row in enumerate(manifest):
        _same(row["visible_fragments"], int(mask[i].sum()), "per-row fragment drift")
    original = {"observations": 180, "parents": 10, "unique_frames": 900, "visible_fragments": 1452, **card["raw_teacher_counts"]}
    _same(result["original_counts"], original, "original summary/card counts drift")
    for field, value in (("observation_count", 180), ("parent_count", 10), ("unique_source_frame_count", 900), ("visible_fragment_count", 1452)):
        _same(card[field], value, "card population drift")
    targets = {}
    for branch in ("gt", "predicted"):
        prefix = branch + "__"
        center_array = arrays[prefix + "centers_m"]
        _require(center_array.ndim == 3 and center_array.shape[0] == 180 and center_array.shape[-1] == 3, "center shape drift")
        m = center_array.shape[1]
        schema = {"centers_m": ((180, m, 3), np.float32), "center_valid": ((180, m), bool),
            "events": ((180, m), np.int64), "event_valid": ((180, m), bool),
            "members": ((180, m, 64), np.float32), "member_valid": ((180, m, 64), bool),
            "label_complete": ((180,), bool)}
        t = {key: array(prefix + key, shape, dtype) for key, (shape, dtype) in schema.items()}
        _require(not t["label_complete"].any(), "complete labels forbidden")
        _require(np.isin(t["events"][t["event_valid"]], [0, 1]).all()
                 and np.all(t["events"][~t["event_valid"]] == -1), "event value/mask drift")
        _require(np.isin(t["members"], [0, 1]).all() and np.all(t["members"][~t["member_valid"]] == 0), "member value/mask drift")
        mapping = array(prefix + "teacher_direction", (180, 64), np.int64)
        direction_valid = array(prefix + "direction_valid", (180, 64), bool)
        input_valid = array(prefix + "input_direction_valid", (180, 64), bool)
        _require(np.all((mapping >= -1) & (mapping < 64)) and np.array_equal(direction_valid, mapping >= 0), "direction range/valid drift")
        _require(np.array_equal(input_valid, tokens_from_axes(axes[branch]).valid.numpy()), "numerical input direction drift")
        _require(not (t["member_valid"] & ~direction_valid[:, None]).any(), "membership on UNKNOWN correspondence")
        records = transport[branch]
        _require(isinstance(records, list) and len(records) == 180, "transport population drift")
        recomputed = []
        for i, record in enumerate(records):
            for key, value in manifest[i].items():
                _same(record.get(key), value, "transport row/frame identity drift")
            known_directions = mapping[i, direction_valid[i]]
            _require(len(np.unique(known_directions)) == len(known_directions)
                     and mask[i, known_directions // 2].all(), "noninjective/inactive direction mapping")
            _same(record["teacher_direction"], mapping[i].tolist(), "transport direction mapping drift")
            ledger = record["direction_alignment"]
            _same(ledger["active_teacher_slots"], int(mask[i].sum()), "alignment population drift")
            _same(ledger["selected_pairs"], int(mask[i].sum()), "alignment population drift")
            _same(ledger["unselected_prediction_slots"], 32 - int(mask[i].sum()), "alignment population drift")
            _same(ledger["transferred_directions"], int(direction_valid[i].sum()), "alignment transfer drift")
            _same(ledger["unique_direction_pairs"] * 2, int(direction_valid[i].sum()), "alignment pair drift")
            reasons = record["direction_reasons"]
            _require(isinstance(reasons, list) and len(reasons) == 32, "direction reasons drift")
            _require(set(reasons) <= {"UNSELECTED_OR_ALTERNATIVE", "AMBIGUOUS_GLOBAL_ASSIGNMENT",
                                     "AMBIGUOUS_AXIS_ORIENTATION", "UNIQUE_GEOMETRY_MATCH_NOT_CONFIDENCE"}, "unknown direction reason")
            for slot, reason in enumerate(reasons):
                pair = mapping[i, 2 * slot:2 * slot + 2]
                if reason == "UNIQUE_GEOMETRY_MATCH_NOT_CONFIDENCE":
                    _require(np.all(pair >= 0) and pair[0] // 2 == pair[1] // 2 and pair[0] != pair[1], "direction endpoint pair drift")
                else:
                    _require(np.all(pair == -1), "unknown direction reason has assignment")
            for key, reason in (("ambiguous_assignment_pairs", "AMBIGUOUS_GLOBAL_ASSIGNMENT"),
                                ("ambiguous_orientation_pairs", "AMBIGUOUS_AXIS_ORIENTATION"),
                                ("unique_direction_pairs", "UNIQUE_GEOMETRY_MATCH_NOT_CONFIDENCE")):
                _same(ledger[key], reasons.count(reason), "direction reason ledger drift")
            _same(ledger["ambiguous_assignment_pairs"] + ledger["ambiguous_orientation_pairs"] + ledger["unique_direction_pairs"],
                  ledger["selected_pairs"], "selected alignment accounting drift")
            usable = t["member_valid"][i] & input_valid[i][None]
            known = t["center_valid"][i] | t["event_valid"][i] | t["member_valid"][i].any(-1)
            eligible = t["center_valid"][i] | t["event_valid"][i] | (usable & (t["members"][i] == 1)).any(-1)
            rr = {"numeric_input_valid_queries": int(input_valid[i].sum()), "known_region_targets": int(known.sum()),
                  "eligible_region_targets": int(eligible.sum()), "region_cardinality_unmatched_lower_bound": max(0, int(eligible.sum()) - int(input_valid[i].sum())),
                  "usable_member_positive": int((usable & (t["members"][i] == 1)).sum()),
                  "usable_member_negative": int((usable & (t["members"][i] == 0)).sum()),
                  "unsupported_transported_members": int((t["member_valid"][i] & ~input_valid[i][None]).sum())}
            for key, value in rr.items():
                _same(record[key], value, "per-row usable ledger drift")
            bridge = record["target_transport"]
            _require(type(bridge["regions"]) is int and 0 <= bridge["regions"] <= m, "region padding ledger drift")
            for key, value in (("center_labels", int(t["center_valid"][i].sum())), ("event_labels", int(t["event_valid"][i].sum()))):
                _same(bridge[key], value, "per-row center/event ledger drift")
            for sign, label in (("positive", 1), ("negative", 0)):
                transferred = int((t["member_valid"][i] & (t["members"][i] == label)).sum())
                _same(bridge["transferred_member_" + sign], transferred, "per-row member ledger drift")
                _same(bridge["original_member_" + sign] - transferred,
                      bridge["unknown_correspondence_member_" + sign], "UNKNOWN accounting drift")
                _require(bridge["unknown_correspondence_member_" + sign] >= 0, "negative UNKNOWN population")
            for key in ("center_valid", "event_valid", "member_valid"):
                _require(not t[key][i, bridge["regions"]:].any(), "valid labels in padded region")
            recomputed.append(rr)
        effective = {key: sum(r[key] for r in recomputed) for key in recomputed[0]}
        for field in ("direction_alignment", "target_transport"):
            effective[field] = {key: sum(r[field][key] for r in records) for key in records[0][field]}
        effective["max_valid_centers_per_observation"] = int(t["center_valid"].sum(1).max())
        effective["all_labels_incomplete"] = True
        _same(result["effective_counts"][branch], effective, "effective summary drift")
        count = effective["target_transport"]
        expected = {"member_positive": effective["usable_member_positive"], "member_negative": effective["usable_member_negative"],
            "unknown_member_positive": count["unknown_correspondence_member_positive"], "unknown_member_negative": count["unknown_correspondence_member_negative"],
            "center_labels": count["center_labels"], "event_labels": count["event_labels"], "numeric_input_valid_queries": effective["numeric_input_valid_queries"]}
        _same(card["effective_counts"][branch], expected, "effective card drift")
        _same(count["original_member_positive"], original["member_positive"], "raw member population drift")
        _same(count["original_member_negative"], original["member_negative"], "raw member population drift")
        _same(count["center_labels"], original["center_labels"], "raw center population drift")
        event_counts = {name: int(((t["events"] == i) & t["event_valid"]).sum()) for i, name in enumerate(("corridor", "junction", "terminal"))}
        _same(event_counts, original["events"], "raw event population drift")
        targets[branch] = RegionTargets(**{key: torch.from_numpy(value.copy()) for key, value in t.items()})
    for key in ("centers_m", "center_valid", "events", "event_valid", "label_complete"):
        _require(torch.equal(getattr(targets["gt"], key), getattr(targets["predicted"], key)), "GT/pred region targets differ")
    for i in range(180):
        for key in ("regions", "original_member_positive", "original_member_negative"):
            _same(transport["gt"][i]["target_transport"][key], transport["predicted"][i]["target_transport"][key],
                  "GT/pred original row targets differ")
        inverse = {}
        for branch in ("gt", "predicted"):
            inverse[branch] = {int(t): q for q, t in enumerate(arrays[branch + "__teacher_direction"][i]) if t >= 0}
        for direction in inverse["gt"].keys() & inverse["predicted"].keys():
            g, p = inverse["gt"][direction], inverse["predicted"][direction]
            for field in ("members", "member_valid"):
                _require(torch.equal(getattr(targets["gt"], field)[i, :, g], getattr(targets["predicted"], field)[i, :, p]),
                         "GT/pred common teacher-direction targets differ")
    return {"axes": axes, "targets": targets, "manifest": manifest, "transport": transport,
            "summary": summary, "read_hashes": reads}
