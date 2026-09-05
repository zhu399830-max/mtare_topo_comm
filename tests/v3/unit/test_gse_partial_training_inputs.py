"""Entirely synthetic four-payload training-input validation."""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import mtare_topo.data.gse_partial_training_inputs as loader
from tests.v3.unit.test_gse_partial_structure_export_runner import fixture as exported, sha


def fixture(tmp_path, monkeypatch):
    runner, spec, run, axes, mask, _ = exported(tmp_path, monkeypatch)
    assert runner.execute(spec, run) == 0
    paths = {"inputs": run / "artifacts/partial_training_inputs.npz", "manifest": run / "artifacts/manifest.json",
             "target_transport": run / "artifacts/target_transport.json", "summary": run / "metrics/summary.json"}
    summary = json.loads(paths["summary"].read_text())["result"]
    manifest = json.loads(paths["manifest"].read_text())
    sources = {key: {"path": str(path.relative_to(tmp_path)), "sha256": sha(path)} for key, path in paths.items()}
    card = {"sources": sources, "sealed_sources": {r["path"]: r["sha256"] for r in sources.values()},
        "selected_rows": [{k: row[k] for k in ("task", "row_index", "source_global_sequence_index")} for row in manifest],
        "tasks": sorted({row["task"] for row in manifest}), "observation_count": 180, "parent_count": 10,
        "unique_source_frame_count": 900, "visible_fragment_count": 1452,
        "raw_teacher_counts": {k: summary["original_counts"][k] for k in ("center_labels", "member_positive", "member_negative", "events")},
        "effective_counts": {}}
    for branch, e in summary["effective_counts"].items():
        t = e["target_transport"]
        card["effective_counts"][branch] = {"member_positive": e["usable_member_positive"], "member_negative": e["usable_member_negative"],
            "unknown_member_positive": t["unknown_correspondence_member_positive"], "unknown_member_negative": t["unknown_correspondence_member_negative"],
            "center_labels": t["center_labels"], "event_labels": t["event_labels"], "numeric_input_valid_queries": e["numeric_input_valid_queries"]}
    monkeypatch.setattr(loader, "validate_partial_structure_training_card", lambda value: SimpleNamespace(passed=True, errors=[]))
    return card, paths, axes, mask


def reseal(card, paths, key):
    digest = sha(paths[key])
    card["sources"][key]["sha256"] = digest
    card["sealed_sources"][card["sources"][key]["path"]] = digest


def mutate_array(card, paths, key, change):
    with np.load(paths["inputs"], allow_pickle=False) as archive:
        arrays = {k: archive[k] for k in archive.files}
    arrays[key] = change(arrays[key].copy())
    np.savez(paths["inputs"], **arrays)
    reseal(card, paths, "inputs")


def test_four_byte_bound_sources_cpu_axes_partial_targets_and_exact_ledger(tmp_path, monkeypatch):
    card, paths, expected, masks = fixture(tmp_path, monkeypatch)
    old = {p: sha(p) for p in paths.values()}
    reads = []
    original = Path.read_bytes
    def guarded(path):
        assert path in paths.values(), f"undeclared read: {path}"
        reads.append(path)
        return original(path)
    with monkeypatch.context() as context:
        context.setattr(Path, "read_bytes", guarded)
        result = loader.load_training_inputs(tmp_path, card)
    assert len(reads) == 4 and set(reads) == set(paths.values())
    assert result["read_hashes"] == card["sealed_sources"]
    assert all(sha(p) == digest for p, digest in old.items())
    np.testing.assert_array_equal(result["axes"]["predicted"].numpy(), expected)
    assert result["axes"]["predicted"].shape == (180, 32, 3, 3)
    assert (result["axes"]["predicted"].numpy()[~masks] != 0).any()
    assert len(result["manifest"]) == 180
    for branch in ("gt", "predicted"):
        assert result["axes"][branch].device.type == "cpu" and not result["axes"][branch].requires_grad
        target = result["targets"][branch]
        assert target.centers_m.dtype == target.members.dtype == torch.float32
        assert target.events.dtype == torch.long
        assert target.member_valid.dtype == target.center_valid.dtype == target.event_valid.dtype == torch.bool
        assert not target.label_complete.any()
        assert target.center_valid.sum() == target.event_valid.sum() == 180
        assert target.member_valid.sum() == 360


@pytest.mark.parametrize("key", ["inputs", "manifest", "target_transport", "summary"])
def test_digest_verified_before_decode(tmp_path, monkeypatch, key):
    card, paths, _, _ = fixture(tmp_path, monkeypatch)
    paths[key].write_bytes(b"not the approved bytes")
    with pytest.raises(ValueError, match="SHA drift"):
        loader.load_training_inputs(tmp_path, card)


@pytest.mark.parametrize("field,change", [
    ("predicted_axes", lambda a: a.astype(np.float64)),
    ("predicted_axes", lambda a: np.full_like(a, np.nan)),
    ("predicted_axes", lambda a: a[:, :8]),
    ("gt_axes", lambda a: a + 1),
    ("loss_only_gt_mask", lambda a: a.astype(np.uint8)),
    ("gt__center_valid", lambda a: a.astype(np.int64)),
    ("gt__events", lambda a: a.astype(np.int32)),
    ("predicted__label_complete", lambda a: ~a),
    ("predicted__input_direction_valid", lambda a: ~a),
    ("predicted__teacher_direction", lambda a: a + 100),
    ("predicted__direction_valid", lambda a: ~a),
    ("predicted__centers_m", lambda a: a + 1),
    ("predicted__members", lambda a: 1 - a),
    ("gt__events", lambda a: np.full_like(a, 2)),
])
def test_resealed_invalid_array_is_not_accepted(tmp_path, monkeypatch, field, change):
    card, paths, _, _ = fixture(tmp_path, monkeypatch)
    mutate_array(card, paths, field, change)
    with pytest.raises(ValueError):
        loader.load_training_inputs(tmp_path, card)


@pytest.mark.parametrize("issue", ["frame", "center_count", "member_count", "mapping", "alignment_count", "unknown_count", "parent"])
def test_resealed_transport_identity_or_ledger_drift(tmp_path, monkeypatch, issue):
    card, paths, _, _ = fixture(tmp_path, monkeypatch)
    value = json.loads(paths["target_transport"].read_text())
    row = value["predicted"][0]
    if issue == "frame": row["frame_rows"] = [6, 7, 8, 9, 10]
    if issue == "parent": row["task"] = "S01_synthetic_C02__c1_mixed"
    if issue == "center_count": row["target_transport"]["center_labels"] += 1
    if issue == "member_count": row["usable_member_positive"] += 1
    if issue == "mapping": row["teacher_direction"][0] = 63
    if issue == "alignment_count": row["direction_alignment"]["unique_direction_pairs"] += 1
    if issue == "unknown_count": row["target_transport"]["unknown_correspondence_member_positive"] += 1
    paths["target_transport"].write_text(json.dumps(value)); reseal(card, paths, "target_transport")
    with pytest.raises(ValueError):
        loader.load_training_inputs(tmp_path, card)


@pytest.mark.parametrize("issue", ["original", "effective", "status", "optimizer", "card_counts", "selected_order", "missing_key"])
def test_summary_card_or_npz_contract_drift(tmp_path, monkeypatch, issue):
    card, paths, _, _ = fixture(tmp_path, monkeypatch)
    value = json.loads(paths["summary"].read_text())
    if issue == "original": value["result"]["original_counts"]["center_labels"] += 1
    if issue == "effective": value["result"]["effective_counts"]["gt"]["usable_member_positive"] += 1
    if issue == "status": value["status"] = "FAILED"
    if issue == "optimizer": value["result"]["optimizer_steps"] = 1
    if issue == "card_counts": card["effective_counts"]["gt"]["member_positive"] += 1
    if issue == "selected_order": card["selected_rows"] = card["selected_rows"][::-1]
    if issue == "missing_key":
        with np.load(paths["inputs"]) as archive: arrays = {k: archive[k] for k in archive.files if k != "gt__events"}
        np.savez(paths["inputs"], **arrays); reseal(card, paths, "inputs")
    paths["summary"].write_text(json.dumps(value)); reseal(card, paths, "summary")
    with pytest.raises(ValueError):
        loader.load_training_inputs(tmp_path, card)


def test_invalid_card_stops_before_any_payload_read(tmp_path, monkeypatch):
    monkeypatch.setattr(loader, "validate_partial_structure_training_card", lambda value: SimpleNamespace(passed=False, errors=["not approved"]))
    monkeypatch.setattr(Path, "read_bytes", lambda _: pytest.fail("payload read before card approval"))
    with pytest.raises(ValueError, match="card rejected"):
        loader.load_training_inputs(tmp_path, {})


def test_source_symlink_escape_rejected_before_bytes(tmp_path, monkeypatch):
    card, paths, _, _ = fixture(tmp_path, monkeypatch)
    outside = tmp_path.parent / (tmp_path.name + "_outside")
    outside.mkdir()
    # Only an artificial temporary destination; no external file is read.
    escaped = outside / "partial_training_inputs.npz"
    escaped.write_bytes(b"synthetic")
    paths["inputs"].unlink()
    paths["inputs"].symlink_to(escaped)
    monkeypatch.setattr(Path, "read_bytes", lambda _: pytest.fail("read before path containment"))
    with pytest.raises(ValueError, match="source escape"):
        loader.load_training_inputs(tmp_path, card)


def test_swapped_known_members_keep_counts_but_violate_shared_teacher_labels(tmp_path, monkeypatch):
    card, paths, _, _ = fixture(tmp_path, monkeypatch)
    def swap(a):
        a[0, 0, 0], a[0, 0, 2] = a[0, 0, 2], a[0, 0, 0]
        return a
    mutate_array(card, paths, "predicted__members", swap)
    with pytest.raises(ValueError, match="common teacher-direction"):
        loader.load_training_inputs(tmp_path, card)
