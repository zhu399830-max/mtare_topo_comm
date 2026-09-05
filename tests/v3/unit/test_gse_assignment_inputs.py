"""Temporary nine-payload attribution fixtures, no models or real runs."""
from copy import deepcopy
from dataclasses import fields
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import mtare_topo.data.gse_assignment_inputs as loader
from mtare_topo.data.gse_partial_training_inputs import load_training_inputs
from mtare_topo.representation.gse_partial_structure_training import BRANCHES, PartialTrainingConfig, paired_schedule
from mtare_topo.representation.gse_region_queries import RegionPrediction, tokens_from_axes
from tests.v3.unit.test_gse_partial_training_inputs import fixture as input_fixture
from tests.v3.unit.test_gse_partial_structure_export_runner import sha


def fixture(tmp_path, monkeypatch):
    embedded, input_paths, _, _ = input_fixture(tmp_path, monkeypatch)
    embedded["training"] = {"seed": 0, "steps_per_branch": 300, "batch_size": 18, "lr": .001, "device": "cuda:0"}
    data = load_training_inputs(tmp_path, embedded)
    paths = dict(input_paths)
    extra = tmp_path / "synthetic_training"; extra.mkdir()
    for branch, source in loader.FINAL_SOURCES.items():
        key = "gt" if branch == "gt_axes" else "predicted"
        q = tokens_from_axes(data["axes"][key]).valid
        pred = RegionPrediction(torch.zeros(180, 64, 3), torch.zeros(180, 64, 3),
            torch.zeros(180, 64), torch.zeros(180, 64, 64), torch.full((180, 64), .5),
            q, q[:, :, None] & q[:, None])
        target = data["targets"][key]
        scored = target.member_valid & q[:, None]
        unique = torch.zeros(180, target.centers_m.shape[1], dtype=torch.int64)
        paths[source] = extra / loader.SOURCE_NAMES[source]
        np.savez(paths[source], **{f.name: getattr(pred, f.name).numpy() for f in fields(pred)},
                 scored_member_mask=scored.numpy(), unique_center_query=unique.numpy())
    schedule = paired_schedule(180, PartialTrainingConfig(0, 300, 18, .001, "cpu"))
    history = {branch: [{"step": i + 1, "optimizer_steps": i + 1, "sample_indices": indices,
        "loss": dict.fromkeys(("total", "center", "event", "membership", "presence", "uncertainty"), .1),
        "counts": {"targets": 1, "matched": 1, "unmatched_targets": 0, "unsupported_member_targets": 0,
                   "unconfirmed_presence_targets": 0, "center": 1, "event": 1, "membership": 2,
                   "presence_positive": 1, "presence_negative": 0}, "matches": [[0, 0, 0]], "gradient_l2": .2,
        "gradient_tensor_count": 26} for i, indices in enumerate(schedule)] for branch in BRANCHES}
    paths["history"] = extra / "training_history.json"; paths["history"].write_text(json.dumps(history))
    summary = {"status": "PARTIAL_STRUCTURE_FIXED_BUDGET_COMPLETE", "error": None,
        "scientific_gate_pass": False, "full_three_class_ready": False,
        "result": {"optimizer_steps": 900, "head_inference_windows": 1080, "backbone_windows": 0,
                   "new_sensor_frames": 0, "same_initial_state_verified": True, "same_schedule_verified": True,
                   "evaluation": {stage: dict.fromkeys(BRANCHES, {}) for stage in ("initial", "final")}}}
    paths["training_summary"] = extra / "summary.json"; paths["training_summary"].write_text(json.dumps(summary))
    sources = {key: {"path": str(path.relative_to(tmp_path)), "sha256": sha(path)} for key, path in paths.items()}
    card = {"training_card": embedded, "sources": sources,
        "sealed_sources": {r["path"]: r["sha256"] for r in sources.values()},
        "observation_count": 180, "selected_rows": deepcopy(embedded["selected_rows"])}
    monkeypatch.setattr(loader, "validate_assignment_attribution_card", lambda c: SimpleNamespace(passed=True, errors=[]))
    return card, paths


def reseal(card, paths, key):
    digest = sha(paths[key]); record = card["sources"][key]
    record["sha256"] = digest; card["sealed_sources"][record["path"]] = digest


def test_exact_nine_byte_reads_cpu_predictions_history_and_scoring_preserved(tmp_path, monkeypatch):
    card, paths = fixture(tmp_path, monkeypatch)
    before = {p: sha(p) for p in paths.values()}
    actual, original = [], Path.read_bytes
    def read(path):
        assert path in paths.values(), f"undeclared data/model read: {path}"
        actual.append(path)
        return original(path)
    with monkeypatch.context() as context:
        context.setattr(Path, "read_bytes", read)
        context.setattr(torch, "load", lambda *a, **kw: pytest.fail("checkpoint read"))
        result = loader.load_assignment_inputs(tmp_path, card)
    assert len(actual) == 9 and set(actual) == set(paths.values())
    assert result["read_hashes"] == card["sealed_sources"]
    assert all(sha(p) == h for p, h in before.items())
    assert sorted(sum(result["last_epoch_batches"], [])) == list(range(180))
    assert [len(b) for b in result["last_epoch_batches"]] == [18] * 10
    for branch in BRANCHES:
        p = result["predictions"][branch]
        assert p.centers_m.shape == (180, 64, 3) and p.centers_m.device.type == "cpu"
        assert p.membership_logits.shape == (180, 64, 64)
        saved = result["saved_scoring"][branch]
        assert saved["unique_center_query"].dtype == torch.int64
        assert saved["scored_member_mask"].dtype == torch.bool
        assert saved["scored_member_mask"].sum() == 360
        assert len(result["history"][branch]) == 300


@pytest.mark.parametrize("key", list(loader.SOURCE_NAMES))
def test_any_payload_hash_drift_refused(tmp_path, monkeypatch, key):
    card, paths = fixture(tmp_path, monkeypatch)
    paths[key].write_bytes(b"unapproved payload")
    with pytest.raises(ValueError, match="SHA drift"):
        loader.load_assignment_inputs(tmp_path, card)


@pytest.mark.parametrize("field,issue", [("centers_m", "dtype"), ("centers_m", "nan"),
    ("event_logits", "shape"), ("query_supported", "mask"), ("member_supported", "mask"),
    ("unique_center_query", "range"), ("unique_center_query", "dtype"),
    ("scored_member_mask", "mask"), ("uncertainty", "range")])
def test_saved_prediction_and_scoring_schema_drift(tmp_path, monkeypatch, field, issue):
    card, paths = fixture(tmp_path, monkeypatch)
    key = "predicted_final"
    with np.load(paths[key]) as archive: arrays = {k: archive[k] for k in archive.files}
    if issue == "dtype": arrays[field] = arrays[field].astype(np.float64)
    if issue == "nan": arrays[field].flat[0] = np.nan
    if issue == "shape": arrays[field] = arrays[field][:, :63]
    if issue == "mask": arrays[field] = ~arrays[field]
    if issue == "range": arrays[field].flat[0] = 100
    np.savez(paths[key], **arrays); reseal(card, paths, key)
    with pytest.raises(ValueError): loader.load_assignment_inputs(tmp_path, card)


@pytest.mark.parametrize("issue", ["missing_step", "step", "batch", "permuted_other_branch", "nan_loss", "gradient", "matches", "counts", "matched_count", "background"])
def test_history_step_schedule_and_finite_evidence_drift(tmp_path, monkeypatch, issue):
    card, paths = fixture(tmp_path, monkeypatch)
    h = json.loads(paths["history"].read_text()); row = h["predicted_axes"][-1]
    if issue == "missing_step": h["gt_axes"].pop()
    if issue == "step": row["step"] = 301
    if issue == "batch": row["sample_indices"][0] = row["sample_indices"][1]
    if issue == "permuted_other_branch": row["sample_indices"] = row["sample_indices"][::-1]
    if issue == "nan_loss": row["loss"]["center"] = float("nan")
    if issue == "gradient": row["gradient_tensor_count"] = 0
    if issue == "matches": row["matches"] = [[18, 0, 0]]
    if issue == "counts": row["counts"]["matched"] = -1
    if issue == "matched_count": row["counts"]["matched"] = 2
    if issue == "background": row["counts"]["presence_negative"] = 1
    paths["history"].write_text(json.dumps(h)); reseal(card, paths, "history")
    with pytest.raises(ValueError): loader.load_assignment_inputs(tmp_path, card)


@pytest.mark.parametrize("field,value", [("optimizer_steps", 899), ("head_inference_windows", 540),
    ("backbone_windows", 1), ("new_sensor_frames", 1), ("same_schedule_verified", False)])
def test_summary_execution_counts_must_match_original_contract(tmp_path, monkeypatch, field, value):
    card, paths = fixture(tmp_path, monkeypatch)
    summary = json.loads(paths["training_summary"].read_text())
    summary["result"][field] = value
    paths["training_summary"].write_text(json.dumps(summary)); reseal(card, paths, "training_summary")
    with pytest.raises(ValueError): loader.load_assignment_inputs(tmp_path, card)


def test_new_authority_required_before_old_training_card_loader_or_bytes(tmp_path, monkeypatch):
    monkeypatch.setattr(loader, "validate_assignment_attribution_card", lambda c: SimpleNamespace(passed=False, errors=["new scope required"]))
    monkeypatch.setattr(loader, "load_training_inputs", lambda *a: pytest.fail("old loader invoked before new authority"))
    monkeypatch.setattr(Path, "read_bytes", lambda *a: pytest.fail("payload read before new authority"))
    with pytest.raises(ValueError, match="card rejected"):
        loader.load_assignment_inputs(tmp_path, {})


def test_old_four_bindings_cannot_be_substituted(tmp_path, monkeypatch):
    card, _ = fixture(tmp_path, monkeypatch)
    card["sources"]["inputs"]["sha256"] = "e" * 64
    with pytest.raises(ValueError, match="embedded four"):
        loader.load_assignment_inputs(tmp_path, card)
