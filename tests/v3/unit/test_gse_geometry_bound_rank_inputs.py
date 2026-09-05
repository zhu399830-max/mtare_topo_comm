"""Temporary synthetic rank payloads; no model, history, or actual run reads."""
from copy import deepcopy
from dataclasses import fields
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import mtare_topo.data.gse_geometry_bound_rank_inputs as loader
from mtare_topo.data.gse_partial_training_inputs import load_training_inputs
from mtare_topo.representation.gse_partial_structure_training import BRANCHES
from mtare_topo.representation.gse_region_queries import RegionPrediction, tokens_from_axes
from tests.v3.unit.test_gse_partial_training_inputs import fixture as input_fixture
from tests.v3.unit.test_gse_partial_structure_export_runner import sha


def fixture(tmp_path, monkeypatch):
    base, input_paths, _, _ = input_fixture(tmp_path, monkeypatch)
    data = load_training_inputs(tmp_path, base)
    paths = dict(input_paths)
    folder = tmp_path / "synthetic_corrective"; folder.mkdir()
    for branch, key in loader.FINAL_SOURCES.items():
        kind = "gt" if branch == "gt_axes" else "predicted"
        support = tokens_from_axes(data["axes"][kind]).valid
        p = RegionPrediction(torch.zeros(180, 64, 3), torch.zeros(180, 64, 3),
                             torch.zeros(180, 64), torch.zeros(180, 64, 64), torch.full((180, 64), .5),
                             support, support[:, :, None] & support[:, None])
        target = data["targets"][kind]
        paths[key] = folder / loader.SOURCE_NAMES[key]
        np.savez(paths[key], **{f.name: getattr(p, f.name).numpy() for f in fields(p)},
                 scored_member_mask=(target.member_valid & support[:, None]).numpy(),
                 unique_center_query=np.zeros(target.center_valid.shape, np.int64))
    summary = dict(status="GEOMETRY_BOUND_FIXED_BUDGET_COMPLETE", error=None,
                   scientific_gate_pass=False, full_three_class_ready=False,
                   result=dict(optimizer_steps=900, head_inference_windows=1080, backbone_windows=0,
                               new_sensor_frames=0, same_initial_state_verified=True, same_schedule_verified=True,
                               evaluation={stage: dict.fromkeys(BRANCHES, {}) for stage in ("initial", "final")}))
    paths["corrective_summary"] = folder / "summary.json"
    paths["corrective_summary"].write_text(json.dumps(summary))
    sources = {key: dict(path=str(path.relative_to(tmp_path)), sha256=sha(path)) for key, path in paths.items()}
    card = dict(corrective_card={"base_training_card": base}, sources=sources,
                sealed_sources={r["path"]: r["sha256"] for r in sources.values()},
                observation_count=180, selected_rows=deepcopy(base["selected_rows"]))
    monkeypatch.setattr(loader, "validate_geometry_bound_rank_card", lambda value: SimpleNamespace(passed=True, errors=[]))
    return card, paths


def reseal(card, paths, key):
    record = card["sources"][key]
    record["sha256"] = sha(paths[key]); card["sealed_sources"][record["path"]] = record["sha256"]


def test_exact_eight_reads_and_complete_cpu_prediction_arrays(tmp_path, monkeypatch):
    card, paths = fixture(tmp_path, monkeypatch)
    before = {path: sha(path) for path in paths.values()}
    calls, original = [], Path.read_bytes
    def read(path):
        assert path in paths.values(), f"undeclared history/weights/data read: {path}"
        calls.append(path)
        return original(path)
    with monkeypatch.context() as context:
        context.setattr(Path, "read_bytes", read)
        context.setattr(torch, "load", lambda *a, **kw: pytest.fail("checkpoint read"))
        result = loader.load_rank_inputs(tmp_path, card)
    assert len(calls) == 8 and set(calls) == set(paths.values())
    assert result["read_hashes"] == card["sealed_sources"]
    assert set(result) == {"data", "predictions", "saved_scoring", "training_summary", "read_hashes"}
    assert all(sha(path) == digest for path, digest in before.items())
    for branch in BRANCHES:
        p = result["predictions"][branch]
        assert p.centers_m.shape == (180, 64, 3) and p.membership_logits.shape == (180, 64, 64)
        assert all(getattr(p, f.name).device.type == "cpu" and not getattr(p, f.name).requires_grad for f in fields(p))
        scoring = result["saved_scoring"][branch]
        assert scoring["unique_center_query"].dtype == torch.int64
        assert scoring["scored_member_mask"].dtype == torch.bool


@pytest.mark.parametrize("key", list(loader.SOURCE_NAMES))
def test_all_payload_hash_drift_fails_before_any_decode(tmp_path, monkeypatch, key):
    card, paths = fixture(tmp_path, monkeypatch)
    paths[key].write_bytes(b"synthetic corruption")
    with monkeypatch.context() as context:
        context.setattr(np, "load", lambda *a, **kw: pytest.fail("NPZ decoded before all hashes checked"))
        with pytest.raises(ValueError, match="SHA drift"):
            loader.load_rank_inputs(tmp_path, card)


@pytest.mark.parametrize("field,issue", [("centers_m", "shape"), ("centers_m", "dtype"),
    ("event_logits", "nan"), ("query_supported", "mask"), ("member_supported", "mask"),
    ("uncertainty", "range"), ("unique_center_query", "range"), ("scored_member_mask", "mask")])
def test_numeric_and_saved_scoring_drift(tmp_path, monkeypatch, field, issue):
    card, paths = fixture(tmp_path, monkeypatch)
    key = "predicted_final"
    with np.load(paths[key], allow_pickle=False) as archive: arrays = {k: archive[k] for k in archive.files}
    if issue == "shape": arrays[field] = arrays[field][:, :63]
    if issue == "dtype": arrays[field] = arrays[field].astype(np.float64)
    if issue == "nan": arrays[field].flat[0] = float("nan")
    if issue == "mask": arrays[field] = ~arrays[field]
    if issue == "range": arrays[field].flat[0] = 100
    np.savez(paths[key], **arrays); reseal(card, paths, key)
    with pytest.raises(ValueError): loader.load_rank_inputs(tmp_path, card)


@pytest.mark.parametrize("field,value", [("optimizer_steps", 899), ("optimizer_steps", 900.),
    ("head_inference_windows", 540), ("backbone_windows", 1), ("new_sensor_frames", 1),
    ("same_initial_state_verified", False), ("same_schedule_verified", False)])
def test_corrective_summary_fixed_population_and_init_schedule(tmp_path, monkeypatch, field, value):
    card, paths = fixture(tmp_path, monkeypatch)
    summary = json.loads(paths["corrective_summary"].read_text())
    summary["result"][field] = value
    paths["corrective_summary"].write_text(json.dumps(summary)); reseal(card, paths, "corrective_summary")
    with pytest.raises(ValueError): loader.load_rank_inputs(tmp_path, card)


def test_old_run_status_cannot_masquerade_as_corrective(tmp_path, monkeypatch):
    card, paths = fixture(tmp_path, monkeypatch)
    summary = json.loads(paths["corrective_summary"].read_text())
    summary["status"] = "PARTIAL_STRUCTURE_FIXED_BUDGET_COMPLETE"
    paths["corrective_summary"].write_text(json.dumps(summary)); reseal(card, paths, "corrective_summary")
    with pytest.raises(ValueError, match="geometry-bound summary"):
        loader.load_rank_inputs(tmp_path, card)


def test_new_card_required_before_old_loader_or_any_read(tmp_path, monkeypatch):
    monkeypatch.setattr(loader, "validate_geometry_bound_rank_card", lambda value: SimpleNamespace(passed=False, errors=["not authorized"]))
    monkeypatch.setattr(loader, "load_training_inputs", lambda *a: pytest.fail("old loader invoked before new card"))
    monkeypatch.setattr(Path, "read_bytes", lambda *a: pytest.fail("data read before new card"))
    with pytest.raises(ValueError, match="card rejected"):
        loader.load_rank_inputs(tmp_path, {})


def test_no_history_or_other_extra_sources(tmp_path, monkeypatch):
    card, paths = fixture(tmp_path, monkeypatch)
    card["sources"]["history"] = dict(path="synthetic/training_history.json", sha256="0"*64)
    with pytest.raises(ValueError, match="exact eight"):
        loader.load_rank_inputs(tmp_path, card)


def test_top_level_old_export_binding_must_equal_base(tmp_path, monkeypatch):
    card, paths = fixture(tmp_path, monkeypatch)
    card["sources"]["inputs"]["sha256"] = "e"*64
    with pytest.raises(ValueError, match="original four"):
        loader.load_rank_inputs(tmp_path, card)
