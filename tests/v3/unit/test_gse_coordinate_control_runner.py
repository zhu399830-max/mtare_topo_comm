"""Synthetic complete runner, with real optimizer/scoring/plots/seal.

Only dataset/cache acquisition and hardware allocation are substituted. No
project sample/checkpoint is opened. GPU parity remains a formal-run check.
"""
import copy
import hashlib
import importlib
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import numpy as np
import pytest
import torch

from mtare_topo.governance import build_run_id
from mtare_topo.governance_coordinate_control import validate_coordinate_control_card
from mtare_topo.representation.gse_point_axis_readout import PointAxisReadout
from tests.v3.unit.test_gse_coordinate_control_card import card


def module(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[3] / "tools/v3"))
    return importlib.import_module("run_gse_coordinate_control_v1")


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def synthetic_cache(records):
    generator = torch.Generator().manual_seed(13)
    entries = []
    for i, row in enumerate(records):
        points = torch.randn(1, 9, 3, generator=generator)
        valid = torch.ones(1, 9, dtype=torch.bool)
        memory = torch.randn(1, 3, 8, generator=generator)
        xyz = points.reshape(1, 3, 3, 3).mean(2)
        indices = (torch.arange(9) // 3)[None]
        slots = torch.randn(1, 32, 8, generator=generator)
        target = torch.randn(1, 32, 3, 3, generator=generator)
        target[:, :, 1] = (target[:, :, 0] + target[:, :, 2]) / 2
        mask = torch.arange(32)[None] < (9 if i < 12 else 8)
        entries.append({"student": (points, valid, memory, xyz, indices, slots),
                        "target": target, "mask": mask, "task": row["task"]})
    return entries


def setup_run(tmp_path, runner):
    value = card()
    for root in value["source_roots"].values(): (tmp_path / root).mkdir(parents=True, exist_ok=True)
    for path in value["sealed_sources"]:
        target = tmp_path / path; target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("Synthetic source placeholder; no actual checkpoint content")
    task = value["selected_rows"][0]["task"]
    chosen = value["source_roots"]["teacher"] + "/" + task + ".zarr/frame_row/.zarray"
    path = tmp_path / chosen; path.parent.mkdir(parents=True, exist_ok=True); path.write_text("{}")
    for role, seal in value["source_seals"].items():
        (tmp_path / seal).write_text((f"{digest(path)}  {chosen}\n" if role == "teacher" else "")
            + "0" * 64 + "  results/forbidden/S01_synthetic_C07__c1_mixed.zarr/frame_row/0\n"
            + "0" * 64 + "  /absolute/unselected/C10\n")
    value["sealed_sources"] = {name: digest(tmp_path / name) for name in value["sealed_sources"]}
    value["checkpoint"]["sha256"] = value["sealed_sources"][value["checkpoint"]["path"]]
    value["approval"]["checkpoint_sha256"] = value["checkpoint"]["sha256"]
    assert validate_coordinate_control_card(value).passed
    write(tmp_path / "configs/card.json", value)
    spec = {"gate": 3, "date": "20260905", "slug": "synthetic_coordinate_control", "seed": 0,
            "operation": "training", "data_card": "configs/card.json", "source_sha256": {},
            "training": copy.deepcopy(value["training"]), "wall_time_cap_s": 1200,
            "expected_versions": runner.environment_versions()}
    spec_path = tmp_path / "configs/spec.json"; write(spec_path, spec)
    run = tmp_path / "results/gate3_semantics" / build_run_id(spec)
    for folder in ("config", "logs", "metrics", "artifacts", "previews"):
        (run / folder).mkdir(parents=True, exist_ok=True)
    write(run / "config/run_spec.json", spec); write(run / "config/data_card.json", value)
    write(run / "RUN_STATE.json", {"state": "CREATED_NOT_EXECUTED"})
    return value, spec_path, run


def configure_synthetic(monkeypatch, tmp_path, runner, value):
    def contained(path):
        assert "C07" not in str(path) and "C10" not in str(path)
        candidate = (tmp_path / path).resolve()
        assert candidate.is_relative_to(tmp_path)
        return candidate
    monkeypatch.setattr(runner, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(runner, "contained", contained)
    frozen = torch.nn.Linear(2, 2).eval().requires_grad_(False)
    monkeypatch.setattr(runner, "load_model", lambda path: frozen)
    monkeypatch.setattr(runner, "device_report", lambda model: {"device": "cpu", "gpu_name": "synthetic_no_GPU_claim"})
    def initial(device):
        torch.manual_seed(0)
        return PointAxisReadout(model_dim=8, point_dim=4).to(device)
    monkeypatch.setattr(runner, "make_initial_head", initial)
    entries = synthetic_cache(value["selected_rows"])
    legacy = np.concatenate([entry["target"].numpy() + .4 for entry in entries])
    def cache(model, reader, records, reference_root, expected, log, guard):
        assert records == value["selected_rows"]
        for task in sorted({row["task"] for row in records}):
            log.write(json.dumps({"stage": "cache", "task": task, "rows": 18}) + "\n")
        return entries, legacy, records
    monkeypatch.setattr(runner, "build_cache", cache)
    return entries


def test_actual_optimizer_full_outputs_plots_ledger_seal_and_no_overwrite(tmp_path, monkeypatch):
    runner = module(monkeypatch); value, spec_path, run = setup_run(tmp_path, runner)
    configure_synthetic(monkeypatch, tmp_path, runner, value)
    immutable = {p: digest(p) for p in tmp_path.rglob("*") if p.is_file() and not p.is_relative_to(run)}
    monkeypatch.setattr(sys, "argv", ["runner", "--spec", str(spec_path), "--run-dir", str(run)])
    assert runner.main() == 0
    summary = json.loads((run / "metrics/summary.json").read_text())
    assert summary["error"] is None and not summary["scientific_gate_pass"]
    assert summary["ledger"] == {"optimizer_steps": 1080, "backbone_full_prediction_windows": 180,
        "backbone_memory_cache_windows": 180, "head_initial_final_evaluation_windows": 720, "zero_update_gradient_windows": 2}
    assert summary["result"]["backbone_unchanged"]
    log = [json.loads(line) for line in (run / "logs/progress.jsonl").read_text().splitlines()]
    for variant in runner.VARIANTS:
        train_rows = [r for r in log if r.get("variant") == variant]
        assert len(train_rows) == 540
        assert [r["observation_index"] for r in train_rows] == runner.sample_schedule()
        saved = torch.load(run / f"artifacts/{variant}_final.pt", weights_only=False)
        assert saved["training"]["steps"] == 540 and "optimizer_state_dict" in saved["training"]
    initial = [torch.load(run / f"artifacts/{v}_initial.pt", weights_only=True) for v in runner.VARIANTS]
    assert all(torch.equal(initial[0][key], initial[1][key]) for key in initial[0])
    with np.load(run / "artifacts/all_predictions.npz") as values:
        assert len(values.files) == 5 and all(values[k].shape == (180, 32, 3, 3) for k in values.files)
        assert not np.array_equal(values["raw_coordinates_initial"], values["mean_broadcast_coordinates_initial"])
    previews = sorted((run / "previews").glob("*.svg")); assert len(previews) == 11
    for preview in previews: ET.parse(preview)
    seal = run / "artifacts/evidence_sha256.txt"; before = digest(seal)
    for line in seal.read_text().splitlines():
        expected, relative = line.split(None, 1); assert digest(tmp_path / relative) == expected
    assert all(digest(path) == expected for path, expected in immutable.items())
    with pytest.raises(ValueError, match="no overwrite"):
        runner.main()
    assert digest(seal) == before


def test_failure_preserves_actual_completed_step_ledger(tmp_path, monkeypatch):
    runner = module(monkeypatch); value, spec_path, run = setup_run(tmp_path, runner)
    configure_synthetic(monkeypatch, tmp_path, runner, value)
    original = runner.fit
    def fail_after_steps(head, cache, schedule, log_step, **kwargs):
        original(head, cache, schedule[:3], log_step, **kwargs)
        raise ValueError("synthetic failure after three actual steps")
    monkeypatch.setattr(runner, "fit", fail_after_steps)
    monkeypatch.setattr(sys, "argv", ["runner", "--spec", str(spec_path), "--run-dir", str(run)])
    assert runner.main() == 1
    summary = json.loads((run / "metrics/summary.json").read_text())
    assert summary["ledger"]["optimizer_steps"] == 3
    assert summary["ledger"]["head_initial_final_evaluation_windows"] == 180
    assert json.loads((run / "RUN_STATE.json").read_text())["state"] == "FAILED"
    assert (run / "artifacts/evidence_sha256.txt").exists()


def test_source_filter_does_not_resolve_unselected_or_undeclared_paths(tmp_path, monkeypatch):
    runner = module(monkeypatch); value = card()
    task = value["selected_rows"][0]["task"]
    good = value["source_roots"]["teacher"] + "/" + task + ".zarr/primitive_mask/0.0"
    bad = [good.replace("C01", "C07"), good.replace("primitive_mask", "node_identity"),
           good.replace("0.0", "../C07"), "/absolute/C10", "../outside"]
    for role, seal in value["source_seals"].items():
        path = tmp_path / seal; path.parent.mkdir(parents=True, exist_ok=True)
        lines = [good] + bad if role == "teacher" else bad
        path.write_text("".join("a" * 64 + "  " + line + "\n" for line in lines))
    resolved = []
    def contained(path):
        assert path not in bad
        resolved.append(path); return tmp_path / path
    monkeypatch.setattr(runner, "contained", contained)
    result = runner.source_index(value)
    assert result == {str(tmp_path / good): "a" * 64}
    assert resolved == [value["source_seals"]["sensor"], value["source_seals"]["teacher"], good,
                        value["source_seals"]["reference"]]


def test_variant_changes_only_coordinates_not_target_mask_or_feature_identity(monkeypatch):
    runner = module(monkeypatch); value = card(); entries = synthetic_cache(value["selected_rows"])
    changed = runner.variant_cache(entries, "mean_broadcast_coordinates")
    for original, control in zip(entries, changed):
        assert control["target"] is original["target"] and control["mask"] is original["mask"]
        assert all(a is b for a, b in zip(control["student"][1:], original["student"][1:]))
        assert not torch.equal(control["student"][0], original["student"][0])


def test_freezer_prepares_independent_contract_without_running_model(tmp_path, monkeypatch):
    runner = module(monkeypatch)
    freezer = importlib.import_module("freeze_gse_coordinate_control_v1")
    value = card()
    # Synthetic predecessor carries the same metadata shape as the real one.
    from tests.v3.unit.test_gse_point_axis_training_card import card as original_card
    oldcard = original_card()
    oldcard["sealed_sources"].update({path: "a" * 64 for path in value["source_seals"].values()})
    old = {"gate": 3, "date": "20260905", "slug": "old", "seed": 0, "operation": "training",
           "data_card": "configs/old_card.json", "sensor_root": value["source_roots"]["sensor"],
           "teacher_root": value["source_roots"]["teacher"], "source_seals": list(value["source_seals"].values()),
           "prediction_reference_root": value["prediction_reference_root"], "source_sha256": {},
           "command": ["env", "A=1", "B=1", "C=1", "D=1", "E=1", "/synthetic/python", "old.py"]}
    write(tmp_path / freezer.SOURCE_SPEC, old); write(tmp_path / old["data_card"], oldcard)
    (tmp_path / "src").mkdir()
    monkeypatch.setattr(freezer, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(freezer, "sha", lambda path: "f" * 64)
    newcard, spec = freezer.documents()
    assert validate_coordinate_control_card(newcard).passed
    assert spec["command"][6] == "/synthetic/python"
    assert spec["training"]["variants"] == ["raw_coordinates", "mean_broadcast_coordinates"]
    assert "Initial coordinate-dependent outputs are NOT required equal" in spec["baseline"]
    assert spec["user_authorization"] == newcard["approval"]
    assert not (tmp_path / freezer.CARD).exists() and not (tmp_path / "results").exists()
