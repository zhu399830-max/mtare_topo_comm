"""Actual tiny CPU training/scoring/save path, entirely synthetic inputs.

Lifecycle checks fail before GPU use. No real sample/cache/checkpoint is read;
the only optimizer steps are six tiny synthetic operator-test updates.
"""
from dataclasses import fields
import hashlib
import importlib
import json
from pathlib import Path
from types import SimpleNamespace
import xml.etree.ElementTree as ET

import numpy as np
import pytest
import torch

from mtare_topo.governance import build_run_id
from mtare_topo.representation.gse_partial_structure_training import BRANCHES
from mtare_topo.representation.gse_region_queries import RegionPrediction, RegionQueryHead, RegionTargets


def module(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[3] / "tools/v3"))
    return importlib.import_module("run_gse_partial_structure_training_v1")


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def folders(run):
    for name in ("config", "logs", "metrics", "artifacts", "previews"):
        (run / name).mkdir(parents=True, exist_ok=True)


def synthetic():
    axes = torch.zeros((3, 32, 3, 3))
    for slot in range(32):
        axes[:, slot] = torch.tensor([[float(slot), -1., 0.], [float(slot), 0., 0.], [float(slot), 1., .2]])
    axes[1, :, :, 2] += .2; axes[2, :, :, 2] += .4
    member = torch.zeros((3, 2, 64))
    member[:, 0, 0] = 1; member[:, 1, 1] = 1
    valid = torch.zeros((3, 2, 64), dtype=torch.bool); valid[:, :, :2] = True
    target = RegionTargets(torch.tensor([[[0., 0., 0.], [5., 0., 0.]]]).repeat(3, 1, 1),
                           torch.ones((3, 2), dtype=torch.bool), torch.tensor([[0, 1]]).repeat(3, 1),
                           torch.ones((3, 2), dtype=torch.bool), member, valid, torch.zeros(3, dtype=torch.bool))
    ledger = dict(original_member_positive=2, original_member_negative=2,
                  transferred_member_positive=2, transferred_member_negative=2,
                  unknown_correspondence_member_positive=0, unknown_correspondence_member_negative=0)
    return dict(axes={"gt": axes, "predicted": axes + .1}, targets={"gt": target, "predicted": target},
                transport={key: [dict(target_transport=dict(ledger)) for _ in range(3)] for key in ("gt", "predicted")},
                manifest=[dict(task="synthetic_parent_a", row=0), dict(task="synthetic_parent_a", row=1),
                          dict(task="synthetic_parent_b", row=2)], read_hashes={})


def settings():
    return dict(seed=3, steps_per_branch=2, batch_size=2, lr=.001, device="cpu", hidden=4,
                parameters_per_head=sum(p.numel() for p in RegionQueryHead(hidden=4).parameters()),
                use_relations={name: name != "predicted_no_relations" for name in BRANCHES})


def test_real_cpu_three_branch_initial_final_scores_predictions_weights_and_previews(tmp_path, monkeypatch):
    runner = module(monkeypatch)
    folders(tmp_path)
    data, config = synthetic(), settings()
    before_axes = {k: v.clone() for k, v in data["axes"].items()}
    updates = []
    result = runner.run_training(tmp_path, data, config, {"member_threshold": .5},
                                 step_callback=lambda branch, record: updates.append((branch, record["step"])))
    assert result["optimizer_steps"] == 6 and len(updates) == 6
    assert updates == [(branch, step) for branch in BRANCHES for step in (1, 2)]
    assert result["head_inference_windows"] == 18
    assert result["backbone_windows"] == result["new_sensor_frames"] == 0
    assert result["same_initial_state_verified"] and result["same_schedule_verified"]
    initial = torch.load(tmp_path / "artifacts/shared_initial_state.pt", weights_only=True)
    history = json.loads((tmp_path / "artifacts/training_history.json").read_text())
    schedule = json.loads((tmp_path / "artifacts/shared_schedule.json").read_text())
    assert len(schedule) == 2 and sorted(sum(schedule, [])) == [0, 1, 2]
    for branch in BRANCHES:
        assert [r["sample_indices"] for r in history[branch]] == schedule
        checkpoint = torch.load(tmp_path / "artifacts" / (branch + "__final.pt"), weights_only=True)
        assert checkpoint["branch"] == branch and checkpoint["step"] == 2
        assert checkpoint["use_relations"] == config["use_relations"][branch]
        assert any(not torch.equal(value, initial[key]) for key, value in checkpoint["state_dict"].items())
        key = "gt" if branch == "gt_axes" else "predicted"
        for stage in ("initial", "final"):
            prefix = branch + "__" + stage
            # Independently reconstruct BOTH heads, rather than trusting the
            # saved metrics or the training loop's Hungarian matches.
            head = RegionQueryHead(hidden=4)
            head.load_state_dict(initial if stage == "initial" else checkpoint["state_dict"])
            head.use_relations = config["use_relations"][branch]
            prediction = runner.predict(head, data["axes"][key], 2, "cpu")
            scored = runner.evaluate_partial_structure(prediction, data["targets"][key], membership_threshold=.5,
                manifest=data["manifest"], direction_bridge_ledger=[r["target_transport"] for r in data["transport"][key]])
            with np.load(tmp_path / "artifacts" / (prefix + "__predictions.npz"), allow_pickle=False) as saved:
                assert set(saved.files) == {f.name for f in fields(RegionPrediction)} | {"scored_member_mask", "unique_center_query"}
                assert saved["centers_m"].shape == (3, 64, 3)
                for field in fields(RegionPrediction):
                    assert np.array_equal(saved[field.name], getattr(prediction, field.name).numpy())
                assert np.array_equal(saved["scored_member_mask"], scored.scored_member_mask.numpy())
                assert np.array_equal(saved["unique_center_query"], scored.unique_center_query.numpy())
            metrics = json.loads((tmp_path / "metrics" / (prefix + ".json")).read_text())
            assert metrics["aggregate"] == scored.summary == result["evaluation"][stage][branch]["aggregate"]
            assert set(metrics["parents"]) == {"synthetic_parent_a", "synthetic_parent_b"}
            observations = json.loads((tmp_path / "artifacts" / (prefix + "__observations.json")).read_text())
            assert len(observations) == 3
            for parent in metrics["parents"]:
                root = ET.parse(tmp_path / "previews" / (prefix + "__" + parent + ".svg")).getroot()
                assert root.tag.endswith("svg")
    assert len(list((tmp_path / "previews").glob("*.svg"))) == 12
    assert len(list((tmp_path / "artifacts").glob("*__predictions.npz"))) == 6
    assert all(torch.equal(value, before_axes[key]) and value.grad is None for key, value in data["axes"].items())


def lifecycle(tmp_path, monkeypatch):
    runner = module(monkeypatch)
    monkeypatch.setattr(runner, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(runner, "validate_partial_structure_training_card", lambda card: SimpleNamespace(passed=True, errors=[]))
    card = dict(training=settings(), evaluation={"member_threshold": .5}, sealed_sources={})
    write(tmp_path / "configs/card.json", card)
    spec = dict(gate=3, date="20260905", slug="synthetic_partial_training", seed=0,
                data_card="configs/card.json", operation="training", wall_time_cap_s=1800,
                training=card["training"], evaluation=card["evaluation"], source_sha256={},
                expected_versions={"python": runner.platform.python_version(), "numpy": runner.np.__version__,
                    "torch": runner.torch.__version__, "zarr": runner.zarr.__version__, "scipy": runner.scipy.__version__})
    run = tmp_path / "results/gate3_semantics" / build_run_id(spec)
    folders(run)
    write(run / "config/run_spec.json", spec); write(run / "config/data_card.json", card)
    write(run / "RUN_STATE.json", {"state": "CREATED_NOT_EXECUTED"})
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    def forbidden(*args, **kwargs): raise AssertionError("no source read, optimizer or GPU permitted in lifecycle failure check")
    monkeypatch.setattr(runner, "load_training_inputs", forbidden)
    monkeypatch.setattr(runner, "run_training", forbidden)
    return runner, spec, run


def test_execute_cuda_absence_seals_failed_and_refuses_retry(tmp_path, monkeypatch):
    runner, spec, run = lifecycle(tmp_path, monkeypatch)
    assert runner.execute(spec, run) == 1
    assert json.loads((run / "RUN_STATE.json").read_text())["state"] == "FAILED"
    summary = json.loads((run / "metrics/summary.json").read_text())
    assert "required CUDA device unavailable" in summary["error"]
    assert not summary["scientific_gate_pass"]
    assert not summary["full_three_class_ready"]
    seal = run / "artifacts/evidence_sha256.txt"
    for line in seal.read_text().splitlines():
        checksum, relative = line.split(None, 1)
        assert digest(tmp_path / relative) == checksum
    before = {str(p.relative_to(run)): digest(p) for p in run.rglob("*") if p.is_file()}
    with pytest.raises(RuntimeError, match="no overwrite/retry"):
        runner.execute(spec, run)
    assert before == {str(p.relative_to(run)): digest(p) for p in run.rglob("*") if p.is_file()}


def test_execute_source_drift_stops_before_hardware_or_data(tmp_path, monkeypatch):
    runner, spec, run = lifecycle(tmp_path, monkeypatch)
    source = tmp_path / "synthetic_source.txt"; source.write_text("synthetic source only")
    spec["source_sha256"] = {"synthetic_source.txt": "0" * 64}
    write(run / "config/run_spec.json", spec)
    def forbidden(): raise AssertionError("hardware should not be queried after source drift")
    monkeypatch.setattr(torch.cuda, "is_available", forbidden)
    assert runner.execute(spec, run) == 1
    assert "frozen source drift" in json.loads((run / "metrics/summary.json").read_text())["error"]
    with pytest.raises(RuntimeError, match="no overwrite/retry"):
        runner.execute(spec, run)


def test_execute_spec_mismatch_preserves_unexecuted_run(tmp_path, monkeypatch):
    runner, spec, run = lifecycle(tmp_path, monkeypatch)
    before = digest(run / "RUN_STATE.json")
    with pytest.raises(RuntimeError, match="exact fresh run"):
        runner.execute({**spec, "wall_time_cap_s": 999}, run)
    assert digest(run / "RUN_STATE.json") == before
    assert not (run / "metrics/summary.json").exists()
