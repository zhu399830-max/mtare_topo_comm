"""Actual synthetic scoring/save path; no maps, cached data or checkpoints."""
import importlib
import json
from pathlib import Path
from types import SimpleNamespace
import xml.etree.ElementTree as ET

import pytest
import torch

from test_gse_assignment_runner import folders, synthetic
from mtare_topo.governance import build_run_id


def module(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[3] / "tools/v3"))
    return importlib.import_module("run_gse_geometry_bound_rank_v1")


def test_full_synthetic_scoring_save_and_all_parents(tmp_path, monkeypatch):
    runner = module(monkeypatch)
    folders(tmp_path)
    data = synthetic(runner)
    result = runner.analyze_and_save(tmp_path, data)
    assert result["cached_prediction_observations"] == 6
    assert result["original_scores_reproduced"] and result["diagnostic_not_calibration"]
    assert result["optimizer_steps"] == result["model_inference"] == result["checkpoint_reads"] == 0
    for branch in runner.BRANCHES:
        r = result["branches"][branch]["all"]
        assert r["members"]["average_precision"] == .25
        assert r["members"]["auroc"] == .5
        assert r["members"]["confusion"] == dict(tp=2, fp=6, fn=0, tn=0)
        assert r["junction"]["average_precision"] == r["junction"]["auroc"] == .5
        assert "fixed_threshold" not in r["junction"] and "confusion" not in r["junction"]
        assert result["branches"][branch]["synthetic_a"]["junction"]["auroc"] is None
        values = json.loads((tmp_path / "artifacts" / (branch + "__scored_probabilities.json")).read_text())
        assert len(values["members"]["label"]) == 8
        assert values["members"]["query"] == [0] * 8
    paths = list((tmp_path / "previews").glob("*.svg"))
    assert len(paths) == 2
    for path in paths:
        assert ET.parse(path).getroot().tag.endswith("svg")
    assert len((tmp_path / "logs/diagnostic.jsonl").read_text().splitlines()) == 3


@pytest.mark.parametrize("issue", ["aggregate", "parent", "query", "mask"])
def test_drift_rejected_not_rescored_as_new_truth(tmp_path, monkeypatch, issue):
    runner = module(monkeypatch)
    folders(tmp_path)
    data = synthetic(runner)
    original = data["training_summary"]["result"]["evaluation"]["final"]["gt_axes"]
    if issue == "aggregate": original["aggregate"]["observations"] += 1
    if issue == "parent": original["parents"]["synthetic_a"]["observations"] += 1
    if issue == "query": data["saved_scoring"]["gt_axes"]["unique_center_query"][0, 0] = 3
    if issue == "mask": data["saved_scoring"]["gt_axes"]["scored_member_mask"][0, 0, 0] = False
    with pytest.raises(ValueError, match="drift"):
        runner.analyze_and_save(tmp_path, data)


def test_bce_stable_for_extreme_logits(monkeypatch):
    runner = module(monkeypatch)
    result = runner.describe(dict(probability=torch.tensor([0., 1.]),
        label=torch.tensor([1., 0.]), logit=torch.tensor([-1000., 1000.])), torch.tensor([True, True]))
    assert result["final_logit_bce"] == 1000
    assert result["average_precision"] == .5 and result["auroc"] == 0
    json.dumps(result, allow_nan=False)


def test_failure_sealed_no_retry(tmp_path, monkeypatch):
    runner = module(monkeypatch)
    monkeypatch.setattr(runner, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(runner, "validate_geometry_bound_rank_card", lambda _: SimpleNamespace(passed=True, errors=[]))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    def forbidden(*args, **kwargs): raise AssertionError("data must not be read")
    monkeypatch.setattr(runner, "load_rank_inputs", forbidden)
    card = {"rank_policy": {}}
    (tmp_path / "card.json").write_text(json.dumps(card))
    spec = dict(gate=3,date="20260905",slug="synthetic_rank",seed=0,operation="data_export",
        data_card="card.json",wall_time_cap_s=300,rank_policy={})
    run = tmp_path / "results/gate3_semantics" / build_run_id(spec)
    folders(run)
    (run / "config/run_spec.json").write_text(json.dumps(spec))
    (run / "config/data_card.json").write_text(json.dumps(card))
    (run / "RUN_STATE.json").write_text(json.dumps({"state":"CREATED_NOT_EXECUTED"}))
    assert runner.execute(spec, run) == 1
    assert json.loads((run / "RUN_STATE.json").read_text())["state"] == "FAILED"
    for line in (run / "artifacts/evidence_sha256.txt").read_text().splitlines():
        digest,relative = line.split(None,1)
        assert runner.sha(tmp_path / relative) == digest
    before = {str(p):runner.sha(p) for p in run.rglob("*") if p.is_file()}
    with pytest.raises(RuntimeError, match="no overwrite/retry"):
        runner.execute(spec,run)
    assert before == {str(p):runner.sha(p) for p in run.rglob("*") if p.is_file()}
