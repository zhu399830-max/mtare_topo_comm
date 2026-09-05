"""Synthetic actual attribution/save flow; no dataset/model/weight access."""
from copy import deepcopy
import hashlib
import importlib
import json
from pathlib import Path
from types import SimpleNamespace
import xml.etree.ElementTree as ET

import pytest
import torch

from mtare_topo.evaluation.gse_partial_structure import evaluate_partial_structure
from mtare_topo.governance import build_run_id
from mtare_topo.representation.gse_region_queries import RegionPrediction, RegionTargets


def module(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[3] / "tools/v3"))
    return importlib.import_module("run_gse_assignment_attribution_v1")


def folders(root):
    for name in ("config", "artifacts", "metrics", "logs", "previews"):
        (root / name).mkdir(parents=True, exist_ok=True)


def synthetic(runner):
    b, n = 2, 4
    support = torch.ones((b, n), dtype=torch.bool)
    prediction = RegionPrediction(torch.tensor([[[0., 0., 0.], [1., 0., 0.], [3., 0., 0.], [4., 0., 0.]]]).repeat(b, 1, 1),
        torch.tensor([[[4., -4., -4.], [-4., 4., -4.], [4., -4., -4.], [-4., 4., -4.]]]).repeat(b, 1, 1),
        torch.zeros((b, n)), torch.zeros((b, n, n)), torch.full((b, n), .5), support,
        support[:, :, None] & support[:, None])
    target = RegionTargets(torch.zeros((b, 1, 3)), torch.ones((b, 1), dtype=torch.bool),
        torch.tensor([[0], [1]]), torch.ones((b, 1), dtype=torch.bool),
        torch.tensor([[[1., 0., 0., 0.]]]).repeat(b, 1, 1),
        torch.ones((b, 1, n), dtype=torch.bool), torch.zeros(b, dtype=torch.bool))
    manifest = [{"task": "synthetic_a", "row_index": 0}, {"task": "synthetic_b", "row_index": 0}]
    ledger = dict(original_member_positive=1, original_member_negative=3, transferred_member_positive=1,
                  transferred_member_negative=3, unknown_correspondence_member_positive=0, unknown_correspondence_member_negative=0)
    bridge = [dict(ledger) for _ in range(b)]
    evaluated = evaluate_partial_structure(prediction, target, membership_threshold=.5, manifest=manifest, direction_bridge_ledger=bridge)
    parents = {row["task"]: evaluate_partial_structure(runner.sliced(prediction, [i]), runner.sliced(target, [i]),
        membership_threshold=.5, manifest=[row], direction_bridge_ledger=[bridge[i]]).summary for i, row in enumerate(manifest)}
    result = dict(data={"manifest": manifest, "targets": {key: deepcopy(target) for key in ("gt", "predicted")},
        "transport": {key: [{"target_transport": dict(x)} for x in bridge] for key in ("gt", "predicted")}},
        predictions={key: deepcopy(prediction) for key in runner.BRANCHES},
        saved_scoring={key: {"scored_member_mask": evaluated.scored_member_mask.clone(), "unique_center_query": evaluated.unique_center_query.clone()} for key in runner.BRANCHES},
        training_summary={"result": {"evaluation": {"final": {key: {"aggregate": deepcopy(evaluated.summary), "parents": deepcopy(parents)} for key in runner.BRANCHES}}}},
        last_epoch_batches=[[1], [0]], read_hashes={})
    return result


def test_real_cpu_core_full_save_original_reproduction_and_all_plots(tmp_path, monkeypatch):
    runner = module(monkeypatch); folders(tmp_path)
    data = synthetic(runner)
    result = runner.analyze_and_save(tmp_path, data)
    assert result["cached_prediction_observations"] == 6
    assert result["original_scores_reproduced"] and result["joint_diagnostic_only"]
    assert result["model_inference"] == result["checkpoint_reads"] == result["optimizer_steps"] == 0
    for branch in runner.BRANCHES:
        saved = json.loads((tmp_path / "artifacts" / (branch + "__attribution.json")).read_text())
        assert saved["geometry_summary"] == data["training_summary"]["result"]["evaluation"]["final"][branch]["aggregate"]
        assert len(saved["target_comparisons"]) == 2
        assert len(saved["junction_table"]) == 1
    assert len(list((tmp_path / "previews").glob("*.svg"))) == 6
    for path in (tmp_path / "previews").glob("*.svg"):
        assert ET.parse(path).getroot().tag.endswith("svg")
    assert len((tmp_path / "logs/diagnostic.jsonl").read_text().splitlines()) == 3


@pytest.mark.parametrize("issue", ["aggregate", "parent", "query", "mask"])
def test_original_scores_and_saved_masks_are_not_silently_replaced(tmp_path, monkeypatch, issue):
    runner = module(monkeypatch); folders(tmp_path)
    data = synthetic(runner)
    old = data["training_summary"]["result"]["evaluation"]["final"]["gt_axes"]
    if issue == "aggregate": old["aggregate"]["observations"] += 1
    if issue == "parent": old["parents"]["synthetic_a"]["observations"] += 1
    if issue == "query": data["saved_scoring"]["gt_axes"]["unique_center_query"][0, 0] = 3
    if issue == "mask": data["saved_scoring"]["gt_axes"]["scored_member_mask"][0, 0, 0] = False
    with pytest.raises(ValueError, match="reproduced"):
        runner.analyze_and_save(tmp_path, data)
    assert not list((tmp_path / "previews").glob("*.svg"))


def test_failure_is_sealed_and_cannot_be_retried(tmp_path, monkeypatch):
    runner = module(monkeypatch); monkeypatch.setattr(runner, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(runner, "validate_assignment_attribution_card", lambda _: SimpleNamespace(passed=True, errors=[]))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    def forbidden(*args, **kwargs): raise AssertionError("failure must precede data/model")
    monkeypatch.setattr(runner, "load_assignment_inputs", forbidden)
    card = {"attribution": {}}
    (tmp_path / "card.json").write_text(json.dumps(card))
    spec = dict(gate=3, date="20260905", slug="synthetic_attribution", seed=0, operation="data_export",
        data_card="card.json", wall_time_cap_s=300, attribution={})
    run = tmp_path / "results/gate3_semantics" / build_run_id(spec); folders(run)
    (run / "config/run_spec.json").write_text(json.dumps(spec))
    (run / "config/data_card.json").write_text(json.dumps(card))
    (run / "RUN_STATE.json").write_text(json.dumps({"state": "CREATED_NOT_EXECUTED"}))
    assert runner.execute(spec, run) == 1
    assert json.loads((run / "RUN_STATE.json").read_text())["state"] == "FAILED"
    assert "CPU-only" in json.loads((run / "metrics/summary.json").read_text())["error"]
    for line in (run / "artifacts/evidence_sha256.txt").read_text().splitlines():
        digest, relative = line.split(None, 1)
        assert hashlib.sha256((tmp_path / relative).read_bytes()).hexdigest() == digest
    before = {str(p): runner.sha(p) for p in run.rglob("*") if p.is_file()}
    with pytest.raises(RuntimeError, match="no overwrite/retry"): runner.execute(spec, run)
    assert before == {str(p): runner.sha(p) for p in run.rglob("*") if p.is_file()}
