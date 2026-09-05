"""End-to-end synthetic factorial training, final scoring and evidence saves."""
from copy import deepcopy
import importlib
import json
from pathlib import Path
from types import SimpleNamespace
import xml.etree.ElementTree as ET

import pytest
import torch

from mtare_topo.governance import build_run_id
from tests.v3.unit.test_gse_partial_structure_training_runner import synthetic,settings,folders,write


def module(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[3]/"tools/v3"))
    return importlib.import_module("run_gse_class_balance_factorial_v1")


def baseline_fixture(tmp_path,monkeypatch):
    runner = module(monkeypatch)
    old = importlib.import_module("run_gse_geometry_bound_training_v1")
    root = tmp_path/"baseline";folders(root)
    data,config = synthetic(),settings()
    result = old.run_training(root,data,config,{"member_threshold":.5})
    ranking = runner.rank_saved_final(root,data,result["evaluation"])
    card = {"training":config,"evaluation":{"member_threshold":.5},
        "balance_policy":{"member_class_counts":[6,6],"event_class_counts":[3,3,0]},
        "baseline_initial_state_sha256":runner.sha(root/"artifacts/shared_initial_state.pt")}
    return runner,data,card,{"result":result},{"result":ranking}


def test_actual_three_factor_training_and_final_ranking(tmp_path,monkeypatch):
    runner,data,card,baseline,rank = baseline_fixture(tmp_path,monkeypatch)
    run = tmp_path/"new";folders(run)
    updates = []
    result = runner.run_factorial(run,data,card,baseline,rank,
        step_callback=lambda factor,branch,record:updates.append((factor,branch,record["step"])))
    assert result["optimizer_steps"] == 18 and result["head_inference_windows"] == 54
    assert result["backbone_windows"] == result["new_sensor_frames"] == result["old_checkpoint_reads"] == 0
    assert result["selected_factor"] is None and not result["threshold_search"]
    assert list(result["factors"]) == ["00","10","01","11"]
    assert result["factors"]["00"]["reused_not_retrained"]
    assert updates == [(f,b,s) for f in runner.FACTORS for b in runner.BRANCHES for s in (1,2)]
    for factor in runner.FACTORS:
        root = run/"artifacts/factors"/factor
        value = result["factors"][factor]
        assert value["evaluation"]["initial"] == baseline["result"]["evaluation"]["initial"]
        assert value["baseline_initial_sha_verified"]
        assert runner.sha(root/"artifacts/shared_initial_state.pt") == card["baseline_initial_state_sha256"]
        assert value["rank"]["original_scores_reproduced"]
        assert len(list((root/"previews").glob("*.svg"))) == 14
        for path in (root/"previews").glob("*.svg"):
            assert ET.parse(path).getroot().tag.endswith("svg")
        for branch in runner.BRANCHES:
            checkpoint = torch.load(root/"artifacts"/(branch+"__final.pt"),weights_only=True)
            assert checkpoint["factor"] == factor and checkpoint["step"] == 2
            assert checkpoint["member_class_counts"] == ([6,6] if factor[0]=="1" else None)
            assert checkpoint["event_class_counts"] == ([3,3,0] if factor[1]=="1" else None)
    effects = json.loads((run/"metrics/factorial_effects.json").read_text())
    assert set(effects) == set(runner.BRANCHES)
    assert set(effects["gt_axes"]["all"]["contrasts"]) == {"10-00","01-00","11-01","11-10","interaction_11-10-01+00"}
    # Undefined single-class populations stay undefined, never zero-filled.
    altered = deepcopy(result["factors"])
    for factor in altered:
        altered[factor]["rank"]["branches"]["gt_axes"]["synthetic_parent_a"]["junction"]["auroc"] = None
    assert runner.factorial_effects(altered)["gt_axes"]["synthetic_parent_a"]["contrasts"]["10-00"]["junction_auc"] is None
    json.dumps(result,allow_nan=False)


@pytest.mark.parametrize("kind",["initial_digest","initial_metric"])
def test_baseline_drift_prevents_new_training(tmp_path,monkeypatch,kind):
    runner,data,card,baseline,rank = baseline_fixture(tmp_path,monkeypatch)
    root = tmp_path/"new";folders(root)
    if kind=="initial_digest":card["baseline_initial_state_sha256"] = "0"*64
    else:baseline["result"]["evaluation"]["initial"]["gt_axes"]["aggregate"]["observations"] += 1
    def forbidden(*args,**kwargs):raise AssertionError("training must not run")
    monkeypatch.setattr(runner,"train_class_balanced_structure",forbidden)
    with pytest.raises(ValueError,match="initial"):
        runner.run_factor(root,data,card["training"],card["evaluation"],"10",[6,6],[3,3,0],
            baseline["result"]["evaluation"]["initial"],card["baseline_initial_state_sha256"])
    assert not list((root/"artifacts").glob("*__final.pt"))


def test_cuda_absence_sealed_once_before_data(tmp_path,monkeypatch):
    runner = module(monkeypatch)
    monkeypatch.setattr(runner,"PROJECT_ROOT",tmp_path)
    monkeypatch.setattr(runner,"validate_class_balance_factorial_card",lambda _:SimpleNamespace(passed=True,errors=[]))
    card = dict(training={},evaluation={},balance_policy={},rank_policy={},sealed_sources={})
    write(tmp_path/"card.json",card)
    spec = dict(gate=3,date="20260905",slug="synthetic_factorial",seed=0,operation="training",
        data_card="card.json",wall_time_cap_s=1800,source_sha256={},
        **{k:card[k] for k in ("training","evaluation","balance_policy","rank_policy")},
        expected_versions={"python":runner.platform.python_version(),"numpy":runner.np.__version__,
            "torch":runner.torch.__version__,"zarr":runner.zarr.__version__,"scipy":runner.scipy.__version__})
    run = tmp_path/"results/gate3_semantics"/build_run_id(spec);folders(run)
    write(run/"config/run_spec.json",spec);write(run/"config/data_card.json",card)
    write(run/"RUN_STATE.json",{"state":"CREATED_NOT_EXECUTED"})
    monkeypatch.setattr(torch.cuda,"is_available",lambda:False)
    def forbidden(*args,**kwargs):raise AssertionError("no input or optimizer")
    monkeypatch.setattr(runner,"load_inputs",forbidden)
    monkeypatch.setattr(runner,"run_factorial",forbidden)
    assert runner.execute(spec,run)==1
    assert "required CUDA unavailable" in json.loads((run/"metrics/summary.json").read_text())["error"]
    assert json.loads((run/"RUN_STATE.json").read_text())["state"]=="FAILED"
    before = {str(p):runner.sha(p) for p in run.rglob("*") if p.is_file()}
    for line in (run/"artifacts/evidence_sha256.txt").read_text().splitlines():
        digest,path = line.split(None,1)
        assert runner.sha(tmp_path/path)==digest
    with pytest.raises(RuntimeError,match="no overwrite/retry"):runner.execute(spec,run)
    assert before=={str(p):runner.sha(p) for p in run.rglob("*") if p.is_file()}
