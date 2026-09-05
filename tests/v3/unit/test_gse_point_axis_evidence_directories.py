import importlib
import json
from pathlib import Path


def test_missing_environment_directory_is_created_without_touching_other_runs(tmp_path,monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[3]/"tools/v3"))
    module=importlib.import_module("run_gse_point_axis_evidence_v1r")
    old=tmp_path/"old";old.mkdir();(old/"RUN_STATE.json").write_text('{"state":"FAILED"}')
    target=tmp_path/"new/environment/execution_environment.json"
    module.write_json_with_parent(target,{"device":"cpu","optimizer_steps":0})
    assert json.loads(target.read_text())=={"device":"cpu","optimizer_steps":0}
    assert (old/"RUN_STATE.json").read_text()=='{"state":"FAILED"}'
    module.write_json_with_parent(target,{"device":"cpu","optimizer_steps":0})
