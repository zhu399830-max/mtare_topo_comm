import hashlib
import importlib
import json
from pathlib import Path

import pytest

from mtare_topo.governance import build_run_id
from mtare_topo.governance_partial_structure_training import validate_partial_structure_training_card
from tests.v3.unit.test_gse_partial_structure_card import card as exported


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,sort_keys=True,indent=2)+"\n")


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def fixture(tmp_path,monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[3]/"tools/v3"))
    freezer=importlib.import_module("freeze_gse_partial_structure_training_v1")
    monkeypatch.setattr(freezer,"PROJECT_ROOT",tmp_path)
    original=exported();cardpath="fixtures/export_card.json"
    write(tmp_path/cardpath,original)
    spec={"gate":3,"date":"20260905","seed":0,"slug":"synthetic_partial_export","data_card":cardpath,
          "command":["env","A=1","B=1","C=1","D=1","/synthetic/bin/python"]}
    write(tmp_path/freezer.EXPORT_SPEC,spec)
    root="results/gate3_semantics/"+build_run_id(spec)
    selected={root+"/config/run_spec.json":sha(tmp_path/freezer.EXPORT_SPEC),root+"/config/data_card.json":sha(tmp_path/cardpath)}
    selected.update({root+("/metrics/" if key=="summary" else "/artifacts/")+name:"e"*64
                     for key,name in freezer.SOURCE_NAMES.items()})
    seal=tmp_path/root/"artifacts/evidence_sha256.txt"
    seal.parent.mkdir(parents=True,exist_ok=True)
    seal.write_text("".join(h+"  "+path+"\n" for path,h in selected.items())+"0"*64+"  /forbidden/C07/model.pt\n")
    # Exact real seal is a validator constant. Mock only seal validation in the
    # freezer, leaving its selector/hash checks covered by export freezer tests.
    def selected_entries(path,wanted,expected):
        assert path==seal and expected==freezer.EXPORT_SEAL_SHA256
        assert wanted==set(selected)
        return dict(selected)
    monkeypatch.setattr(freezer,"selected_seal_entries",selected_entries)
    method=tmp_path/freezer.METHOD_DOC;method.parent.mkdir(parents=True,exist_ok=True);method.write_text("Synthetic training scope")
    tools=tmp_path/"tools/v3";tools.mkdir(parents=True)
    for name in ("run_gse_partial_structure_training_v1.py","freeze_gse_partial_structure_training_v1.py",
                 "freeze_gse_partial_structure_export_v1.py","_bootstrap.py","run_gse_supported_construction_teacher_v1.py"):
        (tools/name).write_text("# synthetic source\n")
    test_file=tmp_path/"tests/v3/unit/test_gse_partial_training_inputs.py"
    test_file.parent.mkdir(parents=True,exist_ok=True);test_file.write_text("# synthetic test\n")
    real_sha=freezer.sha
    def safe_sha(path):
        assert path.suffix not in (".npz",".pt",".ckpt")
        assert path.name not in ("manifest.json","target_transport.json","summary.json")
        return real_sha(path)
    monkeypatch.setattr(freezer,"sha",safe_sha)
    return freezer,original,spec


def test_metadata_only_preparation_independent_training_scope(tmp_path,monkeypatch):
    freezer,original,oldspec=fixture(tmp_path,monkeypatch)
    value,spec=freezer.documents()
    assert validate_partial_structure_training_card(value).passed
    assert value["export_identity_reference"]==original
    assert value["approval"]["authorized_operations"]==["training"]
    assert original["approval"]["authorized_operations"]==["data_export"]
    assert value["training"]["steps_per_branch"]==300 and spec["expected_counts"]["optimizer_steps"]==900
    assert spec["expected_counts"]["head_inference_windows"]==1080
    assert spec["command"][6]=="/synthetic/bin/python"
    assert all(not (tmp_path/s["path"]).exists() for s in value["sources"].values())
    assert not (tmp_path/freezer.CARD).exists() and not (tmp_path/freezer.SPEC).exists()
    assert set(value["sources"])=={"inputs","manifest","target_transport","summary"}
    assert not any("C07" in path for path in value["sealed_sources"])


@pytest.mark.parametrize("which",["spec","card"])
def test_original_export_snapshot_drift_stops(tmp_path,monkeypatch,which):
    freezer,original,spec=fixture(tmp_path,monkeypatch)
    path=tmp_path/(freezer.EXPORT_SPEC if which=="spec" else spec["data_card"])
    original=json.loads(path.read_text());original["tampered"]=True;write(path,original)
    with pytest.raises(ValueError,match="snapshot drift"):freezer.documents()
    assert not (tmp_path/freezer.CARD).exists()


@pytest.mark.parametrize("existing",["CARD","SPEC"])
def test_freeze_never_overwrites(tmp_path,monkeypatch,existing):
    freezer,_,_=fixture(tmp_path,monkeypatch)
    write(tmp_path/getattr(freezer,existing),{"preserved":True})
    monkeypatch.setattr(freezer,"documents",lambda:pytest.fail("existing freeze must be rejected first"))
    with pytest.raises(RuntimeError,match="overwrite"):freezer.main()
