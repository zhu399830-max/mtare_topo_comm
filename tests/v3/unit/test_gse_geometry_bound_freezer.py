import hashlib
import importlib
import json
from pathlib import Path

import pytest

from mtare_topo.governance import build_run_id
from mtare_topo.governance_geometry_bound import validate_geometry_bound_training_card
from tests.v3.unit.test_gse_partial_structure_training_card import card as original_card


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,sort_keys=True,indent=2)+"\n")


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def fixture(tmp_path,monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[3]/"tools/v3"))
    freezer=importlib.import_module("freeze_gse_geometry_bound_training_v1")
    monkeypatch.setattr(freezer,"PROJECT_ROOT",tmp_path)
    base=original_card();tables={}
    for path in freezer.UNCHANGED_SOURCE_FILES:
        source=tmp_path/path;source.parent.mkdir(parents=True,exist_ok=True);source.write_text("# original synthetic source\n")
    for role,specpath,sealhash in (("base",freezer.BASE_SPEC,freezer.TRAINING_SEAL_SHA256),
            ("attribution",freezer.ATTRIBUTION_SPEC,freezer.ATTRIBUTION_SEAL_SHA256)):
        cardpath=f"configs/{role}_card.json";write(tmp_path/cardpath,base if role=="base" else {"metadata_only":True})
        spec={"gate":3,"seed":0,"date":"20260905","slug":"synthetic_"+role,"data_card":cardpath,
            "command":["env","A=1","B=1","C=1","D=1","E=1","/synthetic/bin/python"],
            "expected_counts":{"optimizer_steps":900,"head_inference_windows":1080},
            "source_sha256":{path:sha(tmp_path/path) for path in freezer.UNCHANGED_SOURCE_FILES}}
        write(tmp_path/specpath,spec)
        root="results/gate3_semantics/"+build_run_id(spec)
        tables[sealhash]={root+"/config/run_spec.json":sha(tmp_path/specpath),root+"/config/data_card.json":sha(tmp_path/cardpath)}
    tables[freezer.EXPORT_SEAL_SHA256]={v["path"]:v["sha256"] for v in base["sources"].values()}
    def selector(path,wanted,digest):
        assert wanted==set(tables[digest]);return dict(tables[digest])
    monkeypatch.setattr(freezer,"selected_seal_entries",selector)
    method=tmp_path/freezer.METHOD_DOC;method.parent.mkdir(parents=True,exist_ok=True);method.write_text("Synthetic loss correction")
    scripts=tmp_path/"tools/v3";scripts.mkdir(parents=True)
    for filename in ("run_gse_geometry_bound_training_v1.py","freeze_gse_geometry_bound_training_v1.py",
        "freeze_gse_partial_structure_export_v1.py","run_gse_partial_structure_training_v1.py",
        "run_gse_supported_construction_teacher_v1.py","_bootstrap.py"):(scripts/filename).write_text("# synthetic\n")
    real_sha=freezer.sha
    def metadata_sha(path):
        assert path.suffix not in (".npz",".pt",".ckpt")
        assert path.name not in ("summary.json","training_history.json","manifest.json","target_transport.json")
        return real_sha(path)
    monkeypatch.setattr(freezer,"sha",metadata_sha)
    return freezer


def test_metadata_only_single_variable_freeze_documents(tmp_path,monkeypatch):
    f=fixture(tmp_path,monkeypatch);card,spec=f.documents()
    assert validate_geometry_bound_training_card(card).passed
    assert spec["training"]==card["base_training_card"]["training"]
    assert spec["evaluation"]==card["base_training_card"]["evaluation"]
    assert spec["loss_policy"]=="geometry_only_unique_center_binding_v1"
    assert spec["command"][1]=="CUBLAS_WORKSPACE_CONFIG=:4096:8" and spec["command"][6]=="/synthetic/bin/python"
    assert not any(arg.startswith("CUDA_VISIBLE_DEVICES=") for arg in spec["command"])
    assert not any((tmp_path/v["path"]).exists() for v in card["sources"].values())
    assert "tools/v3/run_gse_partial_structure_training_v1.py" in spec["source_sha256"]
    assert "tools/v3/run_gse_supported_construction_teacher_v1.py" in spec["source_sha256"]
    assert not (tmp_path/f.CARD).exists() and not (tmp_path/f.SPEC).exists()


def test_attribution_metadata_snapshot_drift(tmp_path,monkeypatch):
    f=fixture(tmp_path,monkeypatch);write(tmp_path/"configs/attribution_card.json",{"changed":True})
    with pytest.raises(ValueError,match="snapshot drift"):f.documents()


def test_single_variable_assertion_rejects_changed_original_loss_source(tmp_path,monkeypatch):
    f=fixture(tmp_path,monkeypatch)
    (tmp_path/f.UNCHANGED_SOURCE_FILES[0]).write_text("# changed original loss\n")
    with pytest.raises(ValueError,match="source changed"):f.documents()


def test_no_overwrite_or_execution(tmp_path,monkeypatch):
    f=fixture(tmp_path,monkeypatch);write(tmp_path/f.CARD,{"preserved":True})
    monkeypatch.setattr(f,"documents",lambda:pytest.fail("reject before preparing sources"))
    with pytest.raises(RuntimeError,match="overwrite"):f.main()
