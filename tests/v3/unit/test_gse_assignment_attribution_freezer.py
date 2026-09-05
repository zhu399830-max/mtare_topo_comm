import hashlib
import importlib
import json
from pathlib import Path

import pytest

from mtare_topo.governance import build_run_id
from mtare_topo.governance_assignment_attribution import validate_assignment_attribution_card
from tests.v3.unit.test_gse_partial_structure_training_card import card as training_card


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,sort_keys=True,indent=2)+"\n")


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def fixture(tmp_path,monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[3]/"tools/v3"))
    freezer=importlib.import_module("freeze_gse_assignment_attribution_v1")
    monkeypatch.setattr(freezer,"PROJECT_ROOT",tmp_path)
    old=training_card();oldpath="configs/synthetic_training_card.json";write(tmp_path/oldpath,old)
    spec={"gate":3,"seed":0,"date":"20260905","slug":"synthetic_assignment_source","data_card":oldpath,
        "command":["env","A=1","B=1","C=1","D=1","E=1","/synthetic/bin/python"]}
    write(tmp_path/freezer.TRAINING_SPEC,spec)
    root="results/gate3_semantics/"+build_run_id(spec)
    train_entries={root+"/config/run_spec.json":sha(tmp_path/freezer.TRAINING_SPEC),root+"/config/data_card.json":sha(tmp_path/oldpath)}
    train_entries.update({root+("/metrics/" if k=="training_summary" else "/artifacts/")+freezer.SOURCE_NAMES[k]:"d"*64
        for k in ("gt_final","predicted_final","no_relations_final","history","training_summary")})
    export_entries={v["path"]:v["sha256"] for v in old["sources"].values()}
    def seal_entries(path,wanted,digest):
        if digest==freezer.TRAINING_SEAL_SHA256:entries=train_entries
        else:
            assert digest==freezer.EXPORT_SEAL_SHA256
            entries=export_entries
        assert set(entries)==wanted
        return dict(entries)
    monkeypatch.setattr(freezer,"selected_seal_entries",seal_entries)
    method=tmp_path/freezer.METHOD_DOC;method.parent.mkdir(parents=True,exist_ok=True);method.write_text("Synthetic attribution method")
    scripts=tmp_path/"tools/v3";scripts.mkdir(parents=True)
    for filename in ("run_gse_assignment_attribution_v1.py","freeze_gse_assignment_attribution_v1.py",
                     "freeze_gse_partial_structure_export_v1.py","_bootstrap.py"):(scripts/filename).write_text("# synthetic\n")
    real_sha=freezer.sha
    def metadata_sha(path):
        assert path.suffix not in (".npz",".pt",".ckpt")
        assert path.name not in ("training_history.json","manifest.json","summary.json","target_transport.json")
        return real_sha(path)
    monkeypatch.setattr(freezer,"sha",metadata_sha)
    return freezer,oldpath


def test_nine_payload_hashes_selected_without_decoding_or_freezing(tmp_path,monkeypatch):
    freezer,_=fixture(tmp_path,monkeypatch)
    c,s=freezer.documents()
    assert validate_assignment_attribution_card(c).passed
    assert len(c["sources"])==9 and all(not (tmp_path/v["path"]).exists() for v in c["sources"].values())
    assert s["expected_counts"]["cached_prediction_observations"]==540
    assert s["expected_counts"]["last_epoch_batches"]==10 and s["wall_time_cap_s"]==300
    assert s["command"][5]=="CUDA_VISIBLE_DEVICES="
    assert s["command"][6]=="/synthetic/bin/python"
    assert s["user_authorization"]["authorized_operations"]==["data_export"]
    assert not (tmp_path/freezer.CARD).exists() and not (tmp_path/freezer.SPEC).exists()


def test_original_training_card_snapshot_drift_denied(tmp_path,monkeypatch):
    freezer,oldpath=fixture(tmp_path,monkeypatch)
    value=json.loads((tmp_path/oldpath).read_text());value["drift"]=True;write(tmp_path/oldpath,value)
    with pytest.raises(ValueError,match="snapshot drift"):freezer.documents()


def test_refuse_overwrite_before_source_work(tmp_path,monkeypatch):
    freezer,_=fixture(tmp_path,monkeypatch);write(tmp_path/freezer.SPEC,{"preserve":True})
    monkeypatch.setattr(freezer,"documents",lambda:pytest.fail("must reject overwrite first"))
    with pytest.raises(RuntimeError,match="overwrite"):freezer.main()
