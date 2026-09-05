import hashlib
import importlib
import json
from pathlib import Path

import pytest

from mtare_topo.governance import build_run_id
from mtare_topo.governance_geometry_bound_rank import validate_geometry_bound_rank_card
from tests.v3.unit.test_gse_geometry_bound_card import card as corrective_card


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,sort_keys=True,indent=2)+"\n")


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def fixture(tmp_path,monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[3]/"tools/v3"))
    f=importlib.import_module("freeze_gse_geometry_bound_rank_v1");monkeypatch.setattr(f,"PROJECT_ROOT",tmp_path)
    old=corrective_card();cardpath="configs/corrective_card.json";write(tmp_path/cardpath,old)
    spec={"gate":3,"seed":0,"date":"20260905","slug":"synthetic_rank_source","data_card":cardpath,
        "command":["env","A=1","B=1","C=1","D=1","E=1","/synthetic/bin/python"]}
    write(tmp_path/f.CORRECTIVE_SPEC,spec);root="results/gate3_semantics/"+build_run_id(spec)
    entries={root+"/config/run_spec.json":sha(tmp_path/f.CORRECTIVE_SPEC),root+"/config/data_card.json":sha(tmp_path/cardpath)}
    entries.update({root+("/metrics/" if k=="corrective_summary" else "/artifacts/")+f.SOURCE_NAMES[k]:"d"*64
        for k in ("gt_final","predicted_final","no_relations_final","corrective_summary")})
    exported={v["path"]:v["sha256"] for v in old["sources"].values()}
    def select(path,wanted,digest):
        if digest==f.CORRECTIVE_SEAL_SHA256:chosen=entries
        else:assert digest==f.EXPORT_SEAL_SHA256;chosen=exported
        assert wanted==set(chosen);return dict(chosen)
    monkeypatch.setattr(f,"selected_seal_entries",select)
    method=tmp_path/f.METHOD_DOC;method.parent.mkdir(parents=True,exist_ok=True);method.write_text("Synthetic rank contract")
    scripts=tmp_path/"tools/v3";scripts.mkdir(parents=True)
    for name in ("run_gse_geometry_bound_rank_v1.py","freeze_gse_geometry_bound_rank_v1.py","freeze_gse_partial_structure_export_v1.py",
        "run_gse_partial_structure_training_v1.py","run_gse_supported_construction_teacher_v1.py","_bootstrap.py"):(scripts/name).write_text("# synthetic\n")
    originalsha=f.sha
    def metadata_sha(path):
        assert path.suffix not in (".npz",".pt",".ckpt")
        assert path.name not in ("summary.json","manifest.json","training_history.json","target_transport.json")
        return originalsha(path)
    monkeypatch.setattr(f,"sha",metadata_sha)
    return f,cardpath


def test_metadata_only_rank_preparation_does_not_decode_or_freeze(tmp_path,monkeypatch):
    f,_=fixture(tmp_path,monkeypatch);c,s=f.documents()
    assert validate_geometry_bound_rank_card(c).passed
    assert len(c["sources"])==8 and all(not (tmp_path/v["path"]).exists() for v in c["sources"].values())
    assert s["expected_counts"]["cached_prediction_observations"]==540
    assert s["rank_policy"]["undefined_ap"]==s["rank_policy"]["undefined_auc"]=="null_if_single_class"
    assert s["rank_policy"]["probability_diagnostic"]=="final_logits_bce_vs_constant_prior"
    assert s["command"][5]=="/synthetic/bin/python" and s["wall_time_cap_s"]==300
    assert not (tmp_path/f.CARD).exists() and not (tmp_path/f.SPEC).exists()


def test_corrective_metadata_drift_stops_preparation(tmp_path,monkeypatch):
    f,path=fixture(tmp_path,monkeypatch);write(tmp_path/path,{"drift":True})
    with pytest.raises((ValueError,KeyError)):f.documents()


def test_no_overwrite(tmp_path,monkeypatch):
    f,_=fixture(tmp_path,monkeypatch);write(tmp_path/f.CARD,{"preserve":True})
    monkeypatch.setattr(f,"documents",lambda:pytest.fail("must stop before source preparation"))
    with pytest.raises(RuntimeError,match="overwrite"):f.main()
