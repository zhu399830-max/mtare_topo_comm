import importlib
import json
from pathlib import Path

import pytest

from mtare_topo.governance_supported_teacher import validate_supported_teacher_card
from tests.v3.unit.test_gse_supported_teacher_card import card


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value))


def test_documents_bind_new_teacher_scope_without_opening_new_payloads(tmp_path,monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[3]/"tools/v3"))
    freezer=importlib.import_module("freeze_gse_supported_construction_teacher_v1")
    previous=card();tasks=previous["tasks"]
    construction_root="fixtures/constructions/fit";codebook_root="fixtures/codebooks/fit"
    sensor_seal="fixtures/sensor_seal.txt";teacher_seal="fixtures/teacher_seal.txt"
    write(tmp_path/"configs/probe_card.json",previous);write(tmp_path/"configs/audit_card.json",{})
    write(tmp_path/"fixtures/selection.json",previous["selected_rows"])
    source=tmp_path/sensor_seal
    source.parent.mkdir(parents=True,exist_ok=True)
    source.write_text("".join("c"*64+"  "+root+"/"+task+".json\n" for root in (construction_root,codebook_root) for task in tasks)
                      +"0"*64+"  /forbidden/C07/model.pt\n")
    (tmp_path/teacher_seal).write_text("")
    write(tmp_path/freezer.PROBE_SPEC,{"data_card":"configs/probe_card.json","sensor_root":previous["source_roots"]["sensor"],
        "teacher_root":previous["source_roots"]["teacher"],"source_seals":[sensor_seal,teacher_seal],
        "command":["env","A=1","B=1","C=1","D=1","E=1","/synthetic/python"]})
    write(tmp_path/freezer.AUDIT_SPEC,{"data_card":"configs/audit_card.json","selection_manifest":"fixtures/selection.json",
        "construction_root":construction_root})
    (tmp_path/"src").mkdir();(tmp_path/"tests/v3/unit").mkdir(parents=True)
    monkeypatch.setattr(freezer,"PROJECT_ROOT",tmp_path)
    def sha(path):
        assert not path.suffix in (".pt",".npz",".ckpt")
        if str(path.relative_to(tmp_path))==sensor_seal:return previous["sealed_sources"][sensor_seal]
        if str(path.relative_to(tmp_path))==teacher_seal:return previous["sealed_sources"][teacher_seal]
        return "f"*64
    monkeypatch.setattr(freezer,"sha",sha)
    value,spec=freezer.documents()
    assert validate_supported_teacher_card(value).passed
    assert spec["operation"]=="teacher_generation" and spec["command"][5]=="/synthetic/python"
    assert len(value["task_json_sha256"])==20
    assert value["region_count"] is None and value["event_label_counts"] is None
    assert value["capacity_ready"] is False and not value["terminal_cap_evidence_implemented"]
    assert not any(path.endswith((".pt",".npz")) for path in value["sealed_sources"])
    assert spec["user_authorization"]==value["approval"]
    assert not (tmp_path/freezer.CARD).exists() and not (tmp_path/"results").exists()


def test_freezer_refuses_existing_card_before_documents(tmp_path,monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[3]/"tools/v3"))
    freezer=importlib.import_module("freeze_gse_supported_construction_teacher_v1")
    monkeypatch.setattr(freezer,"PROJECT_ROOT",tmp_path)
    write(tmp_path/freezer.CARD,{"preserved":True})
    monkeypatch.setattr(freezer,"documents",lambda:pytest.fail("must not build over existing card"))
    with pytest.raises(RuntimeError,match="overwrite"):freezer.main()
