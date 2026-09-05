import hashlib
import importlib
import json
from pathlib import Path

import pytest

from mtare_topo.governance import build_run_id
from mtare_topo.governance_partial_structure import PRODUCER_FILES, validate_partial_structure_export_card
from tests.v3.unit.test_gse_partial_structure_card import card


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,sort_keys=True,indent=2)+"\n")


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def fixture(tmp_path,monkeypatch,*,audit_hash_present=True):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[3]/"tools/v3"))
    freezer=importlib.import_module("freeze_gse_partial_structure_export_v1")
    monkeypatch.setattr(freezer,"PROJECT_ROOT",tmp_path)
    old=card()
    for paths in PRODUCER_FILES.values():
        for relative in paths:
            path=tmp_path/relative;path.parent.mkdir(parents=True,exist_ok=True);path.write_text("# sealed synthetic producer\n")
    specs={};roots={};seals={}
    locations={"coordinate_control":freezer.COORDINATE_SPEC,"local_teacher":freezer.AUDIT_SPEC,"supported_teacher":freezer.TEACHER_SPEC}
    for role,location in locations.items():
        card_path="fixtures/"+role+"_card.json"
        original=dict(old)
        if role=="local_teacher" and not audit_hash_present: original.pop("selection_sha256")
        write(tmp_path/card_path,original)
        spec={"gate":3,"date":"20260905","seed":0,"slug":"synthetic_"+role,"data_card":card_path,
              "source_sha256":{p:sha(tmp_path/p) for p in PRODUCER_FILES.get(role,())},
              "command":["env","A=1","B=1","C=1","D=1","E=1","/synthetic/bin/python"]}
        write(tmp_path/location,spec);specs[role]=spec
        root="results/gate3_semantics/"+build_run_id(spec);roots[role]=root
        wanted={root+"/config/run_spec.json":sha(tmp_path/location),root+"/config/data_card.json":sha(tmp_path/card_path)}
        basenames={"coordinate_control":["all_predictions.npz","existing_scoring_targets.npz","sample_manifest.json"],
                   "local_teacher":["observation_audit.json"],"supported_teacher":["observation_targets.json"]}[role]
        wanted.update({root+"/artifacts/"+basename:"e"*64 for basename in basenames})
        seal=tmp_path/root/"artifacts/evidence_sha256.txt"
        seal.parent.mkdir(parents=True,exist_ok=True)
        seal.write_text("".join(digest+"  "+path+"\n" for path,digest in wanted.items())
                        +"0"*64+"  /forbidden/C07/checkpoint.pt\n")
        seals[role]=sha(seal)
    method=tmp_path/freezer.METHOD_DOC;method.parent.mkdir(parents=True,exist_ok=True);method.write_text("Synthetic method only")
    for name in ("run_gse_partial_structure_export_v1.py","freeze_gse_partial_structure_export_v1.py","_bootstrap.py"):
        (tmp_path/"tools/v3"/name).write_text("# synthetic execution source\n")
    monkeypatch.setattr(freezer,"SEAL_HASHES",seals)
    monkeypatch.setattr(freezer,"git_source_sha",lambda commit,path:sha(tmp_path/path))
    original_sha=freezer.sha
    def safe_sha(path):
        assert path.suffix not in (".npz",".pt",".ckpt")
        assert path.name not in ("sample_manifest.json","observation_audit.json","observation_targets.json")
        return original_sha(path)
    monkeypatch.setattr(freezer,"sha",safe_sha)
    return freezer,specs,roots


def test_documents_hash_only_sealed_metadata_never_open_payloads(tmp_path,monkeypatch):
    freezer,old_specs,roots=fixture(tmp_path,monkeypatch)
    value,spec=freezer.documents()
    assert validate_partial_structure_export_card(value).passed
    assert spec["expected_counts"]=={"observations":180,"parents":10,"unique_frames":900,"visible_fragments":1452,
        "center_labels":1206,"member_positive":2155,"member_negative":13510,"events":{"corridor":872,"junction":14,"terminal":0}}
    assert spec["operation"]=="data_export" and spec["wall_time_cap_s"]==120
    assert spec["effective_counts"]==value["effective_counts"]=={"gt":None,"predicted":None}
    assert spec["user_authorization"]==value["approval"]
    assert value["approval"]["authorized_operations"]==["data_export"]
    assert spec["command"][5]=="/synthetic/bin/python"
    assert not any((tmp_path/source["path"]).exists() for source in value["sources"].values())
    assert not (tmp_path/freezer.CARD).exists() and not (tmp_path/freezer.SPEC).exists()
    assert all(source["sha256"]=="e"*64 for source in value["sources"].values())
    assert not any("C07" in p for p in value["sealed_sources"])


def test_original_audit_without_explicit_digest_binds_its_actual_rows_without_mutation(tmp_path,monkeypatch):
    freezer,old_specs,_=fixture(tmp_path,monkeypatch,audit_hash_present=False)
    old_path=tmp_path/old_specs["local_teacher"]["data_card"]
    before=old_path.read_bytes()
    assert "selection_sha256" not in json.loads(before)
    value,_=freezer.documents()
    assert validate_partial_structure_export_card(value).passed
    assert old_path.read_bytes()==before


@pytest.mark.parametrize("issue",["seal","snapshot","git_bytes","missing_entry","duplicate_entry"])
def test_drift_stops_before_new_card_or_run(tmp_path,monkeypatch,issue):
    freezer,old_specs,roots=fixture(tmp_path,monkeypatch)
    seal=tmp_path/roots["coordinate_control"]/"artifacts/evidence_sha256.txt"
    if issue=="seal": seal.write_text("drifted")
    elif issue=="snapshot": (tmp_path/freezer.COORDINATE_SPEC).write_text(json.dumps({**old_specs["coordinate_control"],"drifted":True}))
    elif issue=="git_bytes": monkeypatch.setattr(freezer,"git_source_sha",lambda commit,path:"f"*64)
    elif issue in ("missing_entry","duplicate_entry"):
        lines=seal.read_text().splitlines()
        selected=next(line for line in lines if "all_predictions.npz" in line)
        if issue=="missing_entry": lines.remove(selected)
        else: lines.append(selected)
        seal.write_text("\n".join(lines)+"\n")
        freezer.SEAL_HASHES["coordinate_control"]=sha(seal)
    with pytest.raises(ValueError): freezer.documents()
    assert not (tmp_path/freezer.CARD).exists()


@pytest.mark.parametrize("existing",["CARD","SPEC"])
def test_never_overwrite_existing_freeze(tmp_path,monkeypatch,existing):
    freezer,_,_=fixture(tmp_path,monkeypatch)
    write(tmp_path/getattr(freezer,existing),{"preserve":True})
    monkeypatch.setattr(freezer,"documents",lambda:pytest.fail("must reject before preparing sources"))
    with pytest.raises(RuntimeError,match="overwrite"): freezer.main()
