import copy
import hashlib
import json
from pathlib import Path
import pytest
from mtare_topo import governance_surface_material as g


def fixture(tmp_path,monkeypatch):
    m=g.SOURCE+"/artifacts/input_manifest.json"
    rows=[]; hashes={}
    for task in g.TASKS:
        parent=task.split("__")[0]
        for i in range(16):
            rows.append(dict(task=task,parent_id=parent,source_sequence_id=i,edge_selection_rank=i,
                             frame_rows=list(range(i*5,i*5+5)),split="fit"))
        hashes[g.SOURCE+"/artifacts/inputs/"+task+".npz"]="1"*64
    p=tmp_path/m;p.parent.mkdir(parents=True);p.write_text(json.dumps({"observations":rows}))
    hashes[m]=hashlib.sha256(p.read_bytes()).hexdigest()
    seal=tmp_path/g.SOURCE/"artifacts/evidence_sha256.txt"
    seal.write_text("".join(h+"  "+p+"\n" for p,h in hashes.items())+"f"*64+"  forbidden_C10_does_not_exist\n")
    monkeypatch.setattr(g,"SOURCE_SEAL",hashlib.sha256(seal.read_bytes()).hexdigest())
    return g.compile_scope(tmp_path)


def card(scope):
    h=g.digest(scope)
    return dict(schema_version=g.SCHEMA,card_id="gse_surface_observed_material_v1",operation="data_export",scope=scope,scope_sha256=h,
                approval=dict(status="APPROVED",scope_sha256=h,authorized_operations=["data_export"],authorized_gates=[3],
                              approved_by="synthetic",approved_at="synthetic",scope="synthetic",confirmation_reference="test only"))


def test_metadata_only_fixed30_no_payload_or_hidden_path_reads(tmp_path,monkeypatch):
    scope=fixture(tmp_path,monkeypatch)
    assert len(scope["selected"])==30
    assert all(r["input_row"]==0 for r in scope["selected"])
    assert g.validate_surface_material_card(card(scope)).passed
    # All NPZ files are deliberately absent: preparation only reads metadata.
    assert not list(tmp_path.rglob("*.npz"))


@pytest.mark.parametrize("key,value",[("observations_processed",31),("new_labels",1),("parents",30),("frames_processed",150.)])
def test_re_signed_bad_counts_rejected(tmp_path,monkeypatch,key,value):
    scope=fixture(tmp_path,monkeypatch);scope["counts"]=copy.deepcopy(scope["counts"]);scope["counts"][key]=value
    assert not g.validate_surface_material_card(card(scope)).passed


def test_extra_input_and_training_authority_rejected(tmp_path,monkeypatch):
    scope=fixture(tmp_path,monkeypatch);scope["input_files_sha256"]["C10.npz"]="0"*64
    assert not g.validate_surface_material_card(card(scope)).passed
    scope=fixture(tmp_path/"second",monkeypatch);c=card(scope);c["approval"]["authorized_operations"]=["training"]
    assert not g.validate_surface_material_card(c).passed
