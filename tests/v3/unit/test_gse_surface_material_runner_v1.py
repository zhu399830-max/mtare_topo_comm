import hashlib
import importlib.util
import json
from pathlib import Path
import shlex
import sys
from types import SimpleNamespace
import numpy as np
import pytest


def fixture(tmp_path,monkeypatch):
    tools=Path(__file__).resolve().parents[3]/"tools/v3"
    monkeypatch.syspath_prepend(str(tools))
    spec=importlib.util.spec_from_file_location("surface_material_runner_fixture",tools/"surface_observed_material_v1.py")
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    data=tmp_path/"source.npz"
    np.savez(data,ranges_m=np.zeros((1,5,1,1),np.float32),valid_mask=np.zeros((1,5,1,1),np.uint8),
             relative_translation_current_sensor_m=np.zeros((1,5,3),np.float32),relative_yaw_current_sensor_deg=np.zeros((1,5),np.float32),
             frame_rows=np.arange(5)[None],source_sequence_ids=np.array([1]))
    row=dict(input_path="source.npz",input_row=0,source=dict(frame_rows=list(range(5)),source_sequence_id=1),view_id="test_only")
    seal=tmp_path/"source_seal.txt";seal.write_text("synthetic source index")
    scope=dict(selected=[row],input_files_sha256={"source.npz":m.sha(data)},
               source_seal={"path":"source_seal.txt","sha256":m.sha(seal)})
    cardpath=tmp_path/"card.json";m.write(cardpath,dict(scope=scope))
    font=tmp_path/"font";font.write_bytes(b"synthetic")
    monkeypatch.setattr(m,"FONT",str(font))
    monkeypatch.setattr(m,"compile_scope",lambda root:scope)
    monkeypatch.setattr(m,"validate_surface_material_card",lambda c:SimpleNamespace(passed=True))
    monkeypatch.setattr(m.subprocess,"check_output",lambda *a,**k:"synthetic-freeze")
    import mtare_topo.data.gse_surface_material_v1 as core
    monkeypatch.setattr(core,"build_observed_material",lambda *args:({"test":np.zeros((1,3))},{"surface_patches":0}))
    monkeypatch.setattr(core,"draw_material",lambda a,b,p,t:Path(p).write_bytes(b"synthetic-image"))
    s=dict(gate=3,date="20260907",slug="surface_material_fixture",seed=0,data_card="card.json",command=["synthetic"],
           source_sha256={"card.json":m.sha(cardpath)},environment=dict(python=m.platform.python_version(),
           executable_sha256=m.sha(Path(sys.executable).resolve()),pip_freeze_sha256=hashlib.sha256(b"synthetic-freeze").hexdigest(),font_sha256=m.sha(font)))
    run=tmp_path/"results/gate3_semantics"/m.build_run_id(s)
    for d in ("config","artifacts","metrics","logs","previews"):(run/d).mkdir(parents=True)
    m.write(run/"RUN_STATE.json",dict(state="CREATED_NOT_EXECUTED"));m.write(run/"config/run_spec.json",s);m.write(run/"config/data_card.json",dict(scope=scope))
    (run/"config/command.txt").write_text(shlex.join(s["command"])+"\n")
    return m,s,run,scope


def test_immutable_material_executor_and_full_seal(tmp_path,monkeypatch):
    m,s,run,scope=fixture(tmp_path,monkeypatch)
    assert m.execute(s,run,tmp_path)==0
    assert json.loads((run/"metrics/summary.json").read_text())["completed_observations"]==1
    for line in (run/"artifacts/evidence_sha256.txt").read_text().splitlines():
        h,p=line.split("  ");assert m.sha(tmp_path/p)==h
    with pytest.raises(ValueError,match="fresh"):m.execute(s,run,tmp_path)


def test_scope_failure_stops_before_material(tmp_path,monkeypatch):
    m,s,run,scope=fixture(tmp_path,monkeypatch)
    monkeypatch.setattr(m,"compile_scope",lambda root:{})
    assert m.execute(s,run,tmp_path)==1
    assert json.loads((run/"metrics/summary.json").read_text())["completed_observations"]==0
    assert not list((run/"artifacts").glob("*.npz"))
