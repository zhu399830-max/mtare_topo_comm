"""Runner lifecycle tests with tiny synthetic inputs, no actual source reader.

The real16x5x16x720 six-field reader has its own Zarr integration suite. Here
only compiler/authority/environment/reader are mocked; filesystem, NPZ reopen,
snapshots, checksums, final state and evidence sealing execute for real.
"""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import shlex
import sys
from types import SimpleNamespace
import zipfile

import numpy as np
import pytest

from mtare_topo.data.gse_surface_input_export_v1 import SurfaceTaskInputs


def load_runner():
    folder = Path(__file__).resolve().parents[3] / "tools/v3"
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))
    spec = importlib.util.spec_from_file_location("surface_input_runner_unit_fixture", folder / "run_gse_surface_input_export_v1.py")
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True))


def digest_bytes(value):
    return hashlib.sha256(value).hexdigest()


def fixture(tmp_path, monkeypatch):
    module = load_runner()
    task = "S01_flat_tree_small_C01__ellipse"
    software = tmp_path / "fixtures/software.py"; software.parent.mkdir(); software.write_bytes(b"# synthetic fixture source only\n")
    chunk = tmp_path / "fixtures/selected_chunk"; chunk.write_bytes(b"synthetic input bytes, no actual sensor data")
    metadata = tmp_path / "fixtures/metadata.json"; put(metadata, {"fixture": True})
    environment = {"python": "synthetic-sidecar", "numpy": "fixture"}
    freeze = "synthetic-test-only==0\n"
    source_files = {str(chunk.relative_to(tmp_path)): module.sha(chunk)}
    metadata_reads = {str(metadata.relative_to(tmp_path)): module.sha(metadata)}
    source = {"parent_id": "S01_flat_tree_small_C01", "variant": "ellipse", "partition": "fit",
              "sensor": "fixture-sensor", "teacher": "fixture-relative-motion"}
    manifest = {"observations": [{"task": task, "frame_rows": list(range(i*5,i*5+5))} for i in range(16)]}
    scope = {"task_sources": {task: source}, "file_sha256": source_files,
             "environment": environment, "selection": {"synthetic": True},
             "population": {"observations":16, "unique_variant_source_frames":80}}
    summary = {"fixture": True, "selected_observations":16, "actual_files":1}
    card = {"scope":scope, "synthetic_fixture_only":True}
    card_path = "configs/v3/gate3/data_cards/fixture_surface_input.json"
    put(tmp_path / card_path, card)
    spec = {"gate":3, "date":"20260907", "slug":"gse_surface_input_fixture", "seed":0,
            "operation":"data_export", "data_card":card_path,
            "source_sha256":{str(software.relative_to(tmp_path)):module.sha(software)},
            "environment":environment, "python_executable_sha256":module.sha(Path(sys.executable).resolve()),
            "sidecar_freeze_sha256":digest_bytes(freeze.encode()), "access_summary":summary,
            "command":[sys.executable,"tools/v3/run_gse_surface_input_export_v1.py","--synthetic-only"]}
    run = tmp_path / "results/gate3_semantics" / module.build_run_id(spec)
    for directory in ("config","artifacts","logs","metrics"):
        (run / directory).mkdir(parents=True, exist_ok=True)
    put(run / "RUN_STATE.json", {"state":"CREATED_NOT_EXECUTED"})
    put(run / "config/run_spec.json",spec); put(run / "config/data_card.json",card)
    (run / "config/command.txt").write_text(shlex.join(spec["command"])+"\n")
    ranges = np.arange(80,dtype=np.float32).reshape(16,5,1,1)/2
    valid = np.ones((16,5,1,1),dtype=np.uint8);valid[0]=0;valid[1,0]=0
    batch = SurfaceTaskInputs(task,ranges,valid,np.zeros((16,5,3),dtype=np.float32),
        np.zeros((16,5),dtype=np.float32),np.arange(80,dtype=np.int32).reshape(16,5),
        np.arange(16,dtype=np.int64),{"selected_observations":16,"synthetic_spatial_shape":[1,1]})
    calls={"compiler":0,"reader":0,"tasks":0}
    def compiler(root):
        assert root == tmp_path
        calls["compiler"]+=1
        return copy.deepcopy(scope),copy.deepcopy(manifest),copy.deepcopy(metadata_reads)
    class Reader:
        def __init__(self,root,*,task_sources,selection,sealed_keys):
            assert root == tmp_path and task_sources == scope["task_sources"]
            assert selection == manifest["observations"] and sealed_keys == source_files
            calls["reader"]+=1;self.opened={}
        def read_task(self,key):
            assert key == task
            calls["tasks"]+=1
            self.opened.update(source_files)
            return batch
    monkeypatch.setattr(module,"compile_input_scope",compiler)
    monkeypatch.setattr(module,"access_summary",lambda s: copy.deepcopy(summary))
    monkeypatch.setattr(module,"SurfaceInputReader",Reader)
    monkeypatch.setattr(module,"validate_surface_input_card",lambda c:SimpleNamespace(passed=True,errors=()))
    monkeypatch.setattr(module,"current_environment",lambda:copy.deepcopy(environment))
    monkeypatch.setattr(module.subprocess,"check_output",lambda *args,**kwargs:freeze)
    return SimpleNamespace(module=module,run=run,spec=spec,scope=scope,manifest=manifest,
        batch=batch,calls=calls,software=software,chunk=chunk,metadata=metadata,source_files=source_files,
        metadata_reads=metadata_reads,root=tmp_path)


def assert_sealed(f):
    seal=f.run/"artifacts/evidence_sha256.txt"
    assert seal.is_file()
    entries={}
    for line in seal.read_text().splitlines():
        digest,path=line.split(None,1);entries[path]=digest
        assert f.module.sha(f.root/path)==digest
    expected={str(p.relative_to(f.root)) for p in f.run.rglob("*") if p.is_file() and p!=seal}
    assert set(entries)==expected
    return seal.read_bytes()


def test_success_real_lifecycle_npz_roundtrip_exact_keys_and_no_overwrite(tmp_path,monkeypatch):
    f=fixture(tmp_path,monkeypatch)
    before={p:p.read_bytes() for p in (f.software,f.chunk,f.metadata,f.root/f.spec["data_card"],f.run/"config/run_spec.json")}
    assert f.module.execute(f.spec,f.run,tmp_path)==0
    assert f.calls=={"compiler":1,"reader":1,"tasks":1}
    summary=json.loads((f.run/"metrics/summary.json").read_text())
    assert summary["status"]=="INPUT_EXPORT_COMPLETE" and summary["error"] is None
    assert summary["exported_observations"]==16 and summary["exported_unique_variant_frames"]==80
    assert summary["empty_history_frames"]==6 and summary["empty_windows"]==1
    assert summary["teacher_labels"]==summary["model_windows"]==summary["optimizer_steps"]==0
    assert summary["scientific_gate_pass"] is False
    manifest=json.loads((f.run/"artifacts/input_manifest.json").read_text())
    assert manifest["training_eligible"] is False and manifest["absolute_pose_exported"] is False
    assert manifest["physical_root_labels"]=="PAUSED_ROBOT_CONTRACT_FALLBACK"
    shard=f.run/manifest["task_shards"][0]["path"]
    with np.load(shard,allow_pickle=False) as restored:
        assert set(restored.files)=={"ranges_m","valid_mask","relative_translation_current_sensor_m",
            "relative_yaw_current_sensor_deg","frame_rows","source_sequence_ids"}
        for name in restored.files:
            np.testing.assert_array_equal(restored[name],getattr(f.batch,name))
            assert restored[name].dtype==getattr(f.batch,name).dtype
    with zipfile.ZipFile(f.run/"artifacts/source_snapshot.zip") as source:
        assert source.namelist()==["fixtures/software.py"]
        assert source.read("fixtures/software.py")==before[f.software]
    assert all(p.read_bytes()==content for p,content in before.items())
    assert json.loads((f.run/"RUN_STATE.json").read_text())["state"]=="COMPLETED"
    assert len((f.run/"logs/tasks.jsonl").read_text().splitlines())==1
    sealed=assert_sealed(f)
    with pytest.raises(ValueError,match="no overwrite/retry"):
        f.module.execute(f.spec,f.run,tmp_path)
    assert (f.run/"artifacts/evidence_sha256.txt").read_bytes()==sealed


def test_source_hash_fail_seals_error_and_never_constructs_reader(tmp_path,monkeypatch):
    f=fixture(tmp_path,monkeypatch);f.software.write_bytes(b"modified after source freeze")
    assert f.module.execute(f.spec,f.run,tmp_path)==1
    assert f.calls=={"compiler":0,"reader":0,"tasks":0}
    assert "pinned source drift" in (f.run/"logs/error.log").read_text()
    assert json.loads((f.run/"RUN_STATE.json").read_text())["state"]=="FAILED"
    assert not (f.run/"artifacts/inputs").exists()
    assert_sealed(f)


def test_recomputed_scope_drift_rejected_before_reader_and_payload(tmp_path,monkeypatch):
    f=fixture(tmp_path,monkeypatch)
    def changed(root):
        f.calls["compiler"]+=1;scope=copy.deepcopy(f.scope);scope["population"]["observations"]=17
        return scope,f.manifest,f.metadata_reads
    monkeypatch.setattr(f.module,"compile_input_scope",changed)
    assert f.module.execute(f.spec,f.run,tmp_path)==1
    assert f.calls=={"compiler":1,"reader":0,"tasks":0}
    assert "recomputed exact input/chunk scope" in (f.run/"logs/error.log").read_text()
    assert not (f.run/"artifacts/inputs").exists()
    assert_sealed(f)


@pytest.mark.parametrize("case",["card","environment","command","reader_failure","late_source_drift"])
def test_stage_failures_are_failed_and_sealed(tmp_path,monkeypatch,case):
    f=fixture(tmp_path,monkeypatch)
    if case=="card":
        monkeypatch.setattr(f.module,"validate_surface_input_card",lambda c:SimpleNamespace(passed=False,errors=("synthetic invalid card",)))
    elif case=="environment":
        monkeypatch.setattr(f.module,"current_environment",lambda:{"changed":True})
    elif case=="command":
        (f.run/"config/command.txt").write_text("changed\n")
    else:
        original=f.module.SurfaceInputReader
        class Reader(original):
            def read_task(self,task):
                batch=super().read_task(task)
                if case=="reader_failure":raise ValueError("synthetic codec failure")
                f.chunk.write_bytes(b"changed during export")
                return batch
        monkeypatch.setattr(f.module,"SurfaceInputReader",Reader)
    assert f.module.execute(f.spec,f.run,tmp_path)==1
    state=json.loads((f.run/"RUN_STATE.json").read_text())
    assert state["state"]=="FAILED" and state["error"]
    summary=json.loads((f.run/"metrics/summary.json").read_text())
    assert summary["status"]=="INPUT_EXPORT_FAIL" and summary["error"]
    if case in ("card","environment","command"):
        assert f.calls["reader"]==0
    assert_sealed(f)


def test_preexisting_shard_is_preserved_and_run_fails(tmp_path,monkeypatch):
    f=fixture(tmp_path,monkeypatch)
    folder=f.run/"artifacts/inputs";folder.mkdir();original=folder/(f.batch.task+".npz")
    original.write_bytes(b"do not overwrite")
    assert f.module.execute(f.spec,f.run,tmp_path)==1
    assert original.read_bytes()==b"do not overwrite"
    assert f.calls["tasks"]==0
    assert_sealed(f)


def test_wrong_fresh_run_spec_refused_without_any_state_change(tmp_path,monkeypatch):
    f=fixture(tmp_path,monkeypatch);before=(f.run/"RUN_STATE.json").read_bytes()
    other=copy.deepcopy(f.spec);other["operation"]="training"
    with pytest.raises(ValueError,match="fresh"):
        f.module.execute(other,f.run,tmp_path)
    assert (f.run/"RUN_STATE.json").read_bytes()==before
    assert f.calls=={"compiler":0,"reader":0,"tasks":0}
