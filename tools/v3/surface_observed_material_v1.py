#!/usr/bin/env python3
"""Freeze/execute bounded observed material, never annotate or train."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import platform
import resource
import shlex
import signal
import subprocess
import sys
import time
import traceback
import zipfile

from _bootstrap import PROJECT_ROOT
from mtare_topo.governance import build_run_id, load_json
from mtare_topo.governance_surface_material import (
    SCHEMA, SOURCE, RESOURCES, compile_scope, read_pinned, validate_surface_material_card,
)
from mtare_topo.governance_surface_selection import digest

PYTHON="/home/zeng-workstation/.local/share/mtare_topo_comm/envs/phase3_torch290_cu129_zarr2187_v1/bin/python"
CARD="configs/v3/gate3/data_cards/gse_surface_observed_material_v1.json"
SPEC="configs/v3/gate3/gse_surface_observed_material_v1.json"
FONT="/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"


def sha(path):
    with Path(path).open("rb") as f:return hashlib.file_digest(f,"sha256").hexdigest()


def write(path,value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,sort_keys=True,allow_nan=False)+"\n")


def freeze(root=PROJECT_ROOT):
    root=Path(root).resolve(strict=True)
    if (root/CARD).exists() or (root/SPEC).exists():raise FileExistsError("no refreeze/overwrite")
    scope=compile_scope(root)
    approval={"status":"APPROVED","approved_by":"user-standing-scope-authorization","approved_at":"2026-09-07",
        "authorized_operations":["data_export"],"authorized_gates":[3],"scope_sha256":digest(scope),
        "scope":"Fixed first ranked edge observation for10C01parents x3variants;30observations150frames,source30NPZcontainers480observations2400frames. Observed input material only;no teacher/annotation/training.",
        "confirmation_reference":"User approved surface-relation execution plan and continuous goal; bounded observed-material preparation within registered missing-robot-contract contingency. No invented manual review or approval to train."}
    card={"schema_version":SCHEMA,"card_id":"gse_surface_observed_material_v1","operation":"data_export",
          "scope":scope,"scope_sha256":digest(scope),"approval":approval}
    checked=validate_surface_material_card(card)
    if not checked.passed:raise ValueError(checked.errors)
    files=[CARD,"tools/v3/surface_observed_material_v1.py","tools/v3/_bootstrap.py","tools/v3/preflight.py","tools/v3/create_run.py",
        "src/mtare_topo/__init__.py","src/mtare_topo/data/__init__.py","src/mtare_topo/representation/__init__.py",
        "src/mtare_topo/governance.py","src/mtare_topo/governance_surface_material.py","src/mtare_topo/governance_surface_selection.py",
        "src/mtare_topo/governance_identity_inventory.py","src/mtare_topo/governance_partial_structure_training.py",
        "src/mtare_topo/data/gse_surface_material_v1.py","src/mtare_topo/data/cano_sensor_smoke.py",
        "src/mtare_topo/representation/primitive_relation_model.py","src/mtare_topo/representation/phase3_structural_semantics.py",
        "src/mtare_topo/representation/gse_surface_patches_v1.py","src/mtare_topo/representation/gse_surface_ray_evidence_v1.py"]
    for p in files:
        if p!=CARD and not (root/p).is_file():raise FileNotFoundError(p)
    with (root/CARD).open("x") as f:json.dump(card,f,ensure_ascii=False,indent=2,allow_nan=False)
    run="results/gate3_semantics/gate3_20260907_gse_surface_observed_material_v1_seed20260906"
    cmd=["env","CUDA_VISIBLE_DEVICES=","OMP_NUM_THREADS=1","OPENBLAS_NUM_THREADS=1","MKL_NUM_THREADS=1","PYTHONHASHSEED=20260906",PYTHON,
         "tools/v3/surface_observed_material_v1.py","--spec",str(root/SPEC),"--run-dir",str(root/run)]
    package_freeze=subprocess.check_output([PYTHON,"-m","pip","freeze","--all"],text=True)
    spec={"schema_version":"v3_run_spec_v1","gate":3,"date":"20260907","slug":"gse_surface_observed_material_v1","seed":20260906,
        "operation":"data_export","data_card":CARD,"config_path":CARD,"user_authorization":approval,"command":cmd,
        "question":"Do fixed observed patches and ray evidence preserve the selected30fiveframe inputs, retain unsupported regions, and fit the declared capacity before visible-label work?",
        "method":"Student-exact float32 projection;fixed.5mPCApatches within10m;fixed.25mfirst-return grid;k8observed gaps;all30XY/XZ plots with true central slices.",
        "baseline":"Original same fiveframe returns. No model/GT/physical teacher; preserve raw observations beside geometric summaries.",
        "fallback":"Fail and seal on source/shape/capacity/resource drift;no sample/voxel/radius substitution or rerun. Physical labels remain unknown.",
        "wall_time_cap_s":600,"estimated_cost":{"compute":"CPU only;30observations150frames;read30sealedNPZcontainers,not new raw sources","host_ram_gb":4,"gpu_vram_gb":0,"disk_gb":.5,"wall_time_hours":600/3600},
        "acceptance_criteria":["Exact fixed30observations/10parents/150frames;source hashes unchanged;no hidden map/GT/model reads.",
            "No surface clipping of far returns;unknown maintained;capacity4096/ROI10m/voxels.5and.25unchanged.",
            "30numeric material files and30same-scene previews;no annotation/training/physical safety or sciencePASS assertion.",
            "600s/4GiBhost/.5GiBevidence/zeroGPU;immutable failure evidence and seal."],
        "expected_evidence":["Raw numerical material,metrics,source/method/command/environment hashes,source snapshot,30XYXZ previews,index,logs,RUN_STATE,seal."],
        "source_sha256":{p:sha(root/p) for p in files},"environment":{"python":platform.python_version(),"executable_sha256":sha(Path(PYTHON).resolve()),
            "pip_freeze":package_freeze,"pip_freeze_sha256":hashlib.sha256(package_freeze.encode()).hexdigest(),"font_path":FONT,"font_sha256":sha(FONT)}}
    with (root/SPEC).open("x") as f:json.dump(spec,f,ensure_ascii=False,indent=2,allow_nan=False)
    print(json.dumps({"spec":SPEC,"scope_sha256":digest(scope),"counts":scope["counts"],"executed":False}))


def execute(spec,run,root=PROJECT_ROOT):
    import numpy as np
    from mtare_topo.data.gse_surface_material_v1 import build_observed_material, draw_material
    root,run=Path(root).resolve(strict=True),Path(run).resolve(strict=True)
    if (run!=root/"results/gate3_semantics"/build_run_id(spec) or load_json(run/"RUN_STATE.json")["state"]!="CREATED_NOT_EXECUTED"
        or load_json(run/"config/run_spec.json")!=spec):raise ValueError("fresh exact run required")
    started=time.monotonic(); error=None; entries=[]; reads={}
    def expire(signum,frame):raise TimeoutError("600s observed-material cap")
    prior=signal.signal(signal.SIGALRM,expire);signal.alarm(600)
    write(run/"RUN_STATE.json",{"state":"RUNNING","run_id":run.name})
    try:
        card=load_json(root/spec["data_card"])
        if not validate_surface_material_card(card).passed or card!=load_json(run/"config/data_card.json"):
            raise ValueError("frozen material card mismatch")
        with zipfile.ZipFile(run/"artifacts/source_snapshot.zip","x",compression=zipfile.ZIP_DEFLATED) as archive:
            for p,h in spec["source_sha256"].items():
                archive.writestr(p,read_pinned(root,p,h));reads[p]=h
        env=spec["environment"]
        packages=subprocess.check_output([sys.executable,"-m","pip","freeze","--all"],text=True)
        if (platform.python_version()!=env["python"] or sha(Path(sys.executable).resolve())!=env["executable_sha256"]
            or hashlib.sha256(packages.encode()).hexdigest()!=env["pip_freeze_sha256"] or sha(FONT)!=env["font_sha256"]):
            raise ValueError("material interpreter/package/font drift")
        command=shlex.join(spec["command"])+"\n"
        if (run/"config/command.txt").read_text()!=command:raise ValueError("command snapshot drift")
        write(run/"config/environment.json",{**env,"invoked_python":sys.executable,"command_sha256":hashlib.sha256(command.encode()).hexdigest()})
        scope=compile_scope(root)
        if scope!=card["scope"]:raise ValueError("exact source selection changed before payload")
        reads[scope["source_seal"]["path"]]=scope["source_seal"]["sha256"]
        for p,h in scope["input_files_sha256"].items():
            if p.endswith("/input_manifest.json"):reads[p]=h
        with (run/"logs/observations.jsonl").open("x") as log:
            for row in scope["selected"]:
                p=row["input_path"]; raw=read_pinned(root,p,scope["input_files_sha256"][p]);reads[p]=hashlib.sha256(raw).hexdigest()
                with np.load(io.BytesIO(raw),allow_pickle=False) as data:
                    i=row["input_row"]
                    if not np.array_equal(data["frame_rows"][i],row["source"]["frame_rows"]) or data["source_sequence_ids"][i]!=row["source"]["source_sequence_id"]:
                        raise ValueError("material source row/history identity mismatch")
                    arrays,metrics=build_observed_material(data["ranges_m"][i],data["valid_mask"][i],data["relative_translation_current_sensor_m"][i],data["relative_yaw_current_sensor_deg"][i])
                name=row["view_id"]
                np.savez_compressed(run/"artifacts"/(name+".npz"),**arrays)
                with np.load(run/"artifacts"/(name+".npz"),allow_pickle=False) as restored:
                    if set(restored.files)!=set(arrays) or any(restored[k].dtype!=v.dtype or restored[k].shape!=v.shape or restored[k].tobytes()!=v.tobytes() for k,v in arrays.items()):
                        raise ValueError("observed numeric material roundtrip failed")
                draw_material(arrays,metrics,run/"previews"/(name+".png"),name+"／拟合集观察材料")
                entry={"selection":row,"metrics":metrics,"numeric_file":"artifacts/"+name+".npz","preview":"previews/"+name+".png"}
                entries.append(entry);log.write(json.dumps(entry,ensure_ascii=False,allow_nan=False)+"\n");log.flush()
                print(json.dumps({"completed":len(entries),"patches":metrics["surface_patches"],"elapsed_s":time.monotonic()-started}),flush=True)
                if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024>RESOURCES["host_ram_bytes"]:raise MemoryError("4GiB RSS cap")
        for p,h in {**reads,**scope["input_files_sha256"]}.items():
            if sha(root/p)!=h:raise ValueError("source drift during material export")
        write(run/"artifacts/material_manifest.json",{"observations":entries,"labels":0,"human_reviewed":False,"training_eligible":False})
        html="<!doctype html><meta charset=utf-8><title>五帧观测与面片</title><h1>实际观测材料：不是模型结果或人工标签</h1><p>绿格只表示射线穿过，不能据此断言机器人可通行。右侧为固定中心切片，不是多层地图的压平。全部30例按身份预定选择。</p>"
        for e in entries:html+='<h2>'+e["selection"]["view_id"]+'</h2><img style="max-width:100%" src="'+e["preview"]+'">'
        (run/"index.html").write_text(html)
    except Exception:
        error=traceback.format_exc();(run/"logs/error.log").write_text(error)
    finally:
        signal.alarm(0);signal.signal(signal.SIGALRM,prior)
        summary={"status":"FAILED" if error else "OBSERVED_MATERIAL_COMPLETE","error":error,"completed_observations":len(entries),
                 "elapsed_s":time.monotonic()-started,"peak_rss_bytes":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                 "labels":0,"model_windows":0,"optimizer_steps":0,"scientific_gate_pass":False}
        write(run/"metrics/summary.json",summary);write(run/"artifacts/source_reads_sha256.json",reads)
        write(run/"RUN_STATE.json",{"state":"FAILED" if error else "COMPLETED","run_id":run.name,"error":error})
        seal=run/"artifacts/evidence_sha256.txt"
        def finish_seal():seal.write_text("".join(sha(p)+"  "+str(p.relative_to(root))+"\n" for p in sorted(run.rglob("*")) if p.is_file() and p!=seal))
        finish_seal()
        size=sum(p.stat().st_size for p in run.rglob("*") if p.is_file())
        if error is None and (size>RESOURCES["output_bytes"] or time.monotonic()-started>600 or summary["peak_rss_bytes"]>RESOURCES["host_ram_bytes"]):
            error="Final resource cap exceeded";summary.update(status="FAILED",error=error)
            (run/"logs/error.log").write_text(error+"\n")
            write(run/"metrics/summary.json",summary);write(run/"RUN_STATE.json",{"state":"FAILED","run_id":run.name,"error":error});finish_seal()
    print(json.dumps({**summary,"seal_sha256":sha(seal),"output_bytes":size}),flush=True)
    return int(error is not None)


if __name__=="__main__":
    p=argparse.ArgumentParser(__doc__);p.add_argument("--freeze",action="store_true");p.add_argument("--spec",type=Path);p.add_argument("--run-dir",type=Path)
    a=p.parse_args()
    if a.freeze:freeze()
    elif a.spec and a.run_dir:raise SystemExit(execute(load_json(a.spec),a.run_dir))
    else:p.error("--freeze or --spec/--run-dir required")
