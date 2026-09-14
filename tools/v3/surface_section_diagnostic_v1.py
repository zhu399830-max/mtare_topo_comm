#!/usr/bin/env python3
"""Single frozen source-section diagnostic, no training or label promotion."""
import argparse
import hashlib
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
from mtare_topo.governance import build_run_id,load_json
from mtare_topo.governance_surface_diagnostic import SCHEMA,SLUG,POLICY,validate_card
from mtare_topo.governance_surface_selection import digest
from mtare_topo.governance_surface_material import read_pinned
from mtare_topo.data.gse_surface_teacher_scope_v1 import compile_teacher_source_scope

PYTHON="/home/zeng-workstation/.local/share/mtare_topo_comm/envs/phase3_torch290_cu129_zarr2187_v1/bin/python"
CARD="configs/v3/gate3/data_cards/"+SLUG+".json"
SPEC="configs/v3/gate3/"+SLUG+".json"


def sha(path):
    with Path(path).open("rb") as f:return hashlib.file_digest(f,"sha256").hexdigest()


def write(path,value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,sort_keys=True,allow_nan=False)+"\n")


def freeze():
    root=PROJECT_ROOT
    if (root/CARD).exists() or (root/SPEC).exists():raise FileExistsError("no refreeze")
    scope=compile_teacher_source_scope(root)
    approval=dict(status="APPROVED",approved_by="user-standing-scope-authorization",approved_at="2026-09-07",
        authorized_operations=["data_export"],authorized_gates=[3],scope_sha256=digest(scope),
        scope="Fixed30C01observations150frames;source geometry/pose/code and section witnesses only;no semantic labels or training",
        confirmation_reference="User approved 2026-09-07 surface plan and continuous goal including bounded teacher diagnosis;physical-root fallback remains paused. No invented human annotation.")
    card=dict(schema_version=SCHEMA,card_id=SLUG,operation="data_export",scope=scope,scope_sha256=digest(scope),policy=POLICY,approval=approval)
    report=validate_card(card)
    if not report.passed:raise ValueError(report.errors)
    with (root/CARD).open("x") as f:json.dump(card,f,ensure_ascii=False,indent=2)
    # Snapshot local Python sources once to cover dynamic imports, no datasets.
    files=sorted({str(p.relative_to(root)) for folder in ("src/mtare_topo","tools/v3") for p in (root/folder).rglob("*.py")}|{CARD})
    run="results/gate3_semantics/gate3_20260907_"+SLUG+"_seed20260906"
    command=["env","CUDA_VISIBLE_DEVICES=","OMP_NUM_THREADS=1","OPENBLAS_NUM_THREADS=1","MKL_NUM_THREADS=1","PYTHONHASHSEED=20260906",PYTHON,
        "tools/v3/surface_section_diagnostic_v1.py","--spec",str(root/SPEC),"--run-dir",str(root/run)]
    packages=subprocess.check_output([PYTHON,"-m","pip","freeze","--all"],text=True)
    spec=dict(schema_version="v3_run_spec_v1",gate=3,date="20260907",slug=SLUG,seed=20260906,operation="data_export",
        data_card=CARD,config_path=CARD,user_authorization=approval,command=command,
        question="Which fixed30 source ROI sections have finite causal ray witnesses, and which geometry/source ambiguities prevent using them for visible teacher work?",
        method="All source polylines,10mROI,original.05m/64mesh;exact mesh sections;global finite-ray competition;zero additional range error beyond storage ULP,diagnostic only.",
        baseline="Original source geometry and fixed fiveframe returns;no model performance comparison or semantic teacher success claim.",
        fallback="Fail and seal on source drift,ambiguous crossing/section,open seam or resource cap;no plane shift,no filtering or retry.",
        wall_time_cap_s=1800,estimated_cost=dict(compute="CPU only;30observations150selectedframes",host_ram_gb=4,gpu_vram_gb=0,disk_gb=.5,wall_time_hours=.5),
        acceptance_criteria=["Exact fixed source selection and byte/pose/code equality;no C07-C10 payload or training.",
            "All source candidates preserved;unknown not negative;section evidence never promoted to semantic/physical labels.",
            "Immutable diagnostics and failure provenance;1800s/4GiBhost/512MiBoutput limits."],
        expected_evidence=["Source snapshot,command/environment,per-observation numeric sections and witnesses,raw log,summary,RUN_STATE,SHA256 seal;failure preserves active task."],
        source_sha256={p:sha(root/p) for p in files},environment=dict(python=platform.python_version(),
            executable_sha256=sha(Path(PYTHON).resolve()),pip_freeze=packages,pip_freeze_sha256=hashlib.sha256(packages.encode()).hexdigest()))
    with (root/SPEC).open("x") as f:json.dump(spec,f,ensure_ascii=False,indent=2)
    print(json.dumps({"spec":SPEC,"counts":scope["counts"],"source_files":len(files),"executed":False}))


def execute(spec,run):
    from mtare_topo.data.gse_surface_teacher_reader_v1 import SurfaceTeacherReader
    from mtare_topo.teacher.gse_surface_diagnostic_v1 import diagnose_observation
    root=PROJECT_ROOT;run=run.resolve(strict=True)
    if (run!=root/"results/gate3_semantics"/build_run_id(spec) or load_json(run/"RUN_STATE.json")["state"]!="CREATED_NOT_EXECUTED"
        or load_json(run/"config/run_spec.json")!=spec):raise ValueError("fresh exact run required")
    started=time.monotonic();error=None;entries=[];active=None;reader=None
    def expire(signum,frame):raise TimeoutError("1800s diagnostic limit")
    previous=signal.signal(signal.SIGALRM,expire);signal.alarm(1800)
    write(run/"RUN_STATE.json",dict(state="RUNNING",run_id=run.name))
    try:
        card=load_json(root/spec["data_card"])
        if not validate_card(card).passed or card!=load_json(run/"config/data_card.json"):raise ValueError("card drift")
        with zipfile.ZipFile(run/"artifacts/source_snapshot.zip","x",compression=zipfile.ZIP_DEFLATED) as archive:
            for p,h in spec["source_sha256"].items():archive.writestr(p,read_pinned(root,p,h))
        env=spec["environment"];packages=subprocess.check_output([sys.executable,"-m","pip","freeze","--all"],text=True)
        if (platform.python_version()!=env["python"] or sha(Path(sys.executable).resolve())!=env["executable_sha256"]
            or hashlib.sha256(packages.encode()).hexdigest()!=env["pip_freeze_sha256"]):raise ValueError("environment drift")
        if (run/"config/command.txt").read_text()!=shlex.join(spec["command"])+"\n":raise ValueError("command drift")
        write(run/"config/environment.json",env)
        reader=SurfaceTeacherReader(root,card["scope"])
        with (run/"logs/observations.jsonl").open("x") as log:
            for row in card["scope"]["selected"]:
                active=row["task"]
                write(run/"artifacts/active_observation.json",row)
                bundle=reader.read_task(active)
                output=diagnose_observation(bundle,range_error_bound_m=POLICY["extra_range_error_m"])
                write(run/"artifacts"/(row["view_id"]+".json"),output)
                entry=dict(task=active,sections=len(output["candidate_sections"]),positive_sections=output["section_positive_count"])
                entries.append(entry);log.write(json.dumps(entry)+"\n");log.flush()
                print(json.dumps(dict(completed=len(entries),elapsed_s=time.monotonic()-started,**entry)),flush=True)
                if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024>POLICY["host_ram_bytes"]:raise MemoryError("4GiB RAM limit")
                if sum(p.stat().st_size for p in run.rglob("*") if p.is_file())>POLICY["output_bytes"]:raise RuntimeError("output limit")
        for p,h in reader.opened.items():
            if sha(root/p)!=h:raise ValueError("source changed during run")
    except Exception:
        error=traceback.format_exc();(run/"logs/error.log").write_text(error)
    finally:
        signal.alarm(0);signal.signal(signal.SIGALRM,previous)
        summary=dict(status="FAILED" if error else "SECTION_DIAGNOSTIC_COMPLETE",error=error,active_task=active,
            completed_observations=len(entries),observations=entries,elapsed_s=time.monotonic()-started,
            peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,labels=0,optimizer_steps=0,scientific_gate_pass=False)
        write(run/"metrics/summary.json",summary)
        write(run/"artifacts/source_reads_sha256.json",reader.opened if reader else {})
        write(run/"RUN_STATE.json",dict(state="FAILED" if error else "COMPLETED",run_id=run.name,error=error))
        seal=run/"artifacts/evidence_sha256.txt"
        seal.write_text("".join(sha(p)+"  "+str(p.relative_to(root))+"\n" for p in sorted(run.rglob("*")) if p.is_file() and p!=seal))
    print(json.dumps(summary),flush=True)
    return int(error is not None)


if __name__=="__main__":
    parser=argparse.ArgumentParser(__doc__);parser.add_argument("--freeze",action="store_true");parser.add_argument("--spec",type=Path);parser.add_argument("--run-dir",type=Path)
    args=parser.parse_args()
    if args.freeze:freeze()
    elif args.spec and args.run_dir:raise SystemExit(execute(load_json(args.spec),args.run_dir))
    else:parser.error("--freeze or --spec/--run-dir required")
