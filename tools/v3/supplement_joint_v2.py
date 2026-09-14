#!/usr/bin/env python3
"""Single frozen source-terminal diagnostic, no training or label promotion."""
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
from mtare_topo.governance_supplement_joint_v2 import SCHEMA,SLUG,POLICY,validate_card
from mtare_topo.governance_surface_selection import digest
from mtare_topo.governance_surface_material import read_pinned
from mtare_topo.data.gse_supplement_teacher_scope_v1 import compile_supplement_teacher_scope
from mtare_topo.data.gse_joint_cached_interfaces_v1 import compile_cache,read_cached,compare_prior_targets

PYTHON="/home/zeng-workstation/.local/share/mtare_topo_comm/envs/cano_e1_zarr2187_v1/bin/python"
CARD="configs/v3/gate3/data_cards/"+SLUG+".json"
SPEC="configs/v3/gate3/"+SLUG+".json"


def sha(path):
    with Path(path).open("rb") as f:return hashlib.file_digest(f,"sha256").hexdigest()


def write(path,value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,sort_keys=True,allow_nan=False)+"\n")


def freeze():
    root=PROJECT_ROOT
    if (root/CARD).exists() or (root/SPEC).exists():raise FileExistsError("no refreeze")
    scope=compile_supplement_teacher_scope(root)
    approval=dict(status="APPROVED",approved_by="user-standing-scope-authorization",approved_at="2026-09-07",
        authorized_operations=["data_export"],authorized_gates=[3],scope_sha256=digest(scope),
        scope="Fixed2676 C01-C07 fiveframe observations; original cached scans, poses, constructions and source codes; jointV4 partial anchors/openings/membership with full interface provenance; no qualified labels, model inference or training",
        confirmation_reference="User standing authorization to autonomously implement the approved C01-C07 geometric supervision plan and reuse existing supplementary population. New joint partial target operation, not storage-only replay; source population/splits fixed; historical assets preserved.")
    card=dict(schema_version=SCHEMA,card_id=SLUG,operation="data_export",scope=scope,scope_sha256=digest(scope),policy=POLICY,approval=approval)
    report=validate_card(card)
    if not report.passed:raise ValueError(report.errors)
    with (root/CARD).open("x") as f:json.dump(card,f,ensure_ascii=False,indent=2)
    # Snapshot local Python sources once to cover dynamic imports, no datasets.
    files=sorted({str(p.relative_to(root)) for folder in ("src/mtare_topo","tools/v3") for p in (root/folder).rglob("*.py")}|{CARD})
    run="results/gate3_semantics/gate3_20260907_"+SLUG+"_seed20260906"
    command=["env","CUDA_VISIBLE_DEVICES=","OMP_NUM_THREADS=1","OPENBLAS_NUM_THREADS=1","MKL_NUM_THREADS=1","PYTHONHASHSEED=20260906",PYTHON,
        "tools/v3/supplement_joint_v2.py","--spec",str(root/SPEC),"--run-dir",str(root/run)]
    packages=subprocess.check_output([PYTHON,"-m","pip","freeze","--all"],text=True)
    spec=dict(schema_version="v3_run_spec_v1",gate=3,date="20260907",slug=SLUG,seed=20260906,operation="data_export",
        data_card=CARD,config_path=CARD,user_authorization=approval,command=command,
        question="Does jointV4 partial structural supervision cover the existing supplemental population across independent parents and entities?",
        method="Reuse512 sealed source interfaces, regenerate all jointV4 targets with axis-contour ownership and compare retained geometry/membership; remaining observations compute original source interfaces; partial junction/terminal/window targets; missing witnesses not contradictions, other-node evidence vetoes; per-observation full raw interfaces and provenance stored as lossless gzip; no whole-region label promotion.",
        baseline="Fixed spatial coverage inventory; no learned detector comparison; terminal reference evidence is not complete scene annotation.",
        fallback="Fail and seal on source drift,binding errors or resource cap;ROI clipping contacts remain explicit UNKNOWN;no coordinate shift,sample filtering or retry.",
        wall_time_cap_s=10800,estimated_cost=dict(compute="CPU only;2676 fiveframe observations,13374 selected frames,210 source tasks; cached scans only",host_ram_gb=4,gpu_vram_gb=0,disk_gb=30,wall_time_hours=3),
        acceptance_criteria=["Exact fixed2676 selection,source hashes,poses and membership; no C08-C10, new scan rendering, models, qualified complete labels or training.",
            "All source candidates preserved;unknown not negative;section evidence never promoted to semantic/physical labels.",
            "Immutable diagnostics and failure provenance;10800s/4GiBhost/30GiBoutput limits."],
        expected_evidence=["Source snapshot,command/environment,per-observation cap witnesses and unknown/source rejections,source frame indices,raw log,summary,RUN_STATE,SHA256 seal;failure preserves active task."],
        cached_raw_sha256=compile_cache(root,scope),source_sha256={p:sha(root/p) for p in files},environment=dict(python=platform.python_version(),
            executable_sha256=sha(Path(PYTHON).resolve()),pip_freeze=packages,pip_freeze_sha256=hashlib.sha256(packages.encode()).hexdigest()))
    with (root/SPEC).open("x") as f:json.dump(spec,f,ensure_ascii=False,indent=2)
    print(json.dumps({"spec":SPEC,"counts":scope["counts"],"source_files":len(files),"executed":False}))


def execute(spec,run):
    from mtare_topo.data.gse_supplement_teacher_reader_v1 import SupplementTeacherReader
    from mtare_topo.teacher.gse_junction_interface_diagnostic_v1 import diagnose_observation
    from mtare_topo.teacher.gse_joint_reference_targets_v4 import produce_joint_reference_targets
    from mtare_topo.data.gse_lossless_evidence_v1 import write_evidence
    root=PROJECT_ROOT;run=run.resolve(strict=True)
    if (run!=root/"results/gate3_semantics"/build_run_id(spec) or load_json(run/"RUN_STATE.json")["state"]!="CREATED_NOT_EXECUTED"
        or load_json(run/"config/run_spec.json")!=spec):raise ValueError("fresh exact run required")
    started=time.monotonic();error=None;entries=[];active=None;reader=None
    def expire(signum,frame):raise TimeoutError("10800s diagnostic limit")
    previous=signal.signal(signal.SIGALRM,expire);signal.alarm(10800)
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
        if compile_cache(root,card["scope"])!=spec["cached_raw_sha256"]:raise ValueError("raw cache manifest drift")
        reader=SupplementTeacherReader(root,card["scope"])
        with (run/"logs/observations.jsonl").open("x") as log:
            for row in card["scope"]["entries"]:
                active=row["task"]
                write(run/"artifacts/active_observation.json",row)
                bundles=reader.read_task(active)
                rows=[]
                for bundle in bundles:
                    source=bundle["source"]
                    write(run/"artifacts/active_observation.json",source)
                    cache_key=active+"_"+str(source["source_sequence_id"])
                    pin=spec["cached_raw_sha256"].get(cache_key)
                    old=None
                    if pin:raw,old=read_cached(root,pin,source,reader.opened)
                    else:raw=diagnose_observation(bundle)
                    output=produce_joint_reference_targets(bundle,raw)
                    name=active+"_"+str(source["source_sequence_id"])+".json.gz"
                    comparison=compare_prior_targets(old,output) if old is not None else None
                    evidence=dict(produced_targets=output,prior_comparison=comparison)
                    if pin:evidence["raw_interfaces_reference"]=pin
                    else:evidence["raw_interfaces"]=raw
                    storage=write_evidence(run/"artifacts"/name,evidence)
                    item=dict(source=source,anchors=len(output["record"]["anchors"]),openings=len(output["record"]["openings"]),
                        positive_memberships=sum(v is True for row_ in output["record"]["membership"] for v in row_),
                        target_record_sha256=output["target_record_sha256"],storage=storage,prior_comparison=comparison)
                    rows.append(item)
                    print(json.dumps(dict(task=active,sequence=source["source_sequence_id"],completed_in_task=len(rows),
                        elapsed_s=time.monotonic()-started,anchors=item["anchors"],openings=item["openings"])),flush=True)
                    del raw,output,old,evidence
                    if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024>POLICY["host_ram_bytes"]:raise MemoryError("4GiB RAM limit")
                storage=write_evidence(run/"artifacts"/(active+".json.gz"),dict(task=active,observations=rows))
                entry=dict(task=active,observations=len(rows),anchors=sum(r["anchors"] for r in rows),openings=sum(r["openings"] for r in rows),positive_memberships=sum(r["positive_memberships"] for r in rows))
                entry["storage"]=storage
                del rows,bundles,bundle
                entries.append(entry);log.write(json.dumps(entry)+"\n");log.flush()
                print(json.dumps(dict(completed=len(entries),elapsed_s=time.monotonic()-started,task=active)),flush=True)
                if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024>POLICY["host_ram_bytes"]:raise MemoryError("4GiB RAM limit")
                if sum(p.stat().st_size for p in run.rglob("*") if p.is_file())>POLICY["output_bytes"]:raise RuntimeError("output limit")
        for p,h in reader.opened.items():
            if sha(root/p)!=h:raise ValueError("source changed during run")
    except Exception:
        error=traceback.format_exc();(run/"logs/error.log").write_text(error)
    finally:
        signal.alarm(0);signal.signal(signal.SIGALRM,previous)
        summary=dict(status="FAILED" if error else "SUPPLEMENT_JOINT_PARTIAL_DIAGNOSTIC_COMPLETE",error=error,active_task=active,
            completed_tasks=len(entries),observations=entries,elapsed_s=time.monotonic()-started,
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


