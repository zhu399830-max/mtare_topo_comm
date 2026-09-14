#!/usr/bin/env python3
"""Immutable shared feature export, exact input scope, zero optimizer steps."""
import argparse
import hashlib
import json
from pathlib import Path
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
from mtare_topo.governance_supplement_features_v1 import SCHEMA, SLUG, POLICY, validate_card
from mtare_topo.governance_surface_selection import digest
from mtare_topo.governance_surface_material import read_pinned
from mtare_topo.data.gse_supplement_feature_scope_v1 import compile_feature_scope

PYTHON = "/home/zeng-workstation/.local/share/mtare_topo_comm/envs/phase3_torch290_cu129_zarr2187_v1/bin/python"
CARD = "configs/v3/gate3/data_cards/" + SLUG + ".json"
SPEC = "configs/v3/gate3/" + SLUG + ".json"


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")


def environment():
    packages = subprocess.check_output([PYTHON, "-m", "pip", "freeze", "--all"], text=True)
    return {"executable_sha256": sha(Path(PYTHON).resolve()), "pip_freeze": packages,
            "python": subprocess.check_output([PYTHON, "--version"], text=True).strip()}


def freeze():
    root = PROJECT_ROOT
    if (root / CARD).exists() or (root / SPEC).exists():
        raise FileExistsError("no refreeze")
    scope = compile_feature_scope(root)
    approval = dict(status="APPROVED", approved_by="user-standing-scope-authorization", approved_at="2026-09-08",
        authorized_operations=["data_export"], authorized_gates=[3], scope_sha256=digest(scope),
        confirmation_reference="User approved GSE surface plan: 5090 public encoder extraction on supplemental C01-C07 population, missing features only; original3360 cache unchanged; 设置一个目标一直执行; no routine reapproval.",
        scope="Exact2676 fiveframe inputs; same historical seed0epoch2 frozen encoder; no labels,selection,training,C08-C10 or graphs")
    card = dict(schema_version=SCHEMA, card_id=SLUG, operation="data_export", scope=scope,
                scope_sha256=digest(scope), policy=POLICY, approval=approval)
    report = validate_card(card)
    if not report.passed:
        raise ValueError(report.errors)
    with (root / CARD).open("x") as stream:
        json.dump(card, stream, ensure_ascii=False, indent=2)
    files = sorted({str(p.relative_to(root)) for folder in ("src/mtare_topo", "tools/v3")
                    for p in (root / folder).rglob("*.py")} | {CARD})
    run = "results/gate3_semantics/gate3_20260908_" + SLUG + "_seed0"
    command = ["env", "OMP_NUM_THREADS=1", "OPENBLAS_NUM_THREADS=1", "MKL_NUM_THREADS=1", "PYTHONHASHSEED=0",
               PYTHON, "tools/v3/supplement_features_v1.py", "--spec", str(root / SPEC), "--run-dir", str(root / run)]
    spec = dict(schema_version="v3_run_spec_v1", gate=3, date="20260908", slug=SLUG, seed=0, operation="data_export",
        data_card=CARD, config_path=CARD, user_authorization=approval, command=command,
        question="Extract identical frozen encoder features for the fixed A/B/C structural comparison without teacher input.",
        method="Frozen observable seed0epoch2 _memory only; exact raw fiveframe range+relative motion; microbatch1; compact900x128context,fullXYZ and valid; no pointwise expanded cache.",
        baseline="One shared feature cache for A/B/C; not a perception performance experiment.",
        fallback="Fail and seal on any source,environment,numeric or resource drift; no rerun or data substitution.",
        wall_time_cap_s=1800, estimated_cost=dict(compute="5090D inference only2676windows", host_ram_gb=32, gpu_vram_gb=28, disk_gb=8, wall_time_hours=.5),
        acceptance_criteria=["All2676 exact source windows exported once; no source IDs or labels in encoder input.",
            "Frozen model state unchanged; finite compact features; exact raw-point validity and provenance.",
            "1800s/28GiBGPU/32GiBhost/8GiBoutput bounds; immutable logs and SHA256 seal; zero optimizer steps."],
        expected_evidence=["2676compactcaches,source/weight hashes,per-observation provenance,environment,logs,summary,RUN_STATE,seal"],
        source_sha256={p: sha(root / p) for p in files}, environment=environment())
    with (root / SPEC).open("x") as stream:
        json.dump(spec, stream, ensure_ascii=False, indent=2)
    print(json.dumps(dict(spec=SPEC, counts=scope["counts"], executed=False)))


def execute(spec, run):
    import numpy as np
    import torch
    from mtare_topo.data.gse_supplement_feature_input_v1 import bind_feature_input
    from mtare_topo.representation.gse_surface_encoder_checkpoint_v1 import load_surface_encoder
    from mtare_topo.representation.gse_dual_path_encoder_adapter_v1 import FrozenDualPathEncoderAdapterV1
    from mtare_topo.representation.gse_surface_training_v1 import module_state_sha256
    root, run = PROJECT_ROOT, run.resolve(strict=True)
    if (run != root / "results/gate3_semantics" / build_run_id(spec) or
            load_json(run / "RUN_STATE.json")["state"] != "CREATED_NOT_EXECUTED" or
            load_json(run / "config/run_spec.json") != spec):
        raise ValueError("fresh exact run required")
    start = time.monotonic(); entries = []; opened = {}; error = None; bound = None
    def expired(signum, frame):
        raise TimeoutError("1800s feature export cap")
    old = signal.signal(signal.SIGALRM, expired); signal.alarm(1800)
    write(run / "RUN_STATE.json", dict(state="RUNNING", run_id=run.name))
    try:
        card = load_json(root / spec["data_card"])
        if not validate_card(card).passed or card != load_json(run / "config/data_card.json"):
            raise ValueError("card drift")
        if environment() != spec["environment"] or Path(sys.executable) != Path(PYTHON):
            raise ValueError("environment drift")
        if (run / "config/command.txt").read_text() != shlex.join(spec["command"]) + "\n":
            raise ValueError("command drift")
        with zipfile.ZipFile(run / "artifacts/source_snapshot.zip", "x", compression=zipfile.ZIP_DEFLATED) as archive:
            for path, h in spec["source_sha256"].items():
                archive.writestr(path, read_pinned(root, path, h))
        scope = compile_feature_scope(root)
        if scope != card["scope"]:
            raise ValueError("source scope drift")
        opened.update(scope["metadata_sha256"])
        if not torch.cuda.is_available():
            raise RuntimeError("required CUDA unavailable")
        torch.manual_seed(0); torch.cuda.manual_seed_all(0)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.cuda.set_per_process_memory_fraction(28 * 1024**3 / torch.cuda.get_device_properties(0).total_memory)
        ck = scope["checkpoint"]
        raw = read_pinned(root, ck["path"], ck["sha256"]); opened[ck["path"]] = ck["sha256"]
        bound = load_surface_encoder(raw, expected_sha256=ck["sha256"], expected_epoch=ck["epoch"])
        del raw
        adapter = FrozenDualPathEncoderAdapterV1(bound.backbone, torch.nn.Identity()).to("cuda")
        write(run / "config/environment.json", dict(**spec["environment"], torch=torch.__version__, gpu=torch.cuda.get_device_name(0)))
        output = run / "artifacts/features"; output.mkdir()
        with (run / "logs/observations.jsonl").open("x") as log:
            for task in scope["tasks"]:
                raw = read_pinned(root, task["input_path"], task["input_sha256"])
                opened[task["input_path"]] = task["input_sha256"]
                for row in task["observations"]:
                    observation = bind_feature_input(raw, expected_sha256=task["input_sha256"], task=task["task"],
                        row=row["input_row"], source_sequence_id=row["source_sequence_id"], frame_rows=row["frame_rows"],observation_count=task["observation_count"])
                    args = [torch.from_numpy(a).unsqueeze(0).to("cuda") for a in
                            (observation.range_valid, observation.translation_m, observation.yaw_deg)]
                    features = adapter.extract_compact_features(*args)
                    path = output / (observation.input_binding_sha256 + ".npz")
                    with path.open("xb") as stream:
                        np.savez_compressed(stream, points_xyz_m=features.points_xyz_m.cpu().numpy(),
                            frozen_sensor_context=features.frozen_sensor_context.cpu().numpy(), valid=features.valid.cpu().numpy())
                    entry = dict(source=observation.provenance, input_binding_sha256=observation.input_binding_sha256,
                        frozen_encoder_state_sha256=bound.state_sha256, path=str(path.relative_to(run)), sha256=sha(path))
                    entries.append(entry); log.write(json.dumps(entry) + "\n"); log.flush()
                    del features, args
                    if (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 > POLICY["host_ram_bytes"] or
                            torch.cuda.max_memory_reserved() > POLICY["gpu_bytes"]):
                        raise MemoryError("RAM/VRAM cap")
                print(json.dumps(dict(completed=len(entries), task=task["task"], elapsed_s=time.monotonic()-start)), flush=True)
                if sum(p.stat().st_size for p in run.rglob("*") if p.is_file()) > POLICY["output_bytes"]:
                    raise RuntimeError("output cap")
            if len(entries) != 2676 or module_state_sha256(bound.backbone) != bound.state_sha256:
                raise ValueError("population or frozen model state drift")
        for path, h in opened.items():
            if sha(root / path) != h:
                raise ValueError("source changed during run")
    except Exception:
        error = traceback.format_exc(); (run / "logs/error.log").write_text(error)
    finally:
        signal.alarm(0); signal.signal(signal.SIGALRM, old)
        summary = dict(status="FAILED" if error else "FEATURE_EXPORT_COMPLETE", error=error, observations=len(entries),
            elapsed_s=time.monotonic()-start, peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
            peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved() if torch.cuda.is_initialized() else 0,
            labels=0, optimizer_steps=0, scientific_gate_pass=False)
        write(run / "artifacts/feature_manifest.json", dict(observations=entries, source_reads=opened))
        write(run / "metrics/summary.json", summary)
        write(run / "RUN_STATE.json", dict(state="FAILED" if error else "COMPLETED", run_id=run.name))
        seal = run / "artifacts/evidence_sha256.txt"
        seal.write_text("".join(sha(p) + "  " + str(p.relative_to(root)) + "\n" for p in sorted(run.rglob("*")) if p.is_file() and p != seal))
    print(json.dumps(summary), flush=True)
    return int(error is not None)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--freeze", action="store_true"); parser.add_argument("--spec", type=Path); parser.add_argument("--run-dir", type=Path)
    args = parser.parse_args()
    if args.freeze:
        freeze()
    elif args.spec and args.run_dir:
        raise SystemExit(execute(load_json(args.spec), args.run_dir))
    else:
        parser.error("--freeze or --spec/--run-dir required")
