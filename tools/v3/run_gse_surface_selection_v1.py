#!/usr/bin/env python3
"""Execute one immutable selection from sealed identity metadata, no raw data."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import shlex
import signal
import time
import traceback

from _bootstrap import PROJECT_ROOT
from mtare_topo.governance import build_run_id, load_json
from mtare_topo.governance_surface_selection import (
    INPUT_PATHS, INVENTORY, PARENTS, SEAL_SHA256, validate_surface_selection_card,
)
from mtare_topo.data.gse_surface_selection_v1 import select_parent, summarize_selection


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                   separators=(",", ":"), allow_nan=False) + "\n", encoding="utf-8")


def bounded_path(root, relative):
    if type(relative) is not str or Path(relative).is_absolute() or ".." in Path(relative).parts:
        raise ValueError("explicit project-relative path required")
    result = root / relative
    if result.resolve(strict=True) != result:
        raise ValueError("symlink/alias input is forbidden")
    return result


def execute(spec, run, root=PROJECT_ROOT):
    root, run = Path(root).resolve(strict=True), Path(run).resolve(strict=True)
    if (run != root / "results/gate3_semantics" / build_run_id(spec)
            or load_json(run / "RUN_STATE.json")["state"] != "CREATED_NOT_EXECUTED"
            or load_json(run / "config/run_spec.json") != spec):
        raise ValueError("exact fresh non-overwriting run required")
    started, error, outputs, reads, result = time.monotonic(), None, [], {}, {}
    def expired(signum, frame):
        raise TimeoutError("120s identity-selection cap")
    old_signal = signal.signal(signal.SIGALRM, expired)
    signal.alarm(120)
    write(run / "RUN_STATE.json", {"state": "RUNNING", "run_id": run.name, "pid": os.getpid(),
          "started_at_utc": datetime.now(timezone.utc).isoformat()})
    try:
        card_path = bounded_path(root, spec["data_card"])
        card = load_json(card_path)
        checked = validate_surface_selection_card(card)
        if not checked.passed or spec["operation"] != "audit" or load_json(run / "config/data_card.json") != card:
            raise ValueError("frozen audit card mismatch: " + "; ".join(checked.errors))
        if not spec.get("source_sha256"):
            raise ValueError("source freeze missing")
        for rel, expected in spec["source_sha256"].items():
            path = bounded_path(root, rel)
            reads[rel] = sha(path)
            if reads[rel] != expected:
                raise ValueError("source freeze drift: " + rel)
        command = shlex.join(spec["command"]) + "\n"
        if (run / "config/command.txt").read_text() != command:
            raise ValueError("command snapshot drift")
        write(run / "config/execution_environment.json", {"python": platform.python_version(),
            "python_executable": os.path.realpath(os.sys.executable), "platform": platform.platform(),
            "command_sha256": hashlib.sha256(command.encode()).hexdigest(), "gpu": False})
        seal_rel = INVENTORY + "/artifacts/evidence_sha256.txt"
        reads[seal_rel] = sha(bounded_path(root, seal_rel))
        if reads[seal_rel] != SEAL_SHA256:
            raise ValueError("independently pinned inventory seal drift")
        allowed = card["scope"]["input_files_sha256"]
        sealed = {}
        for line in (root / seal_rel).read_text().splitlines():
            h, rel = line.split("  ", 1)
            if rel in INPUT_PATHS:
                if rel in sealed:
                    raise ValueError("duplicate sealed input entry")
                sealed[rel] = h
        if sealed != allowed:
            raise ValueError("submitted input hashes differ from pinned original seal")
        def read_identity(rel):
            if rel not in allowed:
                raise ValueError("attempted identity input outside exact72file scope")
            path = bounded_path(root, rel)
            raw = path.read_bytes()
            reads[rel] = hashlib.sha256(raw).hexdigest()
            if reads[rel] != allowed[rel]:
                raise ValueError("identity input changed: " + rel)
            return json.loads(raw)
        split = read_identity(INVENTORY + "/artifacts/parent_split.json")
        population = read_identity(INVENTORY + "/artifacts/parent_population.json")
        if (population["parents"] != PARENTS or population["complete_declared_parent_inventory"] is not True
                or sorted(split) != PARENTS or list(split.values()).count("fit") != 60
                or list(split.values()).count("calibration") != 5 or list(split.values()).count("development") != 5):
            raise ValueError("original complete70parent/60-5-5 split drift")
        with (run / "logs/parents.jsonl").open("x", encoding="utf-8") as log:
            for parent in PARENTS:
                report = read_identity(INVENTORY + "/artifacts/" + parent + "_identity_intervals.json")
                selected = select_parent(report, split[parent])
                outputs.append(selected)
                write(run / "artifacts" / (parent + "_selection.json"), selected)
                row = {k: v for k, v in selected.items() if k != "observations"}
                log.write(json.dumps(row, sort_keys=True) + "\n"); log.flush()
                print(json.dumps({"completed_parents": len(outputs), **row}), flush=True)
                if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 > 1073741824:
                    raise MemoryError("1GiB host cap")
        result = summarize_selection(outputs)
        if (result["parents"], result["tasks"], result["physical_edge_units"], result["observations"]) != (70, 210, 1120, 3360):
            raise ValueError("actual selection population differs from fixed contract")
        if result["split_observations"] != card["scope"]["expected_observations"]:
            raise ValueError("actual selected split population drift")
        write(run / "artifacts/selection_manifest.json", {"schema_version": "gse_surface_identity_selection_v1",
            "policy": card["scope"]["policy"], "parent_split": split, "population": result,
            "parents": [{k: v for k, v in p.items() if k != "observations"} for p in outputs],
            "observations": [r for p in outputs for r in p["observations"]]})
        for rel, h in reads.items():
            if sha(bounded_path(root, rel)) != h:
                raise ValueError("input/source changed during selection: " + rel)
    except Exception:
        error = traceback.format_exc()
        (run / "logs/error.log").write_text(error)
    finally:
        signal.alarm(0); signal.signal(signal.SIGALRM, old_signal)
        write(run / "artifacts/source_reads_sha256.json", reads)
        summary = {"status": "IDENTITY_SELECTION_FAIL" if error else "IDENTITY_SELECTION_COMPLETE",
            "elapsed_s": time.monotonic() - started, "peak_host_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
            "completed_parents": len(outputs), "result": result, "error": error}
        write(run / "metrics/summary.json", summary)
        write(run / "RUN_STATE.json", {"state": "FAILED" if error else "COMPLETED", "run_id": run.name, "error": error})
        seal = run / "artifacts/evidence_sha256.txt"
        def seal_evidence():
            seal.write_text("".join(f"{sha(p)}  {p.relative_to(root)}\n" for p in sorted(run.rglob("*")) if p.is_file() and p != seal))
        seal_evidence()
        output_bytes = sum(p.stat().st_size for p in run.rglob("*") if p.is_file())
        if error is None and (time.monotonic() - started > 120 or summary["peak_host_rss_bytes"] > 1073741824 or output_bytes > 33554432):
            error = "Final sealed evidence/time/host budget exceeded;no retry"
            summary.update(status="IDENTITY_SELECTION_FAIL", error=error)
            write(run / "metrics/summary.json", summary)
            write(run / "RUN_STATE.json", {"state": "FAILED", "run_id": run.name, "error": error})
            (run / "logs/error.log").write_text(error + "\n")
            seal_evidence()
    print(json.dumps({"error": error, "completed_parents": len(outputs), "seal_sha256": sha(seal)}), flush=True)
    return int(error is not None)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    args = parser.parse_args()
    resource.setrlimit(resource.RLIMIT_AS, (1073741824, 1073741824))
    raise SystemExit(execute(load_json(args.spec), args.run_dir))
