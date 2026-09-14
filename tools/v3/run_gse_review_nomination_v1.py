#!/usr/bin/env python3
"""One sealed four-strata review-candidate audit; no scans or training labels."""
import argparse
from dataclasses import asdict
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

import numpy as np
import zarr
from _bootstrap import PROJECT_ROOT
from mtare_topo.governance import build_run_id, load_json
from mtare_topo.governance_review_nomination import validate_review_nomination_card
from mtare_topo.data.gse_review_inventory_reader_v1 import project_path, sha_file
from mtare_topo.data.gse_review_nomination_reader_v1 import ReviewNominationReader
from mtare_topo.data.gse_review_nomination_v1 import nominate_interval
from mtare_topo.data.gse_review_quota_assignment_v1 import assign_review_quotas


def write(path, value):
    path.write_text(json.dumps(value, sort_keys=True, ensure_ascii=False,
                              separators=(",", ":"), allow_nan=False) + "\n", encoding="utf-8")


def seal_run(run, root):
    seal = run / "artifacts/evidence_sha256.txt"
    seal.write_text("".join(f"{sha_file(p)}  {p.relative_to(root)}\n" for p in sorted(run.rglob("*"))
        if p.is_file() and p != seal), encoding="utf-8")
    return sha_file(seal)


def execute(spec, run, root=PROJECT_ROOT):
    root, run = Path(root).resolve(strict=True), Path(run).resolve(strict=True)
    if (run != root / "results/gate3_semantics" / build_run_id(spec)
            or load_json(run / "RUN_STATE.json")["state"] != "CREATED_NOT_EXECUTED"
            or load_json(run / "config/run_spec.json") != spec):
        raise ValueError("exact fresh non-overwriting run required")
    start, reads, nominations, per_parent, error = time.monotonic(), {}, [], [], None
    assignment, reader, result = None, None, {}
    def deadline(signum, frame):
        raise TimeoutError("600s nomination audit limit")
    previous = signal.signal(signal.SIGALRM, deadline); signal.alarm(600)
    write(run / "RUN_STATE.json", {"state": "RUNNING", "run_id": run.name, "pid": os.getpid(),
        "started_at_utc": datetime.now(timezone.utc).isoformat()})
    try:
        card = load_json(project_path(root, spec["data_card"]))
        validation = validate_review_nomination_card(card)
        if not validation.passed or spec["operation"] != "audit" or load_json(run / "config/data_card.json") != card:
            raise ValueError("frozen nomination card invalid: " + "; ".join(validation.errors))
        if not spec.get("source_sha256") or spec.get("freeze_status") != "FROZEN_ALL_AUTHORS_STOPPED":
            raise ValueError("source freeze missing")
        for relative, expected in spec["source_sha256"].items():
            actual = sha_file(project_path(root, relative)); reads[relative] = actual
            if actual != expected:
                raise ValueError("bound source changed: " + relative)
        versions = {"python": platform.python_version(), "numpy": np.__version__, "zarr": zarr.__version__}
        if versions != spec["expected_versions"]:
            raise ValueError("frozen sidecar versions drift")
        command = shlex.join(spec["command"]) + "\n"
        if (run / "config/command.txt").read_text() != command:
            raise ValueError("created command snapshot drift")
        write(run / "config/execution_environment.json", {"versions": versions, "platform": platform.platform(),
            "command_sha256": hashlib.sha256(command.encode()).hexdigest(), "pid": os.getpid(),
            "compute": "CPU only; no scans, models, labels or optimizer", "host_ram_cap_bytes": 4 * 1024**3})
        reader = ReviewNominationReader(root, card["scope"], card["scope"]["source_seals"])
        with (run / "logs/parents.jsonl").open("x", encoding="utf-8") as log:
            for parent in card["scope"]["source_scope"]["eligible_parents"]:
                data = reader.read_parent(parent)
                entries, witnesses = [], []
                for interval in data.intervals:
                    nominated = nominate_interval(parent_id=parent, split=data.split, interval=interval,
                        construction=data.construction, overlap_by_variant=data.overlaps_for_interval(interval))
                    entries.extend(nominated.nominations); witnesses.extend(nominated.evidence)
                write(run / "artifacts" / (parent + "_nominations.json"), {
                    "parent_id": parent, "split": data.split, "construction_identity": asdict(data.construction),
                    "nominations": [asdict(n) for n in entries], "witnesses": witnesses,
                    "training_labels_created": 0, "source_identity_not_student_input": True})
                nominations.extend(entries)
                row = {"parent": parent, "split": data.split, "intervals": len(data.intervals),
                    "windows": sum(i.window_count for i in data.intervals), "nominations": len(entries)}
                per_parent.append(row)
                log.write(json.dumps(row, sort_keys=True) + "\n"); log.flush()
                print(json.dumps({"completed_parents": len(per_parent), **row}), flush=True)
                if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 > 4 * 1024**3:
                    raise RuntimeError("4GiB host resource limit")
                if sum(p.stat().st_size for p in run.rglob("*") if p.is_file()) > 250_000_000:
                    raise RuntimeError("250MB evidence limit")
        observed = {"eligible_parents": len(per_parent), "eligible_traversals": sum(p["intervals"] for p in per_parent),
            "overlapping_windows": sum(p["windows"] for p in per_parent)}
        if observed != {"eligible_parents": 55, "eligible_traversals": 918, "overlapping_windows": 2546}:
            raise ValueError("sealed eligible population drift: " + json.dumps(observed))
        assignment = assign_review_quotas(tuple(nominations))
        write(run / "artifacts/quota_assignment.json", asdict(assignment))
        # A maximal incomplete selection is diagnostic only, NOT a training or
        # annotation manifest. No downstream export is executed here even if full.
        result = {**observed, "quota_complete": assignment.complete,
            "selected_candidate_clips": len(assignment.selected), "selection_sha256": assignment.selection_sha256,
            "counts": assignment.counts, "deficits": assignment.deficits,
            "parent_selected_counts": assignment.metadata["parent_selected_counts"],
            "source_payload_files_read": len(reader.opened), "scan_frames_decoded": 0,
            "checkpoint_reads": 0, "optimizer_steps": 0, "human_labels": 0, "structure_labels_created": 0,
            "training_ready": False, "scientific_gate_pass": False, "duration_s": None,
            "continuity_requires_source_audit": True,
            "next_data_action": "FREEZE_SELECTED_SCAN_REVIEW_EXPORT_SCOPE" if assignment.complete else "STOP_AFFECTED_EXPORT_AND_REPORT_QUOTA_DEFICITS"}
        # Hash post-read sources again; no new paths, scopes or payloads added.
        for relative, digest in {**reads, **reader.opened}.items():
            if sha_file(project_path(root, relative)) != digest:
                raise ValueError("source changed during audit: " + relative)
    except Exception:
        error = traceback.format_exc()
        (run / "logs/error.log").write_text(error, encoding="utf-8")
    finally:
        signal.alarm(0); signal.signal(signal.SIGALRM, previous)
        if reader is not None:
            reads.update(reader.opened)
            write(run / "artifacts/chunk_reads.json", reader.chunk_reads)
        write(run / "artifacts/source_reads_sha256.json", reads)
        state = "FAILED" if error else "COMPLETED"
        status = "NOMINATION_AUDIT_FAIL" if error else "REVIEW_CANDIDATE_QUOTAS_FULL" if assignment and assignment.complete else "REVIEW_CANDIDATE_QUOTAS_INSUFFICIENT"
        summary = {"status": status, "elapsed_s": time.monotonic() - start,
            "peak_host_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
            "completed_parents": len(per_parent), "result": result, "error": error}
        write(run / "metrics/summary.json", summary)
        write(run / "RUN_STATE.json", {"state": state, "run_id": run.name, "error": error})
        seal = seal_run(run, root)
        if error is None and (time.monotonic() - start > 600 or summary["peak_host_rss_bytes"] > 4 * 1024**3
                or sum(p.stat().st_size for p in run.rglob("*") if p.is_file()) > 250_000_000):
            error = "Final evidence, memory or time exceeds frozen resource limit; no retry."
            (run / "logs/error.log").write_text(error + "\n", encoding="utf-8")
            summary.update(status="NOMINATION_AUDIT_FAIL", error=error, elapsed_s=time.monotonic() - start)
            write(run / "metrics/summary.json", summary)
            write(run / "RUN_STATE.json", {"state": "FAILED", "run_id": run.name, "error": error})
            seal = seal_run(run, root)
    print(json.dumps({"error": error, "status": summary["status"], "seal_sha256": seal}), flush=True)
    return int(error is not None)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    args = parser.parse_args()
    resource.setrlimit(resource.RLIMIT_AS, (4 * 1024**3, 4 * 1024**3))
    return execute(load_json(args.spec), args.run_dir)


if __name__ == "__main__":
    raise SystemExit(main())
