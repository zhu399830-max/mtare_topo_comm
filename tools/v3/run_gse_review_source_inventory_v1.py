#!/usr/bin/env python3
"""One immutable identity-only inventory; no structural selection or labels."""
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
from mtare_topo.governance_identity_inventory import validate_identity_inventory_card
from mtare_topo.data.gse_review_inventory_reader_v1 import (
    ReviewIdentityInventoryReader, collect_identity_seals, project_path, sha_file,
)
from mtare_topo.data.gse_review_sampling_v1 import ParentMetadata, ParentPopulation, partition_parents


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True,
                               separators=(",", ":"), allow_nan=False) + "\n", encoding="utf-8")


def execute(spec, run, root=PROJECT_ROOT):
    root = Path(root).resolve(strict=True)
    run = Path(run).resolve(strict=True)
    if (run != root / "results/gate3_semantics" / build_run_id(spec)
            or load_json(run / "RUN_STATE.json")["state"] != "CREATED_NOT_EXECUTED"
            or load_json(run / "config/run_spec.json") != spec):
        raise ValueError("exact fresh non-overwriting run required")
    start, error, summaries, source_reads, result = time.monotonic(), None, [], {}, {}
    def deadline(signum, frame):
        raise TimeoutError("600s identity-only inventory limit")
    previous_signal = signal.signal(signal.SIGALRM, deadline)
    signal.alarm(600)
    write(run / "RUN_STATE.json", {"state": "RUNNING", "run_id": run.name, "pid": os.getpid(),
                                   "started_at_utc": datetime.now(timezone.utc).isoformat()})
    try:
        card_path = project_path(root, spec["data_card"])
        card = load_json(card_path)
        report = validate_identity_inventory_card(card)
        if not report.passed or spec["operation"] != "audit" or load_json(run / "config/data_card.json") != card:
            raise ValueError("frozen identity card invalid or changed: " + "; ".join(report.errors))
        if not spec.get("source_sha256"):
            raise ValueError("source freeze missing")
        frozen = {str(project_path(root, path)): digest for path, digest in spec["source_sha256"].items()}
        for path, digest in frozen.items():
            actual = sha_file(path)
            source_reads[path] = actual
            if actual != digest:
                raise ValueError("executor/source freeze drift: " + path)
        source_reads[str(card_path)] = sha_file(card_path)
        versions = {"python": platform.python_version(), "numpy": np.__version__, "zarr": zarr.__version__}
        if versions != spec["expected_versions"]:
            raise ValueError("sidecar versions drift")
        command = shlex.join(spec["command"]) + "\n"
        if (run / "config/command.txt").read_text() != command:
            raise ValueError("created command snapshot drift")
        write(run / "config/execution_environment.json", {"versions": versions, "platform": platform.platform(),
            "command_sha256": hashlib.sha256(command.encode()).hexdigest(), "pid": os.getpid(),
            "compute": "CPU; no GPU or trained model", "maximum_host_ram_bytes": 4 * 1024**3})
        expected, index_reads = collect_identity_seals(root, card["scope"], source_reads)
        source_reads.update(index_reads)
        reader = ReviewIdentityInventoryReader(root, card["scope"], expected)
        with (run / "logs/parents.jsonl").open("x", encoding="utf-8") as log:
            for parent in card["scope"]["parent_ids"]:
                report = reader.read_parent(parent)
                output = run / "artifacts" / (parent + "_identity_intervals.json")
                write(output, asdict(report))
                counts = {"parent_id": parent, "source_partition": report.partition, **report.counts,
                          "active_traversals_with_frames": len(report.intervals),
                          "short_active_traversals": len(report.short_intervals),
                          "artifact_sha256": sha_file(output)}
                summaries.append(counts)
                log.write(json.dumps(counts, sort_keys=True) + "\n"); log.flush()
                print(json.dumps({"completed_parents": len(summaries), "total_parents": 70,
                                  "parent": parent, "logical_sequences": counts["logical_sequences"]}), flush=True)
                if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 > 4 * 1024**3:
                    raise RuntimeError("4GiB host resource limit")
                if sum(p.stat().st_size for p in run.rglob("*") if p.is_file()) > 250_000_000:
                    raise RuntimeError("250MB evidence limit")
        source_reads.update(reader.opened)
        observed = {"parents": len(summaries), "tasks": sum(len(s["variants"]) for s in summaries),
                    "logical_sequences": sum(s["logical_sequences"] for s in summaries),
                    "variant_sequences": sum(s["variant_observations"] for s in summaries)}
        if observed != card["scope"]["expected_counts"]:
            raise ValueError("observed population differs from registered expectations: " + json.dumps(observed))
        population_path = run / "artifacts/parent_population.json"
        write(population_path, {"parents": card["scope"]["parent_ids"], "by_parent": summaries,
                                "observed_counts": observed, "complete_declared_parent_inventory": True})
        population = ParentPopulation(tuple(ParentMetadata(p, p[-3:]) for p in card["scope"]["parent_ids"]),
                                      sha_file(population_path), True)
        write(run / "artifacts/parent_split.json", partition_parents(population))
        result = {**observed, "by_parent": summaries,
            "unique_variant_raw_frames": sum(s["unique_variant_raw_frames"] for s in summaries),
            "active_traversals_with_frames": sum(s["active_traversals_with_frames"] for s in summaries),
            "all_construction_traversal_count": None, "zero_frame_traversals_audited": False,
            "eligible_21_decision_windows": sum(s["eligible_21_decision_windows"] for s in summaries),
            "source_payload_files_read": len(reader.opened), "parent_population_sha256": sha_file(population_path),
            "selected_segments": 0, "scan_frames_decoded": 0, "model_inference": 0, "checkpoint_reads": 0,
            "optimizer_steps": 0, "human_labels": 0, "structure_labels_created": 0,
            "training_ready": False, "scientific_gate_pass": False, "duration_s": None}
        for path, digest in source_reads.items():
            if sha_file(path) != digest:
                raise ValueError("source changed during audit: " + path)
    except Exception:
        error = traceback.format_exc()
        (run / "logs/error.log").write_text(error, encoding="utf-8")
    finally:
        signal.alarm(0); signal.signal(signal.SIGALRM, previous_signal)
        if "reader" in locals():
            source_reads.update(reader.opened)
        # Even a failed run preserves exactly which permitted keys it consumed.
        write(run / "artifacts/source_reads_sha256.json", {str(Path(p).relative_to(root)): h for p, h in source_reads.items()})
        final_summary = {"status": "IDENTITY_INVENTORY_FAIL" if error else "IDENTITY_INVENTORY_COMPLETE",
            "elapsed_s": time.monotonic() - start, "peak_host_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
            "completed_parents": len(summaries), "result": result, "error": error}
        write(run / "metrics/summary.json", final_summary)
        files = [p for p in run.rglob("*") if p.is_file()]
        final_state = {"state": "FAILED" if error else "COMPLETED", "run_id": run.name, "error": error}
        state_size = len((json.dumps(final_state, sort_keys=True, separators=(",", ":")) + "\n").encode())
        seal_size = sum(len(("0" * 64 + "  " + str(p.relative_to(root)) + "\n").encode()) for p in files)
        projected_bytes = sum(p.stat().st_size for p in files) - (run / "RUN_STATE.json").stat().st_size + state_size + seal_size
        if error is None and (projected_bytes > 250_000_000 or final_summary["peak_host_rss_bytes"] > 4 * 1024**3
                              or final_summary["elapsed_s"] > 600):
            error = "Final evidence/host/time budget exceeded; all evidence retained, no retry."
            (run / "logs/error.log").write_text(error + "\n", encoding="utf-8")
            final_summary.update(status="IDENTITY_INVENTORY_FAIL", error=error)
            write(run / "metrics/summary.json", final_summary)
        write(run / "RUN_STATE.json", {"state": "FAILED" if error else "COMPLETED", "run_id": run.name, "error": error})
        seal = run / "artifacts/evidence_sha256.txt"
        seal.write_text("".join(f"{sha_file(p)}  {p.relative_to(root)}\n" for p in sorted(run.rglob("*"))
                                if p.is_file() and p != seal), encoding="utf-8")
        # Verify the actual tail as well as its pre-write estimate. A late
        # overrun is a failed sealed run, never a silent successful overrun.
        if error is None and (time.monotonic() - start > 600
                or resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 > 4 * 1024**3
                or sum(p.stat().st_size for p in run.rglob("*") if p.is_file()) > 250_000_000):
            error = "Final sealed evidence/time/host budget exceeded; no retry."
            (run / "logs/error.log").write_text(error + "\n", encoding="utf-8")
            final_summary.update(status="IDENTITY_INVENTORY_FAIL", error=error, elapsed_s=time.monotonic() - start)
            write(run / "metrics/summary.json", final_summary)
            write(run / "RUN_STATE.json", {"state": "FAILED", "run_id": run.name, "error": error})
            seal.write_text("".join(f"{sha_file(p)}  {p.relative_to(root)}\n" for p in sorted(run.rglob("*"))
                                    if p.is_file() and p != seal), encoding="utf-8")
    print(json.dumps({"error": error, "completed_parents": len(summaries), "seal_sha256": sha_file(seal)}), flush=True)
    return int(error is not None)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    args = parser.parse_args()
    # Limit address space before any source-index payload is decoded. No GPU
    # libraries are imported by this runner, so host AS has a meaningful cap.
    resource.setrlimit(resource.RLIMIT_AS, (4 * 1024**3, 4 * 1024**3))
    return execute(load_json(args.spec), args.run_dir)


if __name__ == "__main__":
    raise SystemExit(main())
