#!/usr/bin/env python3
"""Audit whether observable endpoint relations admit lossless local slots."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import zarr

from mtare_topo.data.primitive_attachment_observability_sidecar import unpack_endpoint_observed
from mtare_topo.evaluation.local_composition_slot_teacher import (
    decompose_observable_attachment,
    select_slot_capacity,
)


EXPECTED = {
    "fit": {"tasks": 180, "rows": 426_552, "positive_pairs": 2_782_487},
    "c07": {"tasks": 30, "rows": 64_644, "positive_pairs": 442_936},
}
CHUNK = 256


def _write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _attachments(neighbors: np.ndarray) -> np.ndarray:
    value = np.asarray(neighbors, dtype=np.int16)
    if value.ndim != 4 or value.shape[1:] != (32, 2, 3):
        raise ValueError("endpoint-neighbor array must be [B,32,2,3]")
    batch = len(value); flat = value.reshape(batch, 64, 3)
    result = np.zeros((batch, 64, 64), dtype=np.bool_)
    b, endpoint, neighbor_slot = np.indices(flat.shape)
    valid = flat >= 0
    if np.any(flat[valid] >= 64):
        raise ValueError("endpoint-neighbor index outside 64 endpoints")
    result[b[valid], endpoint[valid], flat[valid]] = True
    return result


def _endpoint_overlap(packed: np.ndarray) -> np.ndarray:
    primitive = np.unpackbits(
        np.asarray(packed, dtype=np.uint8), axis=2, count=32, bitorder="little",
    ).astype(np.bool_, copy=False)
    return primitive.repeat(2, axis=1).repeat(2, axis=2)


def _empty_split(partition: str) -> dict:
    return {
        "partition": partition, "tasks": 0, "rows": 0,
        "observable_positive_pairs": 0, "clusters": 0,
        "rows_with_cluster": 0, "maximum_clusters_per_row": 0,
        "maximum_endpoints_per_cluster": 0,
        "cluster_count_histogram": {}, "cluster_size_histogram": {},
        "nonclique_rows": 0, "reconstruction_mismatched_pairs": 0,
        "disconnected_overlap_violations": 0,
        "asymmetric_attachment_rows": 0, "deterministic_rederive_mismatches": 0,
        "canonical_label_sha256": None,
    }


def _audit_split(partition: str, teacher_root: Path, observable_root: Path):
    teacher_tasks = sorted(path for path in teacher_root.glob("*.zarr") if path.is_dir())
    observable_tasks = {path.name: path for path in observable_root.glob("*.zarr") if path.is_dir()}
    if {path.name for path in teacher_tasks} != set(observable_tasks):
        raise RuntimeError(f"{partition} Teacher/observability tasks differ")
    summary = _empty_split(partition); per_task = []
    digest = hashlib.sha256()
    for task_path in teacher_tasks:
        teacher = zarr.open_group(str(task_path), mode="r")
        observable = zarr.open_group(str(observable_tasks[task_path.name]), mode="r")
        length = int(teacher["endpoint_neighbor"].shape[0])
        if (
            teacher.attrs.get("partition") != partition
            or observable.attrs.get("partition") != partition
            or observable.attrs.get("hidden_pair_semantics") != "unknown_never_negative"
            or float(observable.attrs.get("support_band_m", -1)) != 0.25
            or observable["endpoint_observed_packed"].shape != (length, 8)
        ):
            raise RuntimeError(f"{partition}/{task_path.name} shard contract drift")
        task = _empty_split(partition); task["task"] = task_path.stem; task["tasks"] = 1
        for start in range(0, length, CHUNK):
            stop = min(start + CHUNK, length)
            attachment = _attachments(teacher["endpoint_neighbor"][start:stop])
            overlap = _endpoint_overlap(teacher["disconnected_overlap_packed"][start:stop])
            observed = unpack_endpoint_observed(observable["endpoint_observed_packed"][start:stop]).reshape(-1, 64)
            sequences = np.asarray(teacher["source_global_sequence_index"][start:stop], dtype="<i8")
            if not np.array_equal(sequences, observable["source_global_sequence_index"][start:stop]):
                raise RuntimeError(f"{partition}/{task_path.name} sequence alignment drift")
            for row in range(stop - start):
                if not np.array_equal(attachment[row], attachment[row].T):
                    task["asymmetric_attachment_rows"] += 1
                    continue
                value = decompose_observable_attachment(attachment[row], observed[row], overlap[row])
                repeated = decompose_observable_attachment(attachment[row], observed[row], overlap[row])
                if not np.array_equal(value.labels, repeated.labels):
                    task["deterministic_rederive_mismatches"] += 1
                positive = int(np.count_nonzero(np.triu(value.observed_attachment, k=1)))
                mismatch = int(np.count_nonzero(np.triu(
                    value.observed_attachment ^ value.reconstructed_attachment, k=1,
                )))
                count = value.cluster_count
                task["rows"] += 1; task["observable_positive_pairs"] += positive
                task["clusters"] += count; task["rows_with_cluster"] += int(count > 0)
                task["maximum_clusters_per_row"] = max(task["maximum_clusters_per_row"], count)
                task["nonclique_rows"] += int(not value.is_clique_partition)
                task["reconstruction_mismatched_pairs"] += mismatch
                task["disconnected_overlap_violations"] += value.overlap_violations
                task["cluster_count_histogram"][str(count)] = task["cluster_count_histogram"].get(str(count), 0) + 1
                for size in value.cluster_sizes:
                    task["maximum_endpoints_per_cluster"] = max(task["maximum_endpoints_per_cluster"], size)
                    task["cluster_size_histogram"][str(size)] = task["cluster_size_histogram"].get(str(size), 0) + 1
                digest.update(sequences[row].tobytes()); digest.update(value.labels.astype("<i2").tobytes())
        per_task.append(task)
        summary["tasks"] += 1
        for key in (
            "rows", "observable_positive_pairs", "clusters", "rows_with_cluster",
            "nonclique_rows", "reconstruction_mismatched_pairs",
            "disconnected_overlap_violations", "asymmetric_attachment_rows",
            "deterministic_rederive_mismatches",
        ):
            summary[key] += task[key]
        for key in ("maximum_clusters_per_row", "maximum_endpoints_per_cluster"):
            summary[key] = max(summary[key], task[key])
        for histogram in ("cluster_count_histogram", "cluster_size_histogram"):
            for key, count in task[histogram].items():
                summary[histogram][key] = summary[histogram].get(key, 0) + count
        print(json.dumps({
            "partition": partition, "task": task_path.stem,
            "rows": task["rows"], "max_clusters": task["maximum_clusters_per_row"],
            "nonclique_rows": task["nonclique_rows"], "mismatched_pairs": task["reconstruction_mismatched_pairs"],
        }), flush=True)
    summary["canonical_label_sha256"] = digest.hexdigest()
    expected = EXPECTED[partition]
    if any(summary[key] != expected[key] for key in ("tasks", "rows")):
        raise RuntimeError(f"{partition} population drift")
    if summary["observable_positive_pairs"] != expected["positive_pairs"]:
        raise RuntimeError(f"{partition} observable-positive population drift")
    return summary, per_task


def _plot(summary: dict, output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.2))
    for partition, color in (("fit", "#287271"), ("c07", "#d37524")):
        values = summary["splits"][partition]["cluster_count_histogram"]
        x = np.asarray(sorted(map(int, values)), dtype=int)
        y = np.asarray([values[str(key)] for key in x], dtype=float)
        axes[0].plot(x, y / y.sum(), marker="o", label=partition, color=color)
        sizes = summary["splits"][partition]["cluster_size_histogram"]
        sx = np.asarray(sorted(map(int, sizes)), dtype=int)
        sy = np.asarray([sizes[str(key)] for key in sx], dtype=float)
        axes[1].bar(sx + (-0.18 if partition == "fit" else 0.18), sy / sy.sum(), width=0.36, label=partition, color=color)
    selected_capacity = summary["capacity"]["selected_capacity"]
    if selected_capacity is not None:
        axes[0].axvline(selected_capacity, linestyle="--", color="black", linewidth=1)
    axes[0].set_xlabel("observable composition clusters per sequence"); axes[0].set_ylabel("row fraction")
    axes[1].set_xlabel("endpoints per composition cluster"); axes[1].set_ylabel("cluster fraction")
    for axis in axes: axis.legend()
    fig.suptitle("Local composition-slot Teacher feasibility")
    fig.tight_layout()
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(output / f"local_composition_slot_teacher_feasibility.{suffix}", dpi=180)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher-root", required=True, type=Path)
    parser.add_argument("--observability-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(); output = args.output_dir.resolve(); output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic(); splits = {}; per_task = []
    for partition in ("fit", "c07"):
        value, tasks = _audit_split(
            partition, args.teacher_root / partition, args.observability_root / partition,
        )
        splits[partition] = value; per_task.extend(tasks); _write(output / f"{partition}.json", value)
    capacity = select_slot_capacity(splits["fit"]["maximum_clusters_per_row"])
    capacity["c07_maximum_clusters"] = splits["c07"]["maximum_clusters_per_row"]
    capacity["c07_overflow_rows"] = sum(
        int(count) for clusters, count in splits["c07"]["cluster_count_histogram"].items()
        if capacity["selected_capacity"] is None or int(clusters) > capacity["selected_capacity"]
    )
    checks = {
        "exact_fit_c07_population": True,
        "exact_observable_positive_populations": True,
        "attachment_symmetric": all(value["asymmetric_attachment_rows"] == 0 for value in splits.values()),
        "components_are_cliques": all(value["nonclique_rows"] == 0 for value in splits.values()),
        "slot_decode_is_lossless": all(value["reconstruction_mismatched_pairs"] == 0 for value in splits.values()),
        "disconnected_overlap_isolated": all(value["disconnected_overlap_violations"] == 0 for value in splits.values()),
        "deterministic_rederive_exact": all(value["deterministic_rederive_mismatches"] == 0 for value in splits.values()),
        "fit_capacity_available": capacity["available"],
        "c07_zero_capacity_overflow": capacity["c07_overflow_rows"] == 0,
        "nonempty_composition_supervision": all(value["clusters"] > 0 for value in splits.values()),
    }
    scientific_pass = all(checks.values())
    summary = {
        "schema_version": "local_composition_slot_teacher_feasibility_v1",
        "overall_status": "PASS_LOCAL_COMPOSITION_SLOT_TEACHER_FEASIBILITY_V1" if scientific_pass else "FAIL_SCIENTIFIC_LOCAL_COMPOSITION_SLOT_TEACHER_FEASIBILITY_V1",
        "scientific_pass": scientific_pass, "checks": checks,
        "splits": splits, "capacity": capacity,
        "decision": "ALLOW_LOCAL_COMPOSITION_SLOT_MODEL_READINESS" if scientific_pass else "STOP_LOCAL_COMPOSITION_SLOT_AND_REASSESS_RELATION_REPRESENTATION",
        "teacher_rows_read": sum(value["rows"] for value in splits.values()),
        "sensor_range_rows_read": 0, "model_forward_rows": 0, "optimizer_steps": 0,
        "c08_rows_read": 0, "c09_c10_worlds_read": 0, "graph_replays": 0, "mtare_worlds_read": 0,
        "duration_seconds": time.monotonic() - started,
    }
    _write(output / "summary.json", summary); _write(output / "per_task.json", per_task)
    with (output / "per_task.csv").open("w", encoding="utf-8", newline="") as stream:
        fields = [key for key in per_task[0] if key not in ("cluster_count_histogram", "cluster_size_histogram")]
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore"); writer.writeheader(); writer.writerows(per_task)
    _write(output / "figure_source.json", {"splits": splits, "capacity": capacity})
    _plot(summary, output)
    print(json.dumps({"overall_status": summary["overall_status"], "checks": checks, "capacity": capacity}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
