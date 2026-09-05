#!/usr/bin/env python3
"""Stream the frozen fit/C07 Teachers and audit connection-cluster feasibility."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import zarr

from mtare_topo.data.primitive_attachment_observability_sidecar import (
    unpack_endpoint_observed,
)
from mtare_topo.evaluation.primitive_connection_hypergraph_teacher import (
    CAPACITY_CANDIDATES,
    MAXIMUM_PRIMITIVES,
    audit_connection_cluster_batch,
    merge_connection_cluster_summaries,
    select_fit_only_cluster_capacity,
)
from mtare_topo.governance import write_json


EXPECTED = {
    "fit": {"parents": 60, "tasks": 180, "rows": 426_552},
    "c07": {"parents": 10, "tasks": 30, "rows": 64_644},
}
EXPECTED_TOTAL_ROWS = 491_196
EXPECTED_C07_OBSERVABLE_POSITIVES = 442_936
BATCH_SIZE = 4096


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _tree_hash(path: Path) -> str:
    digest = hashlib.sha256()
    files = sorted(value for value in path.rglob("*") if value.is_file())
    for value in files:
        relative = value.relative_to(path).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(4, "little"))
        digest.update(relative)
        digest.update(bytes.fromhex(_sha(value)))
    return digest.hexdigest()


def _task_summary(
    task_id: str,
    split: str,
    teacher_root: Path,
    sidecar_root: Path,
):
    teacher_path = teacher_root / split / f"{task_id}.zarr"
    sidecar_path = sidecar_root / split / f"{task_id}.zarr"
    if not teacher_path.is_dir() or not sidecar_path.is_dir():
        raise FileNotFoundError(f"paired Teacher/sidecar shard missing: {task_id}")
    teacher = zarr.open_group(str(teacher_path), mode="r")
    sidecar = zarr.open_group(str(sidecar_path), mode="r")
    rows = int(teacher["primitive_mask"].shape[0])
    if (
        rows != int(sidecar["endpoint_observed_packed"].shape[0])
    ):
        raise RuntimeError(f"paired row-count drift: {task_id}")
    source_tree_sha = _tree_hash(teacher_path)
    if (
        teacher.attrs.get("schema_version") != "primitive_relation_p1b_teacher_shard_v1"
        or sidecar.attrs.get("schema_version") != "primitive_attachment_observability_sidecar_v1"
        or teacher.attrs.get("partition") != split
        or sidecar.attrs.get("partition") != split
        or sidecar.attrs.get("task_id") != task_id
        or sidecar.attrs.get("source_p1b_shard_tree_sha256")
        != source_tree_sha
    ):
        raise RuntimeError(f"paired Teacher provenance drift: {task_id}")

    batches = []
    for start in range(0, rows, BATCH_SIZE):
        stop = min(start + BATCH_SIZE, rows)
        primitive = np.asarray(teacher["primitive_mask"][start:stop], dtype=np.uint8)
        neighbor = np.asarray(teacher["endpoint_neighbor"][start:stop], dtype=np.int8)
        observed = unpack_endpoint_observed(
            np.asarray(sidecar["endpoint_observed_packed"][start:stop], dtype=np.uint8)
        )
        batches.append(audit_connection_cluster_batch(primitive, neighbor, observed))
    summary = merge_connection_cluster_summaries(batches)
    return summary, {
        "parent_id": str(teacher.attrs["parent_id"]),
        "geometry_realization": str(teacher.attrs["geometry_realization"]),
        "source_p1b_shard_tree_sha256": source_tree_sha,
    }


def _overflow_rows(histogram: tuple[int, ...], capacity: int) -> int:
    return int(sum(histogram[capacity + 1:]))


def _plot(summary: dict, output: Path) -> None:
    fit = summary["splits"]["fit"]
    c07 = summary["splits"]["c07"]
    capacity = int(summary["capacity"]["selected_capacity"])
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.3))
    for name, value, color in (("C01–C06 fit", fit, "#2962a3"), ("C07", c07, "#d37524")):
        counts = np.asarray(value["cluster_count_histogram"], dtype=np.int64)
        index = np.flatnonzero(counts)
        axes[0].plot(index, counts[index], marker="o", label=name, color=color)
        sizes = np.asarray(value["cluster_size_histogram"], dtype=np.int64)
        axes[1].bar(
            np.arange(2, len(sizes)) + (-0.18 if name.startswith("C01") else 0.18),
            sizes[2:], width=.35, label=name, color=color,
        )
    axes[0].axvline(capacity, color="black", linestyle="--", label=f"frozen K={capacity}")
    axes[0].set_xlabel("connection clusters in one 5-frame row")
    axes[0].set_ylabel("rows")
    axes[0].set_yscale("symlog", linthresh=1)
    axes[0].legend(fontsize=8)
    axes[1].set_xlabel("physical endpoints per cluster")
    axes[1].set_ylabel("clusters")
    axes[1].set_xticks(range(2, 5))
    axes[1].legend(fontsize=8)

    complexity = summary["complexity"]
    labels = ("pair head\nfixed", "cluster head\nfixed", "pair candidates\nactive", "cluster candidates\nactive")
    values = (
        complexity["fixed_pair_output_values"],
        complexity["fixed_cluster_output_values"],
        complexity["active_pair_candidates"],
        complexity["active_cluster_assignment_candidates"],
    )
    axes[2].bar(labels, values, color=("#a63d40", "#287271", "#d98c3f", "#5b8e7d"))
    axes[2].set_yscale("log")
    axes[2].set_ylabel("candidate scores over all rows")
    axes[2].tick_params(axis="x", labelsize=8)
    fig.suptitle("Primitive Connection Hypergraph Teacher feasibility (fit + C07 only)")
    fig.tight_layout()
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(output / f"primitive_connection_hypergraph_teacher_feasibility.{suffix}", dpi=180)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher-root", required=True, type=Path)
    parser.add_argument("--sidecar-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    per_task = []
    split_values = {}
    for split in ("fit", "c07"):
        teacher_tasks = {
            value.stem for value in (args.teacher_root.resolve() / split).glob("*.zarr")
            if value.is_dir()
        }
        sidecar_tasks = {
            value.stem for value in (args.sidecar_root.resolve() / split).glob("*.zarr")
            if value.is_dir()
        }
        if not teacher_tasks or teacher_tasks != sidecar_tasks:
            raise RuntimeError(f"{split} paired task population differs")
        tasks = sorted(teacher_tasks)
        if len(tasks) != EXPECTED[split]["tasks"]:
            raise RuntimeError(f"{split} task population drift")
        summaries = []
        parents = set()
        for task_id in tasks:
            value, provenance = _task_summary(
                task_id, split, args.teacher_root.resolve(), args.sidecar_root.resolve(),
            )
            summaries.append(value)
            parents.add(provenance["parent_id"])
            row = value.to_dict()
            row.update({
                "split": split,
                "task_id": task_id,
                **provenance,
            })
            per_task.append(row)
        if len(parents) != EXPECTED[split]["parents"]:
            raise RuntimeError(f"{split} parent population drift")
        merged = merge_connection_cluster_summaries(summaries)
        if merged.rows != EXPECTED[split]["rows"]:
            raise RuntimeError(f"{split} row population drift")
        split_values[split] = merged

    fit, c07 = split_values["fit"], split_values["c07"]
    capacity = select_fit_only_cluster_capacity(fit.maximum_clusters_per_row)
    selected = int(capacity["selected_capacity"])
    fit_overflow = _overflow_rows(fit.cluster_count_histogram, selected)
    c07_overflow = _overflow_rows(c07.cluster_count_histogram, selected)
    total_rows = fit.rows + c07.rows
    fixed_pair = total_rows * (2 * MAXIMUM_PRIMITIVES * (MAXIMUM_PRIMITIVES - 1))
    fixed_cluster = total_rows * (2 * MAXIMUM_PRIMITIVES * selected)
    active_pair = fit.pair_candidates + c07.pair_candidates
    selected_index = CAPACITY_CANDIDATES.index(selected)
    active_cluster = (
        fit.dense_assignment_candidates_by_capacity[selected_index]
        + c07.dense_assignment_candidates_by_capacity[selected_index]
    )
    checks = {
        "exact_fit_population": fit.rows == EXPECTED["fit"]["rows"],
        "exact_c07_population": c07.rows == EXPECTED["c07"]["rows"],
        "exact_total_population": total_rows == EXPECTED_TOTAL_ROWS,
        "fit_teacher_union_of_cliques": fit.passed,
        "c07_teacher_union_of_cliques": c07.passed,
        "maximum_cluster_degree_contract": max(
            fit.maximum_cluster_size, c07.maximum_cluster_size,
        ) <= 4,
        "fit_capacity_zero_overflow": fit_overflow == 0,
        "c07_capacity_zero_overflow": c07_overflow == 0,
        "c07_observable_positive_reproduction": (
            c07.observable_undirected_attachment_labels
            == EXPECTED_C07_OBSERVABLE_POSITIVES
        ),
        "nonempty_observable_connection_supervision": (
            fit.observable_undirected_attachment_labels > 0
            and c07.observable_undirected_attachment_labels > 0
        ),
        "fixed_decoder_output_compression": fixed_cluster < fixed_pair,
        "zero_model_or_downstream_work": True,
    }
    scientific_pass = all(checks.values())
    summary = {
        "schema_version": "primitive_connection_hypergraph_teacher_feasibility_v1",
        "scientific_pass": scientific_pass,
        "decision": (
            "PROCEED_TO_ENDPOINT_CLUSTER_DECODER_READINESS"
            if scientific_pass
            else "STOP_PRIMITIVE_CONNECTION_HYPERGRAPH_CANDIDATE"
        ),
        "splits": {name: value.to_dict() for name, value in split_values.items()},
        "capacity": {
            **capacity,
            "fit_overflow_rows": fit_overflow,
            "c07_overflow_rows": c07_overflow,
        },
        "complexity": {
            "fixed_pair_output_values": fixed_pair,
            "fixed_cluster_output_values": fixed_cluster,
            "fixed_output_compression_ratio": fixed_pair / fixed_cluster,
            "active_pair_candidates": active_pair,
            "active_cluster_assignment_candidates": active_cluster,
            "active_candidate_ratio_pair_over_cluster": active_pair / active_cluster,
            "observable_pair_positive_labels": (
                fit.observable_undirected_attachment_labels
                + c07.observable_undirected_attachment_labels
            ),
            "observable_endpoint_cluster_memberships": (
                fit.observable_cluster_memberships + c07.observable_cluster_memberships
            ),
            "interpretation": (
                "Fixed decoder outputs are compared at the frozen 32-primitive contract. "
                "Active counts and positive supervision are reported separately; no claim "
                "is made that every small row has fewer EK than pair candidates."
            ),
        },
        "checks": checks,
        "rows_read": total_rows,
        "fit_rows_read": fit.rows,
        "c07_rows_read": c07.rows,
        "c08_rows_read": 0,
        "c09_c10_worlds_read": 0,
        "model_forward_rows": 0,
        "optimizer_steps": 0,
        "graph_replays": 0,
        "mtare_worlds_read": 0,
    }
    write_json(output / "summary.json", summary)
    write_json(output / "per_task.json", {"tasks": per_task})
    with (output / "per_task.csv").open("w", encoding="utf-8", newline="") as stream:
        fields = [
            "split", "task_id", "parent_id", "geometry_realization", "rows",
            "active_primitives", "observed_endpoints", "physical_clusters",
            "observable_multimember_clusters",
            "observed_singletons_from_physical_multimember",
            "undirected_attachment_labels", "observable_undirected_attachment_labels",
            "maximum_clusters_per_row", "maximum_cluster_size", "passed",
        ]
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader(); writer.writerows(per_task)
    write_json(output / "figure_source.json", {
        "schema_version": "primitive_connection_hypergraph_teacher_figure_source_v1",
        "question": "Does the frozen construction Teacher form bounded observable connection clusters?",
        "splits": summary["splits"], "capacity": summary["capacity"],
        "complexity": summary["complexity"], "checks": checks,
    })
    _plot(summary, output)
    print(json.dumps({
        "scientific_pass": scientific_pass,
        "decision": summary["decision"],
        "capacity": summary["capacity"],
        "checks": checks,
    }, indent=2))
    return 0 if scientific_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
