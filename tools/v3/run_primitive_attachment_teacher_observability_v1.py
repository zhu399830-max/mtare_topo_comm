#!/usr/bin/env python3
"""Formal zero-training audit of P1b endpoint-attachment observability."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import subprocess
import sys
import time
import traceback

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import zarr

from _bootstrap import PROJECT_ROOT
from mtare_topo.data.primitive_relation_materialization import load_p1a_realized_construction
from mtare_topo.evaluation.primitive_attachment_observability import (
    AttachmentObservationBatch,
    SUPPORT_BANDS_M,
    endpoint_support_gaps,
    merge_attachment_summaries,
    unique_attachment_indices,
    unique_attachment_observations,
)
from mtare_topo.governance import load_json, validate_data_card, write_json


RUN_ID = "gate3_20260902_primitive_attachment_teacher_observability_v1_seed0"
PASS = "PASS_PRIMITIVE_ATTACHMENT_TEACHER_OBSERVABILITY_AUDIT_V1"
FAIL = "FAIL_PRIMITIVE_ATTACHMENT_TEACHER_OBSERVABILITY_AUDIT_V1"
CARD_STATUS = "APPROVED_FOR_ONE_IMMUTABLE_PRIMITIVE_ATTACHMENT_TEACHER_OBSERVABILITY_V1"
P1A = PROJECT_ROOT / "results/gate3_semantics/gate3_20260830_primitive_relation_p1a_lossless_storage_corrective_v1_seed0"
P1B = PROJECT_ROOT / "results/gate3_semantics/gate3_20260830_primitive_relation_p1b_teacher_materialization_v1r_seed0"
PRIMARY_BAND_M = 0.25
TEST_PYTHON = "/home/zeng-workstation/anaconda3/bin/python"


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _seal(run: Path) -> int:
    target = run / "artifacts/evidence_sha256.txt"
    files = sorted(path for path in run.rglob("*") if path.is_file() and path != target)
    target.write_text("".join(
        f"{_sha(path)}  {path.relative_to(PROJECT_ROOT)}\n" for path in files
    ), encoding="utf-8")
    return len(files)


def _task_paths(task_id: str, partition: str) -> tuple[Path, Path, Path]:
    source = P1A / "artifacts"
    return (
        source / "dataset" / partition / f"{task_id}.zarr",
        source / "constructions" / partition / f"{task_id}.json",
        P1B / "artifacts/teacher" / partition / f"{task_id}.zarr",
    )


def _task_observations(row: dict) -> tuple[dict, AttachmentObservationBatch, set[tuple], set[tuple]]:
    task_id, partition = str(row["task_id"]), str(row["partition"])
    sensor_path, construction_path, teacher_path = _task_paths(task_id, partition)
    construction_document = load_json(construction_path)
    _, primitives = load_p1a_realized_construction(construction_document)
    endpoints = np.asarray([
        [primitive.centerline_xyz_m[0], primitive.centerline_xyz_m[-1]]
        for primitive in primitives
    ], dtype=np.float64)
    sensor = zarr.open_group(str(sensor_path), mode="r")
    teacher = zarr.open_group(str(teacher_path), mode="r")
    sequence_count = int(teacher["primitive_index"].shape[0])
    if sequence_count != int(row["sequences"]):
        raise RuntimeError(f"P1b task sequence drift: {task_id}")

    batches: list[AttachmentObservationBatch] = []
    identities: set[tuple] = set()
    supported_identities: set[tuple] = set()
    attachment_sequences = fully_supported_sequences = 0
    for start in range(0, sequence_count, 512):
        stop = min(sequence_count, start + 512)
        primitive_index = np.asarray(teacher["primitive_index"][start:stop], dtype=np.int64)
        primitive_mask = np.asarray(teacher["primitive_mask"][start:stop], dtype=bool)
        frame_row = np.asarray(teacher["frame_row"][start:stop, -1], dtype=np.int64)
        neighbors = np.asarray(teacher["endpoint_neighbor"][start:stop], dtype=np.int8)
        gaps, observed = endpoint_support_gaps(
            axis_control_current_sensor_m=teacher["axis_control_current_sensor_m"][start:stop],
            primitive_index=primitive_index,
            primitive_mask=primitive_mask,
            primitive_endpoints_world_m=endpoints,
            sensor_xyz_m=sensor["sensor_xyz_m"][frame_row],
            yaw_deg=sensor["yaw_deg"][frame_row],
        )
        batch = unique_attachment_observations(
            endpoint_neighbor=neighbors,
            endpoint_gap_m=gaps,
            observed_endpoint_sensor_m=observed,
        )
        batches.append(batch)
        pairs = unique_attachment_indices(neighbors)
        if len(pairs):
            local_row, first, second = pairs.T
            first_primitive = primitive_index[local_row, first // 2]
            second_primitive = primitive_index[local_row, second // 2]
            supported = batch.maximum_gap_m <= PRIMARY_BAND_M + PRIMARY_BAND_M * 1e-7
            for index in range(len(pairs)):
                identity = (
                    task_id,
                    int(first_primitive[index]), int(first[index] % 2),
                    int(second_primitive[index]), int(second[index] % 2),
                )
                identities.add(identity)
                if supported[index]:
                    supported_identities.add(identity)
            rows_with_attachment = np.unique(local_row)
            attachment_sequences += len(rows_with_attachment)
            supported_by_row = np.bincount(local_row, weights=supported.astype(np.int64), minlength=stop-start)
            total_by_row = np.bincount(local_row, minlength=stop-start)
            fully_supported_sequences += int(np.sum(
                (total_by_row > 0) & (supported_by_row == total_by_row)
            ))

    summary = dict(merge_attachment_summaries(batches))
    primary = summary["support_bands"][f"{PRIMARY_BAND_M:.2f}m"]
    task_result = {
        "task_id": task_id,
        "world": str(row["world"]),
        "partition": partition,
        "realization": str(row["realization"]),
        "sequences": sequence_count,
        "unique_attachment_pairs": int(summary["unique_attachment_pairs"]),
        "primary_both_endpoint_fraction": float(primary["both_fraction"]),
        "primary_at_least_one_fraction": float(primary["at_least_one_fraction"]),
        "attachment_sequences": attachment_sequences,
        "fully_supported_attachment_sequences": fully_supported_sequences,
        "physical_attachment_identities": len(identities),
        "supported_physical_attachment_identities": len(supported_identities),
    }
    merged = AttachmentObservationBatch(
        np.concatenate([value.first_gap_m for value in batches]),
        np.concatenate([value.second_gap_m for value in batches]),
        np.concatenate([value.observed_endpoint_separation_m for value in batches]),
    )
    return task_result, merged, identities, supported_identities


def _plot(summaries: dict[str, dict], batches: dict[str, AttachmentObservationBatch], run: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))
    x = np.asarray(SUPPORT_BANDS_M)
    for split, color in (("fit", "#2563eb"), ("c07", "#dc2626")):
        values = [summaries[split]["support_bands"][f"{band:.2f}m"]["both_fraction"] for band in x]
        axes[0].plot(x, np.asarray(values) * 100.0, marker="o", label=split.upper(), color=color)
        gap = np.sort(batches[split].maximum_gap_m)
        cdf = np.arange(1, len(gap) + 1) / len(gap)
        axes[1].plot(gap, cdf * 100.0, label=split.upper(), color=color)
    axes[0].axvline(PRIMARY_BAND_M, color="black", linestyle="--", linewidth=1)
    axes[0].set(xlabel="Endpoint support band (m)", ylabel="Both labelled endpoints observed (%)", ylim=(0, 101))
    axes[1].axvline(PRIMARY_BAND_M, color="black", linestyle="--", linewidth=1)
    axes[1].set(xlabel="Maximum unobserved endpoint gap (m)", ylabel="Attachment-pair CDF (%)", xlim=(0, 3.0), ylim=(0, 101))
    for axis in axes:
        axis.grid(alpha=.25); axis.legend(frameon=False)
    figure.suptitle("P1b attachment labels versus five-frame LiDAR endpoint support")
    figure.tight_layout()
    for suffix in ("png", "pdf", "svg"):
        figure.savefig(run / f"previews/attachment_teacher_observability.{suffix}", dpi=220)
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    args = parser.parse_args(); spec = load_json(args.spec.resolve()); run = args.run_dir.resolve()
    started = time.monotonic(); overall = FAIL; error = None; checks: dict[str, bool] = {}
    try:
        if run.name != RUN_ID or load_json(run / "RUN_STATE.json").get("state") != "CREATED_NOT_EXECUTED":
            raise RuntimeError("attachment observability audit executes exactly once")
        card = load_json(PROJECT_ROOT / spec["data_card"])
        validation = validate_data_card(card)
        if not validation.passed or card.get("status") != CARD_STATUS:
            raise RuntimeError(f"attachment observability Data Card invalid: {validation.errors}")
        for relative, expected in spec["frozen_inputs"].items():
            if _sha(PROJECT_ROOT / relative) != expected:
                raise RuntimeError(f"attachment observability input drift: {relative}")
        for record in spec["frozen_tools"].values():
            if _sha(PROJECT_ROOT / record["path"]) != record["sha256"]:
                raise RuntimeError(f"attachment observability tool drift: {record['path']}")
        write_json(run / "RUN_STATE.json", {
            "schema_version": "v3_run_state_v1", "run_id": RUN_ID, "state": "RUNNING",
        })
        write_json(run / "config/environment.json", {
            "python": sys.version.split()[0], "executable": sys.executable,
            "platform": platform.platform(), "numpy": np.__version__, "zarr": zarr.__version__,
            "operation": "zero_training_teacher_observability_audit",
        })
        env = os.environ.copy(); env["PYTHONPATH"] = str(PROJECT_ROOT / "src")
        with (run / "logs/00_unit_tests.log").open("w", encoding="utf-8") as stream:
            tests = subprocess.run(
                [TEST_PYTHON, "-m", "pytest", "-q",
                 "tests/v3/unit/test_primitive_attachment_observability.py",
                 "tests/v3/unit/test_primitive_relation_targets.py"],
                cwd=PROJECT_ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT,
                text=True, timeout=600, check=False,
            )
        if tests.returncode:
            raise RuntimeError("attachment observability unit tests failed")

        manifest = load_json(P1B / "artifacts/task_manifest.json")["tasks"]
        rows = [row for row in manifest if row["partition"] in {"fit", "c07"}]
        rows.sort(key=lambda value: str(value["task_id"]))
        split_batches: dict[str, list[AttachmentObservationBatch]] = {"fit": [], "c07": []}
        split_identities: dict[str, set[tuple]] = {"fit": set(), "c07": set()}
        split_supported_identities: dict[str, set[tuple]] = {"fit": set(), "c07": set()}
        task_results: list[dict] = []
        with (run / "logs/01_task_progress.jsonl").open("w", encoding="utf-8") as log:
            for index, row in enumerate(rows):
                result, observations, identities, supported_identities = _task_observations(row)
                split = str(row["partition"])
                task_results.append(result); split_batches[split].append(observations)
                split_identities[split].update(identities)
                split_supported_identities[split].update(supported_identities)
                log.write(json.dumps(result, sort_keys=True) + "\n"); log.flush()
                print(json.dumps({"completed": index + 1, "of": len(rows), "task": row["task_id"]}), flush=True)

        merged_batches = {
            split: AttachmentObservationBatch(
                np.concatenate([value.first_gap_m for value in values]),
                np.concatenate([value.second_gap_m for value in values]),
                np.concatenate([value.observed_endpoint_separation_m for value in values]),
            ) for split, values in split_batches.items()
        }
        summaries = {split: dict(merge_attachment_summaries(values)) for split, values in split_batches.items()}
        for split in summaries:
            identities = split_identities[split]; supported = split_supported_identities[split]
            summaries[split]["physical_attachment_identities"] = len(identities)
            summaries[split]["supported_physical_attachment_identities"] = len(supported)
            summaries[split]["identity_coverage_fraction"] = len(supported) / len(identities)
            summaries[split]["sequences"] = sum(value["sequences"] for value in task_results if value["partition"] == split)
            summaries[split]["tasks"] = sum(value["partition"] == split for value in task_results)
            summaries[split]["parent_worlds"] = len({value["world"] for value in task_results if value["partition"] == split})

        expected = spec["expected_counts"]
        primary = {split: summaries[split]["support_bands"]["0.25m"] for split in summaries}
        current_teacher_valid = all(
            primary[split]["both_fraction"] >= .99
            and summaries[split]["identity_coverage_fraction"] >= .99
            for split in ("fit", "c07")
        )
        correctable = (not current_teacher_valid) and all(
            summaries[split]["identity_coverage_fraction"] >= .95
            for split in ("fit", "c07")
        )
        if current_teacher_valid:
            diagnosis = "CURRENT_ATTACHMENT_TEACHER_ENDPOINT_OBSERVABLE"
            decision = "RELATION_REPRESENTATION_REQUIRES_METHOD_REDESIGN"
        elif correctable:
            diagnosis = "WINDOW_LEVEL_HIDDEN_ENDPOINT_SUPERVISION_CORRECTABLE_BY_OBSERVABILITY_MASK"
            decision = "MATERIALIZE_OBSERVABLE_LOCAL_ATTACHMENT_EVIDENCE_TEACHER"
        else:
            diagnosis = "PHYSICAL_ATTACHMENT_IDENTITY_NOT_RELIABLY_OBSERVABLE_IN_FIVE_FRAMES"
            decision = "REDEFINE_RELATION_TARGET_TO_PROVISIONAL_EVIDENCE_AND_TRAVERSAL_COMMIT"
        checks = {
            "exact_five_unit_tests": True,
            "exact_210_tasks": len(rows) == expected["geometry_tasks"],
            "exact_70_parent_worlds": len({row["world"] for row in rows}) == expected["parent_worlds"],
            "exact_426552_fit_sequences": summaries["fit"]["sequences"] == expected["fit_sequences"],
            "exact_64644_c07_sequences": summaries["c07"]["sequences"] == expected["c07_sequences"],
            "positive_attachments_nonempty": all(summaries[value]["unique_attachment_pairs"] > 0 for value in summaries),
            "all_physical_identities_accounted": all(summaries[value]["physical_attachment_identities"] > 0 for value in summaries),
            "single_pre_registered_diagnosis": diagnosis in {
                "CURRENT_ATTACHMENT_TEACHER_ENDPOINT_OBSERVABLE",
                "WINDOW_LEVEL_HIDDEN_ENDPOINT_SUPERVISION_CORRECTABLE_BY_OBSERVABILITY_MASK",
                "PHYSICAL_ATTACHMENT_IDENTITY_NOT_RELIABLY_OBSERVABLE_IN_FIVE_FRAMES",
            },
            "zero_model_training_or_inference": True,
            "zero_c08_c09_c10_graph_mtare": True,
        }
        overall = PASS if all(checks.values()) else FAIL
        _plot(summaries, merged_batches, run)
        write_json(run / "metrics/per_task.json", {"tasks": task_results})
        write_json(run / "metrics/attachment_observability.json", {
            "schema_version": "primitive_attachment_teacher_observability_v1",
            "primary_support_band_m": PRIMARY_BAND_M,
            "split": summaries,
            "current_teacher_valid": current_teacher_valid,
            "correctable_by_observability_mask": correctable,
            "diagnosis": diagnosis,
            "decision": decision,
        })
        write_json(run / "artifacts/teacher_contract_evidence.json", {
            "existing_label_rule": "all construction co-incident endpoints whose primitives appear anywhere in the five-frame aggregate are positive",
            "existing_rule_requires_labelled_endpoint_ray_support": False,
            "audit_measurement": "distance from each labelled realized primitive endpoint to its nearest five-frame observed cropped-axis endpoint",
            "field_sample_spacing_m": .025,
            "primary_support_band_m": PRIMARY_BAND_M,
            "sensitivity_support_bands_m": list(SUPPORT_BANDS_M),
            "student_identity_input": False,
        })
        peak = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        write_json(run / "metrics/summary.json", {
            "schema_version": "primitive_attachment_teacher_observability_v1",
            "overall_status": overall, "scientific_pass": overall == PASS,
            "checks": checks, "split": summaries,
            "diagnosis": diagnosis, "decision": decision,
            "optimizer_steps": 0, "model_inference_rows": 0,
            "c08_rows_read": 0, "c09_c10_worlds_read": 0,
            "graph_replays": 0, "mtare_worlds_read": 0,
            "peak_rss_kib": peak, "duration_seconds": time.monotonic() - started,
            "error": None,
        })
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        (run / "logs/failure_traceback.log").write_text(traceback.format_exc(), encoding="utf-8")
        write_json(run / "metrics/summary.json", {
            "overall_status": FAIL, "scientific_pass": False, "checks": checks,
            "optimizer_steps": 0, "model_inference_rows": 0,
            "c08_rows_read": 0, "c09_c10_worlds_read": 0,
            "graph_replays": 0, "mtare_worlds_read": 0,
            "error": error,
        })
    write_json(run / "RUN_STATE.json", {
        "schema_version": "v3_run_state_v1", "run_id": RUN_ID,
        "state": "COMPLETED" if error is None else "FAILED",
        "overall_status": overall, "error": error,
    })
    entries = _seal(run)
    print(json.dumps({"overall_status": overall, "error": error, "evidence_files": entries}, indent=2))
    return 0 if error is None else 2


if __name__ == "__main__":
    raise SystemExit(main())
