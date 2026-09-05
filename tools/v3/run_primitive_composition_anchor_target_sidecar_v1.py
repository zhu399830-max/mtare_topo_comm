#!/usr/bin/env python3
"""Formal immutable fit/C07 composition-anchor target sidecar run."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import shutil
import subprocess
import sys
import time
import traceback

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numcodecs
import numpy as np
import zarr

from _bootstrap import PROJECT_ROOT
from mtare_topo.governance import load_json, validate_data_card, write_json


RUN_ID = "gate3_20260903_primitive_composition_anchor_target_sidecar_v1r_seed0"
PASS = "PASS_PRIMITIVE_COMPOSITION_ANCHOR_TARGET_SIDECAR_V1R"
FAIL = "FAIL_PRIMITIVE_COMPOSITION_ANCHOR_TARGET_SIDECAR_V1R"
CARD_STATUS = "APPROVED_FOR_ONE_IMMUTABLE_PRIMITIVE_COMPOSITION_ANCHOR_TARGET_SIDECAR_V1R"
PYTHON = "/home/zeng-workstation/.local/share/mtare_topo_comm/envs/phase3_torch290_cu129_zarr2187_v1/bin/python"
P1A = PROJECT_ROOT / "results/gate3_semantics/gate3_20260830_primitive_relation_p1a_lossless_storage_corrective_v1_seed0"
P1B = PROJECT_ROOT / "results/gate3_semantics/gate3_20260830_primitive_relation_p1b_teacher_materialization_v1r_seed0"
OBSERVABILITY = PROJECT_ROOT / "results/gate3_semantics/gate3_20260902_primitive_attachment_observability_sidecar_v1_seed0"
ANCHOR_TEACHER = PROJECT_ROOT / "results/gate3_semantics/gate3_20260903_primitive_composition_anchor_teacher_readiness_v1_seed0"
MODEL_READINESS = PROJECT_ROOT / "results/gate3_semantics/gate3_20260903_primitive_composition_anchor_model_readiness_v2_seed0"


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


def _plot(summary: dict, output: Path) -> None:
    fit = summary["split"]["fit"]
    c07 = summary["split"]["c07"]
    edges = np.asarray(fit["anchor_norm_histogram_edges_m"], dtype=np.float64)
    centers = (edges[:-1] + edges[1:]) * .5
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.1))
    for name, row, color in (("C01–C06 fit", fit, "#287271"), ("C07", c07, "#d37524")):
        counts = np.asarray(row["anchor_norm_histogram_counts"], dtype=np.float64)
        axes[0].plot(centers, counts / max(1.0, counts.sum()), label=name, color=color)
    axes[0].set_xlim(0, 55)
    axes[0].set_xlabel("composition-anchor distance from current sensor [m]")
    axes[0].set_ylabel("endpoint fraction")
    axes[0].legend(fontsize=8)
    axes[1].bar(
        ("fit rows", "C07 rows"),
        (fit["sequences"], c07["sequences"]),
        color=("#287271", "#d37524"),
    )
    axes[1].set_ylabel("five-frame sequences")
    axes[1].ticklabel_format(style="plain", axis="y")
    fig.suptitle("Program-derived composition-anchor supervision (Teacher only)")
    fig.tight_layout()
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(output / f"primitive_composition_anchor_target_sidecar.{suffix}", dpi=180)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    args = parser.parse_args()
    spec = load_json(args.spec.resolve())
    run = args.run_dir.resolve()
    started = time.monotonic()
    overall, error = FAIL, None
    checks: dict[str, bool] = {}
    before: dict[str, str] = {}
    try:
        if run.name != RUN_ID or load_json(run / "RUN_STATE.json").get("state") != "CREATED_NOT_EXECUTED":
            raise RuntimeError("composition-anchor sidecar executes exactly once")
        card = load_json(PROJECT_ROOT / spec["data_card"])
        validation = validate_data_card(card)
        if not validation.passed or card.get("status") != CARD_STATUS:
            raise RuntimeError(f"composition-anchor sidecar Data Card invalid: {validation.errors}")
        for relative, expected in spec["frozen_inputs"].items():
            before[relative] = _sha(PROJECT_ROOT / relative)
            if before[relative] != expected:
                raise RuntimeError(f"composition-anchor sidecar input drift: {relative}")
        for record in spec["frozen_tools"].values():
            if _sha(PROJECT_ROOT / record["path"]) != record["sha256"]:
                raise RuntimeError(f"composition-anchor sidecar tool drift: {record['path']}")
        versions = {
            "python": sys.version.split()[0], "executable": sys.executable,
            "numpy": np.__version__, "zarr": zarr.__version__,
            "numcodecs": numcodecs.__version__, "matplotlib": matplotlib.__version__,
        }
        expected_versions = {
            "python": "3.13.5", "executable": PYTHON,
            "numpy": "2.1.3", "zarr": "2.18.7",
            "numcodecs": "0.15.1", "matplotlib": "3.10.0",
        }
        if versions != expected_versions:
            raise RuntimeError(f"composition-anchor sidecar environment drift: {versions}")
        source_checks = {
            "p1a_complete": load_json(P1A / "RUN_STATE.json").get("state") == "COMPLETED",
            "p1b_complete": load_json(P1B / "RUN_STATE.json").get("state") == "COMPLETED",
            "observability_complete": load_json(OBSERVABILITY / "RUN_STATE.json").get("state") == "COMPLETED",
            "anchor_teacher_pass": load_json(
                ANCHOR_TEACHER / "metrics/teacher_readiness/summary.json"
            ).get("scientific_pass") is True,
            "model_readiness_pass": load_json(
                MODEL_READINESS / "metrics/summary.json"
            ).get("scientific_pass") is True,
        }
        if not all(source_checks.values()):
            raise RuntimeError(f"composition-anchor sidecar source gate drift: {source_checks}")
        write_json(run / "config/environment.json", versions)
        write_json(run / "config/source_integrity_before.json", before)
        write_json(run / "RUN_STATE.json", {
            "schema_version": "v3_run_state_v1", "run_id": RUN_ID, "state": "RUNNING",
        })

        env = os.environ.copy()
        env["PYTHONPATH"] = f"{PROJECT_ROOT / 'src'}:{PROJECT_ROOT / 'tools/v3'}"
        tests = subprocess.run(
            [
                PYTHON, "-m", "pytest", "-q",
                "tests/v3/unit/test_primitive_composition_anchor_sidecar.py",
                "tests/v3/unit/test_primitive_composition_anchor_teacher.py",
                "tests/v3/unit/test_primitive_composition_anchor_model.py",
            ],
            cwd=PROJECT_ROOT, env=env,
            stdout=(run / "logs/00_unit_tests.log").open("w", encoding="utf-8"),
            stderr=subprocess.STDOUT, text=True, timeout=600, check=False,
        )
        if tests.returncode:
            raise RuntimeError("composition-anchor sidecar unit tests failed")

        output = run / "artifacts/materialized"
        command = [
            PYTHON, "tools/v3/execute_primitive_composition_anchor_target_sidecar_v1.py",
            "--sensor-root", str(P1A / "artifacts/dataset"),
            "--construction-root", str(P1A / "artifacts/constructions"),
            "--teacher-root", str(P1B / "artifacts/teacher"),
            "--observability-root", str(OBSERVABILITY / "artifacts/endpoint_observability"),
            "--teacher-manifest", str(P1B / "artifacts/task_manifest.json"),
            "--observability-manifest", str(OBSERVABILITY / "artifacts/task_manifest.json"),
            "--output-dir", str(output),
        ]
        write_json(run / "config/executor_command.json", {"command": command})
        with (run / "logs/01_executor.log").open("w", encoding="utf-8") as stream:
            execution = subprocess.run(
                command, cwd=PROJECT_ROOT, env=env, stdout=stream,
                stderr=subprocess.STDOUT, text=True, timeout=7200, check=False,
            )
        if execution.returncode:
            raise RuntimeError(f"composition-anchor sidecar executor failed: {execution.returncode}")
        payload = load_json(output / "summary.json")
        expected = spec["expected_counts"]
        checks = {
            "exact_unit_tests": True,
            "source_runs_complete": all(source_checks.values()),
            "exact_210_tasks": payload["tasks"] == expected["geometry_tasks"],
            "exact_total_sequences": payload["sequences"] == expected["sequences"],
            "exact_fit_population": payload["split"]["fit"]["parents"] == 60
                and payload["split"]["fit"]["tasks"] == 180
                and payload["split"]["fit"]["sequences"] == expected["fit_sequences"],
            "exact_c07_population": payload["split"]["c07"]["parents"] == 10
                and payload["split"]["c07"]["tasks"] == 30
                and payload["split"]["c07"]["sequences"] == expected["c07_sequences"],
            "exact_fit_attachment_pairs": payload["split"]["fit"]["attachment_anchor_pairs_checked"]
                == expected["fit_attachment_pairs"],
            "exact_c07_attachment_pairs": payload["split"]["c07"]["attachment_anchor_pairs_checked"]
                == expected["c07_attachment_pairs"],
            "connected_anchors_exactly_shared": payload["maximum_attachment_anchor_distance_m"] == 0.0,
            "float32_error_below_0p0001m": payload["maximum_float32_error_m"] <= 0.0001,
            "source_sequence_alignment_exact": payload["source_sequence_alignment_exact"] is True,
            "deterministic_full_rederive_exact": payload["deterministic_full_rederive_exact"] is True,
            "inactive_slots_exact_zero": payload["inactive_slots_exact_zero"] is True,
            "output_below_1gib": payload["sidecar_bytes"] <= 1024**3,
            "host_rss_below_4gib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) <= 4 * 1024**2,
            "zero_optimizer_model_c08_c10_graph_mtare": True,
        }
        overall = PASS if all(checks.values()) else FAIL
        _plot(payload, run / "previews")
        shutil.copy2(
            ANCHOR_TEACHER / "previews/primitive_composition_anchor_teacher_readiness.png",
            run / "previews/source_composition_anchor_teacher_readiness.png",
        )
        after = {relative: _sha(PROJECT_ROOT / relative) for relative in before}
        checks["all_frozen_inputs_unchanged"] = after == before
        if not checks["all_frozen_inputs_unchanged"]:
            overall = FAIL
        write_json(run / "config/source_integrity_after.json", after)
        write_json(run / "metrics/summary.json", {
            "schema_version": "primitive_composition_anchor_target_sidecar_formal_v1",
            "overall_status": overall, "scientific_pass": overall == PASS,
            "checks": checks, "materialization": payload,
            "optimizer_steps": 0, "model_inference_rows": 0,
            "c08_rows_read": 0, "c09_c10_worlds_read": 0,
            "graph_replays": 0, "mtare_worlds_read": 0,
            "peak_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
            "duration_seconds": time.monotonic() - started,
            "decision": "ALLOW_COMPOSITION_ANCHOR_HEAD_ONLY_THREE_SEED_TRAINING_SPEC"
                if overall == PASS else "STOP_COMPOSITION_ANCHOR_SIDECAR_FAILED",
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
