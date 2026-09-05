#!/home/zeng-workstation/anaconda3/bin/python
"""Compare sealed V5 cases with the exact composed Gate-6 source matrix."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from _bootstrap import PROJECT_ROOT
from mtare_topo.evaluation.corrected_stochastic_comparison_v2 import (
    analyze_v5_corrected_stochastic_cases,
)
from mtare_topo.evaluation.paper_single_robot_figures import (
    render_single_robot_paper_figures,
)
from mtare_topo.evaluation.paper_qualitative_figures import (
    render_fixed_qualitative_figures,
)
from mtare_topo.evaluation.paper_coverage_curve import (
    load_coverage_curve,
    render_coverage_curve,
)
from mtare_topo.evaluation.paper_single_robot_tables import (
    render_single_robot_paper_tables,
)
from mtare_topo.evaluation.stochastic_closed_loop_v2_bridge import SOURCE_STATUS
from mtare_topo.governance import load_json, write_json
import run_corrected_stochastic_comparison_v1 as comparison_v1
from run_stochastic_v2_compatibility_audit_v1 import (
    _parse_seal,
    _validate_case_summary,
    seal,
    sha256,
)


STATUS_PASS = "PASS_AEE_COMPOSITE_V9_COMBINED_CORRECTED_COMPARISON_V1"
STATUS_FAIL = "FAIL_AEE_COMPOSITE_V9_COMBINED_CORRECTED_COMPARISON_V1"
SOURCE_AUDIT_STATUS = "PASS_COMPOSED_FRONTIER_ATTEMPT_AUDIT_V1"
CORRECTED_STATUS = "PASS_AEE_COMPOSITE_V9_COMBINED_CORRECTION_STOCHASTIC_V1R2"
MECHANISM_FIELDS = (
    "verified_reanchor_count_total",
    "verified_reanchor_case_count",
    "frontier_execution_rejection_count_total",
    "frontier_execution_rejection_case_count",
    "case_count_with_both_correction_types",
    "frontier_execution_outcomes",
)


def _verify_audit_file(
    path: Path, entries: dict[str, str], project_root: Path = PROJECT_ROOT
) -> str:
    relative = path.relative_to(project_root.resolve()).as_posix()
    observed = sha256(path)
    if entries.get(relative) != observed:
        raise RuntimeError(f"composed audit bound evidence drift: {relative}")
    return observed


def load_composed_audit_cases(
    identity: dict[str, Any], *, project_root: Path = PROJECT_ROOT
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    root = project_root.resolve()
    audit = (root / identity["run"]).resolve()
    audit.relative_to(root)
    state_path = audit / "RUN_STATE.json"
    summary_path = audit / "metrics/summary.json"
    manifest_path = audit / "config/source_manifest.json"
    provenance_path = audit / "config/source_provenance.json"
    analysis_path = audit / "metrics/stochastic_analysis.json"
    state, summary, manifest, provenance, analysis = (
        load_json(state_path), load_json(summary_path), load_json(manifest_path),
        load_json(provenance_path), load_json(analysis_path),
    )
    if identity.get("expected_status") != SOURCE_AUDIT_STATUS:
        raise RuntimeError("composed comparison source status contract drift")
    if (
        state.get("state") != "COMPLETED"
        or state.get("overall_status") != SOURCE_AUDIT_STATUS
        or summary.get("overall_status") != SOURCE_AUDIT_STATUS
        or summary.get("source_case_count") != 90
        or summary.get("v9_case_count") != 30
        or manifest.get("case_count") != 90
        or provenance.get("source_mutation_permitted") is not False
        or analysis.get("case_count") != 90
        or analysis.get("compatibility_bridge", {}).get("mapped_field_count") != 90
        or manifest.get("block_count") != 10
        or manifest.get("family_case_counts") != {
            "layered_gt_map_oracle": 30,
            "m1d_topology": 30,
            "original_mtare": 30,
        }
        or manifest.get("source_counts") != {
            "failed_source_run": 69, "recovery_run": 21,
        }
        or manifest.get("schedule_file_sha256") != identity.get("schedule_file_sha256")
        or manifest.get("schedule_content_sha256") != identity.get("schedule_content_sha256")
    ):
        raise RuntimeError("composed comparison source audit is not the exact PASS")
    entries = _parse_seal(audit, identity["seal_sha256"])
    bound = {}
    for path in (state_path, summary_path, manifest_path, provenance_path, analysis_path):
        bound[path.relative_to(root).as_posix()] = _verify_audit_file(path, entries, root)
    rows = manifest.get("cases")
    if not isinstance(rows, list) or len(rows) != 90:
        raise RuntimeError("composed comparison source manifest is not exactly 90 cases")
    if [row.get("index") for row in rows] != list(range(90)):
        raise RuntimeError("composed comparison source indices are not exact and ordered")
    row_blocks = Counter(row.get("block_id") for row in rows)
    row_families = Counter(row.get("method_family") for row in rows)
    row_sources = Counter(row.get("source") for row in rows)
    if len(row_blocks) != 10 or set(row_blocks.values()) != {9}:
        raise RuntimeError("composed comparison source blocks are not ten by nine")
    if row_families != Counter({
        "original_mtare": 30, "m1d_topology": 30,
        "layered_gt_map_oracle": 30,
    }):
        raise RuntimeError("composed comparison manifest family quotas drift")
    if row_sources != Counter({"failed_source_run": 69, "recovery_run": 21}):
        raise RuntimeError("composed comparison manifest source quotas drift")
    summaries = []
    source_files = {}
    for row in rows:
        path = (root / row["summary_path"]).resolve()
        path.relative_to(root)
        observed = sha256(path)
        if observed != row.get("summary_sha256"):
            raise RuntimeError(f"composed source summary drift: {row.get('case_id')}")
        item = load_json(path)
        if item.get("status") != SOURCE_STATUS or item.get("case", {}).get("case_id") != row.get("case_id"):
            raise RuntimeError(f"composed source summary identity drift: {row.get('case_id')}")
        for field in ("block_id", "method_family"):
            if item.get("case", {}).get(field) != row.get(field):
                raise RuntimeError(
                    f"composed source summary manifest drift: {row.get('case_id')}: {field}"
                )
        _validate_case_summary(item, item["case"])
        summaries.append(item)
        source_files[path.relative_to(root).as_posix()] = observed
    quotas = Counter(item["case"]["method_family"] for item in summaries)
    if quotas != Counter({"original_mtare": 30, "m1d_topology": 30, "layered_gt_map_oracle": 30}):
        raise RuntimeError("composed comparison source quotas are not 30/30/30")
    return summaries, {
        "schema_version": "combined_corrected_comparison_source_v1",
        "run": identity["run"], "status": SOURCE_AUDIT_STATUS,
        "seal_sha256": identity["seal_sha256"],
        "case_count": 90, "verified_audit_files": bound,
        "schedule_file_sha256": manifest["schedule_file_sha256"],
        "schedule_content_sha256": manifest["schedule_content_sha256"],
        "cases": rows,
        "source_runs": {
            "failed_source_run": {
                "run": provenance["predecessor_run"],
                "seal_sha256": provenance["predecessor_seal_sha256"],
            },
            "recovery_run": {
                "run": provenance["source_run"],
                "seal_sha256": provenance["source_seal_sha256"],
            },
        },
        "verified_source_summary_files": source_files,
        "source_mutation_permitted": False,
    }


def load_corrected_mechanism_summary(
    identity: dict[str, Any], *, project_root: Path = PROJECT_ROOT
) -> tuple[dict[str, Any], dict[str, Any]]:
    root = project_root.resolve()
    run = (root / identity["run"]).resolve()
    run.relative_to(root)
    summary_path = run / "metrics/summary.json"
    summary = load_json(summary_path)
    if (
        identity.get("expected_status") != CORRECTED_STATUS
        or summary.get("overall_status") != CORRECTED_STATUS
        or summary.get("completed_case_count") != 30
    ):
        raise RuntimeError("corrected V5 mechanism summary is not the exact PASS")
    entries = _parse_seal(run, identity["seal_sha256"])
    relative = summary_path.relative_to(root).as_posix()
    observed = sha256(summary_path)
    if entries.get(relative) != observed:
        raise RuntimeError("corrected V5 mechanism summary drift")
    mechanisms = {"completed_case_count": 30}
    for field in MECHANISM_FIELDS:
        if field not in summary:
            raise RuntimeError(f"corrected V5 mechanism summary lacks {field}")
        mechanisms[field] = summary[field]
    return mechanisms, {
        "schema_version": "corrected_v5_mechanism_manifest_v1",
        "run": identity["run"],
        "status": CORRECTED_STATUS,
        "seal_sha256": identity["seal_sha256"],
        "summary_path": relative,
        "summary_sha256": observed,
        "mechanisms": mechanisms,
        "source_mutation_permitted": False,
    }


def load_sealed_coverage_records(
    source: list[dict[str, Any]],
    corrected: list[dict[str, Any]],
    source_manifest: dict[str, Any],
    corrected_manifest: dict[str, Any],
    *,
    project_root: Path = PROJECT_ROOT,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    root = project_root.resolve()
    source_by_id = {item["case"]["case_id"]: item for item in source}
    corrected_by_id = {item["case"]["case_id"]: item for item in corrected}
    if len(source_by_id) != 90 or len(corrected_by_id) != 30:
        raise RuntimeError("coverage curve inputs are not exact 90+30 summaries")
    source_seals = {}
    for name, identity in source_manifest["source_runs"].items():
        run = (root / identity["run"]).resolve()
        run.relative_to(root)
        source_seals[name] = _parse_seal(run, identity["seal_sha256"])
    corrected_run = (root / corrected_manifest["run"]).resolve()
    corrected_run.relative_to(root)
    corrected_seal = _parse_seal(corrected_run, corrected_manifest["seal_sha256"])

    records, verified = [], {}
    for row in source_manifest["cases"]:
        case_id = row["case_id"]
        summary = source_by_id[case_id]
        path = (root / row["summary_path"]).resolve().parent / "evidence/coverage_curve.jsonl"
        path.relative_to(root)
        relative = path.relative_to(root).as_posix()
        observed = sha256(path)
        entries = source_seals.get(row["source"])
        if entries is None or entries.get(relative) != observed:
            raise RuntimeError(f"source coverage curve seal drift: {case_id}")
        records.append({
            "curve_family": (
                "defective_v9"
                if summary["case"]["method_family"] == "m1d_topology"
                else summary["case"]["method_family"]
            ),
            "case_id": case_id,
            "block_id": summary["case"]["block_id"],
            "samples": load_coverage_curve(
                path, expected_samples=int(summary["metrics"]["synchronized_frame_count"])
            ),
        })
        verified[relative] = observed
    for row in corrected_manifest["cases"]:
        case_id = row["case_id"]
        summary = corrected_by_id[case_id]
        path = (root / row["path"]).resolve().parent / "evidence/coverage_curve.jsonl"
        path.relative_to(root)
        relative = path.relative_to(root).as_posix()
        observed = sha256(path)
        if corrected_seal.get(relative) != observed:
            raise RuntimeError(f"corrected coverage curve seal drift: {case_id}")
        records.append({
            "curve_family": "corrected_v5",
            "case_id": case_id,
            "block_id": summary["case"]["block_id"],
            "samples": load_coverage_curve(
                path, expected_samples=int(summary["metrics"]["synchronized_frame_count"])
            ),
        })
        verified[relative] = observed
    if len(verified) != 120:
        raise RuntimeError("coverage curve provenance is not exactly 120 unique files")
    return records, {
        "source_audit_run": source_manifest["run"],
        "source_audit_seal_sha256": source_manifest["seal_sha256"],
        "corrected_v5_run": corrected_manifest["run"],
        "corrected_v5_seal_sha256": corrected_manifest["seal_sha256"],
        "verified_curve_files": dict(sorted(verified.items())),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    args = parser.parse_args()
    spec, run_dir = load_json(args.spec.resolve()), args.run_dir.resolve()
    if run_dir.name != spec["run_id"] or load_json(run_dir / "RUN_STATE.json").get("state") != "CREATED_NOT_EXECUTED":
        raise RuntimeError("combined corrected comparison run identity/state mismatch")
    started = time.monotonic()
    write_json(run_dir / "RUN_STATE.json", {
        "schema_version": "v3_run_state_v1", "run_id": spec["run_id"], "state": "RUNNING",
    })
    try:
        if spec.get("gate") != 6 or spec.get("operation") != "audit":
            raise RuntimeError("Gate-6 corrected comparison audit scope required")
        if spec.get("user_authorization", {}).get("status") != "APPROVED":
            raise RuntimeError("approved autonomous comparison scope required")
        for name, item in spec["frozen_tools"].items():
            if sha256(PROJECT_ROOT / item["path"]) != item["sha256"]:
                raise RuntimeError(f"frozen combined comparison tool drift: {name}")
        source, source_manifest = load_composed_audit_cases(spec["source_audit"])
        corrected, corrected_manifest = comparison_v1.load_sealed_cases(
            spec["corrected_v5_run"], expected_count=30
        )
        if spec["corrected_v5_run"].get("expected_status") != CORRECTED_STATUS:
            raise RuntimeError("corrected V5 status contract drift")
        mechanisms, mechanism_manifest = load_corrected_mechanism_summary(
            spec["corrected_v5_run"]
        )
        coverage_records, coverage_provenance = load_sealed_coverage_records(
            source, corrected, source_manifest, corrected_manifest
        )
        analysis = analyze_v5_corrected_stochastic_cases(source, corrected)
        write_json(run_dir / "config/source_audit_manifest.json", source_manifest)
        write_json(run_dir / "config/corrected_v5_manifest.json", corrected_manifest)
        write_json(
            run_dir / "config/corrected_v5_mechanism_manifest.json",
            mechanism_manifest,
        )
        write_json(run_dir / "metrics/corrected_stochastic_analysis.json", analysis)
        figure_manifest = render_single_robot_paper_figures(
            analysis, run_dir / "previews"
        )
        write_json(run_dir / "metrics/paper_figure_manifest.json", figure_manifest)
        qualitative_manifest = render_fixed_qualitative_figures(
            source, corrected, source_manifest, corrected_manifest,
            run_dir / "previews/qualitative", project_root=PROJECT_ROOT,
        )
        write_json(
            run_dir / "metrics/qualitative_figure_manifest.json", qualitative_manifest
        )
        table_manifest = render_single_robot_paper_tables(
            analysis,
            mechanisms,
            {
                "source_audit_run": spec["source_audit"]["run"],
                "source_audit_seal_sha256": spec["source_audit"]["seal_sha256"],
                "corrected_v5_run": spec["corrected_v5_run"]["run"],
                "corrected_v5_seal_sha256": spec["corrected_v5_run"]["seal_sha256"],
            },
            run_dir / "previews/tables",
        )
        write_json(run_dir / "metrics/paper_table_manifest.json", table_manifest)
        coverage_manifest = render_coverage_curve(
            coverage_records,
            coverage_provenance,
            run_dir / "previews/coverage",
        )
        write_json(
            run_dir / "metrics/coverage_curve_manifest.json", coverage_manifest
        )
        primary = analysis["corrected_main_analysis"]["primary_metric"]
        summary = {
            "schema_version": "aee_composite_v9_combined_corrected_comparison_summary_v1",
            "overall_status": STATUS_PASS,
            "combined_case_count": 90, "corrected_v5_case_count": 30,
            "reused_original_mtare_case_count": 30,
            "reused_oracle_diagnostic_case_count": 30,
            "block_count": 10, "primary_metric": primary,
            "paper_figure_count": (
                figure_manifest["figure_count"] + qualitative_manifest["figure_count"]
                + coverage_manifest["figure_count"]
            ),
            "paper_figure_file_count": (
                figure_manifest["file_count"] + qualitative_manifest["file_count"]
                + 2
            ),
            "coverage_curve_evidence_file_count": coverage_manifest["file_count"],
            "paper_table_count": table_manifest["table_count"],
            "paper_table_file_count": table_manifest["file_count"],
            "qualitative_selection_uses_performance_outcomes": qualitative_manifest[
                "selection_uses_performance_outcomes"
            ],
            "primary_v5_minus_original": analysis["corrected_main_analysis"]["m1d_comparisons"][primary],
            "primary_v5_minus_defective_v9": analysis["v5_vs_defective_v9"]["metrics"][primary],
            "source_mutation_count": 0, "raw_bag_reads": 0,
            "training_steps": 0, "optimizer_steps": 0,
            "c09_reads": 0, "c10_reads": 0,
            "duration_seconds": time.monotonic() - started,
            "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        write_json(run_dir / "metrics/summary.json", summary)
        write_json(run_dir / "RUN_STATE.json", {
            "schema_version": "v3_run_state_v1", "run_id": spec["run_id"],
            "state": "COMPLETED", "overall_status": STATUS_PASS,
        })
        sealed = seal(run_dir)
        print(json.dumps({**summary, "sealed_files": sealed}, indent=2, sort_keys=True))
        return 0
    except Exception as exc:
        failure = {
            "schema_version": "aee_composite_v9_combined_corrected_comparison_summary_v1",
            "overall_status": STATUS_FAIL, "failure_reason": str(exc),
            "source_mutation_count": 0, "raw_bag_reads": 0,
            "training_steps": 0, "optimizer_steps": 0,
            "c09_reads": 0, "c10_reads": 0,
            "duration_seconds": time.monotonic() - started,
        }
        write_json(run_dir / "metrics/summary.json", failure)
        write_json(run_dir / "RUN_STATE.json", {
            "schema_version": "v3_run_state_v1", "run_id": spec["run_id"],
            "state": "FAILED", "overall_status": STATUS_FAIL,
        })
        seal(run_dir)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
