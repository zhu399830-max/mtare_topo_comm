from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from test_corrected_stochastic_comparison import _corrected, _source


ROOT = Path(__file__).resolve().parents[3]


def _load():
    tools = str(ROOT / "tools/v3")
    if tools not in sys.path:
        sys.path.insert(0, tools)
    spec = importlib.util.spec_from_file_location(
        "run_aee_composite_v9_combined_corrected_comparison_v1",
        ROOT / "tools/v3/run_aee_composite_v9_combined_corrected_comparison_v1.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _audit(tmp_path: Path):
    module = _load()
    audit = tmp_path / "audit"
    (audit / "metrics").mkdir(parents=True)
    (audit / "config").mkdir()
    (audit / "artifacts").mkdir()
    source_root = tmp_path / "source_cases"
    source_root.mkdir()
    source = _source()
    for item in source:
        family = item["case"]["method_family"]
        item["metrics"]["recording_audit"] = {"passed": True}
        item["planner_evidence"] = {
            "failed_cycles": None if family == "original_mtare" else 0
        }
        item["method_identity"] = {"method_family": family}
        item["storage"] = {
            "archive_sha256": "a" * 64,
            "decompressed_sha256": "b" * 64,
            "original_bag_sha256": "b" * 64,
        }
    manifest_rows = []
    for index, item in enumerate(source):
        path = source_root / f"{index:03d}.json"
        path.write_text(json.dumps(item, sort_keys=True) + "\n")
        manifest_rows.append({
            "index": index, "case_id": item["case"]["case_id"],
            "block_id": item["case"]["block_id"],
            "method_family": item["case"]["method_family"],
            "source": "recovery_run" if index >= 69 else "failed_source_run",
            "summary_path": path.relative_to(tmp_path).as_posix(),
            "summary_sha256": module.sha256(path),
        })
    paths = [
        audit / "RUN_STATE.json", audit / "metrics/summary.json",
        audit / "config/source_manifest.json", audit / "config/source_provenance.json",
        audit / "metrics/stochastic_analysis.json",
    ]
    paths[0].write_text(json.dumps({
        "state": "COMPLETED", "overall_status": module.SOURCE_AUDIT_STATUS,
    }) + "\n")
    paths[1].write_text(json.dumps({
        "overall_status": module.SOURCE_AUDIT_STATUS,
        "source_case_count": 90, "v9_case_count": 30,
    }) + "\n")
    paths[2].write_text(json.dumps({
        "case_count": 90,
        "block_count": 10,
        "family_case_counts": {
            "layered_gt_map_oracle": 30,
            "m1d_topology": 30,
            "original_mtare": 30,
        },
        "source_counts": {"failed_source_run": 69, "recovery_run": 21},
        "schedule_file_sha256": "a" * 64,
        "schedule_content_sha256": "b" * 64,
        "cases": manifest_rows,
    }) + "\n")
    paths[3].write_text(json.dumps({
        "source_mutation_permitted": False,
        "predecessor_run": "predecessor",
        "predecessor_seal_sha256": "c" * 64,
        "source_run": "recovery",
        "source_seal_sha256": "d" * 64,
    }) + "\n")
    paths[4].write_text(json.dumps({
        "case_count": 90, "compatibility_bridge": {"mapped_field_count": 90},
    }) + "\n")
    seal_path = audit / "artifacts/evidence_sha256.txt"
    seal_path.write_text("".join(
        f"{module.sha256(path)}  {path.relative_to(tmp_path).as_posix()}\n" for path in paths
    ))
    identity = {
        "run": "audit", "expected_status": module.SOURCE_AUDIT_STATUS,
        "seal_sha256": module.sha256(seal_path),
        "schedule_file_sha256": "a" * 64,
        "schedule_content_sha256": "b" * 64,
    }
    return module, identity, source, manifest_rows


def test_loads_exact_sealed_composed_source_without_copying_bags(tmp_path: Path):
    module, identity, source, _ = _audit(tmp_path)
    loaded, manifest = module.load_composed_audit_cases(identity, project_root=tmp_path)
    assert [item["case"]["case_id"] for item in loaded] == [item["case"]["case_id"] for item in source]
    assert manifest["case_count"] == 90
    assert len(manifest["verified_source_summary_files"]) == 90
    assert manifest["source_mutation_permitted"] is False
    assert len(manifest["cases"]) == 90
    assert set(manifest["source_runs"]) == {"failed_source_run", "recovery_run"}


def test_rejects_source_summary_drift_after_audit_seal(tmp_path: Path):
    module, identity, _, rows = _audit(tmp_path)
    path = tmp_path / rows[0]["summary_path"]
    path.write_text(path.read_text() + "\n")
    with pytest.raises(RuntimeError, match="source summary drift"):
        module.load_composed_audit_cases(identity, project_root=tmp_path)


def test_rejects_manifest_block_identity_drift(tmp_path: Path):
    module, identity, _, rows = _audit(tmp_path)
    manifest_path = tmp_path / "audit/config/source_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["cases"][0]["block_id"] = "wrong_block"
    manifest_path.write_text(json.dumps(manifest) + "\n")
    seal_path = tmp_path / "audit/artifacts/evidence_sha256.txt"
    lines = []
    for line in seal_path.read_text().splitlines():
        _, relative = line.split("  ", 1)
        path = tmp_path / relative
        lines.append(f"{module.sha256(path)}  {relative}\n")
    seal_path.write_text("".join(lines))
    identity["seal_sha256"] = module.sha256(seal_path)
    with pytest.raises(RuntimeError, match="blocks are not ten by nine"):
        module.load_composed_audit_cases(identity, project_root=tmp_path)


def test_loads_only_sealed_corrected_mechanism_summary(tmp_path: Path):
    module = _load()
    run = tmp_path / "corrected"
    (run / "metrics").mkdir(parents=True)
    (run / "artifacts").mkdir()
    summary = {
        "overall_status": module.CORRECTED_STATUS,
        "completed_case_count": 30,
        "verified_reanchor_count_total": 12,
        "verified_reanchor_case_count": 5,
        "frontier_execution_rejection_count_total": 9,
        "frontier_execution_rejection_case_count": 4,
        "case_count_with_both_correction_types": 2,
        "frontier_execution_outcomes": {"departure_mismatch": 9},
    }
    summary_path = run / "metrics/summary.json"
    summary_path.write_text(json.dumps(summary, sort_keys=True) + "\n")
    relative = summary_path.relative_to(tmp_path).as_posix()
    seal_path = run / "artifacts/evidence_sha256.txt"
    seal_path.write_text(f"{module.sha256(summary_path)}  {relative}\n")
    identity = {
        "run": "corrected", "expected_status": module.CORRECTED_STATUS,
        "seal_sha256": module.sha256(seal_path),
    }
    mechanisms, manifest = module.load_corrected_mechanism_summary(
        identity, project_root=tmp_path
    )
    assert mechanisms["verified_reanchor_count_total"] == 12
    assert manifest["summary_sha256"] == module.sha256(summary_path)
    summary_path.write_text(summary_path.read_text() + "\n")
    with pytest.raises(RuntimeError, match="drift"):
        module.load_corrected_mechanism_summary(identity, project_root=tmp_path)


def test_loads_120_coverage_curves_only_when_bound_to_source_seals(tmp_path: Path):
    module = _load()
    source = _source()
    corrected = _corrected(source)
    for item in source + corrected:
        item["metrics"]["synchronized_frame_count"] = 2

    source_rows, source_runs = [], {}
    source_entries = {"failed_source_run": [], "recovery_run": []}
    for index, item in enumerate(source):
        source_name = "failed_source_run" if index < 69 else "recovery_run"
        run = tmp_path / source_name
        case_dir = run / "artifacts/cases" / item["case"]["case_id"]
        curve = case_dir / "evidence/coverage_curve.jsonl"
        curve.parent.mkdir(parents=True, exist_ok=True)
        curve.write_text(
            '{"elapsed_sec":0.1,"explored_volume_m3":1.0}\n'
            '{"elapsed_sec":599.9,"explored_volume_m3":2.0}\n'
        )
        summary = case_dir / "summary.json"
        summary.write_text("{}\n")
        relative = curve.relative_to(tmp_path).as_posix()
        source_entries[source_name].append((curve, relative))
        source_rows.append({
            "case_id": item["case"]["case_id"],
            "summary_path": summary.relative_to(tmp_path).as_posix(),
            "source": source_name,
        })
    for source_name, entries in source_entries.items():
        run = tmp_path / source_name
        seal_path = run / "artifacts/evidence_sha256.txt"
        seal_path.write_text("".join(
            f"{module.sha256(path)}  {relative}\n" for path, relative in entries
        ))
        source_runs[source_name] = {
            "run": source_name, "seal_sha256": module.sha256(seal_path),
        }

    corrected_run = tmp_path / "corrected"
    corrected_rows, corrected_entries = [], []
    for item in corrected:
        case_dir = corrected_run / "artifacts/cases" / item["case"]["case_id"]
        curve = case_dir / "evidence/coverage_curve.jsonl"
        curve.parent.mkdir(parents=True, exist_ok=True)
        curve.write_text(
            '{"elapsed_sec":0.1,"explored_volume_m3":1.5}\n'
            '{"elapsed_sec":599.9,"explored_volume_m3":2.5}\n'
        )
        summary = case_dir / "summary.json"
        summary.write_text("{}\n")
        relative = curve.relative_to(tmp_path).as_posix()
        corrected_entries.append((curve, relative))
        corrected_rows.append({
            "case_id": item["case"]["case_id"],
            "path": summary.relative_to(tmp_path).as_posix(),
        })
    corrected_seal_path = corrected_run / "artifacts/evidence_sha256.txt"
    corrected_seal_path.write_text("".join(
        f"{module.sha256(path)}  {relative}\n" for path, relative in corrected_entries
    ))

    source_manifest = {
        "run": "audit", "seal_sha256": "a" * 64,
        "source_runs": source_runs, "cases": source_rows,
    }
    corrected_manifest = {
        "run": "corrected", "seal_sha256": module.sha256(corrected_seal_path),
        "cases": corrected_rows,
    }
    records, provenance = module.load_sealed_coverage_records(
        source, corrected, source_manifest, corrected_manifest, project_root=tmp_path
    )
    assert len(records) == 120
    assert len(provenance["verified_curve_files"]) == 120
    assert sum(row["curve_family"] == "defective_v9" for row in records) == 30
    first_curve = tmp_path / next(iter(provenance["verified_curve_files"]))
    first_curve.write_text(first_curve.read_text() + "\n")
    with pytest.raises(RuntimeError, match="seal drift"):
        module.load_sealed_coverage_records(
            source, corrected, source_manifest, corrected_manifest,
            project_root=tmp_path,
        )
