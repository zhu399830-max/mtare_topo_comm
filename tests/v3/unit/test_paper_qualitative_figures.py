from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from mtare_topo.evaluation.paper_qualitative_figures import (
    render_fixed_qualitative_figures,
    select_fixed_qualitative_cases,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _summary(block: str, family: str, case_id: str, *, seed=None, repeat=0, volume=100.0):
    return {
        "case": {
            "case_id": case_id,
            "block_id": block,
            "world": block.split("_", 1)[0],
            "environment_seed": 11,
            "method_family": family,
            "checkpoint_seed": seed,
            "execution_repeat": repeat,
            "runtime_sec": 600.0,
        },
        "metrics": {
            "final_explored_volume_m3": volume,
            "traveling_distance_m": volume / 2.0,
        },
        "planner_evidence": (
            {} if family == "original_mtare" else {"topology_snapshot": "planner/topology_snapshot.json"}
        ),
    }


def _write_case(root: Path, run: Path, item: dict, *, graph: bool) -> tuple[Path, list[Path]]:
    case_dir = run / "artifacts/cases" / item["case"]["case_id"]
    summary = case_dir / "summary.json"
    trajectory = case_dir / "evidence/trajectory.jsonl"
    summary.parent.mkdir(parents=True, exist_ok=True)
    trajectory.parent.mkdir(parents=True, exist_ok=True)
    summary.write_text(json.dumps(item) + "\n")
    trajectory.write_text("\n".join(
        json.dumps({"xyz_m": [float(index), float(index % 2), 0.75]})
        for index in range(6)
    ) + "\n")
    paths = [summary, trajectory]
    if graph:
        snapshot = case_dir / "planner/topology_snapshot.json"
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        snapshot.write_text(json.dumps({"runtime": {"graph": {
            "nodes": [
                {"id": 0, "node_kind": "anchor", "xyz_m": [0.0, 0.0, 0.75]},
                {"id": 1, "node_kind": "structural", "xyz_m": [3.0, 1.0, 0.75]},
            ],
            "edges": [{"from": 0, "to": 1}],
        }}}) + "\n")
        paths.append(snapshot)
    return summary.relative_to(root), paths


def _seal(root: Path, run: Path, paths: list[Path]) -> str:
    target = run / "artifacts/evidence_sha256.txt"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("".join(
        f"{_sha(path)}  {path.relative_to(root).as_posix()}\n" for path in paths
    ))
    return _sha(target)


def _fixture(tmp_path: Path):
    source, corrected = [], []
    for block in ("garage_env11", "tunnel_env11"):
        source.append(_summary(block, "original_mtare", f"{block}_original", repeat=0))
        defective = _summary(block, "m1d_topology", f"{block}_m1d0", seed=0, volume=80.0)
        source.append(defective)
        corrected.append(_summary(block, "m1d_topology", f"{block}_m1d0", seed=0, volume=120.0))
    source_run, corrected_run = tmp_path / "source", tmp_path / "corrected"
    source_paths, corrected_paths = [], []
    source_rows, corrected_rows = [], []
    for item in source:
        relative, paths = _write_case(
            tmp_path, source_run, item, graph=item["case"]["method_family"] == "m1d_topology"
        )
        source_paths.extend(paths)
        source_rows.append({
            "case_id": item["case"]["case_id"], "summary_path": relative.as_posix(),
            "source": "failed_source_run",
        })
    for item in corrected:
        relative, paths = _write_case(tmp_path, corrected_run, item, graph=True)
        corrected_paths.extend(paths)
        corrected_rows.append({
            "case_id": item["case"]["case_id"], "summary_path": relative.as_posix(),
        })
    source_seal = _seal(tmp_path, source_run, source_paths)
    corrected_seal = _seal(tmp_path, corrected_run, corrected_paths)
    source_manifest = {
        "cases": source_rows,
        "source_runs": {
            "failed_source_run": {"run": "source", "seal_sha256": source_seal},
        },
    }
    corrected_manifest = {
        "run": "corrected", "seal_sha256": corrected_seal, "cases": corrected_rows,
    }
    return source, corrected, source_manifest, corrected_manifest


def test_identity_only_selection_is_exact_and_outcome_independent(tmp_path: Path):
    source, corrected, _, _ = _fixture(tmp_path)
    selected = select_fixed_qualitative_cases(source, corrected)
    assert [item["block_id"] for item in selected] == ["garage_env11", "tunnel_env11"]
    source[0]["metrics"]["final_explored_volume_m3"] = -999999.0
    assert [item["block_id"] for item in select_fixed_qualitative_cases(source, corrected)] == [
        "garage_env11", "tunnel_env11",
    ]


def test_renders_two_seal_bound_trajectory_graph_panels(tmp_path: Path):
    source, corrected, source_manifest, corrected_manifest = _fixture(tmp_path)
    manifest = render_fixed_qualitative_figures(
        source, corrected, source_manifest, corrected_manifest,
        tmp_path / "figures", project_root=tmp_path,
    )
    assert manifest["selection_uses_performance_outcomes"] is False
    assert manifest["figure_count"] == 2
    assert manifest["file_count"] == 4
    assert all(item["bytes"] > 1000 for item in manifest["files"])


def test_rejects_trajectory_drift_after_source_seal(tmp_path: Path):
    source, corrected, source_manifest, corrected_manifest = _fixture(tmp_path)
    path = tmp_path / "source/artifacts/cases/garage_env11_original/evidence/trajectory.jsonl"
    path.write_text(path.read_text() + "\n")
    with pytest.raises(ValueError, match="seal-bound"):
        render_fixed_qualitative_figures(
            source, corrected, source_manifest, corrected_manifest,
            tmp_path / "figures", project_root=tmp_path,
        )
