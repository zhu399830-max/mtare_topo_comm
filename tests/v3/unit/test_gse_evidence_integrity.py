from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

from mtare_topo.evaluation.gse_evidence_integrity import (
    verify_complete_run_seal,
    verify_failed_component_run_seal,
)


ROOT = Path(__file__).resolve().parents[3]


def _write_run(
    root: Path,
    name: str,
    status: str,
    *,
    strict_test_worlds_read: int = 0,
    mtare_worlds_read: int = 0,
) -> Path:
    run = root / "results" / name
    (run / "metrics").mkdir(parents=True)
    (run / "artifacts").mkdir()
    files = {
        run / "RUN_STATE.json": {
            "run_id": name,
            "state": "COMPLETED",
            "overall_status": status,
        },
        run / "metrics/summary.json": {
            "overall_status": status,
            "strict_test_worlds_read": strict_test_worlds_read,
            "mtare_worlds_read": mtare_worlds_read,
        },
        run / "artifacts/payload.json": {"value": 1},
    }
    for path, payload in files.items():
        path.write_text(json.dumps(payload), encoding="utf-8")
    (run / "artifacts/evidence_sha256.txt").write_text(
        "".join(
            f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.relative_to(root)}\n"
            for path in files
        ),
        encoding="utf-8",
    )
    return run


def _load_evaluator():
    spec = importlib.util.spec_from_file_location(
        "gse_offline_evaluator_evidence_synthetic",
        ROOT / "tools/v3/evaluate_gse_offline_topology_v1.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_complete_run_seal_requires_exact_full_file_coverage(tmp_path: Path) -> None:
    run = _write_run(tmp_path, "source", "PASS_SOURCE")
    result = verify_complete_run_seal(tmp_path, run, "PASS_SOURCE")
    assert result["entries"] == 3
    assert result["exact_full_file_coverage"] is True

    extra = run / "artifacts/unsealed.json"
    extra.write_text("{}\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="does not exactly cover"):
        verify_complete_run_seal(tmp_path, run, "PASS_SOURCE")


def test_complete_run_seal_rejects_path_escape_and_empty_manifest(tmp_path: Path) -> None:
    run = _write_run(tmp_path, "source", "PASS_SOURCE")
    outside = tmp_path / "outside.json"
    outside.write_text("{}\n", encoding="utf-8")
    seal = run / "artifacts/evidence_sha256.txt"
    seal.write_text(
        f"{hashlib.sha256(outside.read_bytes()).hexdigest()}  {outside.relative_to(tmp_path)}\n",
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="escapes its run"):
        verify_complete_run_seal(tmp_path, run, "PASS_SOURCE")

    seal.write_text("", encoding="utf-8")
    with pytest.raises(RuntimeError, match="seal is empty"):
        verify_complete_run_seal(tmp_path, run, "PASS_SOURCE")


def test_complete_run_seal_rejects_forbidden_read_and_run_identity(tmp_path: Path) -> None:
    forbidden = _write_run(tmp_path, "forbidden", "PASS_SOURCE", mtare_worlds_read=1)
    with pytest.raises(RuntimeError, match="leakage-free PASS"):
        verify_complete_run_seal(tmp_path, forbidden, "PASS_SOURCE")

    run = _write_run(tmp_path, "identity", "PASS_SOURCE")
    state = json.loads((run / "RUN_STATE.json").read_text(encoding="utf-8"))
    state["run_id"] = "another_run"
    (run / "RUN_STATE.json").write_text(json.dumps(state), encoding="utf-8")
    with pytest.raises(RuntimeError, match="leakage-free PASS"):
        verify_complete_run_seal(tmp_path, run, "PASS_SOURCE")


def test_failed_component_seal_is_explicit_and_does_not_become_a_pass(tmp_path: Path) -> None:
    run = _write_run(tmp_path, "failed_component", "FAIL_COMPONENT")
    state_path = run / "RUN_STATE.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state["state"] = "FAILED"
    state_path.write_text(json.dumps(state), encoding="utf-8")
    seal = run / "artifacts/evidence_sha256.txt"
    payload = [path for path in run.rglob("*") if path.is_file() and path != seal]
    seal.write_text(
        "".join(
            f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.relative_to(tmp_path)}\n"
            for path in payload
        ),
        encoding="utf-8",
    )
    result = verify_failed_component_run_seal(tmp_path, run, "FAIL_COMPONENT")
    assert result["component_source_only"] is True
    assert result["run_state"] == "FAILED"
    with pytest.raises(RuntimeError, match="leakage-free PASS"):
        verify_complete_run_seal(tmp_path, run, "FAIL_COMPONENT")


def test_offline_evaluator_itself_rejects_forbidden_dataset_source(tmp_path: Path, monkeypatch) -> None:
    evaluator = _load_evaluator()
    monkeypatch.setattr(evaluator, "PROJECT_ROOT", tmp_path)
    training = _write_run(tmp_path, "training", evaluator.TRAINING_STATUS)
    perception = _write_run(tmp_path, "perception", evaluator.PERCEPTION_STATUS)
    dataset = _write_run(
        tmp_path,
        "dataset",
        "PASS_GSE_DEDUPLICATED_DATASET_EXPORT_V1",
        strict_test_worlds_read=1,
    )
    teacher = _write_run(tmp_path, "teacher", "PASS_GSE_TEACHER_MANIFEST_V1")

    with pytest.raises(RuntimeError, match="dataset"):
        evaluator.evaluate(
            training,
            perception,
            dataset,
            teacher,
            tmp_path / "results/output",
        )
