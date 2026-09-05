from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]


def _load():
    tools = str(ROOT / "tools/v3")
    if tools not in sys.path:
        sys.path.insert(0, tools)
    spec = importlib.util.spec_from_file_location(
        "run_aee_composite_v9_combined_correction_stochastic_v1r",
        ROOT / "tools/v3/run_aee_composite_v9_combined_correction_stochastic_v1r.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_v1r_binds_zero_case_interface_failure(tmp_path: Path):
    module = _load()
    source = tmp_path / "failed"
    (source / "metrics").mkdir(parents=True)
    (source / "artifacts").mkdir()
    files = [source / "RUN_STATE.json", source / "metrics/summary.json"]
    files[0].write_text(json.dumps({"state": "FAILED", "overall_status": module.FAILED_STATUS}) + "\n")
    files[1].write_text(json.dumps({"overall_status": module.FAILED_STATUS, "completed_case_count": 0, "failure_reason": "'source_schedule_file_sha256'"}) + "\n")
    seal = source / "artifacts/evidence_sha256.txt"
    seal.write_text("".join(f"{module.base.base.matrix_base.sha256(p)}  {p.relative_to(tmp_path).as_posix()}\n" for p in files))
    spec = {"failed_stochastic_run": "failed", "failed_stochastic_seal_sha256": module.base.base.matrix_base.sha256(seal)}
    evidence = module.verify_failed_source(spec, project_root=tmp_path)
    assert evidence["completed_cases"] == 0
    assert evidence["partial_reuse"] is False
