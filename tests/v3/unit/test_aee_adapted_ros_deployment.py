"""CPU-only contracts for AEE-adapted ROS checkpoint deployment."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import torch

from mtare_topo.deployment.m1d_checkpoint import export_m1d_ros_checkpoint
from mtare_topo.representation.phase3_structural_semantics import StructuralSemanticNet


ROOT = Path(__file__).resolve().parents[3]


def _load():
    tools = str(ROOT / "tools/v3")
    if tools not in sys.path:
        sys.path.insert(0, tools)
    spec = importlib.util.spec_from_file_location(
        "run_aee_adapted_ros_deployment_export_v1",
        ROOT / "tools/v3/run_aee_adapted_ros_deployment_export_v1.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_adapted_export_preserves_mode_seed_and_tensors(tmp_path: Path) -> None:
    runner = _load()
    source, destination = tmp_path / "source.pt", tmp_path / "deployment.pt"
    state = StructuralSemanticNet().state_dict()
    torch.save({"mode": runner.SOURCE_MODE, "seed": 2, "epoch": 7, "model": state}, source)
    result = export_m1d_ros_checkpoint(
        source=source,
        destination=destination,
        expected_source_sha256=runner.sha256(source),
        expected_seed=2,
        expected_mode=runner.SOURCE_MODE,
    )
    deployed = torch.load(destination, map_location="cpu", weights_only=False)
    assert result["mode"] == deployed["mode"] == runner.SOURCE_MODE
    assert deployed["seed"] == 2
    assert all(torch.equal(state[name], deployed["model"][name]) for name in state)


def test_adaptation_source_requires_three_qualified_seed_checkpoints(tmp_path: Path, monkeypatch) -> None:
    runner = _load()
    project = tmp_path / "project"
    run = project / "results/source"
    (run / "artifacts/models").mkdir(parents=True)
    (run / "metrics").mkdir()
    (run / "RUN_STATE.json").write_text(json.dumps({"state": "COMPLETED", "overall_status": runner.STATUS_PASS.replace("ROS_DEPLOYMENT_EXPORT", "HEAD_ADAPTATION")}))
    # Use the exact source status required by the validator.
    (run / "RUN_STATE.json").write_text(json.dumps({"state": "COMPLETED", "overall_status": "PASS_AEE_HEAD_ADAPTATION_V1"}))
    (run / "metrics/summary.json").write_text(json.dumps({"overall_status": "PASS_AEE_HEAD_ADAPTATION_V1", "completed_seeds": 3, "c10_frames_read": 0, "later_sealed_world_frames_read": 0}))
    for seed in (0, 1, 2):
        child = run / f"artifacts/models/m1d_seed{seed}"
        child.mkdir()
        checkpoint = child / "best.pt"
        checkpoint.write_bytes(f"seed{seed}".encode())
        digest = runner.sha256(checkpoint)
        (child / "summary.json").write_text(json.dumps({"status": "PASS_AEE_HEAD_ADAPTATION_SEED_V1", "seed": seed, "best_checkpoint_sha256": digest}))
    monkeypatch.setattr(runner, "PROJECT_ROOT", project)
    seal_path = run / "artifacts/evidence_sha256.txt"
    files = sorted(path for path in run.rglob("*") if path.is_file() and path != seal_path)
    seal_path.write_text("".join(f"{runner.sha256(path)}  {path.relative_to(project)}\n" for path in files))
    records = runner.validate_adaptation_source(run, runner.sha256(seal_path))
    assert [record["seed"] for record in records] == [0, 1, 2]
    bad = json.loads((run / "metrics/summary.json").read_text())
    bad["c10_frames_read"] = 1
    (run / "metrics/summary.json").write_text(json.dumps(bad))
    try:
        runner.validate_adaptation_source(run, runner.sha256(seal_path))
    except RuntimeError as exc:
        assert "forbidden" in str(exc)
    else:
        raise AssertionError("C10 read must be rejected")
