from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
import subprocess

import pytest

from mtare_topo.data.aee_domain_adaptation import enumerate_aee_domain_trajectories


ROOT = Path(__file__).resolve().parents[3]


def load_runner():
    import sys

    tools = ROOT / "tools/v3"
    sys.path.insert(0, str(tools))
    spec = importlib.util.spec_from_file_location("run_aee_domain_sensor_export_v1", tools / "run_aee_domain_sensor_export_v1.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_all_case_commands_use_original_mtare_and_exact_identity() -> None:
    runner = load_runner()
    run_dir = ROOT / "results/gate2_representation/gate2_20260820_aee_domain_sensor_export_v1_seed20260820"
    for item in enumerate_aee_domain_trajectories():
        command, name = runner.case_command(run_dir, item)
        rendered = " ".join(command)
        assert name.endswith(f"{item.index:02d}")
        assert item.trajectory_id in rendered
        assert f"--environment-seed {item.environment_seed}" in rendered
        assert f"--world {item.world}" in rendered
        assert f"--split {item.split}" in rendered


def test_host_archive_is_lossless_and_removes_raw_only_after_verification(tmp_path: Path) -> None:
    runner = load_runner()
    raw = tmp_path / "raw.bag"
    raw.write_bytes((b"aee-domain-bag" * 50000) + bytes(range(256)))
    digest = hashlib.sha256(raw.read_bytes()).hexdigest()
    storage = runner.host_archive(tmp_path, digest)
    assert not raw.exists()
    assert (tmp_path / "raw.bag.zst").is_file()
    assert storage["original_bag_sha256"] == storage["decompressed_sha256"] == digest
    assert storage["host_zstd_sha256"] == hashlib.sha256(Path("/usr/bin/zstd").read_bytes()).hexdigest()


def test_host_archive_rejects_identity_drift_without_deletion(tmp_path: Path) -> None:
    runner = load_runner()
    raw = tmp_path / "raw.bag"
    raw.write_bytes(b"do-not-delete")
    with pytest.raises(RuntimeError, match="identity"):
        runner.host_archive(tmp_path, "0" * 64)
    assert raw.is_file()


def test_outer_runner_has_no_retry_and_gate2_data_export_only() -> None:
    source = (ROOT / "tools/v3/run_aee_domain_sensor_export_v1.py").read_text(encoding="utf-8")
    assert "retry" not in source.lower()
    assert 'spec.get("gate") != 2' in source
    assert 'spec.get("operation") != "data_export"' in source
    assert "teacher_queries\": 0" in source
    assert "raw_frames != 30000" in source and "effective_frames != 6000" in source


def test_teacher_generator_has_no_model_or_future_input() -> None:
    source = (ROOT / "tools/v3/generate_aee_objective_teacher_shard_v1.py").read_text(encoding="utf-8")
    assert "LayeredGTMapOracle" in source
    assert "sensor_xyz_m" in source and "yaw_deg" in source
    assert "torch" not in source
    assert "checkpoint" not in source
    assert "future trajectory" not in source.lower()
