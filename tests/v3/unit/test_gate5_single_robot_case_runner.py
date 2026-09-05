from __future__ import annotations

import argparse
import hashlib
import importlib.util
from pathlib import Path
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[3]


def load_runner():
    tools = ROOT / "tools/v3"
    import sys
    sys.path.insert(0, str(tools))
    spec = importlib.util.spec_from_file_location("run_mtare_single_robot_case_v1", tools / "run_mtare_single_robot_case_v1.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def args(**values):
    defaults = dict(
        method_family="original_mtare", world="tunnel", environment_seed=23,
        checkpoint=None, checkpoint_sha256=None, complete_map=None, complete_map_sha256=None,
    )
    defaults.update(values)
    return argparse.Namespace(**defaults)


def test_original_command_separates_seed_from_complete_test_id() -> None:
    runner = load_runner()
    command = runner.method_command(args(), Path("/evidence/planner"))
    assert "planner_seed:=23" in command
    assert "test_id:=0001" in command
    assert "test_id:=23" not in command
    assert "robot_num:=1" in command


def test_m1d_and_oracle_commands_bind_exact_inputs() -> None:
    runner = load_runner()
    m1d = runner.method_command(
        args(method_family="m1d_topology", checkpoint="models/m.pt", checkpoint_sha256="a" * 64),
        Path("/evidence/planner"),
    )
    assert "semantic_topology_global_node_v3.py" in m1d
    assert "--checkpoint /workspace/models/m.pt" in m1d
    assert "--checkpoint-sha256 " + "a" * 64 in m1d
    oracle = runner.method_command(
        args(method_family="layered_gt_map_oracle", complete_map="/map.ply", complete_map_sha256="b" * 64),
        Path("/evidence/planner"),
    )
    assert "layered_gt_map_global_node_v1.py" in oracle
    assert "--complete-map /map.ply" in oracle
    assert "--complete-map-sha256 " + "b" * 64 in oracle


def test_case_environment_preserves_both_catkin_workspaces() -> None:
    runner = load_runner()
    assert "autonomous_exploration_development_environment/devel/setup.bash" in runner.SOURCE_ENV
    assert "tare_system/devel/setup.bash --extend" in runner.SOURCE_ENV


def test_lossless_zstd_verifier_checks_decompressed_sha(tmp_path: Path) -> None:
    runner = load_runner()
    source = tmp_path / "raw.bag"
    source.write_bytes((b"deterministic-bag-content" * 10000) + bytes(range(256)))
    archive = tmp_path / "raw.bag.zst"
    subprocess.run(["zstd", "-10", "-q", str(source), "-o", str(archive)], check=True)
    expected = hashlib.sha256(source.read_bytes()).hexdigest()
    assert runner.verify_zstd_archive(archive, expected) == expected
    with pytest.raises(RuntimeError):
        runner.verify_zstd_archive(archive, "0" * 64)


def test_case_runner_has_no_retry_path_and_deletes_only_after_verification() -> None:
    source = (ROOT / "tools/v3/run_mtare_single_robot_case_v1.py").read_text(encoding="utf-8")
    assert "retry" not in source.lower()
    assert source.index("verified = verify_zstd_archive") < source.index("bag_path.unlink()")
    assert source.index("write_json(case_dir / \"storage.json\"") < source.index("bag_path.unlink()")


def test_coverage_type_alias_is_python38_compatible() -> None:
    source = (ROOT / "src/mtare_topo/evaluation/mtare_coverage.py").read_text(encoding="utf-8")
    assert "Voxel = Tuple[int, int, int]" in source
    assert "Voxel = tuple[int, int, int]" not in source
