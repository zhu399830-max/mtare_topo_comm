"""CPU-only contracts for the AEE V1R2 ownership handoff replacement."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from mtare_topo.data.aee_domain_adaptation import enumerate_aee_domain_trajectories


ROOT = Path(__file__).resolve().parents[3]


def _load(name: str, path: Path):
    import sys

    tools = str(ROOT / "tools/v3")
    if tools not in sys.path:
        sys.path.insert(0, tools)
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


COLLECTOR = _load(
    "collect_aee_domain_trajectory_v1r2",
    ROOT / "tools/v3/collect_aee_domain_trajectory_v1r2.py",
)
RUNNER = _load(
    "run_aee_domain_sensor_export_v1r2",
    ROOT / "tools/v3/run_aee_domain_sensor_export_v1r2.py",
)


def _allow_evidence_root(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    evidence = tmp_path / "evidence" / "trajectories"
    evidence.mkdir(parents=True)
    real_path = Path

    def path_factory(value):
        if str(value) == "/evidence/trajectories":
            return evidence
        return real_path(value)

    monkeypatch.setattr(COLLECTOR, "Path", path_factory)
    monkeypatch.setattr(COLLECTOR.os, "chown", lambda *args, **kwargs: None)
    monkeypatch.setattr(COLLECTOR.os, "chmod", lambda *args, **kwargs: None)
    return evidence


def test_handoff_tree_accepts_only_evidence_tree_and_records_ownership(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    evidence = _allow_evidence_root(monkeypatch, tmp_path)
    output = evidence / "00_tunnel_seed11"
    output.mkdir()
    (output / "raw.bag").write_bytes(b"raw")
    calls = []
    monkeypatch.setattr(COLLECTOR.os, "chown", lambda path, uid, gid, **kwargs: calls.append((path, uid, gid)))
    result = COLLECTOR.handoff_tree(output, 1000, 1000)
    assert result["status"] == "PASS_AEE_DOMAIN_HOST_OWNERSHIP_HANDOFF_V1"
    assert result["host_uid"] == result["host_gid"] == 1000
    record = json.loads((output / "host_ownership_handoff.json").read_text())
    assert record == result
    assert len(calls) >= 3


def test_handoff_tree_rejects_escape_symlink_and_root_identity(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    evidence = _allow_evidence_root(monkeypatch, tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    with pytest.raises(RuntimeError, match="escapes"):
        COLLECTOR.handoff_tree(outside, 1000, 1000)
    output = evidence / "with-link"
    output.mkdir()
    (output / "link").symlink_to(outside, target_is_directory=True)
    with pytest.raises(RuntimeError, match="symbolic"):
        COLLECTOR.handoff_tree(output, 1000, 1000)
    clean = evidence / "root-id"
    clean.mkdir()
    with pytest.raises(ValueError, match="positive non-root"):
        COLLECTOR.handoff_tree(clean, 0, 1000)


def test_v1r2_case_command_only_replaces_collector_and_adds_fixed_uid_gid(tmp_path: Path) -> None:
    trajectory = enumerate_aee_domain_trajectories()[0]
    base, base_name = RUNNER._base_case_command(tmp_path, trajectory)
    replacement, replacement_name = RUNNER.case_command(tmp_path, trajectory)
    old = "/workspace/tools/v3/collect_aee_domain_trajectory_v1.py"
    new = "/workspace/tools/v3/collect_aee_domain_trajectory_v1r2.py"
    assert base_name == replacement_name
    assert old in base[-1] and old not in replacement[-1]
    assert replacement[-1].count(new) == 1
    assert replacement[-1].endswith("--host-uid 1000 --host-gid 1000")
    for token in ("00_tunnel_seed11", "--world tunnel", "--split train", "--environment-seed 11", "--output-dir"):
        assert token in replacement[-1]
    assert replacement[:-1] == base[:-1]
    assert "3000" not in replacement[-1]  # frame contract remains in frozen collector, not command mutation


def _handoff(case: Path, uid: int = 1000, gid: int = 1000) -> None:
    (case / "host_ownership_handoff.json").write_text(
        json.dumps({
            "schema_version": "aee_domain_host_ownership_handoff_v1",
            "status": "PASS_AEE_DOMAIN_HOST_OWNERSHIP_HANDOFF_V1",
            "host_uid": uid,
            "host_gid": gid,
            "entries": 2,
        })
    )


def test_v1r2_host_archive_rejects_missing_bad_handoff_and_owner(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    case = tmp_path / "case"
    case.mkdir()
    monkeypatch.setattr(RUNNER.os, "getuid", lambda: 1000)
    monkeypatch.setattr(RUNNER.os, "getgid", lambda: 1000)
    with pytest.raises((FileNotFoundError, RuntimeError)):
        RUNNER.host_archive(case, "raw-hash")
    _handoff(case, uid=999)
    with pytest.raises(RuntimeError, match="handoff evidence"):
        RUNNER.host_archive(case, "raw-hash")
    _handoff(case)
    (case / "raw.bag").write_bytes(b"raw")
    monkeypatch.setattr(RUNNER, "EXPECTED_HOST_UID", 999)
    monkeypatch.setattr(RUNNER, "EXPECTED_HOST_GID", 999)
    monkeypatch.setattr(RUNNER.os, "getuid", lambda: 999)
    monkeypatch.setattr(RUNNER.os, "getgid", lambda: 999)
    _handoff(case, uid=999, gid=999)
    with pytest.raises(RuntimeError, match="ownership handoff drift"):
        RUNNER.host_archive(case, "raw-hash")


def test_v1r2_host_archive_is_verified_and_non_overwriting(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    case = tmp_path / "case"
    case.mkdir()
    (case / "raw.bag").write_bytes(b"raw")
    _handoff(case)
    monkeypatch.setattr(RUNNER.os, "getuid", lambda: 1000)
    monkeypatch.setattr(RUNNER.os, "getgid", lambda: 1000)
    digest = RUNNER.implementation.sha256(case / "raw.bag")

    def compress(command, check):
        assert check is False
        assert "-f" not in command
        assert command[-2] == "-o"
        Path(command[-1]).write_bytes(b"compressed")
        return type("Completed", (), {"returncode": 0})()

    monkeypatch.setattr(RUNNER.subprocess, "run", compress)
    monkeypatch.setattr(RUNNER.implementation, "verify_zstd_archive", lambda archive, expected: expected)
    monkeypatch.setattr(RUNNER.implementation, "sha256", lambda path: digest if path.name == "raw.bag" else "artifact-hash")
    result = RUNNER.host_archive(case, digest)
    assert result["archive_sha256"] == "artifact-hash"
    assert result["overwrite_permitted"] is False
    assert result["ownership_handoff"]["status"] == "PASS_AEE_DOMAIN_HOST_OWNERSHIP_HANDOFF_V1"
    assert not (case / "raw.bag").exists()
    assert (case / "storage.json").is_file()


def test_v1r2_host_archive_rejects_preexisting_outputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    case = tmp_path / "case"
    case.mkdir()
    raw = case / "raw.bag"
    raw.write_bytes(b"raw")
    digest = RUNNER.implementation.sha256(raw)
    _handoff(case)
    monkeypatch.setattr(RUNNER.os, "getuid", lambda: 1000)
    monkeypatch.setattr(RUNNER.os, "getgid", lambda: 1000)
    (case / "raw.bag.zst").write_bytes(b"existing")
    with pytest.raises(RuntimeError, match="overwrite forbidden"):
        RUNNER.host_archive(case, digest)


def test_v1r2_main_only_sets_run_id_and_two_hooks(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []
    monkeypatch.setattr(RUNNER.environment, "verify_host_python", lambda: calls.append("verify"))
    monkeypatch.setattr(RUNNER.implementation, "main", lambda: calls.append("base-main") or 17)
    old_case = RUNNER.implementation.case_command
    old_archive = RUNNER.implementation.host_archive
    result = RUNNER.main()
    assert result == 17
    assert calls == ["verify", "base-main"]
    assert RUNNER.implementation.RUN_ID == RUNNER.RUN_ID
    assert RUNNER.implementation.case_command is RUNNER.case_command
    assert RUNNER.implementation.host_archive is RUNNER.host_archive
    assert old_case is not RUNNER.implementation.case_command
    assert old_archive is not RUNNER.implementation.host_archive
