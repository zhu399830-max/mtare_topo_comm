from __future__ import annotations

import json
from pathlib import Path

import pytest

from mtare_topo.evaluation import host_bag_archive as archive


def _handoff(case: Path, uid: int, gid: int) -> None:
    (case / "host_ownership_handoff.json").write_text(json.dumps({
        "status": "PASS_MTARE_CASE_HOST_OWNERSHIP_HANDOFF_V1",
        "host_uid": uid,
        "host_gid": gid,
    }))


def test_host_archive_is_non_overwriting_and_removes_raw_only_after_verify(tmp_path: Path, monkeypatch) -> None:
    case = tmp_path / "case"
    case.mkdir()
    raw = case / "raw.bag"
    raw.write_bytes(b"raw")
    digest = archive.sha256(raw)
    uid, gid = case.stat().st_uid, case.stat().st_gid
    _handoff(case, uid, gid)
    monkeypatch.setattr(archive.os, "getuid", lambda: uid)
    monkeypatch.setattr(archive.os, "getgid", lambda: gid)
    def compress(command, check):
        assert "-f" not in command and check is False
        Path(command[-1]).write_bytes(b"archive")
        return type("Completed", (), {"returncode": 0})()
    monkeypatch.setattr(archive.subprocess, "run", compress)
    monkeypatch.setattr(archive, "verify_zstd_archive", lambda path, expected: expected)
    result = archive.finalize_handed_off_bag(case, expected_uid=uid, expected_gid=gid, expected_raw_sha256=digest)
    assert result["overwrite_permitted"] is False
    assert not raw.exists() and (case / "raw.bag.zst").is_file() and (case / "storage.json").is_file()


def test_host_archive_rejects_bad_handoff_hash_and_existing_output(tmp_path: Path, monkeypatch) -> None:
    case = tmp_path / "case"
    case.mkdir()
    raw = case / "raw.bag"
    raw.write_bytes(b"raw")
    uid, gid = case.stat().st_uid, case.stat().st_gid
    monkeypatch.setattr(archive.os, "getuid", lambda: uid)
    monkeypatch.setattr(archive.os, "getgid", lambda: gid)
    _handoff(case, uid + 1, gid)
    with pytest.raises(RuntimeError, match="handoff evidence"):
        archive.finalize_handed_off_bag(case, expected_uid=uid, expected_gid=gid, expected_raw_sha256=archive.sha256(raw))
    _handoff(case, uid, gid)
    with pytest.raises(RuntimeError, match="identity"):
        archive.finalize_handed_off_bag(case, expected_uid=uid, expected_gid=gid, expected_raw_sha256="0" * 64)
    (case / "raw.bag.zst").write_bytes(b"occupied")
    with pytest.raises(RuntimeError, match="overwrite forbidden"):
        archive.finalize_handed_off_bag(case, expected_uid=uid, expected_gid=gid, expected_raw_sha256=archive.sha256(raw))
