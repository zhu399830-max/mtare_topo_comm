"""Non-overwriting host finalizer for rootless handoff of ROS bag evidence."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_zstd_archive(archive: Path, expected_sha256: str) -> str:
    digest = hashlib.sha256()
    process = subprocess.Popen(["zstd", "-dc", str(archive)], stdout=subprocess.PIPE)
    assert process.stdout is not None
    for block in iter(lambda: process.stdout.read(1024 * 1024), b""):
        digest.update(block)
    return_code = process.wait()
    actual = digest.hexdigest()
    if return_code != 0 or actual != expected_sha256:
        raise RuntimeError("host bag archive decompression identity failed")
    return actual


def finalize_handed_off_bag(
    case_dir: Path,
    *,
    expected_uid: int,
    expected_gid: int,
    expected_raw_sha256: str,
) -> dict[str, Any]:
    case = case_dir.resolve()
    if os.getuid() != expected_uid or os.getgid() != expected_gid:
        raise RuntimeError("formal host archive process identity drift")
    handoff_path = case / "host_ownership_handoff.json"
    handoff = json.loads(handoff_path.read_text(encoding="utf-8"))
    if (
        handoff.get("status") != "PASS_MTARE_CASE_HOST_OWNERSHIP_HANDOFF_V1"
        or handoff.get("host_uid") != expected_uid
        or handoff.get("host_gid") != expected_gid
    ):
        raise RuntimeError("case ownership handoff evidence failed")
    raw = case / "raw.bag"
    for path in (case, raw):
        stat = path.stat()
        if stat.st_uid != expected_uid or stat.st_gid != expected_gid:
            raise RuntimeError(f"case ownership handoff drift: {path.name}")
    if not raw.is_file() or sha256(raw) != expected_raw_sha256:
        raise RuntimeError("raw case bag identity missing or drifted")
    archive = case / "raw.bag.zst"
    storage_path = case / "storage.json"
    if archive.exists() or storage_path.exists():
        raise RuntimeError("host case archive output exists; overwrite forbidden")
    completed = subprocess.run(
        ["zstd", "-10", "-T0", "-q", str(raw), "-o", str(archive)], check=False
    )
    if completed.returncode != 0 or not archive.is_file():
        raise RuntimeError("host case zstd compression failed")
    verified = verify_zstd_archive(archive, expected_raw_sha256)
    storage = {
        "schema_version": "mtare_lossless_host_bag_archive_v2",
        "host_zstd_sha256": sha256(Path("/usr/bin/zstd")),
        "zstd_level": 10,
        "overwrite_permitted": False,
        "original_bag_sha256": expected_raw_sha256,
        "original_bag_bytes": raw.stat().st_size,
        "archive_sha256": sha256(archive),
        "archive_bytes": archive.stat().st_size,
        "decompressed_sha256": verified,
        "ownership_handoff": handoff,
    }
    with storage_path.open("x", encoding="utf-8") as stream:
        json.dump(storage, stream, indent=2, sort_keys=True)
        stream.write("\n")
    raw.unlink()
    return storage


__all__ = ["finalize_handed_off_bag", "verify_zstd_archive"]
