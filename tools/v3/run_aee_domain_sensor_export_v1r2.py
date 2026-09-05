#!/home/zeng-workstation/anaconda3/bin/python
"""Permission-handoff replacement for the sealed V1R archive failure."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Any

import run_aee_domain_sensor_export_v1r as environment


RUN_ID = "gate2_20260820_aee_domain_sensor_export_v1r2_seed20260820"
EXPECTED_HOST_UID = 1000
EXPECTED_HOST_GID = 1000
implementation = environment.implementation
_base_case_command = implementation.case_command


def case_command(run_dir: Path, trajectory: Any) -> tuple[list[str], str]:
    command, name = _base_case_command(run_dir, trajectory)
    old = "/workspace/tools/v3/collect_aee_domain_trajectory_v1.py"
    new = "/workspace/tools/v3/collect_aee_domain_trajectory_v1r2.py"
    if command[-1].count(old) != 1:
        raise RuntimeError("base collector command identity drift")
    command[-1] = (
        command[-1].replace(old, new)
        + f" --host-uid {EXPECTED_HOST_UID} --host-gid {EXPECTED_HOST_GID}"
    )
    return command, name


def host_archive(case_dir: Path, expected_sha256: str) -> dict[str, Any]:
    if os.getuid() != EXPECTED_HOST_UID or os.getgid() != EXPECTED_HOST_GID:
        raise RuntimeError("formal host UID/GID identity drift")
    handoff = implementation.load_json(case_dir / "host_ownership_handoff.json")
    if (
        handoff.get("status") != "PASS_AEE_DOMAIN_HOST_OWNERSHIP_HANDOFF_V1"
        or handoff.get("host_uid") != EXPECTED_HOST_UID
        or handoff.get("host_gid") != EXPECTED_HOST_GID
    ):
        raise RuntimeError("container-to-host ownership handoff evidence failed")
    raw = case_dir / "raw.bag"
    for path in (case_dir, raw):
        stat = path.stat()
        if stat.st_uid != EXPECTED_HOST_UID or stat.st_gid != EXPECTED_HOST_GID:
            raise RuntimeError(f"host ownership handoff drift: {path.name}")
    if not os.access(case_dir, os.W_OK):
        raise RuntimeError("host cannot write the handed-off trajectory directory")
    if not raw.is_file() or implementation.sha256(raw) != expected_sha256:
        raise RuntimeError("raw bag identity missing or drifted before host archive")
    archive = case_dir / "raw.bag.zst"
    storage_path = case_dir / "storage.json"
    if archive.exists() or storage_path.exists():
        raise RuntimeError("host archive output already exists; overwrite forbidden")
    completed = subprocess.run(
        ["zstd", "-10", "-T0", "-q", str(raw), "-o", str(archive)],
        check=False,
    )
    if completed.returncode != 0 or not archive.is_file():
        raise RuntimeError("host zstd compression failed")
    verified = implementation.verify_zstd_archive(archive, expected_sha256)
    storage = {
        "schema_version": "aee_domain_host_archive_v1r2",
        "host_zstd_sha256": implementation.sha256(Path("/usr/bin/zstd")),
        "zstd_level": 10,
        "overwrite_permitted": False,
        "original_bag_sha256": expected_sha256,
        "original_bag_bytes": raw.stat().st_size,
        "archive_sha256": implementation.sha256(archive),
        "archive_bytes": archive.stat().st_size,
        "decompressed_sha256": verified,
        "ownership_handoff": handoff,
    }
    implementation.write_json(storage_path, storage)
    raw.unlink()
    return storage


def main() -> int:
    environment.verify_host_python()
    implementation.RUN_ID = RUN_ID
    implementation.case_command = case_command
    implementation.host_archive = host_archive
    return implementation.main()


if __name__ == "__main__":
    raise SystemExit(main())
