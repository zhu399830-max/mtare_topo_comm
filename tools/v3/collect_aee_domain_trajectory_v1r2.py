#!/usr/bin/env python3
"""V1 collector plus an explicit successful host-ownership handoff."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import collect_aee_domain_trajectory_v1 as implementation


def handoff_tree(output: Path, host_uid: int, host_gid: int) -> dict[str, int | str]:
    root = output.resolve()
    try:
        root.relative_to(Path("/evidence/trajectories"))
    except ValueError as exc:
        raise RuntimeError("ownership handoff target escapes /evidence/trajectories") from exc
    if host_uid <= 0 or host_gid <= 0:
        raise ValueError("formal host UID/GID must be positive non-root identities")
    paths = [root, *sorted(root.rglob("*"))]
    if any(path.is_symlink() for path in paths):
        raise RuntimeError("ownership handoff rejects symbolic links")
    handoff = {
        "schema_version": "aee_domain_host_ownership_handoff_v1",
        "status": "PASS_AEE_DOMAIN_HOST_OWNERSHIP_HANDOFF_V1",
        "host_uid": host_uid,
        "host_gid": host_gid,
        "entries": len(paths),
    }
    implementation.write_json(root / "host_ownership_handoff.json", handoff)
    paths = [root, *sorted(root.rglob("*"))]
    for path in reversed(paths):
        os.chown(path, host_uid, host_gid, follow_symlinks=False)
        os.chmod(path, 0o775 if path.is_dir() else 0o664)
    return handoff


def main() -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--host-uid", required=True, type=int)
    parser.add_argument("--host-gid", required=True, type=int)
    handoff_args, remaining = parser.parse_known_args()
    if "--output-dir" not in remaining:
        raise ValueError("base collector output directory is required")
    output = Path(remaining[remaining.index("--output-dir") + 1])
    sys.argv = [sys.argv[0], *remaining]
    result = implementation.main()
    if result != 0:
        return result
    handoff_tree(output, handoff_args.host_uid, handoff_args.host_gid)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
