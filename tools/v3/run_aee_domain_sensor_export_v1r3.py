#!/home/zeng-workstation/anaconda3/bin/python
"""Metadata-only run-identity replacement for the sealed V1R2 failure."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import run_aee_domain_sensor_export_v1r as environment
import run_aee_domain_sensor_export_v1r2 as permission_replacement


RUN_ID = "gate2_20260820_aee_domain_sensor_export_v1r3_seed20260820"
implementation = environment.implementation


def validate_authorization_timestamp(spec_path: Path, started_at: datetime | None = None) -> datetime:
    payload = json.loads(spec_path.read_text(encoding="utf-8"))
    authorization = payload.get("user_authorization", {})
    if authorization.get("status") != "APPROVED":
        raise RuntimeError("explicit V1R3 approval required")
    try:
        approved_at = datetime.fromisoformat(authorization["approved_at"])
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError("valid ISO-8601 V1R3 approval timestamp required") from exc
    if approved_at.tzinfo is None or approved_at.utcoffset() is None:
        raise RuntimeError("V1R3 approval timestamp must include a timezone")
    observed_start = started_at or datetime.now(timezone.utc)
    if observed_start.tzinfo is None or observed_start.utcoffset() is None:
        raise ValueError("observed start time must include a timezone")
    if approved_at > observed_start:
        raise RuntimeError("V1R3 approval timestamp is later than run start")
    return approved_at


def main() -> int:
    if "--spec" not in sys.argv:
        raise ValueError("formal V1R3 spec path is required")
    validate_authorization_timestamp(Path(sys.argv[sys.argv.index("--spec") + 1]).resolve())
    environment.verify_host_python()
    implementation.RUN_ID = RUN_ID
    implementation.case_command = permission_replacement.case_command
    implementation.host_archive = permission_replacement.host_archive
    return implementation.main()


if __name__ == "__main__":
    raise SystemExit(main())
