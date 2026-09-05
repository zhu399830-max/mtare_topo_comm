#!/usr/bin/env python3
"""Loader-only replacement for the sealed C09 corrective evidence audit V1."""
from __future__ import annotations

import json
from pathlib import Path

import execute_cano_c09_geometry_evidence_corrective_audit_v1 as core


def load_json_any_root(path: Path) -> object:
    """Read a frozen JSON document while preserving its declared root type."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main() -> int:
    core.load_json = load_json_any_root
    core.PASS_STATUS = "PASS_CANO_C09_GEOMETRY_EVIDENCE_CORRECTIVE_AUDIT_V1R"
    return core.main()


if __name__ == "__main__":
    raise SystemExit(main())
