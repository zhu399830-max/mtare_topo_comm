#!/usr/bin/env python3
"""Run and seal the loader-only C09 corrective evidence audit V1R."""
from __future__ import annotations

from _bootstrap import PROJECT_ROOT
import run_cano_c09_geometry_evidence_corrective_audit_v1 as core


def main() -> int:
    core.RUN_ID = "gate4_20260820_cano_c09_geometry_evidence_corrective_audit_v1r_seed0"
    core.EXECUTOR = PROJECT_ROOT / "tools/v3/execute_cano_c09_geometry_evidence_corrective_audit_v1r.py"
    core.PASS_STATUS = "PASS_CANO_C09_GEOMETRY_EVIDENCE_CORRECTIVE_AUDIT_V1R"
    return core.main()


if __name__ == "__main__":
    raise SystemExit(main())
