#!/usr/bin/env python3
"""One-shot V1R runner preserving sparse complete-export global identities."""

from __future__ import annotations

from _bootstrap import PROJECT_ROOT
import run_gse_corrected_causal_teacher_manifest_v1 as base


base.RUN_ID = "gate2_20260827_gse_corrected_causal_teacher_manifest_v1r_seed0"
base.PASS_STATUS = "PASS_GSE_CORRECTED_CAUSAL_TEACHER_MANIFEST_V1R"
base.FAIL_STATUS = "FAIL_GSE_CORRECTED_CAUSAL_TEACHER_MANIFEST_V1R"
base.EXECUTOR = PROJECT_ROOT / "tools/v3/execute_gse_corrected_causal_teacher_manifest_v1r.py"


if __name__ == "__main__":
    raise SystemExit(base.main())
