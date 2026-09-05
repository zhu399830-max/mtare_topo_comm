#!/usr/bin/env python3
"""Immutable runner wrapper for proposal attribution V1R."""
from __future__ import annotations

from pathlib import Path

import run_gse_geometry_anchored_proposal_failure_attribution_v1 as base


base.RUN_ID = "gate3_20260828_gse_geometry_anchored_proposal_failure_attribution_v1r_seed0"
base.PASS = "PASS_GSE_GEOMETRY_ANCHORED_PROPOSAL_FAILURE_ATTRIBUTION_V1R"
base.FAIL = "FAIL_GSE_GEOMETRY_ANCHORED_PROPOSAL_FAILURE_ATTRIBUTION_V1R"
base.EXECUTOR = base.PROJECT_ROOT / "tools/v3/execute_gse_geometry_anchored_proposal_failure_attribution_v1r.py"


if __name__ == "__main__":
    raise SystemExit(base.main())
