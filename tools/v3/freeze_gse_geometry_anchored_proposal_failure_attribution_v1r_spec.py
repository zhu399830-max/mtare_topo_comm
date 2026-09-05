#!/usr/bin/env python3
"""Freeze the serialization-only V1R corrective with unchanged science."""
from __future__ import annotations

import freeze_gse_geometry_anchored_proposal_failure_attribution_v1_spec as base


base.RUN_ID = "gate3_20260828_gse_geometry_anchored_proposal_failure_attribution_v1r_seed0"
base.SPEC = base.PROJECT_ROOT / "configs/v3/gate3/gse_geometry_anchored_proposal_failure_attribution_v1r.json"
base.SLUG = "gse_geometry_anchored_proposal_failure_attribution_v1r"
base.EXECUTOR_TOOL = "tools/v3/execute_gse_geometry_anchored_proposal_failure_attribution_v1r.py"
base.RUNNER_TOOL = "tools/v3/run_gse_geometry_anchored_proposal_failure_attribution_v1r.py"
base.FREEZER_TOOL = "tools/v3/freeze_gse_geometry_anchored_proposal_failure_attribution_v1r_spec.py"


if __name__ == "__main__":
    raise SystemExit(base.main())
