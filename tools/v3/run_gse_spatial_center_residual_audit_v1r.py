#!/usr/bin/env python3
"""Corrective wrapper for the pair-capable identity count semantics."""

from __future__ import annotations

import run_gse_spatial_center_residual_audit_v1 as base


base.RUN_ID = "gate3_20260828_gse_spatial_center_residual_audit_v1r_seed0"
base.PASS = "PASS_GSE_SPATIAL_CENTER_RESIDUAL_AUDIT_V1R"
base.FAIL = "FAIL_GSE_SPATIAL_CENTER_RESIDUAL_AUDIT_V1R"


if __name__ == "__main__":
    raise SystemExit(base.main())
