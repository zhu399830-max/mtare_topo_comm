#!/usr/bin/env python3
"""Immutable V1R entry point for the float32-aware spatial graph audit."""

from __future__ import annotations

import run_gse_spatial_trace_commit_requalification_v1 as implementation


implementation.RUN_ID = "gate3_20260828_gse_spatial_trace_commit_requalification_v1r_seed0"
implementation.CARD_STATUS = "APPROVED_FOR_ONE_IMMUTABLE_GSE_SPATIAL_TRACE_COMMIT_REQUALIFICATION_V1R"
implementation.PASS = "PASS_GSE_SPATIAL_TRACE_COMMIT_REQUALIFICATION_V1R"
implementation.FAIL = "FAIL_GSE_SPATIAL_TRACE_COMMIT_REQUALIFICATION_V1R"


if __name__ == "__main__":
    raise SystemExit(implementation.main())
