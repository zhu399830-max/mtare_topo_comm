#!/usr/bin/env python3
"""Serialization/figure-only corrective wrapper for failure attribution V1."""

from __future__ import annotations

import run_gse_spatial_event_set_failure_attribution_v1 as base


base.RUN_ID = "gate3_20260828_gse_spatial_event_set_failure_attribution_v1r_seed0"
base.CARD_STATUS = "APPROVED_FOR_ONE_IMMUTABLE_GSE_SPATIAL_EVENT_SET_FAILURE_ATTRIBUTION_V1R"


if __name__ == "__main__":
    raise SystemExit(base.main())
