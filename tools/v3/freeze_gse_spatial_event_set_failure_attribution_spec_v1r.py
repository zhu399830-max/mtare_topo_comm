#!/usr/bin/env python3
"""Freeze the serialization/figure-only V1R failure-attribution corrective."""

from __future__ import annotations

import freeze_gse_spatial_event_set_failure_attribution_spec_v1 as base


base.RUN_ID = "gate3_20260828_gse_spatial_event_set_failure_attribution_v1r_seed0"
base.CARD = base.PROJECT_ROOT / "configs/v3/gate3/data_cards/gse_spatial_event_set_failure_attribution_v1r.json"
base.SPEC = base.PROJECT_ROOT / "configs/v3/gate3/gse_spatial_event_set_failure_attribution_v1r.json"
base.CARD_ID = "gse_spatial_event_set_failure_attribution_v1r"
base.CARD_STATUS = "APPROVED_FOR_ONE_IMMUTABLE_GSE_SPATIAL_EVENT_SET_FAILURE_ATTRIBUTION_V1R"
base.SLUG = "gse_spatial_event_set_failure_attribution_v1r"
base.FREEZER_TOOL = "tools/v3/freeze_gse_spatial_event_set_failure_attribution_spec_v1r.py"
base.RUNNER_TOOL = "tools/v3/run_gse_spatial_event_set_failure_attribution_v1r.py"


if __name__ == "__main__":
    raise SystemExit(base.main())
