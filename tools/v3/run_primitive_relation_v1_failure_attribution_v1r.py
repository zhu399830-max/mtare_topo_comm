#!/usr/bin/env python3
"""Representation-neutral Data-Card corrective for V1 failure attribution."""

import run_primitive_relation_v1_failure_attribution as base


base.RUN_ID = "gate3_20260831_primitive_relation_v1_failure_attribution_v1r_seed0"
base.CARD_STATUS = "APPROVED_FOR_ONE_IMMUTABLE_PRIMITIVE_RELATION_V1_FAILURE_ATTRIBUTION_V1R"


if __name__ == "__main__":
    raise SystemExit(base.main())
