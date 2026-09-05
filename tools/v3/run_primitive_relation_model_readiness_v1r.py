#!/usr/bin/env python3
"""Execute the slot-canonicalization-only V1R readiness corrective."""

from __future__ import annotations

import run_primitive_relation_model_readiness_v1 as base


base.RUN_ID = "gate3_20260830_primitive_relation_model_readiness_v1r_seed0"
base.PASS = "PASS_PRIMITIVE_RELATION_MODEL_READINESS_V1R"
base.FAIL = "FAIL_PRIMITIVE_RELATION_MODEL_READINESS_V1R"
base.EXPECTED_TESTS = 30
base.CARD_STATUS = "APPROVED_FOR_ONE_IMMUTABLE_PRIMITIVE_RELATION_MODEL_READINESS_V1R"
base.SCHEMA_VERSION = "primitive_relation_model_readiness_v1r"


if __name__ == "__main__":
    raise SystemExit(base.main())
