#!/usr/bin/env python3
"""Freeze V1R after the exact V1 target-slot tie-breaking failure."""

from __future__ import annotations

import freeze_primitive_relation_model_readiness_v1_spec as base


base.CARD = base.PROJECT_ROOT / "configs/v3/gate3/data_cards/primitive_relation_model_readiness_v1r.json"
base.SPEC = base.PROJECT_ROOT / "configs/v3/gate3/primitive_relation_model_readiness_v1r.json"
base.RUN_ID = "gate3_20260830_primitive_relation_model_readiness_v1r_seed0"
base.CARD_ID = "primitive_relation_model_readiness_v1r"
base.SLUG = "primitive_relation_model_readiness_v1r"
base.RUNNER = "tools/v3/run_primitive_relation_model_readiness_v1r.py"
base.EXPECTED_TESTS = 30
base.CARD_STATUS = "APPROVED_FOR_ONE_IMMUTABLE_PRIMITIVE_RELATION_MODEL_READINESS_V1R"
base.CORRECTIVE_OF = base.PROJECT_ROOT / "results/gate3_semantics/gate3_20260830_primitive_relation_model_readiness_v1_seed0"


if __name__ == "__main__":
    base.main()
