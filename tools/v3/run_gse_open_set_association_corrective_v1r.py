#!/usr/bin/env python3
"""Recovery runner for the sparse-global-ID mapping correction only."""

from __future__ import annotations

import run_gse_open_set_association_corrective_v1 as base


base.RUN_ID = "gate3_20260826_gse_open_set_association_corrective_v1r_seed0"
base.PASS_STATUS = "PASS_GSE_OPEN_SET_ASSOCIATION_CORRECTIVE_V1R"
base.FAIL_STATUS = "FAIL_GSE_OPEN_SET_ASSOCIATION_CORRECTIVE_V1R"
base.DATA_CARD_STATUS = "APPROVED_FOR_ONE_IMMUTABLE_GSE_OPEN_SET_ASSOCIATION_CORRECTIVE_V1R"


if __name__ == "__main__":
    raise SystemExit(base.main())
