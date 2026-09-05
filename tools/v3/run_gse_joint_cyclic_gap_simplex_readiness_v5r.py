#!/usr/bin/env python3
"""V5R: execute V5 with the frozen deterministic cuBLAS workspace contract."""
from __future__ import annotations

import os

# Must be present before the CUDA subprocess starts.  This is the same
# deterministic workspace contract used by the formal training sidecar.
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import run_gse_joint_cyclic_gap_simplex_readiness_v5 as base


base.PASS = "PASS_GSE_JOINT_CYCLIC_GAP_SIMPLEX_READINESS_V5R"
base.FAIL = "FAIL_GSE_JOINT_CYCLIC_GAP_SIMPLEX_READINESS_V5R"
base.CARD_STATUS = "APPROVED_FOR_ONE_IMMUTABLE_GSE_JOINT_CYCLIC_GAP_SIMPLEX_READINESS_V5R"


if __name__ == "__main__":
    raise SystemExit(base.main())
