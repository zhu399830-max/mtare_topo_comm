#!/usr/bin/env python3
"""Batch-rank corrective wrapper for sparse-port C07 failure attribution."""

import run_primitive_relation_sparse_port_failure_attribution_v1 as base


base.RUN_ID = "gate3_20260902_primitive_relation_sparse_port_failure_attribution_v1r_seed0"
base.CARD_STATUS = "APPROVED_FOR_ONE_IMMUTABLE_PRIMITIVE_RELATION_SPARSE_PORT_FAILURE_ATTRIBUTION_V1R"


if __name__ == "__main__":
    raise SystemExit(base.main())
