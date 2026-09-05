#!/usr/bin/env python3
"""V1R wrapper: governance-field-only replacement."""

import run_gse_objective_spatial_graph_rescore_audit_v1 as implementation


implementation.RUN_ID = "gate3_20260828_gse_objective_spatial_graph_rescore_audit_v1r_seed0"
implementation.CARD_STATUS = "APPROVED_FOR_ONE_IMMUTABLE_GSE_OBJECTIVE_SPATIAL_GRAPH_RESCORE_AUDIT_V1R"
implementation.PASS = "PASS_GSE_OBJECTIVE_SPATIAL_GRAPH_RESCORE_AUDIT_V1R"
implementation.FAIL = "FAIL_GSE_OBJECTIVE_SPATIAL_GRAPH_RESCORE_AUDIT_V1R"


if __name__ == "__main__":
    raise SystemExit(implementation.main())
