#!/usr/bin/env python3
"""Corrective V2 runner: identity-aligned C09 causal-node qualification."""

from __future__ import annotations

from _bootstrap import PROJECT_ROOT
from mtare_topo.evaluation.gse_evidence_integrity import verify_failed_component_run_seal
import run_gse_factorized_causal_node_c09_v1 as implementation


implementation.RUN_ID = "gate3_20260828_gse_factorized_causal_node_c09_v2_seed0"
implementation.CARD_STATUS = "APPROVED_FOR_ONE_IMMUTABLE_GSE_FACTORIZED_CAUSAL_NODE_C09_V2"
_base_sources = implementation._sources
_failed_v1 = PROJECT_ROOT / "results/gate3_semantics/gate3_20260828_gse_factorized_causal_node_c09_v1_seed0"


def _sources_with_failed_v1():
    return {
        **_base_sources(),
        "failed_identity_alignment_v1": verify_failed_component_run_seal(
            PROJECT_ROOT, _failed_v1, "FAIL_GSE_FACTORIZED_CAUSAL_NODE_C09_V1"
        ),
    }


implementation._sources = _sources_with_failed_v1


if __name__ == "__main__":
    raise SystemExit(implementation.main())
