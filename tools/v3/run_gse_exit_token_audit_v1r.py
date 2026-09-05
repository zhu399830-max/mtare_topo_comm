#!/usr/bin/env python3
"""Corrective entry point for the cold-import-safe GSE exit-token audit."""

from __future__ import annotations

import run_gse_exit_token_audit_v1 as implementation


implementation.RUN_ID = "gate2_20260824_gse_exit_token_audit_v1r_seed0"


if __name__ == "__main__":
    raise SystemExit(implementation.main())
