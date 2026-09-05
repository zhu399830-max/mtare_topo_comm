#!/usr/bin/env python3
"""Corrective one-shot runner for the sealed short-traversal inventory defect."""

from __future__ import annotations

import run_gse_mesh_teacher_distribution_v1 as base


base.RUN_ID = "gate2_20260824_gse_mesh_teacher_distribution_v1r_seed0"
base.RUN_PASS_STATUS = "PASS_GSE_MESH_TEACHER_DISTRIBUTION_V1R"
base.RUN_FAIL_STATUS = "FAIL_GSE_MESH_TEACHER_DISTRIBUTION_V1R"


if __name__ == "__main__":
    raise SystemExit(base.main())
