#!/usr/bin/env python3
"""Run and seal the approved C09 V1R path-only corrective qualification."""
from __future__ import annotations

import run_cano_c09_route_conditioned_geometry_qualification_v1 as core


def main() -> int:
    core.RUN_ID = "gate4_20260817_cano_c09_route_conditioned_geometry_qualification_v1r_seed0"
    core.EXECUTOR_NAME = "execute_cano_c09_route_conditioned_geometry_qualification_v1r.py"
    core.PASS_STATUS = "PASS_CANO_C09_ROUTE_CONDITIONED_GEOMETRY_QUALIFICATION_V1R"
    core.FAIL_STATUS = "FAIL_CANO_C09_ROUTE_CONDITIONED_GEOMETRY_QUALIFICATION_V1R"
    core.SCHEMA_VERSION = "cano_c09_route_conditioned_geometry_qualification_runner_v1r"
    return core.main()


if __name__ == "__main__":
    raise SystemExit(main())
