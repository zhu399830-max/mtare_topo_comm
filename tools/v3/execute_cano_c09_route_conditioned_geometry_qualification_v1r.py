#!/usr/bin/env python3
"""Execute the approved C09 V1R after correcting only the frozen-evidence path."""
from __future__ import annotations

import execute_cano_c09_route_conditioned_geometry_qualification_v1 as core


def main() -> int:
    core.RUN_ID = "gate4_20260817_cano_c09_route_conditioned_geometry_qualification_v1r_seed0"
    core.PASS_STATUS = "PASS_CANO_C09_ROUTE_CONDITIONED_GEOMETRY_QUALIFICATION_V1R"
    core.FAIL_STATUS = "FAIL_CANO_C09_ROUTE_CONDITIONED_GEOMETRY_QUALIFICATION_V1R"
    core.SCHEMA_VERSION = "cano_c09_route_conditioned_geometry_qualification_v1r"
    core.PREVIEW_TITLE = "C09 route-conditioned geometry qualification V1R"
    return core.main()


if __name__ == "__main__":
    raise SystemExit(main())
