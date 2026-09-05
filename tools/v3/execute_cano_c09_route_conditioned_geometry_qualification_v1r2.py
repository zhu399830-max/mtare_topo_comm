#!/usr/bin/env python3
"""Execute the approved full C09 V1R2 reboot-recovery qualification."""
from __future__ import annotations

import execute_cano_c09_route_conditioned_geometry_qualification_v1 as core


def main() -> int:
    core.RUN_ID = "gate4_20260819_cano_c09_route_conditioned_geometry_qualification_v1r2_seed0"
    core.PASS_STATUS = "PASS_CANO_C09_ROUTE_CONDITIONED_GEOMETRY_QUALIFICATION_V1R2"
    core.SCHEMA_VERSION = "cano_c09_route_conditioned_geometry_qualification_v1r2"
    core.PREVIEW_TITLE = "C09 route-conditioned geometry qualification V1R2"
    return core.main()


if __name__ == "__main__":
    raise SystemExit(main())
