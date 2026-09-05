#!/home/zeng-workstation/anaconda3/bin/python
"""Corrected immutable wrapper for the V1 pre-execution Data Card bug."""

from __future__ import annotations

import run_aee_sensor_operator_parity_v1 as base


base.RUN_ID = "gate2_20260821_aee_sensor_operator_parity_v1r_seed20260820"


if __name__ == "__main__":
    raise SystemExit(base.main())
