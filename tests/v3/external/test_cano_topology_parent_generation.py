#!/usr/bin/env python3
"""Frozen-E1 external checks for native random Cano topology generation."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "external/procedural-subt-gen/src"))
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "tools/v3"))

from execute_cano_100_topology_parent_candidate_audit_v1 import _build_candidate  # noqa: E402
from execute_cano_100_topology_parent_candidate_audit_v2 import (  # noqa: E402
    PARAMETER_DRAW_BUDGET,
    _build_candidate as _build_candidate_v2,
)


class CanoTopologyParentGenerationTests(unittest.TestCase):
    def test_native_flat_candidate_is_replayable_and_actually_flat(self) -> None:
        stratum = {
            "stratum_id": "external_flat_smoke",
            "dimensionality": "flat",
            "grown_tunnels": 1,
            "connector_tunnels": 0,
        }
        candidate = {
            "candidate_id": "external_flat_smoke_C01",
            "topology_seed": 424242,
            "reserved_geometry_seed": 525252,
        }
        first = _build_candidate(stratum, candidate)
        second = _build_candidate(stratum, candidate)
        self.assertTrue(first["generation_succeeded"])
        self.assertEqual(first["graph"], second["graph"])
        self.assertEqual(first["splines"], second["splines"])
        coordinates = np.asarray([node["xyz"] for node in first["graph"]["nodes"]])
        self.assertLessEqual(float(np.ptp(coordinates[:, 2])), 1e-9)
        self.assertEqual(len(first["graph"]["tunnels"]), 1)

    def test_v2_bounded_draw_trace_is_replayable(self) -> None:
        stratum = {
            "stratum_id": "external_v2_flat_smoke",
            "dimensionality": "flat",
            "grown_tunnels": 1,
            "connector_tunnels": 0,
        }
        candidate = {
            "candidate_id": "external_v2_flat_smoke_C01",
            "topology_seed": 620101,
            "reserved_geometry_seed": 720101,
        }
        first = _build_candidate_v2(stratum, candidate)
        second = _build_candidate_v2(stratum, candidate)
        self.assertTrue(first["generation_succeeded"])
        self.assertEqual(first, second)
        operation = first["operations"][0]
        self.assertTrue(operation["success"])
        self.assertGreaterEqual(operation["parameter_draw_count"], 1)
        self.assertLessEqual(operation["parameter_draw_count"], PARAMETER_DRAW_BUDGET)
        self.assertEqual(
            operation["selected_draw_index"], operation["parameter_draw_count"]
        )


if __name__ == "__main__":
    unittest.main()
