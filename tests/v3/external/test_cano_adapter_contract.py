#!/usr/bin/env python3
"""External-environment contract tests for the read-only Cano adapter."""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "external/procedural-subt-gen/src"))
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "tools/v3"))

from generate_cano_audited_bundle import _stable_graph  # noqa: E402
from cano_five_topology_support import (  # noqa: E402
    build_parent,
    canonical_exports,
    template_polyline_length_m,
    topology_identity_preview,
)
from mtare_topo.data.cano_contract_pilot import (  # noqa: E402
    PARENT_REGISTRY,
    anchor_selection_audit,
    select_canonical_anchors,
)
from subt_proc_gen.graph import Node  # noqa: E402
from subt_proc_gen.tunnel import Tunnel, TunnelNetwork, TunnelType  # noqa: E402
from validate_cano_audited_bundle import _axis_conflicts, _graph_audit  # noqa: E402
from run_cano_five_topology_cpu_contract_pilot import (  # noqa: E402
    _approved_project_file,
    _executor_environment,
    _subt_proc_gen_import_identity,
)


class CanoAdapterContractTests(unittest.TestCase):
    def setUp(self) -> None:
        Node.set_global_counter(0)
        Tunnel.counter = 0

    def test_stable_graph_preserves_three_grown_one_connector_cycle(self) -> None:
        origin = Node((0.0, 0.0, 0.0))
        east = Node((20.0, 0.0, 0.0))
        north = Node((0.0, 20.0, 0.0))
        west = Node((-20.0, 0.0, 0.0))
        network = TunnelNetwork(initial_node=False)
        for nodes, tunnel_type in (
            ([origin, east], TunnelType.grown),
            ([origin, north], TunnelType.grown),
            ([origin, west], TunnelType.grown),
            ([east, north], TunnelType.connector),
        ):
            network.add_tunnel(Tunnel(nodes, tunnel_type=tunnel_type))
        graph, _ = _stable_graph(network)
        metrics = _graph_audit(graph)
        self.assertTrue(metrics["pass"])
        self.assertEqual(metrics["connected_components"], 1)
        self.assertEqual(metrics["cycle_rank"], 1)
        self.assertEqual(metrics["tunnel_type_counts"], {"grown": 3, "connector": 1})

    def test_axis_conflict_audit_distinguishes_interior_and_intersection(self) -> None:
        axis = np.array(
            [
                [0, 0, 0, 1, 0, 0, 3, 1, 2],
                [1, 0, 0, 1, 0, 0, 3, 1, 2],
                [0, 0, 0, 0, 1, 0, 3, 1, 4],
                [2, 0, 0, 0, 1, 0, 3, 1, 4],
                [0, 0, 0, 1, 0, 0, 3, 2, 2],
                [0, 0, 0, 0, 1, 0, 3, 2, 4],
            ],
            dtype=float,
        )
        metrics = _axis_conflicts(axis)
        self.assertEqual(metrics["cross_tunnel_conflict_coordinates"], 1)
        self.assertEqual(metrics["cross_tunnel_conflict_rows"], 2)
        self.assertAlmostEqual(metrics["cross_tunnel_conflict_fraction"], 0.5)

    def test_axis_conflict_audit_accepts_distinct_interior_coordinates(self) -> None:
        axis = np.array(
            [
                [0, 0, 0, 1, 0, 0, 3, 1, 2],
                [1, 0, 0, 1, 0, 0, 3, 1, 2],
                [0, 1, 0, 0, 1, 0, 3, 1, 4],
                [0, 2, 0, 0, 1, 0, 3, 1, 4],
                [0, 0, 0, 1, 0, 0, 3, 2, 2],
                [0, 0, 0, 0, 1, 0, 3, 2, 4],
            ],
            dtype=float,
        )
        metrics = _axis_conflicts(axis)
        self.assertEqual(metrics["cross_tunnel_conflict_coordinates"], 0)
        self.assertEqual(metrics["cross_tunnel_conflict_rows"], 0)

    def test_five_topology_templates_have_anchor_length_margin(self) -> None:
        parent_ids = (
            "P01_straight_turn",
            "P02_branch_deadend",
            "P03_loop_bottleneck",
            "P04_chamber_multiexit",
            "P05_slope_multiheight",
        )
        for parent_id in parent_ids:
            with self.subTest(parent_id=parent_id):
                self.assertGreaterEqual(template_polyline_length_m(parent_id), 300.0)

    def test_topology_preview_is_repeatable_and_seed_sensitive(self) -> None:
        parent_id = "P01_straight_turn"
        first = topology_identity_preview(parent_id, 9001)
        self.assertEqual(first, topology_identity_preview(parent_id, 9001))
        self.assertNotEqual(first, topology_identity_preview(parent_id, 9002))

    def test_formal_executor_environment_imports_fixed_checkout(self) -> None:
        identity = _subt_proc_gen_import_identity(_executor_environment())
        self.assertTrue(identity["from_fixed_checkout"])
        self.assertIn("external/procedural-subt-gen/src", identity["module"])

    def test_approval_artifacts_are_resolved_from_spec_paths(self) -> None:
        proposal = _approved_project_file(
            "configs/v3/gate0/cano_five_topology_cpu_contract_pilot_v2_import_fix.proposal.json",
            "config_path",
        )
        data_card = _approved_project_file(
            "configs/v3/gate0/data_cards/cano_five_topology_cpu_contract_pilot_v2_import_fix.json",
            "data_card",
        )
        self.assertEqual(proposal.parent, PROJECT_ROOT / "configs/v3/gate0")
        self.assertEqual(data_card.parent, PROJECT_ROOT / "configs/v3/gate0/data_cards")

    def test_approval_artifact_path_escape_is_rejected(self) -> None:
        for unsafe in ("/etc/passwd", "../outside.json", ""):
            with self.subTest(path=unsafe):
                with self.assertRaises(RuntimeError):
                    _approved_project_file(unsafe, "config_path")

    def test_target_cardinality_selector_passes_frozen_five_topology_inputs(self) -> None:
        sealed_worlds = PROJECT_ROOT / (
            "results/gate0_baseline/"
            "gate0_20260811_cano_five_topology_cpu_contract_pilot_v2b_runner_paths_seed0/"
            "artifacts/worlds"
        )
        for registry in PARENT_REGISTRY:
            parent_id = registry["parent_id"]
            if parent_id == "P05_slope_multiheight":
                network, _, _ = build_parent(parent_id, registry["topology_seed"])
                graph, splines = canonical_exports(network)
            else:
                graph = json.loads((sealed_worlds / parent_id / "graph.json").read_text())
                splines = json.loads((sealed_worlds / parent_id / "splines.json").read_text())
            anchors = select_canonical_anchors(graph, splines, -1.5)
            audit = anchor_selection_audit(graph, splines, anchors)
            with self.subTest(parent_id=parent_id):
                self.assertTrue(audit["passed"], audit)
                self.assertEqual(audit["selected_anchor_count"], 50)
                self.assertGreaterEqual(audit["minimum_pairwise_distance_m"], 5.0)
                self.assertTrue(audit["fill_quotas_match"])
                self.assertLessEqual(
                    audit["maximum_same_tunnel_or_shared_event_coverage_radius_m"],
                    7.5,
                )


if __name__ == "__main__":
    unittest.main()
