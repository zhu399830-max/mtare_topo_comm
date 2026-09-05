from pathlib import Path
import sys
import tempfile
import unittest


SRC_ROOT = Path(__file__).resolve().parents[3] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from mtare_topo.data.worldgen import (
    MeshGenerationError,
    SeedBundle,
    TNGEdge,
    TNGNode,
    TNGParameters,
    TunnelNetworkGraph,
    TunnelMeshParameters,
    Vec3,
    generate_tng,
    generate_tunnel_mesh,
    mesh_stats,
    write_usda,
)


class TunnelMeshTest(unittest.TestCase):
    def setUp(self) -> None:
        self.seeds = SeedBundle.from_master(77)
        self.graph = generate_tng(
            self.seeds,
            TNGParameters(
                target_nodes=10,
                connector_count=1,
                base_segment_length=12.0,
                min_node_clearance=4.0,
                min_edge_clearance=4.0,
                connector_min_length=8.0,
                connector_max_length=35.0,
                collision_samples=13,
                candidate_retries=1024,
            ),
        )
        self.parameters = TunnelMeshParameters(
            voxel_size_m=1.0,
            lateral_half_width_m=1.25,
            lateral_half_width_noise_m=0.0,
            vertical_radius_m=1.25,
            vertical_radius_noise_m=0.0,
            floor_depth_m=0.75,
            geometry_guard_m=1.0,
            minimum_robot_radius_m=0.3,
        )

    def test_mesh_is_reproducible_watertight_and_topology_consistent(self) -> None:
        first = generate_tunnel_mesh(self.graph, self.seeds, self.parameters)
        second = generate_tunnel_mesh(self.graph, self.seeds, self.parameters)
        self.assertEqual(first.canonical_hash(), second.canonical_hash())
        stats = mesh_stats(first)
        self.assertTrue(stats["watertight_edge_incidence"])
        self.assertEqual(stats["mesh_component_count"], 1)
        self.assertEqual(stats["degenerate_face_count"], 0)
        self.assertEqual(stats["genus"], self.graph.stats()["cycle_rank"])

    def test_mesh_rejects_graph_without_cross_section_clearance(self) -> None:
        too_wide = TunnelMeshParameters(
            voxel_size_m=1.0,
            lateral_half_width_m=2.0,
            lateral_half_width_noise_m=0.0,
            vertical_radius_m=2.0,
            vertical_radius_noise_m=0.0,
            floor_depth_m=1.0,
            geometry_guard_m=1.0,
            minimum_robot_radius_m=0.3,
        )
        with self.assertRaises(MeshGenerationError):
            generate_tunnel_mesh(self.graph, self.seeds, too_wide)

    def test_usda_array_chunks_remain_comma_separated(self) -> None:
        mesh = generate_tunnel_mesh(self.graph, self.seeds, self.parameters)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "stage.usda"
            write_usda(mesh, path)
            content = path.read_text(encoding="utf-8")
        self.assertIn(",\n            3, 3", content)

    def test_navigation_grade_mesh_runs_robot_and_path_audit(self) -> None:
        chain_graph = TunnelNetworkGraph(
            nodes=(
                TNGNode("n000", Vec3(0.0, 0.0, 0.0)),
                TNGNode("n001", Vec3(8.0, 0.0, 0.5)),
                TNGNode("n002", Vec3(16.0, 3.0, 1.0)),
                TNGNode("n003", Vec3(24.0, 3.0, 1.5)),
            ),
            edges=(
                TNGEdge("e000", "n000", "n001", "RGTG", 0),
                TNGEdge("e001", "n001", "n002", "RGTG", 0),
                TNGEdge("e002", "n002", "n003", "RGTG", 0),
            ),
            topology_seed=self.seeds.topology,
            parameters=TNGParameters(
                target_nodes=4,
                connector_count=0,
                max_degree=2,
                min_node_clearance=4.0,
                min_edge_clearance=4.0,
            ),
        )
        parameters = TunnelMeshParameters(
            geometry_variant_id="unit_navigation",
            voxel_size_m=0.5,
            lateral_half_width_m=1.25,
            lateral_half_width_noise_m=0.0,
            vertical_radius_m=1.25,
            vertical_radius_noise_m=0.0,
            floor_depth_m=0.75,
            geometry_guard_m=1.0,
            minimum_robot_radius_m=0.3,
            centerline_mode="cubic_hermite",
            centerline_sample_spacing_m=0.5,
            curve_tangent_scale=0.9,
            curve_wander_m=0.0,
            floor_profile_mode="global_z",
            junction_chamber_scale=1.15,
            navigation_audit_enabled=True,
            robot_height_m=0.5,
            robot_safety_margin_m=0.1,
            robot_ground_clearance_m=0.35,
            maximum_path_pitch_rad=0.4,
            minimum_path_turn_radius_m=0.2,
        )
        mesh = generate_tunnel_mesh(chain_graph, self.seeds, parameters)
        self.assertEqual(mesh.navigation_audit["status"], "PASS")
        self.assertEqual(mesh.navigation_audit["robot_probe_failure_count"], 0)
        self.assertGreater(mesh.navigation_audit["minimum_robot_side_clearance_m"], 0)


if __name__ == "__main__":
    unittest.main()
