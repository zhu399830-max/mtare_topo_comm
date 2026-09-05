import json
import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
TOOLS = PROJECT_ROOT / "tools/v3"
sys.path.insert(0, str(PROJECT_ROOT / "external/procedural-subt-gen/src"))
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(TOOLS))


class CanoPerceptionMeshReconstructionExternalTest(unittest.TestCase):
    def test_sealed_m1_failure_has_one_removable_zero_area_face(self):
        import numpy as np
        import open3d as o3d
        from mtare_topo.data.cano_perception_mesh_contract import zero_area_triangle_sanitation

        path = PROJECT_ROOT / (
            "results/gate0_baseline/"
            "gate0_20260811_cano_100_parent_perception_mesh_m1_immutable_assets_seed0/"
            "artifacts/meshes/S05_flat_branch_medium_C05/primary/mesh.obj"
        )
        mesh = o3d.io.read_triangle_mesh(str(path), enable_post_processing=False)
        vertices = np.asarray(mesh.vertices)
        triangles = np.asarray(mesh.triangles)
        keep, evidence = zero_area_triangle_sanitation(vertices, triangles)
        self.assertEqual(evidence["removed_triangle_count"], 1)
        self.assertEqual(evidence["removed_surface_area_sum_m2"], 0.0)
        self.assertEqual(int(np.count_nonzero(keep)), len(triangles) - 1)

    def test_rank_one_parent_reconstructs_exact_source_without_meshing(self):
        import execute_cano_100_parent_perception_mesh_contract_m0 as module

        parents = json.loads(
            (
                PROJECT_ROOT
                / "results/gate0_baseline/gate0_20260811_cano_100_topology_parent_recipe_reclassification_v2r_seed0/artifacts/accepted_parent_manifest.json"
            ).read_text()
        )["parents"]
        parent = next(item for item in parents if item["parent_id"] == "S01_flat_tree_small_C01")
        stratum = module._stratum_registry()[parent["source_stratum_id"]]
        _, graph, splines, operations, _ = module.reconstruct_parent_network(parent, stratum)
        source_graph = json.loads((PROJECT_ROOT / parent["source_graph"]).read_text())
        source_splines = json.loads((PROJECT_ROOT / parent["source_splines"]).read_text())
        source_metric = json.loads(
            (
                module.SOURCE_V2 / "metrics" / f"{parent['parent_id']}.json"
            ).read_text()
        )
        self.assertEqual(module.canonical_json_hash(graph), module.canonical_json_hash(source_graph))
        self.assertEqual(module.canonical_json_hash(splines), module.canonical_json_hash(source_splines))
        self.assertEqual(module.canonical_json_hash(operations), module.canonical_json_hash(source_metric["operations"]))

    def test_sealed_m0_s01_passes_only_noise_bounded_geometric_replay(self):
        import execute_cano_100_parent_perception_mesh_contract_m0r as m0r

        root = PROJECT_ROOT / (
            "results/gate0_baseline/"
            "gate0_20260811_cano_100_parent_perception_mesh_contract_m0_seed0/"
            "artifacts/meshes/S01_flat_tree_small_C01"
        )
        primary = json.loads((root / "primary/materialization.json").read_text())
        replay = json.loads((root / "replay/materialization.json").read_text())
        primary["mesh_path"] = str(root / "primary/mesh.obj")
        replay["mesh_path"] = str(root / "replay/mesh.obj")
        primary["axis_path"] = str(root / "primary/axis.npy")
        replay["axis_path"] = str(root / "replay/axis.npy")
        audit = m0r._geometric_replay(primary, replay)
        self.assertTrue(audit["passed"])
        self.assertFalse(audit["metrics"]["obj_sha256_equal_recorded_not_required"])
        self.assertLessEqual(
            audit["metrics"]["primary_to_replay_nearest_vertex_maximum_m"], 0.75
        )

    def test_sealed_m0r_s02_passes_m0r2_triangle_discretization_contract(self):
        import execute_cano_100_parent_perception_mesh_contract_m0r as m0r

        root = PROJECT_ROOT / (
            "results/gate0_baseline/"
            "gate0_20260811_cano_100_parent_perception_mesh_contract_m0r_geometric_replay_seed0/"
            "artifacts/meshes/S02_3d_tree_small_C01"
        )
        primary = json.loads((root / "primary/materialization.json").read_text())
        replay = json.loads((root / "replay/materialization.json").read_text())
        primary["mesh_path"] = str(root / "primary/mesh.obj")
        replay["mesh_path"] = str(root / "replay/mesh.obj")
        primary["axis_path"] = str(root / "primary/axis.npy")
        replay["axis_path"] = str(root / "replay/axis.npy")
        audit = m0r._geometric_replay_with_triangle_limit(
            primary, replay, triangle_count_relative_limit=0.0001
        )
        self.assertTrue(audit["passed"])
        self.assertEqual(audit["metrics"]["triangle_count_absolute_difference"], 2)
        self.assertLessEqual(
            audit["metrics"]["triangle_count_relative_difference"], 0.0001
        )

    def test_sealed_m0r2_s08_passes_point_to_surface_contract(self):
        import execute_cano_100_parent_perception_mesh_contract_m0r as m0r

        root = PROJECT_ROOT / (
            "results/gate0_baseline/"
            "gate0_20260811_cano_100_parent_perception_mesh_contract_m0r2_triangle_discretization_seed0/"
            "artifacts/meshes/S08_3d_loop_rich_C01"
        )
        primary = json.loads((root / "primary/materialization.json").read_text())
        replay = json.loads((root / "replay/materialization.json").read_text())
        primary["mesh_path"] = str(root / "primary/mesh.obj")
        replay["mesh_path"] = str(root / "replay/mesh.obj")
        primary["axis_path"] = str(root / "primary/axis.npy")
        replay["axis_path"] = str(root / "replay/axis.npy")
        audit = m0r._geometric_replay_with_triangle_limit(
            primary,
            replay,
            triangle_count_relative_limit=0.0001,
            use_point_to_surface=True,
        )
        self.assertTrue(audit["passed"])
        self.assertGreater(
            audit["metrics"]["replay_to_primary_nearest_vertex_maximum_m"], 0.75
        )
        self.assertLessEqual(
            audit["metrics"]["replay_vertices_to_primary_surface_maximum_m"],
            0.75,
        )

    def test_sealed_m0r3_s07_passes_unified_discretization_contract(self):
        import execute_cano_100_parent_perception_mesh_contract_m0r as m0r

        root = PROJECT_ROOT / (
            "results/gate0_baseline/"
            "gate0_20260811_cano_100_parent_perception_mesh_contract_m0r3_point_to_surface_seed0/"
            "artifacts/meshes/S07_flat_loop_rich_C01"
        )
        primary = json.loads((root / "primary/materialization.json").read_text())
        replay = json.loads((root / "replay/materialization.json").read_text())
        for record, role in ((primary, "primary"), (replay, "replay")):
            record["mesh_path"] = str(root / role / "mesh.obj")
            record["axis_path"] = str(root / role / "axis.npy")
        audit = m0r._geometric_replay_with_triangle_limit(
            primary,
            replay,
            vertex_count_relative_limit=0.0001,
            triangle_count_relative_limit=0.0001,
            use_point_to_surface=True,
        )
        self.assertTrue(audit["passed"])
        self.assertEqual(audit["metrics"]["vertex_count_absolute_difference"], 1)


if __name__ == "__main__":
    unittest.main()
