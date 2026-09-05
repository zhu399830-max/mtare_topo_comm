from __future__ import annotations

import unittest

from mtare_topo.data.gse_teacher_inventory import world_teacher_inventory


class GSETeacherInventoryTest(unittest.TestCase):
    def test_both_directions_and_all_sequence_labels_are_preserved(self) -> None:
        graph = {
            "nodes": [
                {"id": "n0", "xyz": [0.0, 0.0, 0.0], "degree": 1, "incident_tunnel_ids": [7]},
                {"id": "n1", "xyz": [15.0, 15.0, 0.0], "degree": 1, "incident_tunnel_ids": [7]},
            ],
            "edges": [{"id": "e0", "node_ids": ["n0", "n1"], "tunnel_ids": [7]}],
        }
        splines = {"tunnels": [{"tunnel_id": 7, "points": [[0.0, 0.0, 0.0], [15.0, 0.0, 0.0], [15.0, 15.0, 0.0]]}]}
        geometry = {"tunnels": [{"tunnel_id": 7, "radius_m": 2.0}]}
        result = world_teacher_inventory(
            parent_id="S01_flat_tree_small_C01",
            split="train",
            graph=graph,
            spline_document=splines,
            geometry_parameters=geometry,
        )
        self.assertEqual(result["edge_count"], 1)
        self.assertEqual(result["directed_traversal_count"], 2)
        self.assertEqual(result["sequence_count"], 52)
        self.assertEqual(result["unique_frame_count"], 60)
        self.assertEqual(sum(result["event_counts"].values()), 52)
        self.assertGreater(result["event_counts"]["turn"], 0)
        self.assertGreater(result["event_counts"]["terminal"], 0)
        self.assertEqual(result["geometry"]["width_m"]["minimum"], 4.0)

    def test_c10_is_rejected_before_asset_use(self) -> None:
        with self.assertRaises(ValueError):
            world_teacher_inventory(
                parent_id="S01_flat_tree_small_C10",
                split="validation",
                graph={"nodes": [], "edges": []},
                spline_document={"tunnels": []},
                geometry_parameters={"tunnels": []},
            )


if __name__ == "__main__":
    unittest.main()
