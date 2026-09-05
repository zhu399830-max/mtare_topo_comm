from __future__ import annotations

import unittest
import json
from pathlib import Path
import tempfile

import numpy as np

from mtare_topo.data.gse_sequence_inventory import (
    causal_anchor_arcs,
    deduplicated_frame_arcs_and_sequence_indices,
    enumerate_directed_traversals,
    inventory_from_registry,
)


class GSESequenceInventoryTest(unittest.TestCase):
    def test_deduplicated_frames_are_referenced_causally(self) -> None:
        frames, anchors, references = deduplicated_frame_arcs_and_sequence_indices(
            8.2,
            global_frame_offset=100,
        )
        np.testing.assert_allclose(frames, np.arange(9, dtype=float))
        np.testing.assert_allclose(anchors, [4.0, 5.0, 6.0, 7.0, 8.0])
        np.testing.assert_array_equal(
            references,
            np.asarray(
                [
                    [100, 101, 102, 103, 104],
                    [101, 102, 103, 104, 105],
                    [102, 103, 104, 105, 106],
                    [103, 104, 105, 106, 107],
                    [104, 105, 106, 107, 108],
                ]
            ),
        )
        self.assertTrue(np.all(np.diff(references, axis=1) == 1))
        self.assertTrue(np.all(references[:, -1] <= 108))

    def setUp(self) -> None:
        self.graph = {
            "nodes": [
                {"id": "n0", "xyz": [0.0, 0.0, 0.0]},
                {"id": "n1", "xyz": [10.0, 0.0, 0.0]},
            ],
            "edges": [{"id": "e0", "node_ids": ["n0", "n1"], "tunnel_ids": [7]}],
        }
        self.splines = {
            "tunnels": [{"tunnel_id": 7, "points": [[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]]}]
        }

    def test_five_frame_anchor_contract(self) -> None:
        np.testing.assert_array_equal(causal_anchor_arcs(10.0), np.arange(4.0, 10.0, 1.0))

    def test_each_edge_is_kept_in_both_directions(self) -> None:
        records = enumerate_directed_traversals(
            parent_id="S01_flat_tree_small_C01",
            split="train",
            graph=self.graph,
            spline_document=self.splines,
        )
        self.assertEqual(len(records), 2)
        self.assertEqual({(item.from_node_id, item.to_node_id) for item in records}, {("n0", "n1"), ("n1", "n0")})
        self.assertEqual([item.sequence_count for item in records], [6, 6])
        self.assertEqual([item.unique_frame_count for item in records], [10, 10])
        self.assertEqual([item.referenced_frame_count for item in records], [30, 30])

    def test_strict_test_parent_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            enumerate_directed_traversals(
                parent_id="S01_flat_tree_small_C10",
                split="validation",
                graph=self.graph,
                spline_document=self.splines,
            )

    def test_table_rows_are_decoded_from_declared_registry_schema(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            mesh_root = root / "meshes"
            rows = []
            for index in range(90):
                split = "train" if index < 80 else "validation"
                parent_id = f"S{index + 1:02d}_C01"
                primary = mesh_root / parent_id / "primary"
                primary.mkdir(parents=True)
                (primary / "graph.json").write_text(json.dumps(self.graph))
                (primary / "splines.json").write_text(json.dumps(self.splines))
                rows.append([parent_id, split])
            registry = {
                "sampling_contract": {"row_schema": ["parent_id", "split"]},
                "rows": rows,
            }
            registry_path = root / "registry.json"
            registry_path.write_text(json.dumps(registry))
            inventory = inventory_from_registry(registry_path=registry_path, mesh_root=mesh_root)
            self.assertEqual(inventory["totals"]["world_count"], 90)
            self.assertEqual(inventory["totals"]["directed_traversal_count"], 180)
            self.assertEqual(inventory["totals"]["unique_frame_count"], 1800)
            self.assertEqual(inventory["totals"]["referenced_frame_count"], 5400)
            self.assertEqual(inventory["strict_test_worlds_read"], 0)


if __name__ == "__main__":
    unittest.main()
