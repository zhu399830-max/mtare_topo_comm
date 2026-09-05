from dataclasses import replace
from pathlib import Path
import sys
import unittest


SRC_ROOT = Path(__file__).resolve().parents[3] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from mtare_topo.data.worldgen import (
    SeedBundle,
    SplitLeakageError,
    TNGParameters,
    generate_tng,
    minimum_nonincident_node_edge_sample_distance,
    minimum_nonincident_edge_sample_distance,
    validate_parent_split,
)


class TNGGeneratorTest(unittest.TestCase):
    def setUp(self) -> None:
        self.seeds = SeedBundle.from_master(0)
        self.parameters = TNGParameters(
            target_nodes=18,
            connector_count=2,
            min_edge_clearance=1.0,
            candidate_retries=512,
        )

    def test_seed_replay_is_byte_stable_at_canonical_level(self) -> None:
        first = generate_tng(self.seeds, self.parameters)
        second = generate_tng(self.seeds, self.parameters)
        self.assertEqual(first.canonical_hash(), second.canonical_hash())
        self.assertEqual(first.canonical_payload(), second.canonical_payload())

    def test_downstream_seeds_do_not_change_topology(self) -> None:
        changed = replace(
            self.seeds,
            geometry=self.seeds.geometry + 1,
            clutter=self.seeds.clutter + 1,
            trajectory=self.seeds.trajectory + 1,
            sensor=self.seeds.sensor + 1,
        )
        first = generate_tng(self.seeds, self.parameters)
        second = generate_tng(changed, self.parameters)
        self.assertEqual(first.canonical_hash(), second.canonical_hash())

    def test_graph_invariants_and_requested_cycles(self) -> None:
        graph = generate_tng(self.seeds, self.parameters)
        graph.validate()
        stats = graph.stats()
        self.assertEqual(stats["node_count"], self.parameters.target_nodes)
        self.assertEqual(stats["rgtg_edge_count"], self.parameters.target_nodes - 1)
        self.assertEqual(stats["ctg_edge_count"], self.parameters.connector_count)
        self.assertEqual(stats["cycle_rank"], self.parameters.connector_count)
        self.assertLessEqual(
            max(len(neighbors) for neighbors in graph.adjacency().values()),
            self.parameters.max_degree,
        )
        self.assertLessEqual(
            stats["maximum_abs_edge_pitch_rad"],
            self.parameters.max_abs_pitch_rad,
        )

    def test_sampled_nonincident_clearance_is_respected(self) -> None:
        graph = generate_tng(self.seeds, self.parameters)
        edge_distance = minimum_nonincident_edge_sample_distance(graph)
        node_edge_distance = minimum_nonincident_node_edge_sample_distance(graph)
        self.assertGreaterEqual(edge_distance, self.parameters.min_edge_clearance)
        self.assertGreaterEqual(node_edge_distance, self.parameters.min_node_clearance)

    def test_parent_split_accepts_variants_in_one_split(self) -> None:
        assignments = validate_parent_split(
            [
                {"topology_parent_id": "tng_a", "split": "train", "geometry": 0},
                {"topology_parent_id": "tng_a", "split": "train", "geometry": 1},
                {"topology_parent_id": "tng_b", "split": "test", "geometry": 0},
            ]
        )
        self.assertEqual(assignments, {"tng_a": "train", "tng_b": "test"})

    def test_parent_split_rejects_cross_split_leakage(self) -> None:
        with self.assertRaises(SplitLeakageError):
            validate_parent_split(
                [
                    {"topology_parent_id": "tng_a", "split": "train"},
                    {"topology_parent_id": "tng_a", "split": "test"},
                ]
            )


if __name__ == "__main__":
    unittest.main()
