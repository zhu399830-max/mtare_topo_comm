import unittest

import numpy as np

from mtare_topo.data.cano_phase2_dataset import candidate_clusters, select_valid_clusters


class Phase2DatasetContractTest(unittest.TestCase):
    def fixture(self):
        graph = {"nodes": [
            {"id":"j","degree":3,"xyz":[10,0,0],"incident_tunnel_ids":[1]},
            {"id":"t","degree":1,"xyz":[15,0,0],"incident_tunnel_ids":[1]},
        ]}
        spline = {"tunnels":[{"tunnel_id":1,"points":[[0,0,0],[30,0,0]]}]}
        return graph, spline

    def test_terminal_overlap_has_independent_flags_and_terminal_primary(self):
        graph, spline = self.fixture()
        rows = candidate_clusters("w", graph, spline)
        overlap = [row for row in rows if row["near_junction"] and row["near_terminal"]]
        self.assertTrue(overlap)
        self.assertTrue(all(row["primary_role"] == "terminal" for row in overlap))

    def test_event_first_selection_and_same_role_replacement(self):
        candidates = [
            {"cluster_id":"j0","primary_role":"junction","junction_event_ids":["j"],"terminal_event_ids":[]},
            {"cluster_id":"j1","primary_role":"junction","junction_event_ids":["j"],"terminal_event_ids":[]},
            {"cluster_id":"t0","primary_role":"terminal","junction_event_ids":[],"terminal_event_ids":["t"]},
            {"cluster_id":"i0","primary_role":"interior","junction_event_ids":[],"terminal_event_ids":[]},
        ]
        selected, rejected = select_valid_clusters(
            candidates, {"interior":1,"junction":1,"terminal":1}, lambda row: row["cluster_id"] != "j0"
        )
        self.assertEqual({row["cluster_id"] for row in selected}, {"j1","t0","i0"})
        self.assertEqual([row["cluster_id"] for row in rejected], ["j0"])

    def test_points_arc_capacity_uses_floor_length_over_five(self):
        graph = {"nodes":[]}
        spline = {"tunnels":[{"tunnel_id":1,"points":[[0,0,0],[12.4,0,0]]}]}
        rows = candidate_clusters("w", graph, spline)
        self.assertEqual(len(rows), 2)
        np.testing.assert_allclose(rows[1]["frame_arcs_m"], [5.5,6.5,7.5,8.5,9.5])


if __name__ == "__main__":
    unittest.main()
