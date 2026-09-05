from __future__ import annotations

from pathlib import Path
import sys
import unittest

SRC_ROOT=Path(__file__).resolve().parents[3]/"src"
if str(SRC_ROOT) not in sys.path:sys.path.insert(0,str(SRC_ROOT))

from mtare_topo.topology.causal_graph_v2 import CausalGraphConfig,CausalTopometricGraphV2,heading_set_distance


class CausalGraphV2Tests(unittest.TestCase):
    def test_heading_set_is_order_invariant(self):
        self.assertEqual(heading_set_distance([10,190],[190,10]),0.0)

    def test_distance_anchor_edge_has_physical_trace_and_accumulated_length(self):
        graph=CausalTopometricGraphV2(CausalGraphConfig(stable_frames=2,minimum_event_travel_m=4,loop_merge_radius_m=1,distance_anchor_interval_m=4,turn_event_deg=180))
        for index,x in enumerate((0.0,2.0,4.0)):
            graph.update(frame_index=index,route_arc_m=x,xyz_m=[x,0,0],yaw_deg=0,headings_robot_deg=[0,180],evaluator_gt_edge_id="e0")
        snapshot=graph.snapshot()
        self.assertEqual(snapshot["node_count"],2);self.assertEqual(snapshot["edge_count"],1)
        traversal=snapshot["edges"][0]["traversals"][0]
        self.assertEqual(traversal["length_m"],4.0);self.assertGreaterEqual(traversal["trace_frame_count"],2);self.assertEqual(traversal["evaluator_gt_edge_ids"],["e0"])

    def test_stable_structural_event_and_loop_merge(self):
        config=CausalGraphConfig(stable_frames=2,minimum_event_travel_m=2,loop_merge_radius_m=1,distance_anchor_interval_m=100,turn_event_deg=180)
        graph=CausalTopometricGraphV2(config)
        frames=[(0,0,[0,180]),(2,2,[0,120,240]),(4,4,[0,120,240]),(6,2,[0,180]),(8,4.1,[0,120,240]),(10,4.1,[0,120,240])]
        for index,(arc,x,heads) in enumerate(frames):
            graph.update(frame_index=index,route_arc_m=arc,xyz_m=[x,0,0],yaw_deg=0,headings_robot_deg=heads,evaluator_gt_edge_id="e0")
        snapshot=graph.snapshot()
        self.assertEqual(snapshot["structural_node_count"],1)
        self.assertTrue(all(edge["kind"]=="verified_traversed" for edge in snapshot["edges"]))

    def test_semantic_probabilities_not_gt_role(self):
        graph=CausalTopometricGraphV2(CausalGraphConfig(distance_anchor_interval_m=100))
        graph.update(frame_index=0,route_arc_m=0,xyz_m=[0,0,0],yaw_deg=0,headings_robot_deg=[0,180],role_probabilities=[0.1,0.8,0.1],branch_count=2)
        self.assertEqual(graph.snapshot()["nodes"][0]["role"],"junction")


if __name__=="__main__":unittest.main()
