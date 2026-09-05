import unittest
import numpy as np

from mtare_topo.data.local_implicit_union import ContinuousLayeredHybridField, LayeredHybridField, LocalUnionWindow, RouteConditionedSupportField, RouteSupportQuery, SampledSweptTubeField, build_local_union_windows, continuous_transition_seam_audit, continuous_transition_semantics_audit, edge_arc_incidence, layered_window_semantics_audit, overlapping_window_pairs, window_boundary_continuity, window_volume_isolation


class LocalImplicitUnionTests(unittest.TestCase):
    def test_windows_are_topology_deterministic(self):
        graph={"nodes":[{"id":"b","xyz":[10,0,0],"degree":3,"incident_tunnel_ids":[3,1,2]},{"id":"a","xyz":[0,0,0],"degree":1,"incident_tunnel_ids":[1]}]}
        geometry={"tunnels":[{"tunnel_id":1,"radius_m":5},{"tunnel_id":2,"radius_m":4},{"tunnel_id":3,"radius_m":3}]}
        windows=build_local_union_windows(graph,geometry)
        self.assertEqual(len(windows),1)
        self.assertEqual(windows[0].incident_tunnel_ids,(1,2,3))
        self.assertAlmostEqual(windows[0].radius_m,5.5)

    def test_overlap_reports_nonidentical_identity(self):
        graph={"nodes":[{"id":"a","xyz":[0,0,0],"degree":3,"incident_tunnel_ids":[1,2,3]},{"id":"b","xyz":[1,0,0],"degree":3,"incident_tunnel_ids":[2,3,4]}]}
        geometry={"tunnels":[{"tunnel_id":value,"radius_m":1} for value in range(1,5)]}
        overlaps=overlapping_window_pairs(build_local_union_windows(graph,geometry))
        self.assertEqual(len(overlaps),1)
        self.assertFalse(overlaps[0]["incident_sets_identical"])

    def test_edge_incidence_keeps_two_arcs_of_one_tunnel_distinct(self):
        graph={"nodes":[{"id":"a","xyz":[0,0,0]},{"id":"b","xyz":[5,0,0]},{"id":"c","xyz":[10,0,0]}],"edges":[{"id":"ab","node_ids":["a","b"],"tunnel_ids":[7]},{"id":"bc","node_ids":["b","c"],"tunnel_ids":[7]}]}
        splines={"tunnels":[{"tunnel_id":7,"points":[[0,0,0],[5,0,0],[10,0,0]]}]}
        records=edge_arc_incidence(graph,splines)
        at_b=[item for item in records if item.node_id=="b"]
        self.assertEqual(len(at_b),2)
        self.assertAlmostEqual(at_b[0].arc_m,5.0)
        self.assertAlmostEqual(at_b[1].arc_m,5.0)
        self.assertAlmostEqual(sum(first*second for first,second in zip(at_b[0].away_tangent,at_b[1].away_tangent)), -1.0)

    def test_sampled_tube_field_and_ray_exit(self):
        samples={1:np.column_stack((np.linspace(-5,5,1001),np.zeros(1001),np.zeros(1001)))}
        field=SampledSweptTubeField(samples,{1:2.0},0.01)
        self.assertAlmostEqual(float(field.signed_distance(np.array([[0,0,0]]))[0]),-2.0,places=6)
        hit=field.ray_exit_distances(np.array([[0,0,0.]]),np.array([[0,1,0.]]),maximum_m=4)
        self.assertLess(abs(float(hit[0])-2.0),0.02)

    def test_window_boundary_continuity_detects_nonincident_tube(self):
        line=np.column_stack((np.linspace(-5,5,1001),np.zeros(1001),np.zeros(1001)))
        other=line+np.array([0,1.5,0])
        field=SampledSweptTubeField({1:line,2:other},{1:1.0,2:1.0},0.01)
        window=build_local_union_windows({"nodes":[{"id":"n","xyz":[0,0,0],"degree":3,"incident_tunnel_ids":[1]}]},{"tunnels":[{"tunnel_id":1,"radius_m":1.0},{"tunnel_id":2,"radius_m":1.0}]})[0]
        self.assertFalse(window_boundary_continuity(field,window,256)["passed"])
        self.assertFalse(window_volume_isolation(field,window,512)["passed"])

    def test_layered_field_excludes_stacked_nonincident_tunnel(self):
        line=np.column_stack((np.linspace(-5,5,1001),np.zeros(1001),np.zeros(1001)))
        stacked=line+np.array([0,0,1.5])
        base=SampledSweptTubeField({1:line,2:stacked},{1:1.0,2:1.0},0.01)
        field=LayeredHybridField(base,())
        point=np.array([[0.0,0.0,0.0]])
        self.assertAlmostEqual(float(field.signed_distance(point,np.array([2]))[0]),0.5,places=6)
        self.assertAlmostEqual(float(field.signed_distance(point,np.array([1]))[0]),-1.0,places=6)

    def test_layered_field_uses_incident_union_only_for_compatible_layer(self):
        first=np.column_stack((np.linspace(-5,5,1001),np.zeros(1001),np.zeros(1001)))
        second=np.column_stack((np.zeros(1001),np.linspace(-5,5,1001),np.zeros(1001)))
        remote=first+np.array([0,0,0.2])
        base=SampledSweptTubeField({1:first,2:second,3:remote},{1:0.4,2:0.4,3:0.4},0.01)
        window=LocalUnionWindow("n",(0.0,0.0,0.0),(1,2),1.1)
        field=LayeredHybridField(base,(window,))
        point=np.array([[0.0,0.8,0.0]])
        self.assertLess(float(field.signed_distance(point,np.array([1]))[0]),-0.39)
        self.assertGreater(float(field.signed_distance(point,np.array([3]))[0]),0.0)
        self.assertTrue(layered_window_semantics_audit(field,window,128)["passed"])

    def test_layered_ray_exit_stays_on_selected_layer(self):
        line=np.column_stack((np.linspace(-5,5,1001),np.zeros(1001),np.zeros(1001)))
        stacked=line+np.array([0,0,1.5])
        field=LayeredHybridField(SampledSweptTubeField({1:line,2:stacked},{1:1.0,2:1.0},0.01),())
        hit=field.ray_exit_distances(
            np.array([[0.0,0.0,1.5]]),np.array([[0.0,0.0,1.0]]),np.array([2]),maximum_m=4
        )
        self.assertLess(abs(float(hit[0])-1.0),0.02)

    def test_layered_ray_exit_refines_coarse_first_outside_bracket(self):
        line=np.column_stack((np.linspace(-5,5,101),np.zeros(101),np.zeros(101)))
        field=LayeredHybridField(SampledSweptTubeField({1:line},{1:1.0},0.1),())
        origins=np.array([[0.0,0.0,0.013],[0.0,0.0,-0.027]])
        directions=np.array([[0.0,0.0,1.0],[0.0,0.0,-1.0]])
        first=field.ray_exit_distances(origins,directions,np.array([1,1]),maximum_m=4)
        second=field.ray_exit_distances(origins,directions,np.array([1,1]),maximum_m=4)
        np.testing.assert_allclose(first,[0.987,0.973],atol=1e-9,rtol=0.0)
        np.testing.assert_array_equal(first,second)

    def test_continuous_transition_has_exact_c1_boundary_weights(self):
        np.testing.assert_allclose(ContinuousLayeredHybridField.incident_weight(np.array([0.0,1.0])),[1.0,0.0])
        np.testing.assert_allclose(ContinuousLayeredHybridField.incident_weight_derivative(np.array([0.0,1.0])),[0.0,0.0])

    def test_continuous_transition_seam_and_dispatch_contract(self):
        first=np.column_stack((np.linspace(-5,5,1001),np.zeros(1001),np.zeros(1001)))
        second=np.column_stack((np.zeros(1001),np.linspace(-5,5,1001),np.zeros(1001)))
        remote=first+np.array([0,0,0.2])
        base=SampledSweptTubeField({1:first,2:second,3:remote},{1:0.4,2:0.4,3:0.4},0.01)
        window=LocalUnionWindow("n",(0.0,0.0,0.0),(1,2),1.1)
        field=ContinuousLayeredHybridField(base,(window,))
        self.assertTrue(continuous_transition_seam_audit(field,window,256)["passed"])
        self.assertTrue(continuous_transition_semantics_audit(field,window,256)["passed"])
        center=np.array([[0.0,0.0,0.0]])
        self.assertAlmostEqual(float(field.signed_distance(center,np.array([1]))[0]),float(base.signed_distance(center,(1,2))[0]))

    @staticmethod
    def _support_documents():
        graph={
            "nodes":[
                {"id":"n","xyz":[0,0,0],"degree":3,"incident_tunnel_ids":[1,2,3]},
                {"id":"a","xyz":[4,0,0],"degree":1,"incident_tunnel_ids":[1]},
                {"id":"b","xyz":[0,4,0],"degree":1,"incident_tunnel_ids":[2]},
                {"id":"c","xyz":[-4,0,0],"degree":1,"incident_tunnel_ids":[3]},
            ],
            "edges":[
                {"id":"na","node_ids":["n","a"],"tunnel_ids":[1]},
                {"id":"nb","node_ids":["n","b"],"tunnel_ids":[2]},
                {"id":"nc","node_ids":["n","c"],"tunnel_ids":[3]},
            ],
        }
        splines={"tunnels":[
            {"tunnel_id":1,"points":[[0,0,0],[4,0,0]]},
            {"tunnel_id":2,"points":[[0,0,0],[0,4,0]]},
            {"tunnel_id":3,"points":[[0,0,0],[-4,0,0]]},
        ]}
        geometry={"fta_distance_m":-1.0,"tunnels":[
            {"tunnel_id":1,"radius_m":1.0},
            {"tunnel_id":2,"radius_m":1.0},
            {"tunnel_id":3,"radius_m":1.0},
        ]}
        return graph,splines,geometry

    def test_route_support_preserves_incident_arcs_and_reverse_traversal(self):
        graph,splines,geometry=self._support_documents()
        windows=build_local_union_windows(graph,geometry)
        field=RouteConditionedSupportField.from_documents(graph,splines,geometry,windows)
        self.assertEqual(len(field.incident_arcs_by_node["n"]),3)
        self.assertEqual({arc.edge_id for arc in field.incident_arcs_by_node["n"]},{"na","nb","nc"})
        points=np.asarray([[2.0,0.0,0.0],[2.0,0.0,0.0]])
        queries=(RouteSupportQuery(1,"na"),RouteSupportQuery(1,"na"))
        np.testing.assert_allclose(field.support_heights(points,queries),[-1.0,-1.0])

    def test_route_support_isolates_stacked_nonincident_spline(self):
        graph,splines,geometry=self._support_documents()
        splines["tunnels"].append({"tunnel_id":4,"points":[[0,0,3],[4,0,3]]})
        geometry["tunnels"].append({"tunnel_id":4,"radius_m":1.0})
        field=RouteConditionedSupportField.from_documents(
            graph,splines,geometry,build_local_union_windows(graph,geometry)
        )
        point=np.asarray([[2.0,0.0,0.0]])
        self.assertAlmostEqual(field.support_heights(point,(RouteSupportQuery(1,"na"),))[0],-1.0)
        self.assertAlmostEqual(field.support_heights(point,(RouteSupportQuery(4,"remote"),))[0],2.0)

    def test_route_support_has_c1_window_boundary_and_exact_floor_distance(self):
        graph,splines,geometry=self._support_documents()
        windows=build_local_union_windows(graph,geometry)
        field=RouteConditionedSupportField.from_documents(graph,splines,geometry,windows)
        window=windows[0]
        boundary=np.asarray([[window.radius_m,0.0,0.0]])
        query=(RouteSupportQuery(1,"na","n"),)
        self.assertAlmostEqual(field.support_heights(boundary,query)[0],-1.0)
        self.assertAlmostEqual(field.incident_weight(np.asarray([1.0]))[0],0.0)
        self.assertAlmostEqual(field.incident_weight_derivative(np.asarray([1.0]))[0],0.0)
        sensor=boundary.copy(); sensor[:,2]=0.0
        self.assertAlmostEqual(field.downward_distances(sensor,boundary,query)[0],1.0)

    def test_route_support_rejects_nonincident_window_membership(self):
        graph,splines,geometry=self._support_documents()
        splines["tunnels"].append({"tunnel_id":4,"points":[[0,0,3],[4,0,3]]})
        geometry["tunnels"].append({"tunnel_id":4,"radius_m":1.0})
        field=RouteConditionedSupportField.from_documents(
            graph,splines,geometry,build_local_union_windows(graph,geometry)
        )
        with self.assertRaisesRegex(ValueError,"non-incident traversal"):
            field.support_heights(np.asarray([[0.0,0.0,0.0]]),(RouteSupportQuery(4,"remote","n"),))


if __name__ == "__main__":
    unittest.main()
