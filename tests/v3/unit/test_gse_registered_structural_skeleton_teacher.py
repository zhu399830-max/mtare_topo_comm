import numpy as np

from mtare_topo.teacher.gse_registered_structural_skeleton_teacher import (
    causal_mesh_visibility,
    ego_connected_component,
    materialize_visible_skeleton,
    sample_physical_skeleton,
)


def _documents():
    graph = {
        "nodes": [
            {"id": "a", "xyz": [0, 0, 0], "degree": 1},
            {"id": "b", "xyz": [5, 0, 0], "degree": 2},
            {"id": "c", "xyz": [10, 0, 0], "degree": 1},
        ],
        "edges": [
            {"id": "e0", "node_ids": ["a", "b"], "tunnel_ids": [1]},
            {"id": "e1", "node_ids": ["b", "c"], "tunnel_ids": [1]},
        ],
    }
    points = [[float(x), 0.0, 0.0] for x in np.arange(0.0, 10.5, 0.5)]
    splines = {"tunnels": [{"tunnel_id": 1, "points": points}]}
    geometry = {"tunnels": [{"tunnel_id": 1, "radius_m": 3.0}]}
    return graph, splines, geometry


def test_physical_sampling_preserves_edge_identity_and_shared_node():
    sampled = sample_physical_skeleton(*_documents(), spacing_m=1.0)
    assert sampled.edge_identity == ("e0", "e1")
    # Six samples per five-metre edge share node b: 6 + 6 - 1.
    assert len(sampled.node_xyz_world_m) == 11
    assert len(sampled.segment_node_indices) == 10


def test_causal_mesh_visibility_unions_past_frames_and_respects_vertical_fov():
    points = np.asarray([[5.0, 0.0, 0.0], [5.0, 0.0, 5.0], [60.0, 0.0, 0.0]])
    sensors = np.asarray([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]])
    def no_occlusion(origins, directions):
        return np.full(len(origins), np.inf)
    visible = causal_mesh_visibility(points, sensors, np.zeros(2), no_occlusion)
    assert visible.tolist() == [True, False, True]
    def wall_at_two_metres(origins, directions):
        return np.full(len(origins), 2.0)
    assert not causal_mesh_visibility(points[:1], sensors[:1], np.zeros(1), wall_at_two_metres)[0]


def test_visible_component_never_bridges_an_invisible_gap():
    sampled = sample_physical_skeleton(*_documents(), spacing_m=1.0)
    visible = np.ones(len(sampled.node_xyz_world_m), dtype=bool)
    ordered = np.argsort(sampled.node_xyz_world_m[:, 0])
    visible[ordered[5]] = False
    component = ego_connected_component(sampled, visible, np.asarray([2.0, 0.0, 0.0]))
    assert np.max(sampled.node_xyz_world_m[component, 0]) < 5.0
    skeleton = materialize_visible_skeleton(
        sampled, component, np.asarray([2.0, 0.0, 0.0]),
        np.asarray([2.0, 0.0, 1.0]), 0.0,
    )
    assert len(skeleton.node_xyz_current_m) == int(component.sum())
    assert len(skeleton.edge_node_indices) == int(component.sum()) - 1
