from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np


SRC_ROOT = Path(__file__).resolve().parents[3] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from mtare_topo.oracle.layered_gt_map import (  # noqa: E402
    LayeredGTMapConfig,
    LayeredGTMapOracle,
    LocalLayerEvidence,
    _MultilayerLocalState,
)
from mtare_topo.oracle.oriented_dae_support import (  # noqa: E402
    DAEUpwardSupportIndex,
    OrientedTriangleMesh,
)


def dense_plane(z_function, extent: float = 12.0, resolution: float = 0.2) -> np.ndarray:
    coordinates = np.arange(0.0, extent + resolution * 0.5, resolution)
    yy, xx = np.meshgrid(coordinates, coordinates, indexing="ij")
    zz = z_function(xx, yy)
    return np.column_stack((xx.ravel(), yy.ravel(), np.asarray(zz).ravel())).astype(np.float32)


def rectangle_mesh(z_values: tuple[float, float, float, float]) -> tuple[np.ndarray, np.ndarray]:
    vertices = np.asarray(
        [[0, 0, z_values[0]], [12, 0, z_values[1]], [12, 12, z_values[2]], [0, 12, z_values[3]]],
        dtype=np.float64,
    )
    triangles = np.asarray([[0, 1, 2], [0, 2, 3]], dtype=np.int64)
    return vertices, triangles


def oriented_oracle(
    obstacle_points: np.ndarray,
    support_surfaces: list[tuple[np.ndarray, np.ndarray]],
) -> LayeredGTMapOracle:
    config = LayeredGTMapConfig(local_size_m=12.0, maximum_reach_m=6.0, exit_minimum_reach_m=3.0)
    base = LayeredGTMapOracle(obstacle_points, config)
    vertices: list[np.ndarray] = []
    triangles: list[np.ndarray] = []
    surface_ids: list[np.ndarray] = []
    offset = 0
    for surface_id, (surface_vertices, surface_triangles) in enumerate(support_surfaces):
        vertices.append(surface_vertices)
        triangles.append(surface_triangles + offset)
        surface_ids.append(np.full(len(surface_triangles), surface_id, dtype=np.int64))
        offset += len(surface_vertices)
    all_vertices = np.concatenate(vertices)
    all_triangles = np.concatenate(triangles)
    faces = all_vertices[all_triangles]
    cross = np.cross(faces[:, 1] - faces[:, 0], faces[:, 2] - faces[:, 0])
    normals = cross / np.linalg.norm(cross, axis=1)[:, None]
    support = DAEUpwardSupportIndex.from_mesh(
        OrientedTriangleMesh(
            all_vertices,
            all_triangles,
            normals,
            triangle_surface_ids=np.concatenate(surface_ids),
            surface_names=tuple(f"surface{index}" for index in range(len(surface_ids))),
        ),
        base.origin_xy,
        base.shape_xy,
        config.resolution_m,
    )
    return LayeredGTMapOracle(obstacle_points, config, oriented_support_index=support)


class AEEMultilayerSupportGraphTests(unittest.TestCase):
    def test_horizontal_floor_and_overlapping_ramp_do_not_require_tie_break(self):
        lower_points = dense_plane(lambda x, y: np.zeros_like(x))
        ramp_points = dense_plane(lambda x, y: 0.3 + 0.1 * (x - 6.0))
        lower_mesh = rectangle_mesh((0.0, 0.0, 0.0, 0.0))
        ramp_mesh = rectangle_mesh((-0.3, 0.9, 0.9, -0.3))
        oracle = oriented_oracle(np.r_[lower_points, ramp_points], [lower_mesh, ramp_mesh])
        prediction = oracle.predict((6.0, 6.0, 1.05), 0.0)
        self.assertAlmostEqual(prediction.evidence.selected_center_support_z_m, 0.3, places=5)
        self.assertGreater(prediction.evidence.multilayer_cell_count, 0)
        self.assertGreater(prediction.evidence.support_node_count, prediction.evidence.reachable_support_node_count)
        self.assertEqual(prediction.evidence.maximum_reachable_layers_per_cell, 2)
        self.assertGreaterEqual(prediction.exit_count, 1)

    def test_center_height_tie_still_fails_closed(self):
        obstacle_points = dense_plane(lambda x, y: np.zeros_like(x))
        lower_mesh = rectangle_mesh((0.0, 0.0, 0.0, 0.0))
        upper_mesh = rectangle_mesh((0.5, 0.5, 0.5, 0.5))
        oracle = oriented_oracle(obstacle_points, [lower_mesh, upper_mesh])
        with self.assertRaisesRegex(RuntimeError, "ambiguous upward-facing support"):
            oracle.local_layer((6.0, 6.0, 1.0), 0.0)

    def test_direction_ray_cannot_jump_between_reachable_layers(self):
        size = 60
        center = size // 2
        counts = np.zeros(size * size, dtype=np.int64)
        cells: list[int] = []
        heights: list[float] = []
        for column in range(center, center + 10):
            cells.append(center * size + column)
            heights.append(0.0)
        for column in range(center + 10, center + 25):
            cells.append(center * size + column)
            heights.append(4.0)
        for cell in cells:
            counts[cell] = 1
        offsets = np.r_[0, np.cumsum(counts)]
        node_z = np.empty(len(cells), dtype=np.float64)
        for cell, height in zip(cells, heights):
            node_z[offsets[cell]] = height
        valid = counts.reshape(size, size) > 0
        evidence = LocalLayerEvidence(
            support_height_m=np.where(valid, 0.0, np.nan).astype(np.float32),
            support_valid=valid,
            obstacle=np.zeros((size, size), dtype=bool),
            inflated_obstacle=np.zeros((size, size), dtype=bool),
            connected_traversable=valid,
            snapped_center_rc=(center, center),
            expected_support_z_m=0.0,
            selected_center_support_z_m=0.0,
        )
        state = _MultilayerLocalState(
            evidence=evidence,
            cell_node_offsets=offsets,
            node_z_m=node_z,
            node_reachable=np.ones(len(node_z), dtype=bool),
        )
        config = LayeredGTMapConfig(
            resolution_m=0.2,
            local_size_m=12.0,
            maximum_reach_m=6.0,
            exit_minimum_reach_m=3.0,
        )
        oracle = LayeredGTMapOracle(dense_plane(lambda x, y: np.zeros_like(x)), config)
        prediction = oracle._predict_multilayer(state)
        self.assertLess(prediction.reachable_distance_m[0], 2.1)
        self.assertGreater(prediction.reachable_distance_m[0], 1.5)

    def test_multilayer_result_is_deterministic_under_ply_permutation(self):
        lower = dense_plane(lambda x, y: np.zeros_like(x))
        upper = dense_plane(lambda x, y: np.full_like(x, 4.0))
        lower_mesh = rectangle_mesh((0.0, 0.0, 0.0, 0.0))
        upper_mesh = rectangle_mesh((4.0, 4.0, 4.0, 4.0))
        points = np.r_[lower, upper]
        first = oriented_oracle(points, [lower_mesh, upper_mesh]).predict((6.0, 6.0, 0.75), 0.0)
        permutation = np.random.default_rng(17).permutation(len(points))
        second = oriented_oracle(points[permutation], [lower_mesh, upper_mesh]).predict(
            (6.0, 6.0, 0.75), 0.0
        )
        np.testing.assert_array_equal(first.direction_logits, second.direction_logits)
        np.testing.assert_array_equal(first.reachable_distance_m, second.reachable_distance_m)
        self.assertEqual(first.evidence.support_node_count, second.evidence.support_node_count)
        self.assertEqual(
            first.evidence.reachable_support_node_count,
            second.evidence.reachable_support_node_count,
        )


if __name__ == "__main__":
    unittest.main()
