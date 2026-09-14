"""Synthetic execution-boundary check, not a real-scene qualification."""
import numpy as np


def test_explicit_thread_budgets_preserve_simple_intersections():
    try:
        import open3d as o
    except ImportError:
        import pytest
        pytest.skip('Open3D sidecar required')
    vertices = o.core.Tensor([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=o.core.Dtype.Float32)
    triangles = o.core.Tensor([[0, 1, 2]], dtype=o.core.Dtype.UInt32)
    rays = o.core.Tensor([[.2, .2, 1, 0, 0, -1], [2, 2, 1, 0, 0, -1]], dtype=o.core.Dtype.Float32)
    results = []
    for threads in (1, 2):
        scene = o.t.geometry.RaycastingScene(nthreads=threads)
        scene.add_triangles(vertices, triangles)
        hits = {key: value.numpy().copy() for key, value in scene.list_intersections(rays, nthreads=threads).items()}
        assert hits['ray_ids'].tolist() == [0]
        assert hits['ray_splits'].tolist() == [0, 1, 1]
        # Analytic distance is 1; Embree executes float32, not exact arithmetic.
        assert abs(float(hits['t_hit'][0]) - 1.) <= np.finfo(np.float32).eps
        results.append(hits)
    for key in results[0]:
        np.testing.assert_array_equal(results[0][key], results[1][key])
