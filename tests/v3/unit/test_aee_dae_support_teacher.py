from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np


SRC_ROOT = Path(__file__).resolve().parents[3] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from mtare_topo.oracle.oriented_dae_support import (  # noqa: E402
    DAEUpwardSupportIndex,
    OrientedTriangleMesh,
    load_collada_triangle_mesh,
    load_gazebo_collision_mesh_binding,
)


def collada(vertices: list[tuple[float, float, float]], triangles: list[tuple[int, int, int]], matrix: np.ndarray | None = None) -> str:
    transform = np.eye(4) if matrix is None else np.asarray(matrix, dtype=float)
    positions = " ".join(str(value) for vertex in vertices for value in vertex)
    indices = " ".join(f"{index} 0 0" for triangle in triangles for index in triangle)
    matrix_text = " ".join(str(value) for value in transform.ravel())
    return f'''<?xml version="1.0" encoding="utf-8"?>
<COLLADA xmlns="http://www.collada.org/2005/11/COLLADASchema" version="1.4.1">
  <asset><unit name="meter" meter="1"/><up_axis>Z_UP</up_axis></asset>
  <library_geometries><geometry id="geometry"><mesh>
    <source id="positions"><float_array id="positions-array" count="{len(vertices) * 3}">{positions}</float_array>
      <technique_common><accessor source="#positions-array" count="{len(vertices)}" stride="3"><param name="X"/><param name="Y"/><param name="Z"/></accessor></technique_common>
    </source>
    <source id="normals"><float_array id="normals-array" count="3">0 0 1</float_array>
      <technique_common><accessor source="#normals-array" count="1" stride="3"><param name="X"/><param name="Y"/><param name="Z"/></accessor></technique_common>
    </source>
    <source id="uv"><float_array id="uv-array" count="2">0 0</float_array>
      <technique_common><accessor source="#uv-array" count="1" stride="2"><param name="S"/><param name="T"/></accessor></technique_common>
    </source>
    <vertices id="vertices"><input semantic="POSITION" source="#positions"/></vertices>
    <triangles count="{len(triangles)}"><input semantic="VERTEX" source="#vertices" offset="0"/><input semantic="NORMAL" source="#normals" offset="1"/><input semantic="TEXCOORD" source="#uv" offset="2"/><p>{indices}</p></triangles>
  </mesh></geometry></library_geometries>
  <library_visual_scenes><visual_scene id="scene"><node id="node"><matrix>{matrix_text}</matrix><instance_geometry url="#geometry"/></node></visual_scene></library_visual_scenes>
  <scene><instance_visual_scene url="#scene"/></scene>
</COLLADA>'''


class DAESupportTeacherTests(unittest.TestCase):
    def load(self, document: str) -> OrientedTriangleMesh:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fixture.dae"
            path.write_text(document, encoding="utf-8")
            return load_collada_triangle_mesh(path)

    def test_upward_floor_is_selected_over_downward_underside(self):
        mesh = self.load(
            collada(
                [(0, 0, 0), (2, 0, 0), (0, 2, 0), (0, 0, -0.2), (0, 2, -0.2), (2, 0, -0.2)],
                [(0, 1, 2), (3, 4, 5)],
            )
        )
        self.assertEqual(mesh.triangle_count, 2)
        self.assertEqual(mesh.upward_triangle_count, 1)
        index = DAEUpwardSupportIndex.from_mesh(mesh, np.asarray([0.0, 0.0]), (3, 3), 1.0)
        np.testing.assert_array_equal(index.z_values(0), np.asarray([0.0], dtype=np.float32))

    def test_downward_only_surface_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "no upward-facing"):
            self.load(collada([(0, 0, 0), (0, 2, 0), (2, 0, 0)], [(0, 1, 2)]))

    def test_ramp_support_interpolates_height(self):
        mesh = self.load(collada([(0, 0, 0), (2, 0, 2), (0, 2, 0)], [(0, 1, 2)]))
        index = DAEUpwardSupportIndex.from_mesh(mesh, np.asarray([0.0, 0.0]), (3, 3), 1.0)
        np.testing.assert_array_equal(index.z_values(0), np.asarray([0.0], dtype=np.float32))
        np.testing.assert_array_equal(index.z_values(1), np.asarray([1.0], dtype=np.float32))

    def test_full_instance_transform_is_applied_before_orientation(self):
        transform = np.asarray(
            [[0.0, -2.0, 0.0, 10.0], [3.0, 0.0, 0.0, 20.0], [0.0, 0.0, 1.5, 4.0], [0.0, 0.0, 0.0, 1.0]]
        )
        first = self.load(collada([(0, 0, 0), (1, 0, 0), (0, 1, 0)], [(0, 1, 2)], transform))
        second = self.load(collada([(0, 0, 0), (1, 0, 0), (0, 1, 0)], [(0, 1, 2)], transform))
        expected = np.asarray([[10.0, 20.0, 4.0], [10.0, 23.0, 4.0], [8.0, 20.0, 4.0]])
        np.testing.assert_array_equal(first.vertices_xyz_m, expected)
        np.testing.assert_array_equal(first.vertices_xyz_m, second.vertices_xyz_m)
        np.testing.assert_array_equal(first.triangle_normals, second.triangle_normals)
        np.testing.assert_array_equal(first.triangle_normals, [[0.0, 0.0, 1.0]])

    def test_ambiguous_nearest_support_policy_can_detect_exact_tie(self):
        # The index preserves distinct stacked upward layers for the oracle to
        # resolve against the expected sensor height; it never collapses them.
        vertices = np.asarray(
            [[0, 0, 0], [2, 0, 0], [0, 2, 0], [0, 0, 2], [2, 0, 2], [0, 2, 2]], dtype=float
        )
        triangles = np.asarray([[0, 1, 2], [3, 4, 5]], dtype=np.int64)
        normals = np.asarray([[0, 0, 1], [0, 0, 1]], dtype=float)
        index = DAEUpwardSupportIndex.from_mesh(
            OrientedTriangleMesh(vertices, triangles, normals), np.asarray([0.0, 0.0]), (3, 3), 1.0
        )
        np.testing.assert_array_equal(index.z_values(0), np.asarray([0.0, 2.0], dtype=np.float32))

    def test_sheet_identity_merges_exact_duplicate_seams_but_separates_stacked_surface(self):
        # Blender may duplicate position indices on a continuous triangle
        # seam.  Exact world coordinates still identify that seam without a
        # tunable distance tolerance.  A disconnected sheet in the same DAE
        # primitive must retain a different identity.
        mesh = self.load(
            collada(
                [
                    (0, 0, 0),
                    (2, 0, 0),
                    (0, 2, 0),
                    (2, 0, 0),
                    (2, 2, 0),
                    (0, 2, 0),
                    (0, 0, 2),
                    (2, 0, 2),
                    (0, 2, 2),
                ],
                [(0, 1, 2), (3, 4, 5), (6, 7, 8)],
            )
        )
        first, second, stacked = mesh.triangle_surface_ids.tolist()
        self.assertEqual(first, second)
        self.assertNotEqual(first, stacked)
        self.assertEqual(len(mesh.surface_names), 2)

    def test_singular_instance_transform_is_rejected(self):
        transform = np.eye(4)
        transform[2, 2] = 0.0
        with self.assertRaisesRegex(ValueError, "invertible"):
            self.load(collada([(0, 0, 0), (1, 0, 0), (0, 1, 0)], [(0, 1, 2)], transform))

    def test_exact_zero_area_exporter_residue_is_skipped_and_recorded(self):
        mesh = self.load(
            collada(
                [(0, 0, 0), (2, 0, 0), (0, 2, 0), (3, 3, 1)],
                [(0, 1, 2), (3, 3, 3)],
            )
        )
        self.assertEqual(mesh.source_triangle_count, 2)
        self.assertEqual(mesh.triangle_count, 1)
        self.assertEqual(mesh.skipped_zero_area_source_indices, (1,))
        self.assertEqual(mesh.skipped_zero_area_source_faces, ("geometry:primitive0:face1",))
        index = DAEUpwardSupportIndex.from_mesh(mesh, np.asarray([0.0, 0.0]), (3, 3), 1.0)
        provenance = index.provenance()
        self.assertEqual(provenance["source_mesh_triangles"], 2)
        self.assertEqual(provenance["mesh_triangles"], 1)
        self.assertEqual(provenance["skipped_zero_area_triangle_indices"], [1])
        self.assertEqual(provenance["skipped_zero_area_source_faces"], ["geometry:primitive0:face1"])

    def test_all_zero_area_faces_still_fail_closed(self):
        with self.assertRaisesRegex(ValueError, "no nonzero-area"):
            self.load(collada([(0, 0, 0)], [(0, 0, 0)]))

    def test_numerically_tilted_vertical_face_is_not_upward_support(self):
        # The second face is physically vertical.  Its 1e-18 XY determinant
        # models transform round-off seen in the real tunnel DAE.
        vertices = np.asarray(
            [[0, 0, 0], [2, 0, 0], [0, 2, 0], [5, 0, 0], [5, 1e-18, 2], [5, 0, 2]],
            dtype=float,
        )
        triangles = np.asarray([[0, 1, 2], [3, 4, 5]], dtype=np.int64)
        faces = vertices[triangles]
        cross = np.cross(faces[:, 1] - faces[:, 0], faces[:, 2] - faces[:, 0])
        normals = cross / np.linalg.norm(cross, axis=1)[:, None]
        mesh = OrientedTriangleMesh(vertices, triangles, normals)
        self.assertEqual(mesh.upward_triangle_count, 1)
        index = DAEUpwardSupportIndex.from_mesh(mesh, np.asarray([0.0, 0.0]), (7, 3), 1.0)
        self.assertEqual(index.upward_triangles, 1)

    def test_gazebo_world_model_collision_chain_is_applied(self):
        world = '''<sdf version="1.6"><world name="default"><include><uri>model://fixture</uri></include></world></sdf>'''
        model = '''<sdf version="1.6"><model name="fixture"><pose>-22 92.5 0 0 0 0</pose><link name="link"><collision name="collision"><geometry><mesh><uri>model://fixture/meshes/fixture.dae</uri></mesh></geometry></collision></link></model></sdf>'''
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            world_path = root / "fixture.world"
            model_path = root / "model.sdf"
            dae_path = root / "fixture.dae"
            world_path.write_text(world, encoding="utf-8")
            model_path.write_text(model, encoding="utf-8")
            dae_path.write_text(
                collada([(0, 0, 0), (1, 0, 0), (0, 1, 0)], [(0, 1, 2)]), encoding="utf-8"
            )
            binding = load_gazebo_collision_mesh_binding(
                world_path,
                model_path,
                "model://fixture",
                "model://fixture/meshes/fixture.dae",
            )
            mesh = load_collada_triangle_mesh(dae_path, binding.world_from_mesh)
        np.testing.assert_array_equal(binding.model_pose_xyz_rpy, [-22.0, 92.5, 0, 0, 0, 0])
        np.testing.assert_array_equal(
            mesh.vertices_xyz_m,
            [[-22.0, 92.5, 0.0], [-21.0, 92.5, 0.0], [-22.0, 93.5, 0.0]],
        )

    def test_gazebo_chain_composes_include_link_collision_pose_and_scale(self):
        world = '''<sdf version="1.6"><world name="default"><include><uri>model://fixture</uri><pose>1 0 0 0 0 0</pose></include></world></sdf>'''
        model = '''<sdf version="1.6"><model name="fixture"><pose>0 2 0 0 0 0</pose><link name="link"><pose>0 0 3 0 0 0</pose><collision name="collision"><pose>4 0 0 0 0 0</pose><geometry><mesh><uri>model://fixture/meshes/fixture.dae</uri><scale>2 3 4</scale></mesh></geometry></collision></link></model></sdf>'''
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "world").write_text(world, encoding="utf-8")
            (root / "model").write_text(model, encoding="utf-8")
            binding = load_gazebo_collision_mesh_binding(
                root / "world", root / "model", "model://fixture", "model://fixture/meshes/fixture.dae"
            )
        np.testing.assert_array_equal(
            binding.world_from_mesh,
            [[2, 0, 0, 5], [0, 3, 0, 2], [0, 0, 4, 3], [0, 0, 0, 1]],
        )

    def test_gazebo_relative_frame_and_wrong_mesh_uri_fail_closed(self):
        world = '''<sdf version="1.6"><world name="default"><include><uri>model://fixture</uri></include></world></sdf>'''
        relative_model = '''<sdf version="1.6"><model name="fixture"><pose relative_to="x">0 0 0 0 0 0</pose><link name="link"><collision name="collision"><geometry><mesh><uri>model://fixture/meshes/fixture.dae</uri></mesh></geometry></collision></link></model></sdf>'''
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "world").write_text(world, encoding="utf-8")
            (root / "model").write_text(relative_model, encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "relative_to"):
                load_gazebo_collision_mesh_binding(
                    root / "world", root / "model", "model://fixture", "model://fixture/meshes/fixture.dae"
                )
            (root / "model").write_text(
                relative_model.replace(' relative_to="x"', ""), encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "URI"):
                load_gazebo_collision_mesh_binding(
                    root / "world", root / "model", "model://fixture", "model://fixture/meshes/wrong.dae"
                )


if __name__ == "__main__":
    unittest.main()
