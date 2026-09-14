"""Deterministic upward-facing support layers from an AEE COLLADA scene.

The AEE preview PLY contains positions only, so two sides of a garage slab
cannot be distinguished by orientation.  Gazebo uses the sibling DAE as its
collision surface.  This module reads the exact, deliberately narrow COLLADA
schema used by the frozen AEE assets and rasterizes only upward-facing
triangles onto the same XY lattice as :mod:`layered_gt_map`.

The loader resolves the active visual scene and applies every finite,
invertible 4x4 instance transform before computing face orientation.  It fails
closed on units, axes, unsupported transform encodings or primitives,
ambiguous sources/instances and non-finite data.  Exactly zero-area exporter
residues are removed by an explicit, provenance-preserving rule; no nonzero
area threshold is used.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np


COLLADA_NAMESPACE = "http://www.collada.org/2005/11/COLLADASchema"
_NS = {"c": COLLADA_NAMESPACE}
UPWARD_NORMAL_Z_NUMERICAL_TOLERANCE = 64.0 * np.finfo(np.float64).eps


def _upward_mask(normals: np.ndarray) -> np.ndarray:
    """Exclude numerically tilted vertical faces without a physical angle knob."""

    return np.asarray(normals)[:, 2] > UPWARD_NORMAL_Z_NUMERICAL_TOLERANCE


def _readonly(values: np.ndarray) -> np.ndarray:
    result = np.ascontiguousarray(values)
    result.setflags(write=False)
    return result


def _numbers(element: ET.Element, dtype: np.dtype | type) -> np.ndarray:
    values = np.fromstring(element.text or "", sep=" ", dtype=dtype)
    if "count" in element.attrib and values.size != int(element.attrib["count"]):
        raise ValueError("COLLADA array count does not match its payload")
    return values


def _validated_affine(values: np.ndarray, label: str) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
        raise ValueError(f"{label} must be a finite 4x4 matrix")
    tolerance = 64.0 * np.finfo(np.float64).eps
    if not np.allclose(matrix[3], [0.0, 0.0, 0.0, 1.0], rtol=0.0, atol=tolerance):
        raise ValueError(f"{label} must be affine")
    if abs(float(np.linalg.det(matrix[:3, :3]))) <= np.finfo(np.float64).tiny:
        raise ValueError(f"{label} must be invertible")
    return matrix


def _pose_matrix(xyz_rpy: np.ndarray) -> np.ndarray:
    x, y, z, roll, pitch, yaw = np.asarray(xyz_rpy, dtype=np.float64)
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rotation = np.asarray(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ],
        dtype=np.float64,
    )
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = rotation
    result[:3, 3] = [x, y, z]
    return result


def _sdf_pose(parent: ET.Element, label: str) -> tuple[np.ndarray, np.ndarray]:
    poses = parent.findall("pose")
    if not poses:
        values = np.zeros(6, dtype=np.float64)
        return values, np.eye(4, dtype=np.float64)
    if len(poses) != 1 or poses[0].attrib:
        raise ValueError(f"{label} pose must be unique and must not use relative_to")
    values = np.fromstring(poses[0].text or "", sep=" ", dtype=np.float64)
    if values.size != 6 or not np.all(np.isfinite(values)):
        raise ValueError(f"{label} pose must contain finite xyz/rpy")
    return values, _pose_matrix(values)


@dataclass(frozen=True)
class GazeboCollisionMeshBinding:
    model_name: str
    include_uri: str
    mesh_uri: str
    include_pose_xyz_rpy: np.ndarray
    model_pose_xyz_rpy: np.ndarray
    link_pose_xyz_rpy: np.ndarray
    collision_pose_xyz_rpy: np.ndarray
    mesh_scale_xyz: np.ndarray
    world_from_mesh: np.ndarray

    def __post_init__(self) -> None:
        for name in (
            "include_pose_xyz_rpy",
            "model_pose_xyz_rpy",
            "link_pose_xyz_rpy",
            "collision_pose_xyz_rpy",
        ):
            values = np.asarray(getattr(self, name), dtype=np.float64)
            if values.shape != (6,) or not np.all(np.isfinite(values)):
                raise ValueError(f"{name} must be finite xyz/rpy")
            object.__setattr__(self, name, _readonly(values))
        scale = np.asarray(self.mesh_scale_xyz, dtype=np.float64)
        if scale.shape != (3,) or not np.all(np.isfinite(scale)) or np.any(scale <= 0.0):
            raise ValueError("Gazebo mesh scale must be finite and positive")
        object.__setattr__(self, "mesh_scale_xyz", _readonly(scale))
        object.__setattr__(
            self, "world_from_mesh", _readonly(_validated_affine(self.world_from_mesh, "world_from_mesh"))
        )

    def provenance(self) -> dict[str, str | list[float] | list[list[float]]]:
        return {
            "model_name": self.model_name,
            "include_uri": self.include_uri,
            "mesh_uri": self.mesh_uri,
            "include_pose_xyz_rpy": self.include_pose_xyz_rpy.tolist(),
            "model_pose_xyz_rpy": self.model_pose_xyz_rpy.tolist(),
            "link_pose_xyz_rpy": self.link_pose_xyz_rpy.tolist(),
            "collision_pose_xyz_rpy": self.collision_pose_xyz_rpy.tolist(),
            "mesh_scale_xyz": self.mesh_scale_xyz.tolist(),
            "world_from_mesh": self.world_from_mesh.tolist(),
        }


def load_gazebo_collision_mesh_binding(
    world_path: str | Path,
    model_sdf_path: str | Path,
    expected_include_uri: str,
    expected_mesh_uri: str,
) -> GazeboCollisionMeshBinding:
    """Resolve the strict Gazebo world→model→collision mesh coordinate chain."""

    world_root = ET.parse(Path(world_path)).getroot()
    if world_root.tag != "sdf" or world_root.attrib.get("version") != "1.6":
        raise ValueError("Gazebo world must be SDF 1.6")
    worlds = world_root.findall("world")
    if len(worlds) != 1:
        raise ValueError("Gazebo file must contain exactly one world")
    includes = [
        item for item in worlds[0].findall("include")
        if (item.findtext("uri") or "").strip() == expected_include_uri
    ]
    if len(includes) != 1:
        raise ValueError("Gazebo world mesh-model include is missing or ambiguous")
    include = includes[0]
    allowed_include_children = {"uri", "pose", "name", "static", "placement_frame"}
    if any(child.tag not in allowed_include_children for child in include):
        raise ValueError("Gazebo include contains an unsupported coordinate field")
    if include.find("placement_frame") is not None:
        raise ValueError("Gazebo include placement_frame is unsupported")
    include_pose, include_matrix = _sdf_pose(include, "include")

    model_root = ET.parse(Path(model_sdf_path)).getroot()
    if model_root.tag != "sdf" or model_root.attrib.get("version") != "1.6":
        raise ValueError("Gazebo model must be SDF 1.6")
    models = model_root.findall("model")
    if len(models) != 1:
        raise ValueError("Gazebo model file must contain exactly one model")
    model = models[0]
    model_name = model.attrib.get("name", "")
    if not model_name or expected_include_uri != f"model://{model_name}":
        raise ValueError("Gazebo include URI does not match model identity")
    if model.findall("model") or model.findall("frame") or model.findall("joint"):
        raise ValueError("nested models, explicit frames and joints are unsupported")
    model_pose, model_matrix = _sdf_pose(model, "model")
    links = model.findall("link")
    if len(links) != 1:
        raise ValueError("Gazebo support model must contain exactly one link")
    link = links[0]
    if link.findall("frame") or link.findall("joint"):
        raise ValueError("link frames and joints are unsupported")
    link_pose, link_matrix = _sdf_pose(link, "link")
    collisions = [
        item
        for item in link.findall("collision")
        if (item.findtext("geometry/mesh/uri") or "").strip() == expected_mesh_uri
    ]
    if len(collisions) != 1:
        raise ValueError("Gazebo collision mesh URI is missing or ambiguous")
    collision = collisions[0]
    collision_pose, collision_matrix = _sdf_pose(collision, "collision")
    mesh = collision.find("geometry/mesh")
    if mesh is None:
        raise ValueError("Gazebo collision mesh is missing")
    scale_elements = mesh.findall("scale")
    if not scale_elements:
        scale = np.ones(3, dtype=np.float64)
    elif len(scale_elements) == 1 and not scale_elements[0].attrib:
        scale = np.fromstring(scale_elements[0].text or "", sep=" ", dtype=np.float64)
        if scale.size != 3 or not np.all(np.isfinite(scale)) or np.any(scale <= 0.0):
            raise ValueError("Gazebo mesh scale must contain three positive values")
    else:
        raise ValueError("Gazebo mesh scale must be unique and unframed")
    scale_matrix = np.eye(4, dtype=np.float64)
    scale_matrix[:3, :3] = np.diag(scale)
    transform = include_matrix @ model_matrix @ link_matrix @ collision_matrix @ scale_matrix
    return GazeboCollisionMeshBinding(
        model_name=model_name,
        include_uri=expected_include_uri,
        mesh_uri=expected_mesh_uri,
        include_pose_xyz_rpy=include_pose,
        model_pose_xyz_rpy=model_pose,
        link_pose_xyz_rpy=link_pose,
        collision_pose_xyz_rpy=collision_pose,
        mesh_scale_xyz=scale,
        world_from_mesh=transform,
    )


@dataclass(frozen=True)
class OrientedTriangleMesh:
    vertices_xyz_m: np.ndarray
    triangle_vertex_indices: np.ndarray
    triangle_normals: np.ndarray
    source_triangle_count: int | None = None
    skipped_zero_area_source_indices: tuple[int, ...] = ()
    skipped_zero_area_source_faces: tuple[str, ...] = ()
    triangle_surface_ids: np.ndarray | None = None
    surface_names: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        vertices = np.asarray(self.vertices_xyz_m, dtype=np.float64)
        triangles = np.asarray(self.triangle_vertex_indices, dtype=np.int64)
        normals = np.asarray(self.triangle_normals, dtype=np.float64)
        if vertices.ndim != 2 or vertices.shape[1] != 3 or not len(vertices):
            raise ValueError("COLLADA vertices must be nonempty [N,3]")
        if triangles.ndim != 2 or triangles.shape[1] != 3 or not len(triangles):
            raise ValueError("COLLADA triangles must be nonempty [M,3]")
        if normals.shape != triangles.shape:
            raise ValueError("COLLADA triangle normals must have shape [M,3]")
        if not np.all(np.isfinite(vertices)) or not np.all(np.isfinite(normals)):
            raise ValueError("COLLADA geometry must be finite")
        if int(triangles.min()) < 0 or int(triangles.max()) >= len(vertices):
            raise ValueError("COLLADA triangle index is out of bounds")
        lengths = np.linalg.norm(normals, axis=1)
        if not np.allclose(lengths, 1.0, rtol=0.0, atol=32.0 * np.finfo(np.float64).eps):
            raise ValueError("COLLADA triangle normals must be unit length")
        source_count = len(triangles) if self.source_triangle_count is None else int(self.source_triangle_count)
        skipped = tuple(int(index) for index in self.skipped_zero_area_source_indices)
        skipped_faces = tuple(str(value) for value in self.skipped_zero_area_source_faces)
        if source_count != len(triangles) + len(skipped):
            raise ValueError("source triangle count must equal retained plus skipped triangles")
        if len(set(skipped)) != len(skipped) or any(index < 0 or index >= source_count for index in skipped):
            raise ValueError("skipped zero-area source indices must be unique and in range")
        if skipped_faces and len(skipped_faces) != len(skipped):
            raise ValueError("skipped source-face identities must align with skipped indices")
        if self.triangle_surface_ids is None:
            surface_ids = np.zeros(len(triangles), dtype=np.int64)
            surface_names = ("surface0",)
        else:
            surface_ids = np.asarray(self.triangle_surface_ids, dtype=np.int64)
            surface_names = tuple(str(value) for value in self.surface_names)
            if surface_ids.shape != (len(triangles),) or not surface_names:
                raise ValueError("triangle surface identities must align with triangles")
            if int(surface_ids.min()) < 0 or int(surface_ids.max()) >= len(surface_names):
                raise ValueError("triangle surface identity is out of range")
        object.__setattr__(self, "vertices_xyz_m", _readonly(vertices))
        object.__setattr__(self, "triangle_vertex_indices", _readonly(triangles))
        object.__setattr__(self, "triangle_normals", _readonly(normals))
        object.__setattr__(self, "source_triangle_count", source_count)
        object.__setattr__(self, "skipped_zero_area_source_indices", skipped)
        object.__setattr__(self, "skipped_zero_area_source_faces", skipped_faces)
        object.__setattr__(self, "triangle_surface_ids", _readonly(surface_ids))
        object.__setattr__(self, "surface_names", surface_names)

    @property
    def triangle_count(self) -> int:
        return int(len(self.triangle_vertex_indices))

    @property
    def upward_triangle_count(self) -> int:
        return int(np.count_nonzero(_upward_mask(self.triangle_normals)))

    @property
    def skipped_zero_area_triangle_count(self) -> int:
        return len(self.skipped_zero_area_source_indices)


def _position_array(mesh: ET.Element) -> tuple[np.ndarray, str]:
    vertices_elements = mesh.findall("c:vertices", _NS)
    if len(vertices_elements) != 1:
        raise ValueError("COLLADA mesh must contain exactly one vertices element")
    vertices = vertices_elements[0]
    inputs = vertices.findall("c:input", _NS)
    positions = [item for item in inputs if item.attrib.get("semantic") == "POSITION"]
    if len(inputs) != 1 or len(positions) != 1:
        raise ValueError("COLLADA vertices must contain exactly one POSITION input")
    source_id = positions[0].attrib.get("source", "")
    if not source_id.startswith("#"):
        raise ValueError("COLLADA POSITION source must be a local reference")
    source_id = source_id[1:]
    sources = [item for item in mesh.findall("c:source", _NS) if item.attrib.get("id") == source_id]
    if len(sources) != 1:
        raise ValueError("COLLADA POSITION source is missing or ambiguous")
    source = sources[0]
    arrays = source.findall("c:float_array", _NS)
    accessor = source.find("c:technique_common/c:accessor", _NS)
    if len(arrays) != 1 or accessor is None:
        raise ValueError("COLLADA POSITION source must have one float array and accessor")
    stride = int(accessor.attrib.get("stride", "0"))
    count = int(accessor.attrib.get("count", "0"))
    parameters = [item.attrib.get("name") for item in accessor.findall("c:param", _NS)]
    if stride != 3 or parameters != ["X", "Y", "Z"]:
        raise ValueError("COLLADA POSITION accessor must be XYZ stride 3")
    values = _numbers(arrays[0], np.float64)
    if values.size != count * stride:
        raise ValueError("COLLADA POSITION accessor count does not match its array")
    return values.reshape(count, stride), vertices.attrib.get("id", "")


def _triangle_indices(mesh: ET.Element, vertices_id: str) -> tuple[np.ndarray, tuple[tuple[int, int], ...]]:
    unsupported = {
        child.tag.rsplit("}", 1)[-1]
        for child in mesh
        if child.tag.rsplit("}", 1)[-1]
        in {"lines", "linestrips", "polygons", "polylist", "trifans", "tristrips"}
    }
    if unsupported:
        raise ValueError(f"unsupported COLLADA mesh primitives: {sorted(unsupported)}")
    groups: list[np.ndarray] = []
    identities: list[tuple[int, int]] = []
    for primitive_index, triangles in enumerate(mesh.findall("c:triangles", _NS)):
        count = int(triangles.attrib.get("count", "-1"))
        if count < 0:
            raise ValueError("COLLADA triangle group is missing a valid count")
        inputs = triangles.findall("c:input", _NS)
        offsets = [int(item.attrib.get("offset", "-1")) for item in inputs]
        if not inputs or min(offsets) < 0:
            raise ValueError("COLLADA triangle inputs require nonnegative offsets")
        stride = max(offsets) + 1
        vertex_inputs = [item for item in inputs if item.attrib.get("semantic") == "VERTEX"]
        if len(vertex_inputs) != 1 or vertex_inputs[0].attrib.get("source") != f"#{vertices_id}":
            raise ValueError("COLLADA triangles require one local VERTEX input")
        vertex_offset = int(vertex_inputs[0].attrib["offset"])
        payloads = triangles.findall("c:p", _NS)
        if len(payloads) != 1:
            raise ValueError("COLLADA triangle group must contain exactly one index payload")
        payload = _numbers(payloads[0], np.int64)
        if payload.size != count * 3 * stride:
            raise ValueError("COLLADA triangle count does not match its index payload")
        groups.append(payload.reshape(count * 3, stride)[:, vertex_offset].reshape(count, 3))
        identities.extend((primitive_index, face_index) for face_index in range(count))
    if not groups:
        raise ValueError("COLLADA mesh contains no triangles")
    return np.concatenate(groups, axis=0), tuple(identities)


def _node_matrix(node: ET.Element) -> np.ndarray:
    transforms = [
        child
        for child in node
        if child.tag.rsplit("}", 1)[-1]
        in {"matrix", "translate", "rotate", "scale", "skew", "lookat"}
    ]
    if not transforms:
        return np.eye(4, dtype=np.float64)
    if len(transforms) != 1 or transforms[0].tag.rsplit("}", 1)[-1] != "matrix":
        raise ValueError("COLLADA nodes must encode their transform as one matrix")
    values = _numbers(transforms[0], np.float64)
    if values.size != 16:
        raise ValueError("COLLADA node matrix must contain 16 values")
    matrix = values.reshape(4, 4)
    return _validated_affine(matrix, "COLLADA node transform")


def _active_visual_scene(root: ET.Element) -> ET.Element:
    scene_instances = root.findall("c:scene/c:instance_visual_scene", _NS)
    if len(scene_instances) != 1:
        raise ValueError("COLLADA must select exactly one visual scene")
    reference = scene_instances[0].attrib.get("url", "")
    if not reference.startswith("#"):
        raise ValueError("COLLADA visual-scene reference must be local")
    scenes = [
        item
        for item in root.findall("c:library_visual_scenes/c:visual_scene", _NS)
        if item.attrib.get("id") == reference[1:]
    ]
    if len(scenes) != 1:
        raise ValueError("COLLADA active visual scene is missing or ambiguous")
    return scenes[0]


def _refine_upward_sheet_identities(
    vertices: np.ndarray,
    triangles: np.ndarray,
    normals: np.ndarray,
    coarse_surface_ids: np.ndarray,
    coarse_surface_names: tuple[str, ...],
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Split each DAE primitive into connected upward-facing sheets."""

    count = len(triangles)
    parent = np.arange(count, dtype=np.int64)

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = int(parent[index])
        return index

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root == right_root:
            return
        if left_root < right_root:
            parent[right_root] = left_root
        else:
            parent[left_root] = right_root

    upward = _upward_mask(normals)
    first_by_vertex: dict[tuple[int, float, float, float], int] = {}
    for face_index in np.flatnonzero(upward):
        coarse_id = int(coarse_surface_ids[face_index])
        for vertex_index in triangles[face_index]:
            vertex = vertices[int(vertex_index)]
            key = (coarse_id, float(vertex[0]), float(vertex[1]), float(vertex[2]))
            previous = first_by_vertex.get(key)
            if previous is None:
                first_by_vertex[key] = int(face_index)
            else:
                union(int(face_index), previous)

    refined = np.empty(count, dtype=np.int64)
    names: list[str] = []
    identity_to_id: dict[tuple[int, str, int], int] = {}
    per_coarse_sheet_count: dict[int, int] = {}
    root_to_sheet_number: dict[tuple[int, int], int] = {}
    for face_index in range(count):
        coarse_id = int(coarse_surface_ids[face_index])
        if upward[face_index]:
            root = find(face_index)
            root_key = (coarse_id, root)
            if root_key not in root_to_sheet_number:
                number = per_coarse_sheet_count.get(coarse_id, 0)
                per_coarse_sheet_count[coarse_id] = number + 1
                root_to_sheet_number[root_key] = number
            identity = (coarse_id, "upward_sheet", root_to_sheet_number[root_key])
            label = f"{coarse_surface_names[coarse_id]}:upward_sheet{identity[2]}"
        else:
            identity = (coarse_id, "non_support", 0)
            label = f"{coarse_surface_names[coarse_id]}:non_support"
        if identity not in identity_to_id:
            identity_to_id[identity] = len(names)
            names.append(label)
        refined[face_index] = identity_to_id[identity]
    return refined, tuple(names)


def load_collada_triangle_mesh(
    path: str | Path,
    world_from_mesh: np.ndarray | None = None,
) -> OrientedTriangleMesh:
    source = Path(path)
    root = ET.parse(source).getroot()
    if root.tag != f"{{{COLLADA_NAMESPACE}}}COLLADA" or root.attrib.get("version") != "1.4.1":
        raise ValueError("only COLLADA 1.4.1 is accepted")
    unit = root.find("c:asset/c:unit", _NS)
    up_axis = root.findtext("c:asset/c:up_axis", default="", namespaces=_NS)
    if unit is None or not math.isclose(float(unit.attrib.get("meter", "nan")), 1.0, rel_tol=0.0, abs_tol=0.0):
        raise ValueError("COLLADA unit must be exactly one metre")
    if up_axis.strip() != "Z_UP":
        raise ValueError("COLLADA axis must be Z_UP")
    geometry_data: dict[str, tuple[np.ndarray, np.ndarray, tuple[tuple[int, int], ...]]] = {}
    for geometry in root.findall("c:library_geometries/c:geometry", _NS):
        geometry_id = geometry.attrib.get("id", "")
        mesh = geometry.find("c:mesh", _NS)
        if not geometry_id or mesh is None or geometry_id in geometry_data:
            raise ValueError("COLLADA geometry ids must be unique and contain a mesh")
        vertices, vertices_id = _position_array(mesh)
        triangles, face_identities = _triangle_indices(mesh, vertices_id)
        if int(triangles.min()) < 0 or int(triangles.max()) >= len(vertices):
            raise ValueError("COLLADA triangle index is out of bounds")
        geometry_data[geometry_id] = (vertices, triangles, face_identities)
    if not geometry_data:
        raise ValueError("COLLADA asset contains no geometry")

    resolved_vertices: list[np.ndarray] = []
    resolved_triangles: list[np.ndarray] = []
    resolved_face_identities: list[str] = []
    resolved_surface_ids: list[int] = []
    surface_name_to_id: dict[str, int] = {}
    instance_counts = {geometry_id: 0 for geometry_id in geometry_data}

    def visit(node: ET.Element, parent_transform: np.ndarray) -> None:
        transform = parent_transform @ _node_matrix(node)
        for instance in node.findall("c:instance_geometry", _NS):
            reference = instance.attrib.get("url", "")
            if not reference.startswith("#") or reference[1:] not in geometry_data:
                raise ValueError("COLLADA geometry instance has an unknown local reference")
            geometry_id = reference[1:]
            instance_counts[geometry_id] += 1
            vertices, triangles, face_identities = geometry_data[geometry_id]
            homogeneous = np.column_stack((vertices, np.ones(len(vertices), dtype=np.float64)))
            world = (homogeneous @ transform.T)[:, :3]
            if not np.all(np.isfinite(world)):
                raise ValueError("COLLADA transformed geometry must be finite")
            offset = sum(len(item) for item in resolved_vertices)
            resolved_vertices.append(world)
            resolved_triangles.append(triangles + offset)
            resolved_face_identities.extend(
                f"{geometry_id}:primitive{primitive_index}:face{face_index}"
                for primitive_index, face_index in face_identities
            )
            for primitive_index, _ in face_identities:
                surface_name = f"{geometry_id}:primitive{primitive_index}"
                if surface_name not in surface_name_to_id:
                    surface_name_to_id[surface_name] = len(surface_name_to_id)
                resolved_surface_ids.append(surface_name_to_id[surface_name])
        for child in node.findall("c:node", _NS):
            visit(child, transform)

    scene = _active_visual_scene(root)
    outer_transform = (
        np.eye(4, dtype=np.float64)
        if world_from_mesh is None
        else _validated_affine(world_from_mesh, "world_from_mesh")
    )
    for node in scene.findall("c:node", _NS):
        visit(node, outer_transform)
    if any(count != 1 for count in instance_counts.values()):
        raise ValueError("each COLLADA geometry must have exactly one active scene instance")
    vertices = np.concatenate(resolved_vertices, axis=0)
    triangles = np.concatenate(resolved_triangles, axis=0)
    faces = vertices[triangles]
    cross = np.cross(faces[:, 1] - faces[:, 0], faces[:, 2] - faces[:, 0])
    lengths = np.linalg.norm(cross, axis=1)
    if np.any(~np.isfinite(lengths)):
        raise ValueError("COLLADA mesh contains a non-finite triangle area")
    # Blender/Gazebo assets can contain collapsed exporter residues.  They do
    # not define a surface normal or cover any 2-D support area.  The approved
    # rule removes only an exactly zero cross-product norm, records every
    # source index and deliberately introduces no tunable epsilon.
    zero_area = lengths == 0.0
    skipped = tuple(int(index) for index in np.flatnonzero(zero_area))
    skipped_faces = tuple(resolved_face_identities[index] for index in skipped)
    source_triangle_count = int(len(triangles))
    surface_ids = np.asarray(resolved_surface_ids, dtype=np.int64)
    if skipped:
        triangles = triangles[~zero_area]
        cross = cross[~zero_area]
        lengths = lengths[~zero_area]
        surface_ids = surface_ids[~zero_area]
    if not len(triangles):
        raise ValueError("COLLADA mesh contains no nonzero-area triangles")
    normals = cross / lengths[:, None]
    surface_ids, refined_surface_names = _refine_upward_sheet_identities(
        vertices,
        triangles,
        normals,
        surface_ids,
        tuple(surface_name_to_id),
    )
    result = OrientedTriangleMesh(
        vertices,
        triangles,
        normals,
        source_triangle_count=source_triangle_count,
        skipped_zero_area_source_indices=skipped,
        skipped_zero_area_source_faces=skipped_faces,
        triangle_surface_ids=surface_ids,
        surface_names=refined_surface_names,
    )
    if result.upward_triangle_count == 0:
        raise ValueError("COLLADA mesh contains no upward-facing support triangles")
    return result


@dataclass(frozen=True)
class DAEUpwardSupportIndex:
    origin_xy_m: np.ndarray
    shape_xy: tuple[int, int]
    resolution_m: float
    flat_cells: np.ndarray
    starts: np.ndarray
    ends: np.ndarray
    z_sorted_m: np.ndarray
    surface_id_sorted: np.ndarray
    mesh_vertices: int
    mesh_triangles: int
    upward_triangles: int
    source_mesh_triangles: int
    skipped_zero_area_triangle_indices: tuple[int, ...]
    skipped_zero_area_source_faces: tuple[str, ...]
    surface_names: tuple[str, ...]

    def __post_init__(self) -> None:
        origin = np.asarray(self.origin_xy_m, dtype=np.float64)
        flat = np.asarray(self.flat_cells, dtype=np.int64)
        starts = np.asarray(self.starts, dtype=np.int64)
        ends = np.asarray(self.ends, dtype=np.int64)
        z_values = np.asarray(self.z_sorted_m, dtype=np.float32)
        surface_ids = np.asarray(self.surface_id_sorted, dtype=np.int64)
        if origin.shape != (2,) or not np.all(np.isfinite(origin)):
            raise ValueError("support-index origin must be finite XY")
        if len(self.shape_xy) != 2 or min(self.shape_xy) <= 0:
            raise ValueError("support-index shape must contain positive XY dimensions")
        if not math.isfinite(self.resolution_m) or self.resolution_m <= 0.0:
            raise ValueError("support-index resolution must be positive")
        if flat.ndim != 1 or starts.shape != flat.shape or ends.shape != flat.shape:
            raise ValueError("support-index CSR arrays have inconsistent shapes")
        if z_values.ndim != 1 or (len(flat) and (starts[0] != 0 or ends[-1] != len(z_values))):
            raise ValueError("support-index CSR offsets are invalid")
        if len(flat) and (np.any(np.diff(flat) <= 0) or np.any(ends <= starts)):
            raise ValueError("support-index cells must be unique, sorted and nonempty")
        if not np.all(np.isfinite(z_values)):
            raise ValueError("support-index heights must be finite")
        if surface_ids.shape != z_values.shape:
            raise ValueError("support-index surface identities must align with heights")
        if len(surface_ids) and (
            not self.surface_names
            or int(surface_ids.min()) < 0
            or int(surface_ids.max()) >= len(self.surface_names)
        ):
            raise ValueError("support-index surface identity is out of range")
        if self.source_mesh_triangles != self.mesh_triangles + len(self.skipped_zero_area_triangle_indices):
            raise ValueError("support-index source triangle provenance is inconsistent")
        if len(self.skipped_zero_area_source_faces) != len(self.skipped_zero_area_triangle_indices):
            raise ValueError("support-index skipped source-face provenance is inconsistent")
        object.__setattr__(self, "origin_xy_m", _readonly(origin))
        object.__setattr__(self, "flat_cells", _readonly(flat))
        object.__setattr__(self, "starts", _readonly(starts))
        object.__setattr__(self, "ends", _readonly(ends))
        object.__setattr__(self, "z_sorted_m", _readonly(z_values))
        object.__setattr__(self, "surface_id_sorted", _readonly(surface_ids))
        object.__setattr__(self, "surface_names", tuple(str(value) for value in self.surface_names))

    @classmethod
    def from_mesh(
        cls,
        mesh: OrientedTriangleMesh,
        origin_xy_m: np.ndarray,
        shape_xy: tuple[int, int],
        resolution_m: float,
    ) -> "DAEUpwardSupportIndex":
        origin = np.asarray(origin_xy_m, dtype=np.float64)
        shape = (int(shape_xy[0]), int(shape_xy[1]))
        if origin.shape != (2,) or not np.all(np.isfinite(origin)) or min(shape) <= 0:
            raise ValueError("invalid support-index lattice")
        if not math.isfinite(resolution_m) or resolution_m <= 0.0:
            raise ValueError("support-index resolution must be positive")
        faces = mesh.vertices_xyz_m[mesh.triangle_vertex_indices]
        upward = np.flatnonzero(_upward_mask(mesh.triangle_normals))
        flat_chunks: list[np.ndarray] = []
        z_chunks: list[np.ndarray] = []
        surface_chunks: list[np.ndarray] = []
        epsilon = 64.0 * np.finfo(np.float64).eps
        for face_index in upward:
            face = faces[face_index]
            minimum = np.ceil((face[:, :2].min(axis=0) - origin) / resolution_m - epsilon).astype(np.int64)
            maximum = np.floor((face[:, :2].max(axis=0) - origin) / resolution_m + epsilon).astype(np.int64)
            minimum = np.maximum(minimum, 0)
            maximum = np.minimum(maximum, np.asarray(shape, dtype=np.int64) - 1)
            if np.any(minimum > maximum):
                continue
            ix = np.arange(minimum[0], maximum[0] + 1, dtype=np.int64)
            iy = np.arange(minimum[1], maximum[1] + 1, dtype=np.int64)
            grid_x, grid_y = np.meshgrid(ix, iy, indexing="xy")
            x = origin[0] + grid_x.ravel() * resolution_m
            y = origin[1] + grid_y.ravel() * resolution_m
            x1, y1 = face[0, :2]
            x2, y2 = face[1, :2]
            x3, y3 = face[2, :2]
            # Work in face-local XY.  The algebraically equivalent global
            # formula loses the determinant for large-coordinate thin faces.
            dx21, dy21 = x2 - x1, y2 - y1
            dx31, dy31 = x3 - x1, y3 - y1
            denominator = dx21 * dy31 - dy21 * dx31
            if denominator <= 0.0:
                raise ValueError("upward-facing COLLADA triangle has non-positive XY orientation")
            px, py = x - x1, y - y1
            b = (px * dy31 - py * dx31) / denominator
            c = (dx21 * py - dy21 * px) / denominator
            a = 1.0 - b - c
            tolerance = epsilon * max(1.0, float(np.max(np.abs(face[:, :2]))))
            inside = (a >= -tolerance) & (b >= -tolerance) & (c >= -tolerance)
            if not np.any(inside):
                continue
            flat_chunks.append((grid_y.ravel()[inside] * shape[0] + grid_x.ravel()[inside]).astype(np.int64))
            z = a[inside] * face[0, 2] + b[inside] * face[1, 2] + c[inside] * face[2, 2]
            z_chunks.append(z.astype(np.float32))
            surface_chunks.append(
                np.full(np.count_nonzero(inside), mesh.triangle_surface_ids[face_index], dtype=np.int64)
            )
        if not flat_chunks:
            raise ValueError("upward-facing COLLADA triangles do not intersect the support lattice")
        flat = np.concatenate(flat_chunks)
        z_values = np.concatenate(z_chunks)
        surface_ids = np.concatenate(surface_chunks)
        order = np.lexsort((surface_ids, z_values, flat))
        flat = flat[order]
        z_values = z_values[order]
        surface_ids = surface_ids[order]
        keep = np.ones(len(flat), dtype=bool)
        keep[1:] = (
            (flat[1:] != flat[:-1])
            | (z_values[1:] != z_values[:-1])
            | (surface_ids[1:] != surface_ids[:-1])
        )
        flat = flat[keep]
        z_values = z_values[keep]
        surface_ids = surface_ids[keep]
        cells, starts = np.unique(flat, return_index=True)
        ends = np.r_[starts[1:], len(flat)]
        return cls(
            origin_xy_m=origin,
            shape_xy=shape,
            resolution_m=float(resolution_m),
            flat_cells=cells,
            starts=starts,
            ends=ends,
            z_sorted_m=z_values,
            surface_id_sorted=surface_ids,
            mesh_vertices=int(len(mesh.vertices_xyz_m)),
            mesh_triangles=mesh.triangle_count,
            upward_triangles=mesh.upward_triangle_count,
            source_mesh_triangles=int(mesh.source_triangle_count),
            skipped_zero_area_triangle_indices=mesh.skipped_zero_area_source_indices,
            skipped_zero_area_source_faces=mesh.skipped_zero_area_source_faces,
            surface_names=mesh.surface_names,
        )

    @classmethod
    def from_dae(
        cls,
        path: str | Path,
        origin_xy_m: np.ndarray,
        shape_xy: tuple[int, int],
        resolution_m: float,
        world_from_mesh: np.ndarray | None = None,
    ) -> "DAEUpwardSupportIndex":
        return cls.from_mesh(
            load_collada_triangle_mesh(path, world_from_mesh), origin_xy_m, shape_xy, resolution_m
        )

    def z_values(self, flat_cell: int) -> np.ndarray:
        z_values, _ = self.candidate_values(flat_cell)
        if not len(z_values):
            return z_values
        return np.unique(z_values)

    def candidate_values(self, flat_cell: int) -> tuple[np.ndarray, np.ndarray]:
        position = int(np.searchsorted(self.flat_cells, int(flat_cell)))
        if position >= len(self.flat_cells) or int(self.flat_cells[position]) != int(flat_cell):
            return np.empty(0, dtype=np.float32), np.empty(0, dtype=np.int64)
        span = slice(self.starts[position], self.ends[position])
        return self.z_sorted_m[span], self.surface_id_sorted[span]

    def provenance(self) -> dict[str, int | float | list[float] | list[int]]:
        return {
            "origin_xy_m": self.origin_xy_m.astype(float).tolist(),
            "shape_xy": list(self.shape_xy),
            "resolution_m": self.resolution_m,
            "mesh_vertices": self.mesh_vertices,
            "mesh_triangles": self.mesh_triangles,
            "source_mesh_triangles": self.source_mesh_triangles,
            "upward_triangles": self.upward_triangles,
            "upward_normal_z_numerical_tolerance": UPWARD_NORMAL_Z_NUMERICAL_TOLERANCE,
            "skipped_zero_area_triangles": len(self.skipped_zero_area_triangle_indices),
            "skipped_zero_area_triangle_indices": list(self.skipped_zero_area_triangle_indices),
            "skipped_zero_area_source_faces": list(self.skipped_zero_area_source_faces),
            "support_surfaces": len(self.surface_names),
            "support_surface_names": list(self.surface_names),
            "support_cells": int(len(self.flat_cells)),
            "support_height_samples": int(len(self.z_sorted_m)),
        }


__all__ = [
    "DAEUpwardSupportIndex",
    "GazeboCollisionMeshBinding",
    "OrientedTriangleMesh",
    "load_collada_triangle_mesh",
    "load_gazebo_collision_mesh_binding",
]
