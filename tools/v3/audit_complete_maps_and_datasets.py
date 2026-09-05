#!/usr/bin/env python3
"""Create a source-geometry map panel and an audit of the legacy datasets.

This tool deliberately renders the complete SDF/mesh assets.  It does not read
an accumulated scan map and it does not launch Gazebo or generate samples.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon
from matplotlib.collections import PolyCollection


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUN = ROOT / "results/gate0_baseline/gate0_20260810_interface_benchmark_audit_v1_seed0"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_obj_mesh(path: Path) -> tuple[np.ndarray, np.ndarray]:
    vertices: list[list[float]] = []
    faces: list[list[int]] = []
    with path.open("r", encoding="utf-8", errors="ignore") as stream:
        for line in stream:
            if line.startswith("v "):
                values = np.fromstring(line[2:], sep=" ", count=3)
                if values.size == 3:
                    vertices.append(values.tolist())
            elif line.startswith("f "):
                polygon = [int(token.split("/", 1)[0]) - 1 for token in line[2:].split()]
                for index in range(1, len(polygon) - 1):
                    faces.append([polygon[0], polygon[index], polygon[index + 1]])
    return np.asarray(vertices, dtype=np.float64), np.asarray(faces, dtype=np.int64)


def read_collada_mesh(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Resolve Blender COLLADA geometry instances into model coordinates."""
    root = ET.parse(path).getroot()
    namespace = {"c": "http://www.collada.org/2005/11/COLLADASchema"}
    geometries: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for geometry in root.findall(".//c:library_geometries/c:geometry", namespace):
        positions = None
        for source in geometry.findall(".//c:source", namespace):
            if "positions" in source.get("id", ""):
                positions = source.find("c:float_array", namespace)
                break
        if positions is not None and positions.text:
            values = np.fromstring(positions.text, sep=" ")
            if values.size % 3 == 0:
                vertices = values.reshape(-1, 3)
                faces: list[np.ndarray] = []
                for triangles in geometry.findall(".//c:triangles", namespace):
                    inputs = triangles.findall("c:input", namespace)
                    if not inputs:
                        continue
                    stride = max(int(item.get("offset", "0")) for item in inputs) + 1
                    vertex_input = next((item for item in inputs if item.get("semantic") == "VERTEX"), None)
                    packed = triangles.find("c:p", namespace)
                    if vertex_input is None or packed is None or not packed.text:
                        continue
                    indices = np.fromstring(packed.text, sep=" ", dtype=np.int64)
                    vertex_indices = indices.reshape(-1, stride)[:, int(vertex_input.get("offset", "0"))]
                    if vertex_indices.size % 3 == 0:
                        faces.append(vertex_indices.reshape(-1, 3))
                geometries[geometry.get("id", "")] = (
                    vertices,
                    np.vstack(faces) if faces else np.empty((0, 3), dtype=np.int64),
                )

    resolved: list[np.ndarray] = []
    resolved_faces: list[np.ndarray] = []

    def visit(node: ET.Element, parent: np.ndarray) -> None:
        local = np.eye(4)
        matrix = node.find("c:matrix", namespace)
        if matrix is not None and matrix.text:
            local = np.fromstring(matrix.text, sep=" ").reshape(4, 4)
        transform = parent @ local
        for instance in node.findall("c:instance_geometry", namespace):
            geometry_data = geometries.get(instance.get("url", "").lstrip("#"))
            if geometry_data is None:
                continue
            vertices, faces = geometry_data
            homogeneous = np.column_stack((vertices, np.ones(len(vertices))))
            offset = sum(len(item) for item in resolved)
            resolved.append((homogeneous @ transform.T)[:, :3])
            resolved_faces.append(faces + offset)
        for child in node.findall("c:node", namespace):
            visit(child, transform)

    for scene in root.findall(".//c:library_visual_scenes/c:visual_scene", namespace):
        for node in scene.findall("c:node", namespace):
            visit(node, np.eye(4))
    if not resolved:
        raise RuntimeError(f"No instantiated COLLADA geometry in {path}")
    return np.vstack(resolved), np.vstack(resolved_faces)


def read_unseen_mine_boxes(path: Path) -> list[dict[str, object]]:
    root = ET.parse(path).getroot()
    boxes: list[dict[str, object]] = []
    for collision in root.findall(".//collision"):
        if "roof" in collision.get("name", ""):
            continue
        box = collision.find("./geometry/box/size")
        if box is None or not box.text:
            continue
        pose_node = collision.find("pose")
        pose = np.fromstring(pose_node.text if pose_node is not None else "0 0 0 0 0 0", sep=" ")
        size = np.fromstring(box.text, sep=" ")
        boxes.append(
            {
                "name": collision.get("name", ""),
                "center": [float(pose[0]), float(pose[1])],
                "size": [float(size[0]), float(size[1])],
                "yaw": float(pose[5]) if pose.size >= 6 else 0.0,
            }
        )
    return boxes


def box_corners(box: dict[str, object]) -> np.ndarray:
    width, height = box["size"]
    local = np.asarray(
        [[-width / 2, -height / 2], [width / 2, -height / 2],
         [width / 2, height / 2], [-width / 2, height / 2]]
    )
    yaw = float(box["yaw"])
    rotation = np.asarray([[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]])
    return local @ rotation.T + np.asarray(box["center"])


def bounds(points: np.ndarray) -> dict[str, list[float]]:
    return {
        "min_xyz": [round(float(value), 4) for value in points.min(axis=0)],
        "max_xyz": [round(float(value), 4) for value in points.max(axis=0)],
        "span_xyz_m": [round(float(value), 4) for value in np.ptp(points, axis=0)],
    }


def plot_points(
    ax, points: np.ndarray, start: tuple[float, float], title: str, faces: np.ndarray | None = None
) -> None:
    if faces is not None and len(faces):
        face_limit = 120_000
        shown_faces = faces
        if len(faces) > face_limit:
            indices = np.linspace(0, len(faces) - 1, face_limit, dtype=np.int64)
            shown_faces = faces[indices]
        triangles = points[shown_faces]
        colors = triangles[:, :, 2].mean(axis=1)
        collection = PolyCollection(
            triangles[:, :, :2], array=colors, cmap="viridis", edgecolors="#17202a",
            linewidths=0.08, alpha=0.72, rasterized=True,
        )
        ax.add_collection(collection)
    maximum = 250_000
    if len(points) > maximum:
        indices = np.linspace(0, len(points) - 1, maximum, dtype=np.int64)
        shown = points[indices]
    else:
        shown = points
    z_low, z_high = np.percentile(shown[:, 2], [2, 98])
    ax.scatter(
        shown[:, 0], shown[:, 1], c=np.clip(shown[:, 2], z_low, z_high),
        cmap="viridis", s=0.35, alpha=0.58, linewidths=0, rasterized=True,
    )
    ax.scatter(*start, marker="*", s=95, color="#d62728", edgecolor="white", linewidth=0.6, zorder=5)
    ax.set_title(title, fontsize=10)


def plot_boxes(ax, boxes: list[dict[str, object]], start: tuple[float, float], title: str) -> None:
    all_corners = []
    for box in boxes:
        corners = box_corners(box)
        all_corners.append(corners)
        ax.add_patch(Polygon(corners, closed=True, facecolor="#485563", edgecolor="#17202a", linewidth=0.8))
    corners = np.vstack(all_corners)
    ax.set_xlim(corners[:, 0].min() - 5, corners[:, 0].max() + 5)
    ax.set_ylim(corners[:, 1].min() - 5, corners[:, 1].max() + 5)
    ax.scatter(*start, marker="*", s=95, color="#d62728", edgecolor="white", linewidth=0.6, zorder=5)
    ax.set_title(title, fontsize=10)


def finish_axis(ax) -> None:
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Gazebo x (m)", fontsize=8)
    ax.set_ylabel("Gazebo y (m)", fontsize=8)
    ax.grid(True, linewidth=0.35, alpha=0.35)
    ax.tick_params(labelsize=7)


def parse_semantic_filename(path: Path) -> tuple[str, str]:
    name = path.stem
    world = next((item for item in ("tunnel", "garage", "forest", "campus", "indoor") if f"_{item}_" in name), "unknown")
    match = re.search(r"_(traj\d+)_", name)
    trajectory = match.group(1) if match else "named_transfer_trajectory"
    return world, trajectory


def npz_schema(path: Path) -> dict[str, object]:
    with np.load(path, allow_pickle=False) as archive:
        return {
            key: {"shape": list(archive[key].shape), "dtype": str(archive[key].dtype)}
            for key in archive.files
        }


def legacy_dataset_audit() -> dict[str, object]:
    semantic_root = ROOT / "results/topological_semantic_dataset_v4_supervision_fixed"
    structural_root = ROOT / "results/structural_dataset_v4_multienv"

    semantic_files = sorted(semantic_root.rglob("*.npz"))
    semantic_splits = Counter(path.parent.name for path in semantic_files)
    semantic_worlds: dict[str, Counter] = defaultdict(Counter)
    for path in semantic_files:
        world, trajectory = parse_semantic_filename(path)
        semantic_worlds[path.parent.name][world] += 1

    structural_files = sorted(structural_root.rglob("*.npz"))
    structural_splits = Counter(path.parent.name for path in structural_files)
    structural_worlds: dict[str, Counter] = defaultdict(Counter)
    trajectory_counts = Counter()
    structural_pattern = re.compile(r"^(train|val|test)_([^_]+)_([^_]+)_\d+$")
    for path in structural_files:
        match = structural_pattern.match(path.stem)
        if match:
            split, environment, trajectory = match.groups()
            structural_worlds[split][environment] += 1
            trajectory_counts[f"{environment}:{trajectory}"] += 1

    deployment_manifest = ROOT / "results/semantic_bottleneck_role/20260808_underground_crossval_v1/deployment_manifest.json"
    checkpoint = ROOT / "results/semantic_bottleneck_role/20260807_polar_semantic_v5_final/semantic_bottleneck_underground_deployment_epoch70.pt"
    return {
        "schema_version": "gate0_dataset_audit_v1",
        "v3_formal_dataset": {
            "status": "NONE",
            "meaning": "No V3 training/validation/test sample set has been approved or generated at Gate 0.",
        },
        "legacy_deployed_semantic_model": {
            "role": "Historical semantic closed-loop model; reference only, not accepted V3 training data.",
            "checkpoint": str(checkpoint.relative_to(ROOT)),
            "checkpoint_sha256": sha256(checkpoint),
            "deployment_manifest": str(deployment_manifest.relative_to(ROOT)),
            "dataset_root": str(semantic_root.relative_to(ROOT)),
            "sample_format": "One compressed NumPy .npz per trajectory-local sample, not one world per file.",
            "sample_count": len(semantic_files),
            "split_counts": dict(sorted(semantic_splits.items())),
            "split_world_counts": {key: dict(sorted(value.items())) for key, value in sorted(semantic_worlds.items())},
            "actual_training_worlds": ["tunnel", "garage"],
            "validation_worlds": ["forest"],
            "test_worlds": ["campus", "indoor"],
            "training_trajectory_count": 4,
            "training_trajectories": ["tunnel:traj01", "tunnel:traj02", "garage:traj01", "garage:traj02"],
            "representative_schema": npz_schema(semantic_files[0]),
        },
        "legacy_lamp_structural_dataset": {
            "role": "Earlier structural-representation experiments; not the deployed semantic closed-loop model's training set.",
            "dataset_root": str(structural_root.relative_to(ROOT)),
            "raw_source": "/mnt/nas_znfy/Shared-2/dataset/SLAM/subt/LAMP",
            "sample_count": len(structural_files),
            "split_counts": dict(sorted(structural_splits.items())),
            "split_environment_counts": {key: dict(sorted(value.items())) for key, value in sorted(structural_worlds.items())},
            "trajectory_counts": dict(sorted(trajectory_counts.items())),
            "representative_schema": npz_schema(structural_files[0]),
        },
        "interpretation": [
            "4,849 semantic NPZ files are strongly correlated local windows sampled along four training trajectories, not 4,849 independent maps.",
            "The historical deployed semantic model trained on only two geometries (tunnel and garage).",
            "Forest/campus/indoor supplied domain checks but did not enlarge the deployed model's training geometry diversity.",
            "The LAMP dataset exists and was used by a separate older structural pipeline; it was not used to train the deployed semantic model cited above.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_RUN / "artifacts/complete_world_maps")
    parser.add_argument("--tunnel-dae", type=Path, default=Path("/tmp/mtare_complete_map_assets/tunnel.dae"))
    parser.add_argument("--garage-dae", type=Path, default=Path("/tmp/mtare_complete_map_assets/garage.dae"))
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    unseen_path = ROOT / "integration/cmu_unseen_mine/unseen_mine.world"
    cave_path = ROOT / "external/gazebo_cave_world/worlds/models/cave_world/meshes/cave_world.obj"
    op1_path = ROOT / "external/SubTGraph/benchmark/operational/01/subtgraph.obj"
    op2_path = ROOT / "external/SubTGraph/benchmark/operational/02/subtgraph.obj"

    tunnel, tunnel_faces = read_collada_mesh(args.tunnel_dae)
    tunnel = tunnel + np.asarray([-22.0, 92.5, 0.0])
    garage, garage_faces = read_collada_mesh(args.garage_dae)
    cave_raw, cave_faces = read_obj_mesh(cave_path)
    cave = np.column_stack((cave_raw[:, 0] + 5.66804, -cave_raw[:, 2] - 21.9271, cave_raw[:, 1]))
    op1_raw, op1_faces = read_obj_mesh(op1_path)
    op1 = np.column_stack((op1_raw[:, 0], -op1_raw[:, 2], op1_raw[:, 1]))
    op2_raw, op2_faces = read_obj_mesh(op2_path)
    op2 = np.column_stack((op2_raw[:, 0], -op2_raw[:, 2], op2_raw[:, 1]))
    unseen_boxes = read_unseen_mine_boxes(unseen_path)

    maps = [
        ("tunnel", tunnel, tunnel_faces, (0.0, 0.0), "Development | M-TARE tunnel"),
        ("unseen_mine", unseen_boxes, None, (0.0, 0.0), "Development | custom unseen_mine"),
        ("external_cave", cave, cave_faces, (10.0, -21.0), "Development | external cave"),
        ("subtgraph_operational_01", op1, op1_faces, (0.0, 40.0), "Regression | SubTGraph operational 01"),
        ("subtgraph_operational_02", op2, op2_faces, (0.0, 40.0), "Regression | SubTGraph operational 02"),
        ("garage", garage, garage_faces, (0.0, 0.0), "Smoke only | M-TARE garage"),
    ]

    figure, axes = plt.subplots(2, 3, figsize=(18, 11), constrained_layout=True)
    for ax, (identifier, geometry, faces, start, title) in zip(axes.flat, maps):
        if identifier == "unseen_mine":
            plot_boxes(ax, geometry, start, title)
        else:
            plot_points(ax, geometry, start, title, faces)
        finish_axis(ax)
    figure.suptitle(
        "Gate 0 complete source-geometry map inventory\n"
        "Full SDF/mesh extents in Gazebo coordinates; red star = currently configured start; not accumulated scans",
        fontsize=15,
    )
    panel_path = output_dir / "current_benchmark_complete_source_maps.png"
    figure.savefig(panel_path, dpi=220)
    plt.close(figure)

    individual_images: dict[str, Path] = {}
    for identifier, geometry, faces, start, title in maps:
        item_figure, item_axis = plt.subplots(figsize=(10, 8), constrained_layout=True)
        if identifier == "unseen_mine":
            plot_boxes(item_axis, geometry, start, title)
        else:
            plot_points(item_axis, geometry, start, title, faces)
        finish_axis(item_axis)
        item_figure.suptitle(
            "Complete source geometry (not an accumulated scan)\nred star = configured start",
            fontsize=13,
        )
        item_path = output_dir / f"{identifier}_complete_source_map.png"
        item_figure.savefig(item_path, dpi=220)
        plt.close(item_figure)
        individual_images[identifier] = item_path

    map_manifest = {
        "schema_version": "gate0_complete_source_map_manifest_v1",
        "rendering_contract": "Complete SDF collision primitives or all mesh vertices transformed into Gazebo coordinates; never accumulated sensor scans.",
        "maps": [],
        "composite": str(panel_path.relative_to(ROOT)),
    }
    source_lookup = {
        "tunnel": ("container:vehicle_simulator/mesh/tunnel/meshes/tunnel.dae", args.tunnel_dae, tunnel),
        "garage": ("container:vehicle_simulator/mesh/garage/meshes/garage.dae", args.garage_dae, garage),
        "external_cave": (str(cave_path.relative_to(ROOT)), cave_path, cave),
        "subtgraph_operational_01": (str(op1_path.relative_to(ROOT)), op1_path, op1),
        "subtgraph_operational_02": (str(op2_path.relative_to(ROOT)), op2_path, op2),
    }
    for identifier, geometry, faces, start, title in maps:
        if identifier == "unseen_mine":
            corners = np.vstack([box_corners(box) for box in unseen_boxes])
            pseudo = np.column_stack((corners, np.zeros(len(corners))))
            entry_bounds = bounds(pseudo)
            source_label = str(unseen_path.relative_to(ROOT))
            hash_source = unseen_path
            source_type = "SDF collision boxes (roof excluded from top-down footprint)"
            vertex_count = None
            primitive_count = len(unseen_boxes)
        else:
            source_label, hash_source, points = source_lookup[identifier]
            entry_bounds = bounds(points)
            source_type = "COLLADA mesh" if hash_source.suffix == ".dae" else "OBJ mesh"
            vertex_count = int(len(points))
            primitive_count = None
        map_manifest["maps"].append(
            {
                "id": identifier,
                "title": title,
                "source": source_label,
                "render_input": str(hash_source),
                "source_type": source_type,
                "source_sha256": sha256(hash_source),
                "individual_image": str(individual_images[identifier].relative_to(ROOT)),
                "configured_start_xy_m": list(start),
                "vertex_count": vertex_count,
                "primitive_count": primitive_count,
                **entry_bounds,
            }
        )
    (output_dir / "complete_source_map_manifest.json").write_text(
        json.dumps(map_manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    dataset_audit = legacy_dataset_audit()
    (output_dir / "dataset_inventory.json").write_text(
        json.dumps(dataset_audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    report = f"""# 完整地图与数据集审计（Gate 0）

## 先说结论

- 当前 V3 正式数据集：**尚未建立（NONE）**。Gate 0 只做接口、地图和旧证据审计，没有批准生成新训练样本。
- 旧闭环语义模型的实际训练数据：`results/topological_semantic_dataset_v4_supervision_fixed`，共 {dataset_audit['legacy_deployed_semantic_model']['sample_count']} 个局部 `.npz`。
- 真正进入旧模型训练集的场景只有 `tunnel + garage`；`forest` 是验证，`campus + indoor` 是测试。
- 这些 `.npz` 是四条训练轨迹沿线截取的局部观测窗口，不是 3024 张独立地图，更不是 3024 个独立地下环境。
- 项目里另有 {dataset_audit['legacy_lamp_structural_dataset']['sample_count']} 个 LAMP 结构表征样本，但属于更早的另一条结构学习流水线，并未训练上述旧闭环语义模型。

## 当前拟议实验地图

完整源几何总图：`current_benchmark_complete_source_maps.png`

| 用途 | 地图 |
|---|---|
| 开发 | `tunnel`, `unseen_mine`, `external_cave` |
| 回归开发 | `subtgraph_operational_01`, `subtgraph_operational_02` |
| 接口冒烟，不作地下结论 | `garage` |
| V3 严格未见测试 | 暂无；仍需隔离新增至少两个不同几何族地下世界 |

图中展示的是 `.world/.dae/.obj` 的完整范围，红星是当前配置起点。它不是历史轨迹图，也不是机器人扫到哪里才出现哪里的累计点云。

## 旧语义数据拆分

| split | 数量 | 世界 |
|---|---:|---|
| train | 3024 | tunnel 1545；garage 1479 |
| val | 356 | forest 356 |
| test | 1469 | campus 694；indoor 775 |

训练轨迹只有四条：`tunnel:traj01`, `tunnel:traj02`, `garage:traj01`, `garage:traj02`。因此样本数量看似为 3024，几何多样性仍只有两个世界，而且同一轨迹相邻样本高度相关。

## `.npz` 是什么

`.npz` 是 NumPy 的压缩归档格式，一个文件可以保存多个命名数组。旧语义数据的代表性样本包含：

- 当前局部输入 `student_input_current`: `4×100×100`；
- 8 帧历史 `student_input_history`: `8×4×100×100`；
- 32 个方向的可通行性、可达距离、可达面积和出口监督；
- 出口中心/宽度/长度；
- 连续 openness、bottleneck、main-path-continuity 分数；
- 64 维 canonical/topological role 教师目标；
- 位姿、时间戳、世界、轨迹、split 和数据契约字段。

所以“监督标签”来自离线教师规则/几何处理结果，并非人工逐帧真值。V3 如果改为 AI 辅助人工标注有监督学习，必须重新定义标签本体、标注输入、复核协议和世界隔离，不能直接把这些旧 teacher 标签当作人工真值。

## 文件位置

- 完整地图清单：`complete_source_map_manifest.json`
- 数据集机器可读清单：`dataset_inventory.json`
- 旧语义数据：`results/topological_semantic_dataset_v4_supervision_fixed/`
- 旧部署权重：`results/semantic_bottleneck_role/20260807_polar_semantic_v5_final/semantic_bottleneck_underground_deployment_epoch70.pt`
- 旧部署声明：`results/semantic_bottleneck_role/20260808_underground_crossval_v1/deployment_manifest.json`
- 旧 LAMP 结构数据：`results/structural_dataset_v4_multienv/`

## 当前暴露的问题

1. 旧部署模型的训练几何只有两个，而且 `garage` 不是地下环境。
2. 旧地下交叉验证只是同一 `tunnel/garage` 世界里的轨迹互换，不是世界级隔离验证。
3. `forest/campus/indoor` 没有增强训练，只能说明做过域外验证/测试；它们也不适合作为地下任务主证据。
4. 所有现有地下候选地图都已被历史开发或结果观察污染，因此没有严格未见测试集。
5. SubTGraph 两个 OBJ 的完整横向跨度达到公里量级，而当前起点配置在 `(0, 40)`；这与历史 no-motion/spawn 问题一致，必须先做单位、连通域和可达面审计，不能直接纳入正式公平实验。
"""
    (output_dir / "DATASET_AND_COMPLETE_MAP_AUDIT.md").write_text(report, encoding="utf-8")
    print(panel_path)
    print(output_dir / "complete_source_map_manifest.json")
    print(output_dir / "dataset_inventory.json")
    print(output_dir / "DATASET_AND_COMPLETE_MAP_AUDIT.md")


if __name__ == "__main__":
    main()
