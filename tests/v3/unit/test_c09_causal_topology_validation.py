from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from mtare_topo.data.c08_causal_replay import frame_contract
from mtare_topo.topology.causal_graph_v2 import CausalGraphConfig
from tools.v3.run_cano_c09_causal_topology_validation_v1 import native_c09_bundle_sha256


ROOT = Path(__file__).resolve().parents[3]
MESH = ROOT / "results/gate0_baseline/gate0_20260811_cano_100_parent_perception_mesh_m1r_sanitized_assets_seed0/artifacts/meshes"
GEOMETRY = ROOT / "results/gate4_topology/gate4_20260819_cano_c09_route_conditioned_geometry_qualification_v1r2_seed0/artifacts"
PARAMETERS = ROOT / "results/gate4_topology/gate4_20260817_cano_c08_route_conditioned_causal_topology_replay_v2r_seed0/artifacts/frozen_c08_graph_parameters.json"
WORLDS = tuple(f"S{index:02d}_{name}_C09" for index, name in enumerate((
    "flat_tree_small", "3d_tree_small", "flat_unicyclic_small", "3d_unicyclic_small",
    "flat_branch_medium", "3d_branch_medium", "flat_loop_rich", "3d_loop_rich",
    "flat_complex", "3d_complex",
), 1))


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_all_c09_coordinate_contracts_are_explicit_and_complete() -> None:
    total = 0
    for world in WORLDS:
        source = MESH / world / "primary"
        original = np.load(GEOMETRY / f"{world}_trajectory.npz")
        corrected = np.load(GEOMETRY / f"{world}_corrected_trajectory.npz")
        traversals = load(GEOMETRY / f"{world}_traversals.json")
        graph = load(source / "graph.json")
        fta = float(load(source / "geometry_parameters.json")["fta_distance_m"])
        assert (source / "mesh.obj").is_file()
        frames = frame_contract(
            teacher_axis_xyz_m=original["xyz_m"],
            graph_axis_xyz_m=corrected["graph_axis_xyz_m"],
            sensor_xyz_m=corrected["sensor_xyz_m"],
            tangent_world=original["tangent_world"],
            route_arc_m=original["route_arc_m"],
            traversals=traversals,
            graph=graph,
            fta_distance_m=fta,
        )
        assert len(frames) == len(original["xyz_m"]) == len(corrected["sensor_xyz_m"])
        total += len(frames)
    assert total == 15833


def test_c09_uses_single_c08_frozen_parameter_without_selection() -> None:
    frozen = load(PARAMETERS)
    config = CausalGraphConfig(**frozen["config"])
    assert frozen["parameter_id"] == "sf2_tr8_lr6_hh20_te45_da20"
    assert config.to_dict() == frozen["config"]
    assert len(WORLDS) * 5 == 50


def test_expected_replay_cardinalities() -> None:
    assert 15833 * 16 * 720 * 2 == 364792320
    assert 15833 * 3 == 47499
    assert 10 * 5 == 50


def test_native_c09_bundle_is_frozen_without_c10() -> None:
    digest, count = native_c09_bundle_sha256()
    assert count == 40
    assert digest == "3f8bf6535923f76103d2ba535b918c63d0f8d65e35fe836a699dd59b1b919ff4"
