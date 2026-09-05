"""Pack sealed GSE observations and visible exit tokens into training arrays."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from mtare_topo.data.gse_sensor_export import WorldUniqueFramePoses
from mtare_topo.data.gse_training_targets import heading_deg_to_unit, world_vector_to_robot
from mtare_topo.semantics.geometric_semantics import EVENT_NAMES


MAXIMUM_EXIT_TOKENS = 6
EVENT_TO_INDEX = {name: index for index, name in enumerate(EVENT_NAMES)}


@dataclass(frozen=True)
class PackedWorldTeacher:
    arrays: Mapping[str, np.ndarray]
    sequence_manifest: tuple[dict[str, Any], ...]

    def __post_init__(self) -> None:
        sequence_count = len(self.sequence_manifest)
        if not self.arrays or any(len(value) != sequence_count for value in self.arrays.values()):
            raise ValueError("packed teacher arrays must align with the sequence manifest")


def pack_world_teacher_targets(
    *,
    parent_id: str,
    observations: Sequence[Mapping[str, Any]],
    exit_token_rows: Sequence[Mapping[str, Any]],
    poses: WorldUniqueFramePoses,
    association_identity_to_index: Mapping[str, int],
    exit_identity_to_index: Mapping[str, int],
    maximum_exit_tokens: int = MAXIMUM_EXIT_TOKENS,
) -> PackedWorldTeacher:
    """Pack complete robot-frame targets while retaining explicit loss masks."""

    if not parent_id or not 1 <= maximum_exit_tokens <= 255:
        raise ValueError("parent identity and exit-token capacity are invalid")
    if len(set(int(value) for value in poses.global_frame_indices)) != len(poses.global_frame_indices):
        raise ValueError("world frame poses contain duplicate global indices")
    pose_by_global = {
        int(global_index): index for index, global_index in enumerate(poses.global_frame_indices)
    }
    token_by_observation: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in exit_token_rows:
        if str(row["parent_id"]) != parent_id:
            raise ValueError("exit-token row crosses parent identity")
        token_by_observation[str(row["observation_id"])].append(row)
    for rows in token_by_observation.values():
        rows.sort(key=lambda row: int(row["candidate_index"]))

    ordered = sorted(
        observations,
        key=lambda row: (int(row["global_sequence_index"]), str(row["observation_id"])),
    )
    count = len(ordered)
    arrays = {
        "global_sequence_index": np.empty(count, dtype=np.int64),
        "global_frame_references": np.empty((count, 5), dtype=np.int64),
        "local_frame_references": np.empty((count, 5), dtype=np.int32),
        "event_index": np.empty(count, dtype=np.uint8),
        "local_axis_robot": np.empty((count, 3), dtype=np.float32),
        "geometry": np.empty((count, 4), dtype=np.float32),
        "geometry_valid_mask": np.empty((count, 4), dtype=np.uint8),
        "association_identity": np.full(count, -1, dtype=np.int64),
        "association_valid_mask": np.zeros(count, dtype=np.uint8),
        "exit_mask": np.zeros((count, maximum_exit_tokens), dtype=np.uint8),
        "exit_heading_unit": np.zeros((count, maximum_exit_tokens, 2), dtype=np.float32),
        "exit_opening_width_m": np.zeros((count, maximum_exit_tokens), dtype=np.float32),
        "exit_width_valid_mask": np.zeros((count, maximum_exit_tokens), dtype=np.uint8),
        "exit_vertical_profile_m": np.zeros((count, maximum_exit_tokens, 4), dtype=np.float32),
        "exit_identity": np.full((count, maximum_exit_tokens), -1, dtype=np.int64),
        "exit_candidate_count": np.empty(count, dtype=np.uint8),
        "exit_invisible_count": np.empty(count, dtype=np.uint8),
    }
    manifest: list[dict[str, Any]] = []
    seen_observations: set[str] = set()
    for row_index, observation in enumerate(ordered):
        observation_id = str(observation["observation_id"])
        if observation_id in seen_observations or str(observation["parent_id"]) != parent_id:
            raise ValueError("teacher observation identity is duplicate or crosses parent")
        seen_observations.add(observation_id)
        references = np.asarray(observation["global_frame_references"], dtype=np.int64)
        if references.shape != (5,):
            raise ValueError("teacher observation must reference five frames")
        pose_indices = np.asarray([pose_by_global.get(int(value), -1) for value in references], dtype=np.int64)
        if np.any(pose_indices < 0) or not np.all(np.diff(pose_indices) == 1):
            raise RuntimeError("teacher frame references do not resolve contiguously inside the world")
        anchor_pose = int(pose_indices[-1])
        if (
            poses.traversal_ids[anchor_pose] != str(observation["traversal_id"])
            or int(poses.local_frame_indices[anchor_pose]) != int(observation["frame_index"])
        ):
            raise RuntimeError("teacher observation and anchor pose identity drift")
        event = str(observation["event"])
        if event not in EVENT_TO_INDEX:
            raise ValueError("unknown structural event")
        geometry_valid = bool(observation["geometry_valid"])
        width = float(observation["width_m"]) if geometry_valid else 0.0
        height = float(observation["height_m"]) if geometry_valid else 0.0
        geometry = np.asarray(
            (width, height, float(observation["slope_deg"]), float(observation["curvature_per_m"])),
            dtype=np.float32,
        )
        if not np.all(np.isfinite(geometry)):
            raise ValueError("teacher geometry must be finite after masking")
        association_identity = observation.get("identity")
        if association_identity is not None:
            identity = str(association_identity)
            if identity not in association_identity_to_index:
                raise ValueError("structural association identity is absent from the frozen map")
            arrays["association_identity"][row_index] = int(association_identity_to_index[identity])
            arrays["association_valid_mask"][row_index] = 1

        candidate_rows = token_by_observation.pop(observation_id, [])
        if not candidate_rows or any(
            str(candidate["observation_id"]) != observation_id
            or str(candidate["parent_id"]) != parent_id
            or int(candidate["candidate_count"]) != len(candidate_rows)
            or int(candidate["candidate_index"]) != candidate_index
            or str(candidate["candidate"]["identity"]) != str(candidate["token"]["identity"])
            for candidate_index, candidate in enumerate(candidate_rows)
        ):
            raise RuntimeError("exit candidate rows are missing or not a complete ordered set")
        visible_rows = [candidate for candidate in candidate_rows if bool(candidate["token"]["visible"])]
        if len(visible_rows) > maximum_exit_tokens:
            raise RuntimeError("visible exit set exceeds model token capacity")
        for token_index, candidate in enumerate(visible_rows):
            token = candidate["token"]
            identity = str(token["identity"])
            if identity not in exit_identity_to_index:
                raise ValueError("directed exit identity is absent from the frozen map")
            arrays["exit_mask"][row_index, token_index] = 1
            arrays["exit_heading_unit"][row_index, token_index] = heading_deg_to_unit(
                float(token["heading_robot_deg"])
            )
            width_valid = bool(token["opening_width_valid"])
            arrays["exit_width_valid_mask"][row_index, token_index] = int(width_valid)
            arrays["exit_opening_width_m"][row_index, token_index] = (
                float(token["opening_width_m"]) if width_valid else 0.0
            )
            vertical_profile = np.asarray(token["vertical_profile_m"], dtype=np.float32)
            if vertical_profile.shape != (4,) or not np.all(np.isfinite(vertical_profile)):
                raise ValueError("visible exit vertical profile must contain four finite values")
            arrays["exit_vertical_profile_m"][row_index, token_index] = vertical_profile
            arrays["exit_identity"][row_index, token_index] = int(exit_identity_to_index[identity])

        arrays["global_sequence_index"][row_index] = int(observation["global_sequence_index"])
        arrays["global_frame_references"][row_index] = references
        arrays["local_frame_references"][row_index] = pose_indices.astype(np.int32)
        arrays["event_index"][row_index] = EVENT_TO_INDEX[event]
        arrays["local_axis_robot"][row_index] = world_vector_to_robot(
            poses.tangent_world_xyz[anchor_pose],
            robot_yaw_deg=float(poses.yaw_deg[anchor_pose]),
        ).astype(np.float32)
        arrays["geometry"][row_index] = geometry
        arrays["geometry_valid_mask"][row_index] = np.asarray(
            (geometry_valid, geometry_valid, True, True), dtype=np.uint8
        )
        arrays["exit_candidate_count"][row_index] = len(candidate_rows)
        arrays["exit_invisible_count"][row_index] = len(candidate_rows) - len(visible_rows)
        manifest.append(
            {
                "observation_id": observation_id,
                "parent_id": parent_id,
                "split": str(observation["split"]),
                "traversal_id": str(observation["traversal_id"]),
                "world_sequence_row": row_index,
                "global_sequence_index": int(observation["global_sequence_index"]),
                "global_anchor_frame_index": int(references[-1]),
                "world_anchor_frame_row": anchor_pose,
            }
        )
    if token_by_observation:
        raise RuntimeError("exit-token audit contains observations absent from the teacher manifest")
    if count and (
        np.any(arrays["exit_width_valid_mask"] > arrays["exit_mask"])
        or np.any(arrays["exit_identity"][arrays["exit_mask"] == 0] != -1)
        or not np.all(np.isfinite(arrays["local_axis_robot"]))
        or not np.allclose(
            np.linalg.norm(arrays["local_axis_robot"], axis=1),
            1.0,
            rtol=0.0,
            atol=1e-6,
        )
        or np.any(arrays["local_frame_references"][:, 0] < 0)
    ):
        raise RuntimeError("packed teacher mask, identity or robot-frame axis contract failed")
    return PackedWorldTeacher(arrays=arrays, sequence_manifest=tuple(manifest))


__all__ = [
    "EVENT_TO_INDEX",
    "MAXIMUM_EXIT_TOKENS",
    "PackedWorldTeacher",
    "pack_world_teacher_targets",
]
