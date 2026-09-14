"""Teacher and readiness utilities for linear endpoint composition anchors.

Each procedural sweep primitive has two endpoints and each endpoint belongs to
exactly one construction operation.  The proposed learned interface predicts
one shared physical anchor and uncertainty per visible endpoint; attachment is
then a calibrated probability of shared anchor rather than an independent
binary output for every endpoint pair.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from mtare_topo.evaluation.primitive_attachment_observability import world_to_sensor


MAXIMUM_PRIMITIVES = 32
MAXIMUM_ENDPOINTS = 64


@dataclass(frozen=True)
class ConstructionEndpointAnchors:
    primitive_ids: tuple[str, ...]
    node_ids: np.ndarray
    endpoint_world_m: np.ndarray
    anchor_world_m: np.ndarray

    def __post_init__(self) -> None:
        count = len(self.primitive_ids)
        if len(set(self.primitive_ids)) != count:
            raise ValueError("construction primitive identities must be unique")
        if self.node_ids.shape != (count, 2):
            raise ValueError("construction node identities must be [primitive,2]")
        for name in ("endpoint_world_m", "anchor_world_m"):
            value = np.asarray(getattr(self, name))
            if value.shape != (count, 2, 3) or not np.all(np.isfinite(value)):
                raise ValueError(f"{name} must be finite [primitive,2,3]")


@dataclass(frozen=True)
class ObservableAnchorPairSlice:
    observed_endpoint_residual_m: np.ndarray
    observed_axis_pair_distance_m: np.ndarray
    target_anchor_pair_distance_m: np.ndarray
    attachment_target: np.ndarray
    overlap_hard_negative: np.ndarray

    def __post_init__(self) -> None:
        residual = np.asarray(self.observed_endpoint_residual_m)
        arrays = tuple(np.asarray(value) for value in (
            self.observed_axis_pair_distance_m,
            self.target_anchor_pair_distance_m,
            self.attachment_target,
            self.overlap_hard_negative,
        ))
        if residual.ndim != 1 or any(value.ndim != 1 for value in arrays):
            raise ValueError("anchor readiness slices must be one-dimensional")
        if len({len(value) for value in arrays}) != 1:
            raise ValueError("anchor pair slice lengths disagree")
        if (
            not np.all(np.isfinite(residual))
            or np.any(residual < 0)
            or not np.all(np.isfinite(arrays[0]))
            or not np.all(np.isfinite(arrays[1]))
            or np.any(arrays[0] < 0)
            or np.any(arrays[1] < 0)
        ):
            raise ValueError("anchor distances must be finite and nonnegative")


def construction_endpoint_anchor_table(
    document: Mapping[str, Any],
) -> ConstructionEndpointAnchors:
    """Validate the realized-construction document and return endpoint anchors."""

    base = document.get("base_construction")
    realized = document.get("realized_primitives")
    if not isinstance(base, Mapping) or not isinstance(realized, list):
        raise ValueError("realized construction document is incomplete")
    primitives = base.get("primitives")
    operations = base.get("composition_operations")
    if not isinstance(primitives, list) or not primitives or not isinstance(operations, list):
        raise ValueError("construction primitives and operations must be nonempty")
    primitive_ids = tuple(str(value["primitive_id"]) for value in primitives)
    realized_ids = tuple(str(value["primitive_id"]) for value in realized)
    if primitive_ids != realized_ids:
        raise ValueError("base and realized primitive ordering differs")
    lookup = {value: index for index, value in enumerate(primitive_ids)}
    node_ids = np.empty((len(primitives), 2), dtype=object)
    endpoints = np.empty((len(primitives), 2, 3), dtype=np.float64)
    anchors = np.empty_like(endpoints)
    endpoint_records: dict[tuple[str, int], Mapping[str, Any]] = {}
    for primitive_index, primitive in enumerate(primitives):
        values = primitive.get("endpoints")
        if not isinstance(values, list) or len(values) != 2:
            raise ValueError("every construction primitive must have two endpoints")
        for endpoint in values:
            endpoint_index = int(endpoint["endpoint_index"])
            key = (str(endpoint["primitive_id"]), endpoint_index)
            if key[0] != primitive_ids[primitive_index] or endpoint_index not in (0, 1) or key in endpoint_records:
                raise ValueError("construction endpoint identity is invalid or duplicated")
            endpoint_records[key] = endpoint
            node_ids[primitive_index, endpoint_index] = str(endpoint["node_id"])
            endpoints[primitive_index, endpoint_index] = endpoint["xyz_m"]
            anchors[primitive_index, endpoint_index] = endpoint["composition_anchor_xyz_m"]

    operation_membership: dict[tuple[str, int], str] = {}
    for operation in operations:
        node_id = str(operation["node_id"])
        anchor = np.asarray(operation["anchor_xyz_m"], dtype=np.float64)
        members = operation.get("member_endpoints")
        if not isinstance(members, list) or len(members) != int(operation["degree"]):
            raise ValueError("composition degree/member population differs")
        for member in members:
            key = (str(member["primitive_id"]), int(member["endpoint_index"]))
            if key not in endpoint_records or key in operation_membership:
                raise ValueError("endpoint has zero, duplicate or unknown composition membership")
            primitive_index = lookup[key[0]]
            if (
                str(member["node_id"]) != node_id
                or node_ids[primitive_index, key[1]] != node_id
                or not np.array_equal(
                    np.asarray(member["composition_anchor_xyz_m"], dtype=np.float64), anchor,
                )
                or not np.array_equal(anchors[primitive_index, key[1]], anchor)
            ):
                raise ValueError("composition member anchor or node identity differs")
            operation_membership[key] = node_id
    if set(operation_membership) != set(endpoint_records):
        raise ValueError("every primitive endpoint must belong to exactly one composition")
    return ConstructionEndpointAnchors(primitive_ids, node_ids, endpoints, anchors)


def current_sensor_anchor_targets(
    *,
    primitive_index: np.ndarray,
    primitive_mask: np.ndarray,
    anchor_world_m: np.ndarray,
    sensor_xyz_m: np.ndarray,
    yaw_deg: np.ndarray,
) -> np.ndarray:
    """Gather unique endpoint anchors and transform them to current sensor frames."""

    index = np.asarray(primitive_index, dtype=np.int64)
    mask = np.asarray(primitive_mask, dtype=np.uint8)
    anchors = np.asarray(anchor_world_m, dtype=np.float64)
    if index.ndim != 2 or index.shape[1:] != (MAXIMUM_PRIMITIVES,) or mask.shape != index.shape:
        raise ValueError("primitive anchor index/mask must be [batch,32]")
    if np.any((mask != 0) & (mask != 1)) or np.any((index >= 0) != mask.astype(bool)):
        raise ValueError("primitive anchor index/mask differs")
    if anchors.ndim != 3 or anchors.shape[1:] != (2, 3):
        raise ValueError("construction anchors must be [primitive,2,3]")
    active = mask.astype(bool)
    if np.any(index[active] >= len(anchors)):
        raise ValueError("primitive anchor index is out of range")
    safe = np.where(active, index, 0)
    result = world_to_sensor(anchors[safe], sensor_xyz_m, yaw_deg)
    result[~active] = np.nan
    return result


def _attachment_matrix(endpoint_neighbor: np.ndarray) -> np.ndarray:
    neighbor = np.asarray(endpoint_neighbor, dtype=np.int16)
    if neighbor.ndim != 4 or neighbor.shape[1:] != (32, 2, 3):
        raise ValueError("endpoint neighbors must be [batch,32,2,3]")
    batch = len(neighbor)
    result = np.zeros((batch, MAXIMUM_ENDPOINTS, MAXIMUM_ENDPOINTS), dtype=bool)
    flat = neighbor.reshape(batch, MAXIMUM_ENDPOINTS, 3)
    valid = flat >= 0
    if np.any(flat[valid] >= MAXIMUM_ENDPOINTS):
        raise ValueError("endpoint neighbor is out of range")
    row, source, column = np.nonzero(valid)
    result[row, source, flat[row, source, column]] = True
    return result


def observable_anchor_pair_slice(
    *,
    axis_control_current_sensor_m: np.ndarray,
    anchor_current_sensor_m: np.ndarray,
    primitive_mask: np.ndarray,
    endpoint_observed: np.ndarray,
    endpoint_neighbor: np.ndarray,
    disconnected_overlap: np.ndarray,
) -> ObservableAnchorPairSlice:
    """Return observed endpoint residuals and all eligible anchor pair distances."""

    axis = np.asarray(axis_control_current_sensor_m, dtype=np.float64)
    anchors = np.asarray(anchor_current_sensor_m, dtype=np.float64)
    primitive = np.asarray(primitive_mask, dtype=np.uint8)
    observed = np.asarray(endpoint_observed, dtype=np.uint8)
    overlap = np.asarray(disconnected_overlap, dtype=np.uint8)
    batch = len(primitive)
    if axis.shape != (batch, 32, 3, 3) or anchors.shape != (batch, 32, 2, 3):
        raise ValueError("axis controls and anchors have invalid shapes")
    if primitive.shape != (batch, 32) or observed.shape != (batch, 32, 2):
        raise ValueError("primitive and endpoint masks have invalid shapes")
    if overlap.shape != (batch, 32, 32):
        raise ValueError("disconnected overlap must be [batch,32,32]")
    if any(np.any((value != 0) & (value != 1)) for value in (primitive, observed, overlap)):
        raise ValueError("anchor readiness masks must be binary")
    active = primitive.astype(bool)
    endpoint_active = active[:, :, None]
    endpoint_observed_bool = observed.astype(bool)
    if np.any(endpoint_observed_bool & ~endpoint_active):
        raise ValueError("inactive endpoint cannot be observed")
    if not np.all(np.isfinite(axis[active])) or not np.all(np.isfinite(anchors[active])):
        raise ValueError("active axis controls and anchor targets must be finite")

    observed_axis = axis[:, :, (0, 2), :].reshape(batch, MAXIMUM_ENDPOINTS, 3)
    flat_anchor = anchors.reshape(batch, MAXIMUM_ENDPOINTS, 3)
    flat_observed = endpoint_observed_bool.reshape(batch, MAXIMUM_ENDPOINTS)
    residual = np.linalg.norm(observed_axis - flat_anchor, axis=2)[flat_observed]
    endpoint = np.arange(MAXIMUM_ENDPOINTS)
    primitive_id = endpoint // 2
    upper = np.triu(primitive_id[:, None] != primitive_id[None, :], k=1)
    eligible = flat_observed[:, :, None] & flat_observed[:, None, :] & upper[None]
    axis_distance = np.linalg.norm(
        observed_axis[:, :, None, :] - observed_axis[:, None, :, :], axis=3,
    )[eligible]
    anchor_distance = np.linalg.norm(
        flat_anchor[:, :, None, :] - flat_anchor[:, None, :, :], axis=3,
    )[eligible]
    attachment = _attachment_matrix(endpoint_neighbor)[eligible]
    endpoint_overlap = np.repeat(np.repeat(overlap.astype(bool), 2, axis=1), 2, axis=2)
    hard_negative = endpoint_overlap[eligible] & ~attachment
    return ObservableAnchorPairSlice(
        residual.astype(np.float32, copy=False),
        axis_distance.astype(np.float32, copy=False),
        anchor_distance.astype(np.float32, copy=False),
        attachment.astype(np.bool_, copy=False),
        hard_negative.astype(np.bool_, copy=False),
    )


__all__ = [
    "ConstructionEndpointAnchors",
    "ObservableAnchorPairSlice",
    "construction_endpoint_anchor_table",
    "current_sensor_anchor_targets",
    "observable_anchor_pair_slice",
]
