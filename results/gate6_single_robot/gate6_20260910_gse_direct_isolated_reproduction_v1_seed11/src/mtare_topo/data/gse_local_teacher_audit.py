"""Read-only checks of existing visible geometry and construction relations.

This is an evaluation module, not a target generator or deployment loader.
In particular, angular overlap is NOT physical overlap, a visible primitive
fragment is NOT a visible physical endpoint, and Hungarian assignment is NOT
evidence that the detector is accurate. No new event labels are returned.
"""
from types import SimpleNamespace
from pathlib import Path

import numpy as np
import torch
import zarr

from mtare_topo.data.gse_scoped_inventory import EvidenceStore, validate_selection
from mtare_topo.representation.primitive_relation_losses import (
    PrimitiveRelationLossTargets, match_primitives,
)


FIELDS = frozenset(("frame_row", "source_global_sequence_index", "primitive_index",
    "primitive_mask", "axis_control_current_sensor_m", "endpoint_half_axes_m",
    "endpoint_shape_exponent", "support_ray_count", "temporal_visibility",
    "endpoint_neighbor", "disconnected_overlap_packed"))
PREDICTION_FIELDS = ("axis_control_current_sensor_m", "endpoint_half_axes_m",
    "endpoint_shape_exponent", "existence_logits", "geometry_uncertainty",
    "endpoint_evidence_logits")


def _binary(array, name):
    if not np.isfinite(array).all() or not np.isin(array, (0, 1)).all():
        raise ValueError(f"{name} must be binary")


def audit_support(mask, counts, temporal):
    mask, counts, temporal = map(np.asarray, (mask, counts, temporal))
    if (mask.ndim != 2 or mask.shape[1] != 32 or counts.shape != mask.shape
            or temporal.shape != (len(mask), 5, 32)):
        raise ValueError("support shape drift")
    _binary(mask, "mask"); _binary(temporal, "temporal")
    if (not np.isfinite(counts).all() or (counts < 0).any()
            or (counts != np.floor(counts)).any()):
        raise ValueError("support counts must be nonnegative integers")
    if (not np.array_equal(mask, counts > 0)
            or not np.array_equal(mask, temporal.any(axis=1))
            or (counts < temporal.sum(axis=1)).any()):
        raise ValueError("visible fragment/support/temporal mismatch")


def decode_and_validate_relations(neighbors, angular_packed, mask):
    neighbors, angular_packed, mask = map(np.asarray, (neighbors, angular_packed, mask))
    if (mask.ndim != 2 or mask.shape[1] != 32
            or neighbors.shape != (len(mask), 32, 2, 3)
            or angular_packed.shape != (len(mask), 32, 4)):
        raise ValueError("relation shape drift")
    _binary(mask, "mask")
    if neighbors.dtype != np.int8 or angular_packed.dtype != np.uint8:
        raise ValueError("packed relation dtype drift")
    if ((neighbors < -1) | (neighbors >= 64)).any():
        raise ValueError("endpoint neighbor out of bounds")
    active = mask.astype(bool)
    attachment = np.zeros((len(mask), 32, 2, 32, 2), np.uint8)
    for b, slot, end in np.ndindex(len(mask), 32, 2):
        values = neighbors[b, slot, end]
        valid = values[values >= 0]
        if len(np.unique(valid)) != len(valid):
            raise ValueError("duplicate neighbor")
        for value in valid:
            other, other_end = divmod(int(value), 2)
            if other == slot or not active[b, slot] or not active[b, other]:
                raise ValueError("self relation or inactive relation slot")
            attachment[b, slot, end, other, other_end] = 1
    if not np.array_equal(attachment, attachment.transpose(0, 3, 4, 1, 2)):
        raise ValueError("asymmetric attachment")
    angular = np.unpackbits(angular_packed, axis=2, count=32, bitorder="little")
    if (not np.array_equal(angular, angular.transpose(0, 2, 1))
            or np.diagonal(angular, axis1=1, axis2=2).any()
            or (angular & ~(active[:, :, None] & active[:, None, :])).any()
            or (angular & attachment.any(axis=(2, 4))).any()):
        raise ValueError("invalid angular overlap (not physical collision)")
    return attachment, angular


def expected_construction_attachment(primitive_indices, mask, construction):
    """All visible GT members, used solely to check old annotation identity."""
    indices, mask = np.asarray(primitive_indices), np.asarray(mask)
    if mask.ndim != 2 or mask.shape[1] != 32 or indices.shape != mask.shape:
        raise ValueError("construction index shape drift")
    _binary(mask, "mask")
    if indices.dtype.kind not in "iu" or not np.array_equal(mask, indices >= 0):
        raise ValueError("construction primitive index/mask drift")
    primitives = construction["realized_primitives"]
    lookup = {str(p["primitive_id"]): i for i, p in enumerate(primitives)}
    if len(lookup) != len(primitives):
        raise ValueError("duplicate construction primitive id")
    groups = []
    endpoints_seen = set()
    for composition in construction["base_construction"]["composition_operations"]:
        members = []
        for member in composition["member_endpoints"]:
            end = member["endpoint_index"]
            if type(end) is not int or end not in (0, 1):
                raise ValueError("invalid construction endpoint")
            pair = (lookup[str(member["primitive_id"])], end)
            if pair in endpoints_seen:
                raise ValueError("ambiguous construction endpoint membership")
            endpoints_seen.add(pair); members.append(pair)
        if not 1 <= len(members) <= 4 or len(members) != composition["degree"]:
            raise ValueError("construction incidence/degree drift")
        groups.append(members)
    expected = np.zeros((len(mask), 32, 2, 32, 2), np.uint8)
    for b in range(len(mask)):
        active = np.flatnonzero(mask[b])
        values = indices[b, active]
        if len(np.unique(values)) != len(values) or (values >= len(primitives)).any():
            raise ValueError("invalid or duplicate visible primitive index")
        slot_for = dict(zip(values.tolist(), active.tolist()))
        for group in groups:
            visible = [(slot_for[p], e) for p, e in group if p in slot_for]
            for slot, end in visible:
                for other, other_end in visible:
                    if slot != other:
                        expected[b, slot, end, other, other_end] = 1
    return expected


class ScopedLocalTeacherReader:
    def __init__(self, root, records, *, expected_sha256):
        self.selection = validate_selection(records)
        if sum(map(len, self.selection.values())) > 180:
            raise ValueError("audit exceeds 180 observations")
        self.root = Path(root).resolve(strict=True)
        self.expected_sha256, self.opened = expected_sha256, {}

    def read_task(self, task):
        if task not in self.selection:
            raise PermissionError("unselected task")
        path = self.root / (task + ".zarr")
        if path.resolve().parent != self.root:
            raise PermissionError("teacher task escapes selected root")
        group = zarr.open_group(store=EvidenceStore(path, FIELDS, self.opened,
            self.expected_sha256), mode="r")
        if (group.attrs.get("parent_id") != task.split("__")[0]
                or group.attrs.get("partition") != "fit"
                or group.attrs.get("geometry_realization") != "c1_mixed"
                or group.attrs.get("student_identity_input_forbidden") is not True):
            raise ValueError("teacher task/split contract drift")
        records = self.selection[task]
        indices = np.array([r["row_index"] for r in records], dtype=np.int64)
        if (indices >= group["frame_row"].shape[0]).any():
            raise ValueError("selected row outside shard")
        output = {key: np.asarray(group[key].oindex[indices]) for key in sorted(FIELDS)}
        b = len(records)
        shapes = {"frame_row": (b, 5), "source_global_sequence_index": (b,),
            "primitive_index": (b, 32), "primitive_mask": (b, 32),
            "axis_control_current_sensor_m": (b, 32, 3, 3), "endpoint_half_axes_m": (b, 32, 2, 2),
            "endpoint_shape_exponent": (b, 32, 2), "support_ray_count": (b, 32),
            "temporal_visibility": (b, 5, 32), "endpoint_neighbor": (b, 32, 2, 3),
            "disconnected_overlap_packed": (b, 32, 4)}
        for key, shape in shapes.items():
            if output[key].shape != shape or not np.isfinite(output[key]).all():
                raise ValueError(f"teacher shape/nonfinite: {key}")
        if not np.array_equal(output["source_global_sequence_index"],
                [r["source_global_sequence_index"] for r in records]):
            raise ValueError("teacher source row identity drift")
        rows = output["frame_row"]
        if (rows < 0).any() or (np.diff(rows, axis=1) <= 0).any():
            raise ValueError("noncausal teacher frame rows")
        active = output["primitive_mask"].astype(bool)
        if ((output["endpoint_half_axes_m"][active] <= 0).any()
                or (output["endpoint_shape_exponent"][active] <= 0).any()):
            raise ValueError("nonpositive visible geometry")
        audit_support(output["primitive_mask"], output["support_ray_count"], output["temporal_visibility"])
        output["endpoint_attachment"], output["angular_overlap"] = decode_and_validate_relations(
            output["endpoint_neighbor"], output["disconnected_overlap_packed"], output["primitive_mask"])
        return output


def score_cached_geometry(prediction, teacher):
    """Historical oracle Hungarian alignment + physical errors, no acceptance."""
    b = len(teacher["primitive_mask"])
    shapes = {"axis_control_current_sensor_m": (b, 32, 3, 3), "endpoint_half_axes_m": (b, 32, 2, 2),
        "endpoint_shape_exponent": (b, 32, 2), "existence_logits": (b, 32),
        "geometry_uncertainty": (b, 32), "endpoint_evidence_logits": (b, 32, 2)}
    for key, shape in shapes.items():
        if key not in prediction or prediction[key].shape != shape or not np.isfinite(prediction[key]).all():
            raise ValueError(f"cached prediction shape/nonfinite: {key}")
    if (prediction["endpoint_half_axes_m"] <= 0).any() or (prediction["endpoint_shape_exponent"] <= 0).any():
        raise ValueError("nonpositive predicted geometry")
    pred = SimpleNamespace(**{key: torch.from_numpy(np.array(value, copy=True)) for key, value in prediction.items()})
    target_fields = {key: torch.from_numpy(np.array(teacher[key], copy=True)) for key in (
        "primitive_mask", "axis_control_current_sensor_m", "endpoint_half_axes_m",
        "endpoint_shape_exponent", "temporal_visibility", "endpoint_attachment")}
    targets = PrimitiveRelationLossTargets(**target_fields,
        disconnected_overlap=torch.from_numpy(teacher["angular_overlap"]))
    with torch.inference_mode():
        assignments = match_primitives(pred, targets)
    scored = []
    for i, assignment in enumerate(assignments):
        matches = []
        for predicted, truth in enumerate(assignment.predicted_to_target.tolist()):
            if truth < 0:
                continue
            reverse = bool(assignment.endpoint_reversed[predicted])
            axis, axes, exponent = (teacher[key][i, truth] for key in (
                "axis_control_current_sensor_m", "endpoint_half_axes_m", "endpoint_shape_exponent"))
            if reverse:
                axis, axes, exponent = axis[::-1], axes[::-1], exponent[::-1]
            residual = prediction["axis_control_current_sensor_m"][i, predicted] - axis
            matches.append({"prediction_slot": predicted, "teacher_slot_scoring_only": truth,
                "reversed": reverse, "axis_coordinate_mae_m": float(np.abs(residual).mean()),
                "axis_control_point_mean_euclidean_m": float(np.linalg.norm(residual, axis=1).mean()),
                "half_axes_mae_m": float(np.abs(prediction["endpoint_half_axes_m"][i, predicted] - axes).mean()),
                "shape_exponent_mae": float(np.abs(prediction["endpoint_shape_exponent"][i, predicted] - exponent).mean()),
                "existence_probability": float(torch.sigmoid(pred.existence_logits[i, predicted])),
                "geometry_uncertainty_uncalibrated": float(prediction["geometry_uncertainty"][i, predicted]),
                "teacher_support_rays_nonexclusive": int(teacher["support_ray_count"][i, truth]),
                "teacher_support_frames": int(teacher["temporal_visibility"][i, :, truth].sum())})
        scored.append(matches)
    return scored
