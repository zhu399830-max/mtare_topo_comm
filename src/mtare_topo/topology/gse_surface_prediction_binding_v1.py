"""Owned forward -> immutable observation binding, synthetic/development only.

Five packets carry independently supplied payload digests. The wrapper checks
their actual bytes, owns defensive input copies, aligns frames, extracts the
same-input patches, and itself invokes the model without teacher/identity/pose
arguments. Shape/mask agreement is only validation, never source identity.

Digests do not authenticate a lying upstream reader or prove features came from
a particular encoder. No real-reader/weight provenance is certified here. All
readout thresholds are explicit uncalibrated diagnostics. Returned geometric
reachability is a prediction, not a safety or traversal certificate.
"""
from dataclasses import asdict, dataclass, fields
import hashlib
import json
import math

import numpy as np
import torch

from mtare_topo.representation.gse_surface_observed_inputs_v1 import build_observed_surface_inputs
from mtare_topo.representation.gse_surface_relation_model_v1 import (
    SurfaceRelationPrediction,
)
from mtare_topo.representation.gse_surface_ray_evidence_v1 import (
    EVIDENCE_ORDER,
)
from mtare_topo.semantics.gse_local_structure_v2 import SourceFrameV2
from .gse_graph_frames import _rotation
from .gse_surface_graph_v1 import KnownSurfacePoseV1, SurfaceSegmentedGraphV1
from .gse_surface_observation_v1 import (
    SurfaceAnchorV1, SurfaceOpeningV1, SurfaceObservationV1, _number, _probability, _vector,
)


def _json(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=True, separators=(",", ":"), allow_nan=False)


def _sha(value):
    return hashlib.sha256(_json(value).encode()).hexdigest()


def _tensor_record(value):
    if not torch.is_tensor(value):
        raise ValueError("actual tensor required")
    raw = value.detach().cpu().contiguous()
    return {"shape": list(raw.shape), "dtype": str(raw.dtype),
            "sha256": hashlib.sha256(raw.numpy().tobytes()).hexdigest()}


@dataclass(frozen=True)
class SurfaceRigidTransformV1:
    rotation: tuple[tuple[float, float, float], ...]
    translation_m: tuple[float, float, float]

    def __post_init__(self):
        if type(self.rotation) is not tuple or len(self.rotation) != 3:
            raise ValueError("immutable proper rotation required")
        for row in self.rotation:
            _vector(row, 3)
        _rotation(self.rotation); _vector(self.translation_m, 3)


@dataclass(frozen=True)
class SurfaceFramePacketV1:
    source: SourceFrameV2
    points_frame_sensor_m: torch.Tensor      # N,3; actual input bytes remain hash-bound
    frozen_point_context: torch.Tensor       # N,128; upstream encoder provenance not authenticated here
    valid: torch.Tensor                     # N bool observed first-return validity, not synthetic max-range rays
    local_sensor_token_index: torch.Tensor   # N long0..179; wrapper adds180*history slot
    current_sensor_from_frame: SurfaceRigidTransformV1
    payload_sha256: str                     # must be supplied independently of the forward call


def frame_payload_sha256(packet: SurfaceFramePacketV1):
    """Reader/fixture helper; hashing is not evidence of actual file provenance."""
    if type(packet) is not SurfaceFramePacketV1 or type(packet.source) is not SourceFrameV2:
        raise ValueError("typed frame packet and opaque source identity required")
    return _sha({"source": asdict(packet.source), "current_sensor_from_frame": asdict(packet.current_sensor_from_frame),
        "tensors": {k: _tensor_record(getattr(packet, k)) for k in (
            "points_frame_sensor_m", "frozen_point_context", "valid", "local_sensor_token_index")}})


@dataclass(frozen=True)
class SurfaceForwardReadoutV1:
    dimension_evidence_threshold: float
    opening_support_threshold: float
    relation_validity_threshold: float
    reachability_threshold: float
    voxel_size_m: float
    roi_radius_m: float
    max_patches: int
    supervision_policy: str = "partial_geometry_v1"

    def __post_init__(self):
        if self.supervision_policy not in ('partial_geometry_v1','synthetic_all_heads_v1'):
            raise ValueError('explicit partial supervision or synthetic diagnostic policy required')
        for p in (self.dimension_evidence_threshold, self.opening_support_threshold,
                  self.relation_validity_threshold, self.reachability_threshold):
            _probability(p)
        for value in (self.voxel_size_m, self.roi_radius_m):
            _number(value)
            if value <= 0:
                raise ValueError("positive explicit patch scales required")
        if self.roi_radius_m > 10.:
            raise ValueError("patch ROI cannot exceed fixed10m observed-ray cube; never truncate evidence")
        if type(self.max_patches) is not int or not 1 <= self.max_patches <= 4096:
            raise ValueError("explicit patch capacity1..4096 required")


@dataclass(frozen=True)
class BoundSurfaceForwardV1:
    observation: SurfaceObservationV1
    proof_json: str
    raw_prediction_json: str
    proof_sha256: str


_FLOAT_SHAPES = {
    "anchor_position_m": (1,32,3), "anchor_presence_logits": (1,32), "anchor_uncertainty_m": (1,32,3),
    "opening_position_m": (1,64,3), "opening_presence_logits": (1,64), "opening_direction": (1,64,3),
    "opening_dimensions_m": (1,64,2), "opening_dimension_evidence_logits": (1,64,2),
    "opening_support_logits": (1,64), "reachability_logits": (1,64,3),
    "membership_logits": (1,64,32), "membership_validity_logits": (1,64,32),
}


class SurfacePredictionBindingV1:
    """There is deliberately no API accepting a caller-supplied prediction.

    Issued objects are delivered only through this instance and only with their
    exact bound known pose. This prevents accidental output/pose substitution;
    it is not a security boundary against arbitrary Python object mutation.
    """
    def __init__(self, model, readout: SurfaceForwardReadoutV1):
        if not isinstance(model, torch.nn.Module) or type(readout) is not SurfaceForwardReadoutV1:
            raise ValueError("model module and explicit uncalibrated readout required")
        self.model, self.readout = model, readout
        self._issued = {}

    def _model_digest(self):
        return _sha({"class": type(self.model).__module__ + "." + type(self.model).__qualname__,
                     "state": {k: _tensor_record(v) for k, v in sorted(self.model.state_dict().items())}})

    def forward(self, *, packets: tuple[SurfaceFramePacketV1, ...], runtime_stream_key: str,
                decision_index: int, current_source: SourceFrameV2, pose: KnownSurfacePoseV1,
                robot_from_current_sensor: SurfaceRigidTransformV1,
                timestamp_s: float | None = None) -> BoundSurfaceForwardV1:
        if self.model.training:
            raise ValueError("binding requires explicit eval mode; no training/optimizer is performed")
        if type(packets) is not tuple or len(packets) != 5 or any(type(p) is not SurfaceFramePacketV1 for p in packets):
            raise ValueError("exactly five typed frame packets required")
        if type(pose) is not KnownSurfacePoseV1 or type(robot_from_current_sensor) is not SurfaceRigidTransformV1:
            raise ValueError("known robot pose and explicit sensor extrinsic required")
        # Validate frame causality with the same strict DTO before any model.
        sources = tuple(p.source for p in packets)
        if type(current_source) is not SourceFrameV2 or any(type(s) is not SourceFrameV2 for s in sources):
            raise ValueError("explicit current decision source and typed frame identities required")
        if sources[-1] != current_source:
            raise ValueError("five-frame current source differs from explicit decision source; future/source substitution")
        SurfaceObservationV1(runtime_stream_key, decision_index, sources, current_source.order_index,
                             "0" * 64, (), (), timestamp_s)
        if ((pose.coordinate.kind == "order_index" and pose.coordinate.value != decision_index)
                or (pose.coordinate.kind == "seconds" and pose.coordinate.value != timestamp_s)):
            raise ValueError("forward decision timestamp does not match known pose")
        last_transform = packets[-1].current_sensor_from_frame
        if (type(last_transform) is not SurfaceRigidTransformV1
                or last_transform.rotation != ((1.,0.,0.),(0.,1.,0.),(0.,0.,1.))
                or last_transform.translation_m != (0.,0.,0.)):
            raise ValueError("current frame relative transform must be identity, not a future/other pose")
        points, contexts, validity, layout, frame_index, origins, source_digests = [], [], [], [], [], [], []
        dtype = device = None
        total_points = 0
        for slot, packet in enumerate(packets):
            if type(packet.current_sensor_from_frame) is not SurfaceRigidTransformV1:
                raise ValueError("typed motion alignment required")
            x, c, valid, index = (getattr(packet, k) for k in (
                "points_frame_sensor_m", "frozen_point_context", "valid", "local_sensor_token_index"))
            if not all(torch.is_tensor(v) for v in (x,c,valid,index)):
                raise ValueError("actual frame/context/layout tensors required")
            n = x.shape[0] if x.ndim == 2 else 0
            if (n < 1 or x.shape != (n,3) or c.shape != (n,128) or valid.shape != (n,) or index.shape != (n,)
                    or x.dtype not in (torch.float32, torch.float64) or c.dtype != x.dtype
                    or valid.dtype != torch.bool or index.dtype != torch.long
                    or any(v.device != x.device for v in (c,valid,index))
                    or bool(((index < 0) | (index >= 180)).any())):
                raise ValueError("strict same-frame point/context/valid/local layout contract")
            if dtype is None:
                dtype, device = x.dtype, x.device
            total_points += n
            if total_points > 57600:
                raise ValueError("one five-frame forward exceeds57600 raw points")
            if dtype != x.dtype or device != x.device:
                raise ValueError("five frames must share dtype/device")
            actual = frame_payload_sha256(packet)
            if packet.payload_sha256 != actual:
                raise ValueError("actual frame input bytes or provenance differ from reader digest")
            # Owned copies break writable caller aliases; no supplied patch can
            # silently describe another set of points with the same valid mask.
            x, c, valid, index = (v.detach().clone() for v in (x,c,valid,index))
            if not bool(torch.isfinite(x[valid]).all() and torch.isfinite(c[valid]).all()):
                raise ValueError("nonfinite supported source input")
            rotation = x.new_tensor(packet.current_sensor_from_frame.rotation)
            translation = x.new_tensor(packet.current_sensor_from_frame.translation_m)
            aligned = torch.where(valid[:,None], x @ rotation.T + translation, 0.)
            if not bool(torch.isfinite(aligned).all()):
                raise ValueError("motion alignment overflow")
            points.append(aligned); contexts.append(torch.where(valid[:,None], c, 0.)); validity.append(valid)
            layout.append(index + slot * 180); frame_index.extend([slot] * n)
            origins.append(translation.expand(n,3)); source_digests.append(actual)
        xyz, context, valid, tokens = (torch.cat(v, 0)[None] for v in (points, contexts, validity, layout))
        ray_origins = torch.cat(origins, 0)
        if xyz.shape[1] > 57600:
            raise ValueError("one five-frame forward exceeds57600 raw points")
        patches, patch_batch, grid, gap = build_observed_surface_inputs(
            xyz[0].cpu().numpy(), valid[0].cpu().numpy(), np.asarray(frame_index, dtype=np.int64),
            ray_origins.cpu().numpy(), voxel_size_m=self.readout.voxel_size_m,
            roi_radius_m=self.readout.roi_radius_m, max_patches=self.readout.max_patches,
            device=device, dtype=dtype)
        m = len(patches.centers_m)
        ray_evidence = {
            "schema_version": "gse_surface_owned_ray_evidence_v1",
            "coordinate_frame": "current_sensor", "cube_half_extent_m": 10., "voxel_size_m": .25,
            "feature_order": list(EVIDENCE_ORDER), "feature_semantics": "observed_cell_fractions_not_reachability_probabilities",
            "undefined_feature": [0., 0., 1.], "physical_connectivity": False,
            "grid_source_geometry_sha256": grid.source_geometry_sha256,
            "grid_content_sha256": grid.content_sha256,
            "grid_numerical_bound_m": grid.numerical_bound_m,
            "grid_state_cell_counts": [int((grid.state == state).sum()) for state in (0, 1, 2)],
            "grid_state_order": ["unknown", "observed_free", "observed_occupied"],
            "first_return_count": grid.first_return_count, "ignored_ray_count": grid.ignored_ray_count,
            "ambiguous_ray_count": grid.ambiguous_ray_count,
            "patch_centers_m": patches.centers_m.tolist(),
            "neighbor_index": patch_batch.neighbor_index[0, :m].cpu().tolist(),
            "gap_counts": gap.counts.tolist(), "gap_fractions": gap.fractions.tolist(),
            "gap_defined": gap.gap_defined.tolist(), "message_neighbor_valid": gap.valid.tolist(),
            "gap_endpoint_cell_count": gap.endpoint_cell_count.tolist(),
            "gap_ambiguous_cell_count": gap.ambiguous_cell_count.tolist(),
            "actual_ray_relation_features": patch_batch.relation[0, :m, :, -3:].cpu().tolist(),
        }
        def actual_inputs():
            return {"xyz": _tensor_record(xyz), "context": _tensor_record(context),
                "valid": _tensor_record(valid), "layout": _tensor_record(tokens),
                "patches": {f.name: _tensor_record(getattr(patch_batch, f.name)) for f in fields(patch_batch)}}
        input_record = actual_inputs(); model_sha = self._model_digest()
        with torch.no_grad():
            prediction = self.model(xyz, context, valid, patch_batch, sensor_token_index=tokens)
        if self._model_digest() != model_sha or actual_inputs() != input_record:
            raise ValueError("forward mutated model state or owned input tensors")
        if type(prediction) is not SurfaceRelationPrediction:
            raise ValueError("typed SurfaceRelationPrediction required")
        for name, shape in _FLOAT_SHAPES.items():
            tensor = getattr(prediction, name)
            if (not torch.is_tensor(tensor) or tuple(tensor.shape) != shape or tensor.dtype != dtype
                    or tensor.device != device or not bool(torch.isfinite(tensor).all())):
                raise ValueError("prediction shape/dtype/device/nonfinite contract: " + name)
        for name, shape in (("opening_direction_valid", (1,64)), ("observation_supported", (1,))):
            t = getattr(prediction, name)
            if not torch.is_tensor(t) or tuple(t.shape) != shape or t.dtype != torch.bool or t.device != device:
                raise ValueError("prediction boolean contract: " + name)
        if bool(prediction.observation_supported[0]) != bool(valid.any()):
            raise ValueError("output support disagrees with actual owned input; not an identity proof")
        raw, anchors, openings = self._readout(prediction, robot_from_current_sensor)
        raw["observed_ray_evidence"] = ray_evidence
        raw_json = _json(raw)
        proof = {"schema_version": "gse_surface_owned_forward_proof_v1", "runtime_stream_key": runtime_stream_key,
            "decision_index": decision_index, "source_frames": [asdict(s) for s in sources],
            "frame_payload_sha256": source_digests, "actual_model_inputs": input_record,
            "actual_model_state_sha256": model_sha, "pose": asdict(pose),
            "robot_from_current_sensor": asdict(robot_from_current_sensor), "readout": asdict(self.readout),
            "raw_prediction_sha256": hashlib.sha256(raw_json.encode()).hexdigest(),
            "observed_ray_evidence_sha256": _sha(ray_evidence),
            "grid_source_geometry_sha256": grid.source_geometry_sha256,
            "grid_content_sha256": grid.content_sha256,
            "model_absolute_pose_input": False, "teacher_input": False,
            "upstream_file_encoder_provenance_authenticated": False,
            "robot_uncertainty_available": self.readout.supervision_policy == 'synthetic_all_heads_v1' and robot_from_current_sensor.rotation == ((1.,0.,0.),(0.,1.,0.),(0.,0.,1.)),
            "uncertainty_note": "partial policy: untrained scale unavailable; synthetic policy: sensor-axis metres only, rotated scale unknown; no covariance fabricated"}
        proof_json = _json(proof); digest = hashlib.sha256(proof_json.encode()).hexdigest()
        observation = SurfaceObservationV1(runtime_stream_key, decision_index, sources, sources[-1].order_index,
            digest, anchors, openings, timestamp_s)
        result = BoundSurfaceForwardV1(observation, proof_json, raw_json, digest)
        self._issued[id(result)] = (result, _sha(asdict(pose)))
        return result

    def _readout(self, pred, transform):
        raw = {f.name: getattr(pred, f.name).detach().cpu()[0].tolist() for f in fields(pred)}
        probability_fields = ("anchor_presence", "opening_presence", "opening_dimension_evidence", "opening_support",
                              "membership", "membership_validity")
        for name in probability_fields:
            raw[name + "_probability"] = getattr(pred, name + "_logits").sigmoid().detach().cpu()[0].tolist()
        raw["reachability_probability"] = pred.reachability_logits.softmax(-1).detach().cpu()[0].tolist()
        supported = bool(raw["observation_supported"])
        partial = self.readout.supervision_policy == 'partial_geometry_v1'
        raw['readout_supervision_policy'] = self.readout.supervision_policy
        raw['reliability_status'] = 'UNTRAINED_UNCALIBRATED'
        rotation, translation = np.asarray(transform.rotation), np.asarray(transform.translation_m)
        rotate_uncertainty = transform.rotation != ((1.,0.,0.),(0.,1.,0.),(0.,0.,1.))
        def point(p):
            return tuple(float(v) for v in rotation @ np.asarray(p) + translation)
        anchors = tuple(SurfaceAnchorV1(i, point(raw["anchor_position_m"][i]) if supported else None,
            raw["anchor_presence_probability"][i],
            tuple(raw["anchor_uncertainty_m"][i]) if supported and not rotate_uncertainty and not partial else None,
            supported) for i in range(32))
        openings = []
        for i in range(64):
            # Numerical support is only availability of observed input, not a
            # per-query visibility certificate. Presence is gated by graph.
            valid = supported and (partial or raw["opening_support_probability"][i] >= self.readout.opening_support_threshold)
            # This categorical UNKNOWN is a readout abstention, not a learned
            # calibrated probability. Original logits remain in raw evidence.
            probabilities = (0.,0.,1.) if partial else tuple(raw["reachability_probability"][i])
            maximum = max(probabilities)
            winners = [j for j, p in enumerate(probabilities) if p == maximum]
            reachability = ("traversable", "blocked", "unknown")[winners[0]] if (
                not partial and valid and len(winners) == 1 and maximum >= self.readout.reachability_threshold) else "unknown"
            dimensions = [float(raw["opening_dimensions_m"][i][j]) if (
                not partial and valid and raw["opening_dimension_evidence_probability"][i][j] >= self.readout.dimension_evidence_threshold) else None for j in range(2)]
            direction = tuple(float(v) for v in rotation @ np.asarray(raw["opening_direction"][i])) if (
                valid and raw["opening_direction_valid"][i]) else None
            openings.append(SurfaceOpeningV1(i, point(raw["opening_position_m"][i]) if valid else None,
                direction, *dimensions, raw["opening_presence_probability"][i], valid, probabilities, reachability,
                tuple(raw["membership_probability"][i]), tuple(valid and (partial or p >= self.readout.relation_validity_threshold)
                    for p in raw["membership_validity_probability"][i])))
        return raw, anchors, tuple(openings)

    def update_graph(self, result: BoundSurfaceForwardV1, graph: SurfaceSegmentedGraphV1,
                     pose: KnownSurfacePoseV1, *, continuous=True):
        if (type(graph) is SurfaceSegmentedGraphV1 and self.readout.supervision_policy == 'partial_geometry_v1'
                and graph.config.anchor_uncertainty_policy != 'known_pose_partial_diagnostic'):
            raise ValueError('partial readout requires explicit known-pose partial graph policy')
        item = self._issued.get(id(result))
        if item is None or item[0] is not result or type(graph) is not SurfaceSegmentedGraphV1 or type(pose) is not KnownSurfacePoseV1:
            raise ValueError("only this wrapper's owned forward result may be delivered to a graph")
        if item[1] != _sha(asdict(pose)):
            raise ValueError("old output cannot be attached to a different pose, frame or timestamp")
        return graph.update(result.observation, pose, continuous=continuous)
