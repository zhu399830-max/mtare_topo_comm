"""Versioned complete relative-SE3 wiring for unchanged frozen token models.

No files, checkpoint reads, model selection, new parameters or training. A single
rotation tensor enters BOTH compact XYZ registration and the existing backbone
_memory. Geometry and 128D context are rederived together; yaw caches must not be
reused. The existing yaw pose embedding is retained (atan2(R10,R00) convention).
This changes input preprocessing, not model architecture. Software consistency
does not establish transfer quality or exact rotation equivariance of voxel/CNN
features. The old yaw-only entry points and their rejection evidence are intact.
"""
import numpy as np
import torch

from mtare_topo.integration.frozen_structural_tokens import (
    FrozenStructuralTokenExtractor, FrozenStructuralTokens, _student_copy, _readonly,
)
from mtare_topo.representation.branch_relation_learning import LocalBranchEvidence
from mtare_topo.representation.gse_common_observation import assemble_common_observation
from mtare_topo.representation.gse_dual_path_encoder_adapter_v1 import (
    FrozenDualPathEncoderAdapterV1, CompactFrozenDualPathFeatures, expand_features,
)
from mtare_topo.representation.gse_patch_observed_features import pool_patch_observed_features
from mtare_topo.representation.gse_structure_context import CausalFrameOrderContext
from mtare_topo.representation.gse_structural_representation import bind_structural_patches
from mtare_topo.representation.primitive_relation_model import register_causal_lidar_points, _token_xyz


def relative_sensor_motion_se3(world_from_sensor):
    """Pure full-pose conversion; no robot/world geometry or timestamp matching.

    Caller binds these five sensor (not robot-base) poses to actual raw frames.
    Roll/pitch are retained. atan2 yaw is only the unchanged pose-embedding input.
    """
    p = world_from_sensor
    if (type(p) is not np.ndarray or p.shape != (5, 4, 4) or p.dtype != np.float64 or
            not np.isfinite(p).all() or not np.array_equal(p[:, 3], np.tile([0., 0., 0., 1.], (5, 1)))):
        raise ValueError('five finite float64 homogeneous sensor poses required')
    r = p[:, :3, :3]
    if (not np.allclose(r.transpose(0, 2, 1) @ r, np.eye(3), atol=1e-10, rtol=0) or
            not np.allclose(np.linalg.det(r), 1, atol=1e-10, rtol=0)):
        raise ValueError('proper sensor rotations required')
    rotation = r[-1].T @ r
    translation = (p[:, :3, 3] - p[-1, :3, 3]) @ r[-1]
    yaw = np.rad2deg(np.arctan2(rotation[:, 1, 0], rotation[:, 0, 0]))
    # These are exact mathematical identities, not a tilt simplification.
    rotation[-1] = np.eye(3); translation[-1] = 0; yaw[-1] = 0
    return translation.astype(np.float32), yaw.astype(np.float32), rotation


def _validate_rotation_tensor(rotation, range_valid, yaw):
    b = range_valid.shape[0] if range_valid.ndim else 0
    if (not torch.is_tensor(rotation) or rotation.shape != (b, 5, 3, 3) or
            rotation.dtype != range_valid.dtype or rotation.device != range_valid.device or
            not bool(torch.isfinite(rotation).all())):
        raise ValueError('explicit same-dtype/device B,5,3,3 full rotation required')
    if not torch.equal(rotation[:, -1], torch.eye(3, device=rotation.device, dtype=rotation.dtype).expand(b, 3, 3)):
        raise ValueError('exact current identity required in direct SE3 adapter')
    angle = torch.deg2rad(yaw)
    derived = torch.atan2(rotation[..., 1, 0], rotation[..., 0, 0])
    if (not torch.allclose(angle.sin(), derived.sin(), atol=8*torch.finfo(rotation.dtype).eps, rtol=0) or
            not torch.allclose(angle.cos(), derived.cos(), atol=8*torch.finfo(rotation.dtype).eps, rtol=0)):
        raise ValueError('yaw embedding must match full rotation atan2 convention')
    # Proper rotation and exact current identity are also checked by the existing
    # register function below, without weakening its numerical checks.


class FrozenSE3EncoderAdapterV1(FrozenDualPathEncoderAdapterV1):
    """Same supplied modules/parameter layout; R is mandatory, never default yaw."""
    @torch.no_grad()
    def extract_compact_features(self, range_valid, relative_translation_current_sensor_m,
                                 relative_yaw_current_sensor_deg, *, relative_rotation_current_sensor):
        self.backbone.eval()
        if any(p.requires_grad for p in self.backbone.parameters()):
            raise ValueError('backbone was unfrozen outside the adapter')
        inputs = (range_valid, relative_translation_current_sensor_m, relative_yaw_current_sensor_deg)
        if (not all(torch.is_tensor(v) for v in inputs) or range_valid.dtype not in (torch.float32, torch.float64) or
                any(v.dtype != range_valid.dtype or v.device != range_valid.device for v in inputs) or
                range_valid.ndim != 5 or len(range_valid) < 1 or range_valid.shape[1:] != (5, 2, 16, 720) or
                relative_translation_current_sensor_m.shape != (len(range_valid), 5, 3) or
                relative_yaw_current_sensor_deg.shape != (len(range_valid), 5)):
            raise ValueError('common floating student tensor dtype/device/layout required')
        rotation = relative_rotation_current_sensor
        _validate_rotation_tensor(rotation, range_valid, relative_yaw_current_sensor_deg)
        points, valid = register_causal_lidar_points(*inputs, relative_rotation_current_sensor=rotation)
        b = len(points)
        expected_xyz, token_valid = _token_xyz(points, valid)
        supported_frames = token_valid.any(-1)
        keep = supported_frames.any(-1)
        if bool((keep & ~supported_frames.all(-1)).any()):
            raise ValueError('legacy encoder requires all five frames supported; partial empty history')
        t = torch.arange(5, device=points.device)[:, None, None]
        az = torch.arange(720, device=points.device)[None, None]
        index = (t*180 + az//4).expand(5, 16, 720).reshape(1, 57600).expand(b, -1).clone()
        context = points.new_zeros((b, 900, 128))
        if bool(keep.any()):
            output = self.backbone._memory(*(v[keep] for v in inputs),
                                           relative_rotation_current_sensor=rotation[keep])
            if not isinstance(output, (tuple, list)) or len(output) != 4:
                raise ValueError('legacy memory four-field contract required')
            memory, memory_valid, memory_xyz, frame_memory = output
            k = int(keep.sum())
            shapes = ((k, 900, 128), (k, 900), (k, 900, 3), (k, 5, 180, 128))
            if any(not torch.is_tensor(v) or v.shape != shape for v, shape in zip(output, shapes)):
                raise ValueError('legacy memory shape drift')
            if (memory_valid.dtype != torch.bool or any(v.device != points.device for v in output) or
                    any(v.dtype != points.dtype for v in (memory, memory_xyz, frame_memory))):
                raise ValueError('legacy memory dtype/device drift')
            if not torch.equal(memory_valid, token_valid[keep].reshape(k, 900)):
                raise ValueError('legacy token validity disagrees with SE3 registered returns')
            if not torch.equal(memory_xyz, expected_xyz[keep].reshape(k, 900, 3)):
                raise ValueError('legacy token coordinates disagree with SE3 registered returns')
            clean = torch.where(memory_valid[..., None], memory, 0.)
            clean_frame = torch.where(token_valid[keep][..., None], frame_memory, 0.)
            if not bool(torch.isfinite(clean).all() and torch.isfinite(clean_frame).all()):
                raise ValueError('nonfinite supported legacy context')
            context[keep] = clean
        return CompactFrozenDualPathFeatures(points.reshape(b, 57600, 3).detach(), context.detach(),
                                             valid.reshape(b, 57600).detach(), index.detach())

    def extract_features(self, *args, relative_rotation_current_sensor):
        return expand_features(self.extract_compact_features(*args,
            relative_rotation_current_sensor=relative_rotation_current_sensor))

    def forward(self, *args, relative_rotation_current_sensor):
        f = self.extract_features(*args, relative_rotation_current_sensor=relative_rotation_current_sensor)
        return self.model(f.points_xyz_m, f.frozen_point_context, f.valid)


def _rotation_array(rotation):
    if (type(rotation) is not np.ndarray or rotation.shape != (5, 3, 3) or
            rotation.dtype not in (np.dtype('float32'), np.dtype('float64')) or not np.isfinite(rotation).all()):
        raise ValueError('five finite floating complete relative rotations required')
    if not np.array_equal(rotation[-1], np.eye(3)):
        raise ValueError('exact current identity required before dtype conversion')
    if (not np.allclose(rotation.transpose(0, 2, 1) @ rotation, np.eye(3), atol=1e-6, rtol=0) or
            not np.allclose(np.linalg.det(rotation), 1., atol=1e-6, rtol=0)):
        raise ValueError('proper relative rotations required')
    return np.array(rotation, dtype=np.float32, copy=True)


@torch.no_grad()
def prepare_common_observation_se3(adapter, student, *, relative_rotation_current_sensor, device):
    if not isinstance(adapter, FrozenSE3EncoderAdapterV1):
        raise ValueError('explicit full-SE3 adapter required')
    student = _student_copy(student)
    rotation = _rotation_array(relative_rotation_current_sensor)
    rv = np.stack((student.ranges_m/np.float32(50), student.valid_mask.astype(np.float32)), axis=1)
    args = [torch.from_numpy(np.array(a, copy=True))[None].to(device) for a in
            (rv, student.relative_translation_current_sensor_m, student.relative_yaw_current_sensor_deg)]
    compact = adapter.extract_compact_features(*args,
        relative_rotation_current_sensor=torch.from_numpy(rotation)[None].to(device))
    return assemble_common_observation(compact, student.relative_translation_current_sensor_m)


class FrozenStructuralSE3TokenExtractor(FrozenStructuralTokenExtractor):
    """Same all-ray DTO and empty-pair C path, with fully realigned geometry."""
    preprocessing_version = 'full_relative_se3_v1'

    def __init__(self, adapter, c500_model, *, device='cpu'):
        if not isinstance(adapter, FrozenSE3EncoderAdapterV1):
            raise ValueError('full-SE3 adapter required, not old yaw-only adapter')
        super().__init__(adapter, c500_model, device=device)

    @torch.inference_mode()
    def extract(self, student, *, context, source_refs, relative_rotation_current_sensor):
        if (type(context) is not CausalFrameOrderContext or context.coordinate_frame != 'sensor_current' or
                context.observation_frame_id != context.source_frame_ids[-1]):
            raise ValueError('same latest-frame sensor_current context required')
        if (type(source_refs) is not tuple or len(source_refs) != 5 or
                any(type(v) is not str or not v.strip() for v in source_refs) or len(set(source_refs)) != 5):
            raise ValueError('five explicit original source references required')
        student = _student_copy(student)
        for module in (self.adapter, self.model):
            if any(p.requires_grad for p in module.parameters()):
                raise ValueError('extractor module externally unfrozen')
            module.eval()
        obs = prepare_common_observation_se3(self.adapter, student,
            relative_rotation_current_sensor=relative_rotation_current_sensor, device=self.device)
        patches = bind_structural_patches([obs.surface_patches], device=self.device)
        features = pool_patch_observed_features(obs)
        if features.shape[1] == 0:
            features = features.new_zeros((1, 1, 128))
        ids = np.flatnonzero(student.valid_mask[-1].reshape(-1)).astype(np.int64)
        indices = 4*180 + (ids % 720)//4
        xyz = torch.from_numpy(np.array(obs.registered_returns_xyz_m[4*11520+ids], copy=True)).to(self.device)
        evidence = LocalBranchEvidence(torch.from_numpy(ids).to(self.device), xyz,
            obs.full_sensor_context[indices], features, patches, context)
        prediction = self.model(evidence, torch.empty((0, 2), device=self.device, dtype=torch.long))
        n, k = len(ids), min(8, len(obs.surface_patches.centers_m))
        if (prediction.ray_tokens.shape != (n, 128) or not bool(torch.isfinite(prediction.ray_tokens).all()) or
                prediction.geometric_support.shape != (n,) or prediction.geometric_support.dtype != torch.bool or
                prediction.nearest_patch_indices.shape != (n, k) or prediction.nearest_patch_indices.dtype != torch.long or
                prediction.logits.shape != (0,) or prediction.pair_indices.shape != (0, 2) or
                bool((prediction.geometric_support != (k > 0)).any()) or
                bool(((prediction.nearest_patch_indices < 0) |
                      (prediction.nearest_patch_indices >= len(obs.surface_patches.centers_m))).any())):
            raise ValueError('full-SE3 token output finite/layout/support drift')
        return FrozenStructuralTokens(context, source_refs, _readonly(ids), _readonly(xyz),
            _readonly(prediction.ray_tokens), _readonly(prediction.geometric_support),
            _readonly(prediction.nearest_patch_indices), _readonly(obs.surface_patches.centers_m),
            _readonly(obs.surface_patches.frame_support), _readonly(indices))
