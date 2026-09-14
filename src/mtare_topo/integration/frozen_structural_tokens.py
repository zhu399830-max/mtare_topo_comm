"""Frozen, per-return C tokens for structural assistance; no graph decisions.

The producer loads and authenticates the epoch2 common encoder and C500 state.
This module performs no checkpoint/file I/O and cannot authenticate their IDs.
Source references bind caller-declared provenance, not a proof of byte identity.
Current-sensor XYZ uses the existing yaw-only five-frame registration contract;
sensor extrinsics and deployment pose quality remain producer responsibilities.
Geometric support means nearest observed patches exist, NOT membership, line of
sight, physical accessibility or calibrated confidence. No global mean token or
same-group probability is exported. Ordinary voxel extraction is not yaw exact.
"""
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import torch

from mtare_topo.representation.branch_relation_learning import LocalBranchEvidence, RayBranchRelationModel
from mtare_topo.representation.gse_common_observation import prepare_common_observation
from mtare_topo.representation.gse_dual_path_encoder_adapter_v1 import FrozenDualPathEncoderAdapterV1
from mtare_topo.representation.gse_patch_observed_features import pool_patch_observed_features
from mtare_topo.representation.gse_structure_context import CausalFrameOrderContext
from mtare_topo.representation.gse_structural_representation import bind_structural_patches


@dataclass(frozen=True)
class FrozenStructuralTokens:
    context: CausalFrameOrderContext
    source_refs: tuple[str, str, str, str, str]
    ray_ids: np.ndarray
    endpoints_current_sensor_m: np.ndarray
    ray_tokens: np.ndarray
    geometric_support: np.ndarray
    nearest_patch_indices: np.ndarray
    patch_centers_current_sensor_m: np.ndarray
    patch_frame_support: np.ndarray
    sensor_token_indices: np.ndarray
    coordinate_frame: str = 'current_sensor_m'
    support_is_membership_probability: bool = False


def _readonly(value):
    if torch.is_tensor(value):
        value = value.detach().cpu().numpy()
    value = np.asarray(value)
    # Immutable backing bytes, not only a reversible ndarray WRITEABLE flag.
    return np.frombuffer(value.tobytes(order='C'), dtype=value.dtype).reshape(value.shape)


def _student_copy(student):
    fields = {'ranges_m': ((5, 16, 720), np.float32),
              'valid_mask': ((5, 16, 720), None),
              'relative_translation_current_sensor_m': ((5, 3), np.float32),
              'relative_yaw_current_sensor_deg': ((5,), np.float32)}
    clean = {}
    for key, (shape, dtype) in fields.items():
        value = getattr(student, key, None)
        if (type(value) is not np.ndarray or value.shape != shape or
                (dtype is not None and value.dtype != dtype)):
            raise ValueError('student shape/dtype: ' + key)
        if not np.isfinite(value).all():
            raise ValueError('nonfinite student: ' + key)
        clean[key] = value.copy()
    mask = clean['valid_mask']
    if mask.dtype not in (np.dtype('bool'), np.dtype('uint8')) or np.any((mask != 0) & (mask != 1)):
        raise ValueError('binary bool/uint8 validity required')
    ranges = clean['ranges_m']
    if np.any((ranges < 0) | (ranges > 50)) or np.any(ranges[mask.astype(bool)] <= 0):
        raise ValueError('valid return must have range in (0,50] m')
    if (np.any(clean['relative_translation_current_sensor_m'][-1] != 0) or
            clean['relative_yaw_current_sensor_deg'][-1] != 0):
        raise ValueError('current-frame relative motion must be zero')
    return SimpleNamespace(**clean)


class FrozenStructuralTokenExtractor:
    """Owns already-loaded modules; freezes mode/grad, never changes weights.

    All five empty frames produce an empty token set. Partly empty histories
    fail under the unchanged legacy encoder contract, never synthetic fill.
    An empty ROI retains distant current returns with raw-context tokens and
    false geometric support. Patch capacity overflow is not truncated.
    """
    def __init__(self, adapter, c500_model, *, device='cpu'):
        if not isinstance(adapter, FrozenDualPathEncoderAdapterV1):
            raise ValueError('common frozen encoder adapter required')
        if not isinstance(c500_model, RayBranchRelationModel) or c500_model.variant != 'C':
            raise ValueError('C RayBranchRelationModel required; checkpoint producer verifies step500')
        self.device = torch.device(device)
        self.adapter = adapter.to(self.device).eval().requires_grad_(False)
        self.model = c500_model.to(self.device).eval().requires_grad_(False)
        if any(p.dtype != torch.float32 for module in (self.adapter, self.model) for p in module.parameters()):
            raise ValueError('frozen float32 models required')

    @torch.inference_mode()
    def extract(self, student, *, context, source_refs, relative_rotation_current_sensor=None):
        """Optional full rotations must agree with the legacy yaw-only contract.

        A producer holding full rigid poses MUST pass their relative rotations;
        omission declares a yaw-only source, not proof of physical zero tilt.
        Original source frame keys are preserved verbatim in source_refs.
        """
        if (type(context) is not CausalFrameOrderContext or context.coordinate_frame != 'sensor_current' or
                context.observation_frame_id != context.source_frame_ids[-1]):
            raise ValueError('same latest-frame sensor_current context required')
        if (type(source_refs) is not tuple or len(source_refs) != 5 or
                any(type(v) is not str or not v.strip() for v in source_refs)):
            raise ValueError('five explicit source references required')
        student = _student_copy(student)
        if relative_rotation_current_sensor is not None:
            rotation = relative_rotation_current_sensor
            if (type(rotation) is not np.ndarray or rotation.shape != (5, 3, 3) or
                    rotation.dtype not in (np.dtype('float32'), np.dtype('float64')) or
                    not np.isfinite(rotation).all()):
                raise ValueError('finite five-frame relative rotation matrices required')
            angle = np.deg2rad(student.relative_yaw_current_sensor_deg.astype(np.float64))
            expected = np.zeros((5, 3, 3), dtype=np.float64)
            expected[:, 0, 0] = expected[:, 1, 1] = np.cos(angle)
            expected[:, 1, 0] = np.sin(angle)
            expected[:, 0, 1] = -np.sin(angle)
            expected[:, 2, 2] = 1
            # Float32 pose representation roundoff only, not a tilt threshold.
            if not np.allclose(rotation, expected, rtol=0, atol=8 * np.finfo(np.float32).eps):
                raise ValueError('roll/pitch or relative rotation/yaw mismatch: legacy path is yaw-only')
        for module in (self.adapter, self.model):
            if any(p.requires_grad for p in module.parameters()):
                raise ValueError('extractor module externally unfrozen')
            module.eval()
        obs = prepare_common_observation(self.adapter, student, device=self.device)
        patches = bind_structural_patches([obs.surface_patches], device=self.device)
        features = pool_patch_observed_features(obs)
        # The public collator uses one masked padding row for an empty ROI;
        # pad features only, do not invent an observed patch or remove returns.
        if features.shape[1] == 0:
            features = features.new_zeros((1, 1, 128))
        ids = np.flatnonzero(student.valid_mask[-1].reshape(-1)).astype(np.int64)
        indices = 4 * 180 + (ids % 720) // 4
        tensor_ids = torch.from_numpy(ids).to(self.device)
        xyz = torch.from_numpy(np.array(obs.registered_returns_xyz_m[4 * 11520 + ids], copy=True)).to(self.device)
        evidence = LocalBranchEvidence(tensor_ids, xyz, obs.full_sensor_context[indices], features,
                                       patches, context)
        empty_pairs = torch.empty((0, 2), device=self.device, dtype=torch.long)
        prediction = self.model(evidence, empty_pairs)
        n = len(ids)
        k = min(8, len(obs.surface_patches.centers_m))
        if (prediction.ray_tokens.shape != (n, 128) or prediction.geometric_support.shape != (n,) or
                prediction.geometric_support.dtype != torch.bool or
                prediction.nearest_patch_indices.shape != (n, k) or
                prediction.nearest_patch_indices.dtype != torch.long or
                prediction.logits.shape != (0,) or prediction.pair_indices.shape != (0, 2) or
                not bool(torch.isfinite(prediction.ray_tokens).all()) or
                bool((prediction.geometric_support != (k > 0)).any())):
            raise ValueError('token output/support shape or finite drift')
        if bool(((prediction.nearest_patch_indices < 0) |
                 (prediction.nearest_patch_indices >= len(obs.surface_patches.centers_m))).any()):
            raise ValueError('nearest patch provenance drift')
        return FrozenStructuralTokens(context, source_refs, _readonly(ids), _readonly(xyz),
            _readonly(prediction.ray_tokens), _readonly(prediction.geometric_support),
            _readonly(prediction.nearest_patch_indices), _readonly(obs.surface_patches.centers_m),
            _readonly(obs.surface_patches.frame_support), _readonly(indices))
