"""Explicit full-SE3 production path, reusing v1 source/IO/weight safeguards.

Same raw buffers and full sensor poses, newly registered XYZ through both
coordinate and encoder paths. This is a versioned preprocessing change, not an
equivalence claim or relaxation of the old yaw compatibility guard. No weights,
teacher, ROI, token count or optimizer changes; v1 defaults stay unchanged.
"""
from torch import nn
from mtare_topo.integration import frozen_structural_token_worker as legacy
from mtare_topo.integration.frozen_structural_se3_v1 import (
    relative_sensor_motion_se3, FrozenSE3EncoderAdapterV1, FrozenStructuralSE3TokenExtractor)

VERSION = 'full_relative_se3_v1'


def _checked_motion(poses):
    try:
        return relative_sensor_motion_se3(poses)
    except ValueError as exc:
        raise legacy.RejectedRecord('INVALID_FULL_SE3_SENSOR_POSES: '+str(exc)) from exc


def prepare_window(entry, raw_pointcloud_bytes, world_from_sensor):
    return legacy.prepare_window(entry, raw_pointcloud_bytes, world_from_sensor,
                                 _pose_converter=_checked_motion)


def load_registered_extractor(root, *, device):
    loaded = legacy.load_registered_extractor(root, device=device)
    adapter = FrozenSE3EncoderAdapterV1(loaded.adapter.backbone, nn.Identity())
    return FrozenStructuralSE3TokenExtractor(adapter, loaded.model, device=device)


def run_worker(root, manifest_reference, output_dir, *, device):
    return legacy.run_worker(root, manifest_reference, output_dir, device=device,
        _prepare=prepare_window, _extractor_loader=load_registered_extractor,
        _preprocessing_version=VERSION)
