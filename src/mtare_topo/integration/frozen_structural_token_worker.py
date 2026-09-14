"""File-only frozen structural-token producer; never launches ROS or control.

CLI: python -m mtare_topo.integration.frozen_structural_token_worker
  --project-root ROOT --manifest FILE --manifest-sha256 SHA --output NEWDIR
  --device cpu|cuda

Manifest schema frozen_structural_token_windows_v1: {schema_version, windows}.
Each window has window_id, sequence_id, input:{path,sha256}, pose_binding:
{mode:tf|commanded_pose,reference}, frames:[{frame_order,stamp_ns,source_key,
frame_id,pose_source_key,pointcloud_layout}] (exactly five in causal order).
Input NPZ has ONLY raw_pointcloud_bytes uint8[5,123200] and world_from_sensor
float64[5,4,4]. The bag producer binds full sensor poses to the original raw
timestamps using real extrinsics; this worker never equates registered and raw
cloud timestamps. Layout is actual PointCloud2 metadata including all fields.
Source records, transforms, stamps and adaptation audits accompany every output.

The original encoder is yaw-only. Non-yaw relative rotations are explicitly
REJECTED_RECORD before weights are read, not flattened, removed or interpolated.
No window selection is performed here. The frozen manifest supplies all windows.
This is worker evidence, not create_run/preflight or an execution authorization.
"""
from __future__ import annotations

import argparse
from dataclasses import fields
import hashlib
import io
import json
from pathlib import Path
from types import SimpleNamespace
import zipfile

import numpy as np
import torch
from torch import nn

from mtare_topo.integration.aee_organized_scan_adapter import aee_organized_pointcloud2_to_range_image
from mtare_topo.integration.frozen_structural_tokens import FrozenStructuralTokenExtractor
from mtare_topo.representation.branch_relation_learning import RayBranchRelationModel
from mtare_topo.representation.gse_dual_path_encoder_adapter_v1 import FrozenDualPathEncoderAdapterV1
from mtare_topo.representation.gse_structure_context import CausalFrameOrderContext
from mtare_topo.representation.gse_surface_encoder_checkpoint_v1 import load_surface_encoder


ENCODER = ('results/gate3_semantics/gate3_20260902_primitive_relation_observable_three_seed_training_v1_seed0/artifacts/models/seed0/selected.pt',
           '8d5d2e0ec9779c28b068d0c7db80aeed38ebdb22dff5b781c26234d9001287fb')
RELATION = ('results/gate3_semantics/gate3_20260913_gse_branch_core_fit_v1_seed0/artifacts/step_0500.pt',
            'ac1badf396804debb2b7daa505375bb0885a9d79bd918832a9f66e16697f6a76')


class RejectedRecord(ValueError):
    """Preserve the window and reason; do not substitute another observation."""


def _sha(payload):
    return hashlib.sha256(payload).hexdigest()


def _json_bytes(payload):
    def unique(pairs):
        result = {}
        for k, v in pairs:
            if k in result:
                raise ValueError('duplicate JSON key: ' + k)
            result[k] = v
        return result
    return json.loads(payload, object_pairs_hook=unique,
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError('nonfinite JSON')))


def _read_bound(root, reference, *, limit):
    if type(reference) is not dict or set(reference) != {'path', 'sha256'}:
        raise ValueError('exact path/SHA reference required')
    path, expected = reference['path'], reference['sha256']
    if type(path) is not str or not path or Path(path).is_absolute() or '..' in Path(path).parts:
        raise ValueError('project-relative exact file path required')
    current = Path(root).resolve()
    for part in Path(path).parts:
        current = current / part
        if current.is_symlink():
            raise ValueError('symlink source forbidden')
    if current.stat().st_size > limit:
        raise ValueError('source size cap exceeded')
    payload = current.read_bytes()
    if _sha(payload) != expected:
        raise ValueError('source SHA-256 drift: ' + path)
    return payload


class _BufferPointCloud2:
    @staticmethod
    def read_points(message, *, field_names, skip_nans):
        if field_names != ('x', 'y', 'z', 'ring') or skip_nans:
            raise ValueError('retain every original beam record')
        dtype = np.dtype({'names': ['x', 'y', 'z', 'ring'],
                          'formats': ['<f4', '<f4', '<f4', '<u2'],
                          'offsets': [0, 4, 8, 16], 'itemsize': 22})
        return ((row['x'], row['y'], row['z'], row['ring'])
                for row in np.frombuffer(message.data, dtype=dtype))


def _relative_pose(poses):
    if type(poses) is not np.ndarray or poses.shape != (5, 4, 4) or poses.dtype != np.float64 or not np.isfinite(poses).all():
        raise RejectedRecord('INVALID_FULL_SENSOR_POSES')
    if not np.array_equal(poses[:, 3], np.tile([0., 0., 0., 1.], (5, 1))):
        raise RejectedRecord('INVALID_HOMOGENEOUS_POSES')
    rotation = poses[:, :3, :3]
    if (not np.allclose(rotation.transpose(0, 2, 1) @ rotation, np.eye(3), atol=1e-10, rtol=0) or
            not np.allclose(np.linalg.det(rotation), 1, atol=1e-10, rtol=0)):
        raise RejectedRecord('INVALID_PROPER_ROTATION')
    relative_rotation = rotation[-1].T @ rotation
    translation = (poses[:, :3, 3] - poses[-1, :3, 3]) @ rotation[-1]
    yaw = np.rad2deg(np.arctan2(relative_rotation[:, 1, 0], relative_rotation[:, 0, 0]))
    angle = np.deg2rad(yaw)
    expected = np.zeros_like(relative_rotation)
    expected[:, 0, 0] = expected[:, 1, 1] = np.cos(angle)
    expected[:, 1, 0] = np.sin(angle)
    expected[:, 0, 1] = -np.sin(angle)
    expected[:, 2, 2] = 1
    if not np.allclose(relative_rotation, expected, atol=8 * np.finfo(np.float32).eps, rtol=0):
        raise RejectedRecord('INCOMPATIBLE_FULL_ROTATION: frozen common encoder supports relative yaw only')
    # Exact mathematical identity for the current frame, not sensor tilt removal.
    translation[-1] = 0
    yaw[-1] = 0
    relative_rotation[-1] = np.eye(3)
    return translation.astype(np.float32), yaw.astype(np.float32), relative_rotation


def prepare_window(entry, raw_pointcloud_bytes, world_from_sensor, *, _pose_converter=None):
    """Pure raw-buffer adaptation, pose compatibility and source-order checks."""
    expected_keys = {'window_id', 'sequence_id', 'input', 'pose_binding', 'frames'}
    if type(entry) is not dict or set(entry) != expected_keys:
        raise RejectedRecord('WINDOW_SCHEMA')
    binding = entry['pose_binding']
    if (type(binding) is not dict or set(binding) != {'mode', 'reference'} or
            binding['mode'] not in ('tf', 'commanded_pose') or
            type(binding['reference']) is not str or not binding['reference'].strip()):
        raise RejectedRecord('EXPLICIT_SENSOR_POSE_BINDING_REQUIRED')
    frames = entry['frames']
    if type(frames) is not list or len(frames) != 5:
        raise RejectedRecord('EXACT_FIVE_SOURCE_FRAMES_REQUIRED')
    for frame in frames:
        if type(frame) is not dict or set(frame) != {'frame_order', 'stamp_ns', 'source_key', 'frame_id', 'pose_source_key', 'pointcloud_layout'}:
            raise RejectedRecord('FRAME_SCHEMA')
        if any(type(frame[k]) is not int or frame[k] < 0 for k in ('frame_order', 'stamp_ns')):
            raise RejectedRecord('FRAME_ORDER_TIMESTAMP_TYPE')
        if any(type(frame[k]) is not str or not frame[k].strip() for k in ('source_key', 'frame_id', 'pose_source_key')):
            raise RejectedRecord('ORIGINAL_FRAME_POSE_REFERENCES_REQUIRED')
    for key in ('frame_order', 'stamp_ns'):
        if any(a[key] >= b[key] for a, b in zip(frames, frames[1:])):
            raise RejectedRecord('NONCAUSAL_OR_DUPLICATE_FRAME_ORDER')
    if len({f['source_key'] for f in frames}) != 5 or len({f['frame_id'] for f in frames}) != 1:
        raise RejectedRecord('DUPLICATE_SOURCE_OR_SENSOR_FRAME_DRIFT')
    if type(raw_pointcloud_bytes) is not np.ndarray or raw_pointcloud_bytes.dtype != np.uint8 or raw_pointcloud_bytes.shape != (5, 123200):
        raise RejectedRecord('EXACT_FIVE_RAW_POINTCLOUD_BYTE_BUFFERS_REQUIRED')
    converter = _relative_pose if _pose_converter is None else _pose_converter
    translation, yaw, rotations = converter(world_from_sensor)
    ranges, masks, audits = [], [], []
    for index, frame in enumerate(frames):
        layout = frame['pointcloud_layout']
        if type(layout) is not dict or set(layout) != {'width', 'height', 'point_step', 'row_step', 'is_bigendian', 'is_dense', 'fields'}:
            raise RejectedRecord('POINTCLOUD_LAYOUT_SCHEMA')
        if any(type(layout[k]) is not int for k in ('width', 'height', 'point_step', 'row_step')) or any(type(layout[k]) is not bool for k in ('is_bigendian', 'is_dense')):
            raise RejectedRecord('POINTCLOUD_LAYOUT_TYPES')
        fs = layout['fields']
        if type(fs) is not list or any(type(f) is not dict or set(f) != {'name', 'offset', 'datatype', 'count'} for f in fs):
            raise RejectedRecord('POINTCLOUD_FIELDS_SCHEMA')
        if any(type(f['name']) is not str or type(f['count']) is not int or f['count'] != 1 or
               type(f['offset']) is not int or type(f['datatype']) is not int for f in fs) or len({f['name'] for f in fs}) != len(fs):
            raise RejectedRecord('POINTCLOUD_FIELDS_TYPES_OR_DUPLICATES')
        message = SimpleNamespace(**{k: v for k, v in layout.items() if k != 'fields'},
                                  fields=[SimpleNamespace(**f) for f in fs], data=raw_pointcloud_bytes[index].tobytes())
        try:
            r, v, audit = aee_organized_pointcloud2_to_range_image(message, _BufferPointCloud2)
        except ValueError as exc:
            raise RejectedRecord('RAW_ORGANIZED_ADAPTER: ' + str(exc)) from exc
        ranges.append(r); masks.append(v); audits.append(audit.to_dict())
    student = SimpleNamespace(ranges_m=np.stack(ranges), valid_mask=np.stack(masks),
        relative_translation_current_sensor_m=translation, relative_yaw_current_sensor_deg=yaw)
    context = CausalFrameOrderContext('sensor_current', tuple(f['frame_order'] for f in frames),
                                     frames[-1]['frame_order'], entry['sequence_id'])
    return student, context, tuple(f['source_key'] for f in frames), rotations, audits


def load_registered_extractor(root, *, device):
    """Pinned immutable bytes, safe deserialize, strict tensors, no fallback."""
    encoder_bytes = _read_bound(root, dict(zip(('path', 'sha256'), ENCODER)), limit=512 * 1024**2)
    relation_bytes = _read_bound(root, dict(zip(('path', 'sha256'), RELATION)), limit=512 * 1024**2)
    bound = load_surface_encoder(encoder_bytes, expected_sha256=ENCODER[1], expected_epoch=2)
    state = torch.load(io.BytesIO(relation_bytes), map_location='cpu', weights_only=True)
    if (type(state) is not dict or type(state.get('step')) is not int or state['step'] != 500 or
            type(state.get('seed')) is not int or state['seed'] != 0 or
            state.get('purpose') != 'PARTIAL_AI_CORE_FIT_NOT_GENERALIZATION' or not isinstance(state.get('model'), dict)):
        raise ValueError('C500 checkpoint metadata drift')
    with torch.random.fork_rng(devices=[]):
        model = RayBranchRelationModel('C')
    reference = model.state_dict()
    if set(reference) != set(state['model']):
        raise ValueError('C500 tensor keys drift')
    for k, value in reference.items():
        actual = state['model'][k]
        if not torch.is_tensor(actual) or actual.layout != torch.strided or actual.shape != value.shape or actual.dtype != value.dtype or not bool(torch.isfinite(actual).all()):
            raise ValueError('C500 tensor drift: ' + k)
    model.load_state_dict(state['model'], strict=True)
    adapter = FrozenDualPathEncoderAdapterV1(bound.backbone, nn.Identity())
    return FrozenStructuralTokenExtractor(adapter, model, device=device)


def _write_json(path, value):
    with path.open('x', encoding='utf8') as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')


def run_worker(root, manifest_reference, output_dir, *, device,
               _prepare=None, _extractor_loader=None, _preprocessing_version=None):
    """Process every preregistered window, preserving rejection and source rows."""
    payload = _read_bound(root, manifest_reference, limit=16 * 1024**2)
    manifest = _json_bytes(payload)
    if type(manifest) is not dict or set(manifest) != {'schema_version', 'windows'} or manifest['schema_version'] != 'frozen_structural_token_windows_v1':
        raise ValueError('manifest schema')
    windows = manifest['windows']
    if type(windows) is not list or not windows or any(type(w) is not dict or type(w.get('window_id')) is not str or not w['window_id'] for w in windows):
        raise ValueError('nonempty explicit window population required')
    if len({w['window_id'] for w in windows}) != len(windows):
        raise ValueError('duplicate window IDs')
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    prepare = prepare_window if _prepare is None else _prepare
    load_extractor = load_registered_extractor if _extractor_loader is None else _extractor_loader
    rows, extractor, error = [], None, None
    _write_json(output/'request.json', manifest)
    try:
        for index, entry in enumerate(windows):
            record = dict(window_id=entry['window_id'], source=entry, model_forward=False)
            if _preprocessing_version is not None:
                record['preprocessing_version'] = _preprocessing_version
            try:
                raw = _read_bound(root, entry['input'], limit=8 * 1024**2)
                with zipfile.ZipFile(io.BytesIO(raw)) as archive:
                    if sum(i.file_size for i in archive.infolist()) > 8 * 1024**2:
                        raise ValueError('input NPZ expansion cap')
                with np.load(io.BytesIO(raw), allow_pickle=False) as data:
                    if sorted(data.files) != ['raw_pointcloud_bytes', 'world_from_sensor']:
                        raise RejectedRecord('INPUT_NPZ_EXACT_TWO_FIELDS_REQUIRED')
                    buffers, poses = data['raw_pointcloud_bytes'], data['world_from_sensor']
                student, context, refs, rotations, audits = prepare(entry, buffers, poses)
                if extractor is None:
                    extractor = load_extractor(root, device=device)
                record['model_forward'] = True  # attempted call, including rejected calls
                try:
                    result = extractor.extract(student, context=context, source_refs=refs,
                                               relative_rotation_current_sensor=rotations)
                except ValueError as exc:
                    raise RejectedRecord('FROZEN_INPUT_INCOMPATIBLE: ' + str(exc)) from exc
                filename = f'tokens_{index:06d}.npz'
                arrays = {f.name: getattr(result, f.name) for f in fields(result)
                          if isinstance(getattr(result, f.name), np.ndarray)}
                arrays.update(world_from_sensor=poses, relative_rotation_current_sensor=rotations,
                              source_stamp_ns=np.array([f['stamp_ns'] for f in entry['frames']], dtype=np.int64))
                with (output/filename).open('xb') as stream:
                    np.savez_compressed(stream, **arrays)
                record.update(status='PROCESSED', model_forward=True, ray_count=len(result.ray_ids),
                              token_ref={'path': filename, 'sha256': _sha((output/filename).read_bytes())},
                              stamp_ns=entry['frames'][-1]['stamp_ns'], adaptation_audits=audits,
                              coordinate_frame=result.coordinate_frame)
            except RejectedRecord as exc:
                record.update(status='REJECTED_RECORD', reason=str(exc))
            except Exception as exc:
                record.update(status='FATAL_SOURCE_OR_EXECUTION_ERROR', reason=type(exc).__name__ + ': ' + str(exc))
                rows.append(record)
                _write_json(output/f'record_{index:06d}.json', record)
                raise
            rows.append(record)
            _write_json(output/f'record_{index:06d}.json', record)
    except Exception as exc:
        error = type(exc).__name__ + ': ' + str(exc)
    summary = dict(schema_version='frozen_structural_token_worker_summary_v1', error=error,
        status='FAILED' if error else 'COMPLETED', requested_windows=len(windows), recorded_windows=len(rows),
        processed_windows=sum(r['status'] == 'PROCESSED' for r in rows),
        rejected_windows=sum(r['status'] == 'REJECTED_RECORD' for r in rows),
        model_forwards=sum(r['model_forward'] for r in rows),
        model_forward_counter_semantics='extractor call attempts, including rejected calls; not a GPU kernel count',
        no_valid_transfer=not any(r.get('ray_count', 0) > 0 for r in rows),
        training_steps=0, relation_probabilities_exported=False, manifest=manifest_reference,
        checkpoints={'encoder': dict(zip(('path', 'sha256'), ENCODER)), 'relation': dict(zip(('path', 'sha256'), RELATION))})
    if _preprocessing_version is not None:
        summary['preprocessing_version'] = _preprocessing_version
        summary['old_yaw_outputs_overwritten'] = False
    _write_json(output/'summary.json', summary)
    _write_json(output/'files_sha256.json', {p.name: _sha(p.read_bytes()) for p in sorted(output.iterdir()) if p.is_file()})
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root', required=True)
    parser.add_argument('--manifest', required=True, help='project-relative path')
    parser.add_argument('--manifest-sha256', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', choices=('cpu', 'cuda'), required=True)
    args = parser.parse_args()
    summary = run_worker(args.project_root, {'path': args.manifest, 'sha256': args.manifest_sha256}, args.output, device=args.device)
    print(json.dumps(summary, sort_keys=True))
    return int(summary['error'] is not None or summary['no_valid_transfer'])


if __name__ == '__main__':
    raise SystemExit(main())
