"""Synthetic raw PointCloud2 buffers and poses; no actual checkpoint/bag reads."""
import copy
from dataclasses import dataclass
import hashlib
import io
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from mtare_topo.integration import frozen_structural_token_worker as worker


def fixture():
    layout = dict(width=16, height=350, point_step=22, row_step=352,
                  is_bigendian=False, is_dense=False,
                  fields=[dict(name=n, offset=o, datatype=t, count=1)
                          for n, o, t in [('x', 0, 7), ('y', 4, 7), ('z', 8, 7), ('ring', 16, 4)]])
    entry = dict(window_id='synthetic-0', sequence_id='raw-bag', input={'path': 'input.npz', 'sha256': 'unused'},
                 pose_binding={'mode': 'tf', 'reference': 'synthetic-exact-raw-stamp-T-world-sensor'},
                 frames=[dict(frame_order=i, stamp_ns=1000000000+i*200000000,
                              source_key=f'raw-topic@{i}', frame_id='sensor', pose_source_key=f'pose@{i}',
                              pointcloud_layout=copy.deepcopy(layout)) for i in range(5)])
    dtype = np.dtype({'names': ['x', 'y', 'z', 'ring'], 'formats': ['<f4', '<f4', '<f4', '<u2'],
                      'offsets': [0, 4, 8, 16], 'itemsize': 22})
    records = np.zeros((5, 350, 16), dtype=dtype)
    records['x'] = 4; records['ring'] = np.arange(16)
    raw = records.view(np.uint8).reshape(5, 123200).copy()
    poses = np.tile(np.eye(4), (5, 1, 1))
    poses[:, 0, 3] = np.arange(5)*.1
    return entry, raw, poses


def test_raw_buffer_real_adapter_and_full_pose_keep_sources():
    entry, raw, poses = fixture()
    student, context, refs, rotations, audits = worker.prepare_window(entry, raw, poses)
    assert student.ranges_m.shape == (5, 16, 720)
    assert student.ranges_m.dtype == np.float32
    np.testing.assert_allclose(student.ranges_m, 4)
    assert student.valid_mask.all()
    np.testing.assert_allclose(student.relative_translation_current_sensor_m[:, 0], [-.4, -.3, -.2, -.1, 0])
    np.testing.assert_allclose(rotations, np.tile(np.eye(3), (5, 1, 1)))
    assert refs == tuple(f['source_key'] for f in entry['frames'])
    assert context.source_frame_ids == (0, 1, 2, 3, 4)
    assert all(a['input_points'] == 5600 for a in audits)


def test_valid_full_rotation_is_not_silently_reduced_to_yaw():
    entry, raw, poses = fixture()
    theta = .05
    poses[0, :3, :3] = [[1, 0, 0], [0, np.cos(theta), -np.sin(theta)], [0, np.sin(theta), np.cos(theta)]]
    with pytest.raises(worker.RejectedRecord, match='INCOMPATIBLE_FULL_ROTATION'):
        worker.prepare_window(entry, raw, poses)


@pytest.mark.parametrize('problem', ['stamp', 'key', 'frame', 'pose', 'binding', 'ring', 'layout', 'field'])
def test_malformed_or_noncausal_record_rejected(problem):
    entry, raw, poses = fixture()
    if problem == 'stamp': entry['frames'][2]['stamp_ns'] = entry['frames'][1]['stamp_ns']
    elif problem == 'key': entry['frames'][1]['source_key'] = entry['frames'][0]['source_key']
    elif problem == 'frame': entry['frames'][1]['frame_id'] = 'map'
    elif problem == 'pose': poses[1, 0, 0] = 2
    elif problem == 'binding': entry['pose_binding'] = {}
    elif problem == 'ring': raw[0, 16] = 9
    elif problem == 'layout': entry['frames'][0]['pointcloud_layout']['width'] = 720
    elif problem == 'field': entry['frames'][0]['pointcloud_layout']['fields'].append(dict(name='x', offset=0, datatype=7, count=1))
    with pytest.raises(worker.RejectedRecord): worker.prepare_window(entry, raw, poses)


def write_manifest(tmp_path, entries):
    windows = []
    for i, (entry, raw, poses) in enumerate(entries):
        entry = copy.deepcopy(entry); entry['window_id'] = f'window-{i}'
        path = tmp_path/f'input{i}.npz'
        np.savez_compressed(path, raw_pointcloud_bytes=raw, world_from_sensor=poses)
        entry['input'] = {'path': path.name, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
        windows.append(entry)
    manifest = {'schema_version': 'frozen_structural_token_windows_v1', 'windows': windows}
    path = tmp_path/'manifest.json'; path.write_text(json.dumps(manifest))
    return {'path': path.name, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


@dataclass
class FakeOutput:
    ray_ids: np.ndarray
    ray_tokens: np.ndarray
    coordinate_frame: str = 'current_sensor_m'


def test_worker_real_raw_read_decode_write_and_rejection_preserved(tmp_path, monkeypatch):
    entry, raw, poses = fixture()
    bad = poses.copy(); bad[0, :3, :3] = [[1, 0, 0], [0, 0, -1], [0, 1, 0]]
    manifest = write_manifest(tmp_path, [(entry, raw, poses), (entry, raw, bad)])
    calls = []
    class FakeExtractor:
        def extract(self, student, **kwargs):
            calls.append((student, kwargs))
            assert kwargs['relative_rotation_current_sensor'].shape == (5, 3, 3)
            return FakeOutput(np.arange(2), np.ones((2, 128), dtype=np.float32))
    monkeypatch.setattr(worker, 'load_registered_extractor', lambda root, device: FakeExtractor())
    output = tmp_path/'output'
    summary = worker.run_worker(tmp_path, manifest, output, device='cpu')
    assert summary['processed_windows'] == summary['rejected_windows'] == 1
    assert summary['recorded_windows'] == 2 and summary['model_forwards'] == 1 and len(calls) == 1
    assert not summary['no_valid_transfer'] and summary['training_steps'] == 0
    record = json.loads((output/'record_000001.json').read_text())
    assert record['status'] == 'REJECTED_RECORD' and 'INCOMPATIBLE_FULL_ROTATION' in record['reason']
    with np.load(output/'tokens_000000.npz') as data:
        np.testing.assert_array_equal(data['world_from_sensor'], poses)
        assert data['ray_tokens'].shape == (2, 128)
        assert 'logits' not in data.files
    for path, digest in json.loads((output/'files_sha256.json').read_text()).items():
        assert hashlib.sha256((output/path).read_bytes()).hexdigest() == digest
    with pytest.raises(FileExistsError): worker.run_worker(tmp_path, manifest, output, device='cpu')


def test_all_rejected_never_load_weights_and_source_drift_is_fatal(tmp_path, monkeypatch):
    entry, raw, poses = fixture()
    poses[0, :3, :3] = [[1, 0, 0], [0, 0, -1], [0, 1, 0]]
    manifest = write_manifest(tmp_path, [(entry, raw, poses)])
    monkeypatch.setattr(worker, 'load_registered_extractor', lambda *a, **k: pytest.fail('weights must not load'))
    summary = worker.run_worker(tmp_path, manifest, tmp_path/'rejected', device='cpu')
    assert summary['no_valid_transfer'] and summary['rejected_windows'] == 1
    assert summary['model_forwards'] == 0
    (tmp_path/'input0.npz').write_bytes(b'drift')
    summary = worker.run_worker(tmp_path, manifest, tmp_path/'drift', device='cpu')
    assert summary['status'] == 'FAILED' and 'SHA-256 drift' in summary['error']


def test_weight_sha_checked_before_deserialization(tmp_path, monkeypatch):
    (tmp_path/'encoder.pt').write_bytes(b'not a checkpoint')
    monkeypatch.setattr(worker, 'ENCODER', ('encoder.pt', '0'*64))
    monkeypatch.setattr(worker, 'load_surface_encoder', lambda *a, **k: pytest.fail('must not deserialize'))
    monkeypatch.setattr(torch, 'load', lambda *a, **k: pytest.fail('must not deserialize'))
    with pytest.raises(ValueError, match='SHA-256 drift'):
        worker.load_registered_extractor(tmp_path, device='cpu')


def test_c500_wrong_step_rejected_after_synthetic_sha_authentication(tmp_path, monkeypatch):
    (tmp_path/'encoder.pt').write_bytes(b'synthetic encoder loader fixture')
    stream = io.BytesIO(); torch.save({'step': 499, 'seed': 0, 'model': {}}, stream)
    (tmp_path/'relation.pt').write_bytes(stream.getvalue())
    monkeypatch.setattr(worker, 'ENCODER', ('encoder.pt', hashlib.sha256((tmp_path/'encoder.pt').read_bytes()).hexdigest()))
    monkeypatch.setattr(worker, 'RELATION', ('relation.pt', hashlib.sha256(stream.getvalue()).hexdigest()))
    monkeypatch.setattr(worker, 'load_surface_encoder', lambda *a, **k: SimpleNamespace(backbone=None))
    with pytest.raises(ValueError, match='C500 checkpoint metadata drift'):
        worker.load_registered_extractor(tmp_path, device='cpu')


def test_real_safe_c500_loader_strict_state_on_synthetic_weights(tmp_path, monkeypatch):
    from tests.v3.unit.test_frozen_structural_tokens import FakeBackbone
    torch.set_num_threads(1)
    model = worker.RayBranchRelationModel('C')
    state = dict(step=500, seed=0, purpose='PARTIAL_AI_CORE_FIT_NOT_GENERALIZATION', model=model.state_dict())
    stream = io.BytesIO(); torch.save(state, stream)
    encoder_payload = b'synthetic encoder bytes authenticated before fake constructor'
    (tmp_path/'encoder.pt').write_bytes(encoder_payload)
    (tmp_path/'relation.pt').write_bytes(stream.getvalue())
    monkeypatch.setattr(worker, 'ENCODER', ('encoder.pt', hashlib.sha256(encoder_payload).hexdigest()))
    monkeypatch.setattr(worker, 'RELATION', ('relation.pt', hashlib.sha256(stream.getvalue()).hexdigest()))
    def fake_encoder(payload, *, expected_sha256, expected_epoch):
        assert payload == encoder_payload and expected_epoch == 2
        assert expected_sha256 == hashlib.sha256(payload).hexdigest()
        return SimpleNamespace(backbone=FakeBackbone())
    monkeypatch.setattr(worker, 'load_surface_encoder', fake_encoder)
    rng = torch.get_rng_state().clone()
    result = worker.load_registered_extractor(tmp_path, device='cpu')
    assert torch.equal(rng, torch.get_rng_state())
    assert not result.model.training and not result.adapter.training
    assert all(not p.requires_grad for p in result.model.parameters())
    assert all(torch.equal(v, result.model.state_dict()[k]) for k, v in state['model'].items())


def test_cli_all_rejected_preserves_no_valid_transfer(tmp_path, monkeypatch, capsys):
    import sys
    entry, raw, poses = fixture()
    poses[0, :3, :3] = [[1, 0, 0], [0, 0, -1], [0, 1, 0]]
    ref = write_manifest(tmp_path, [(entry, raw, poses)])
    monkeypatch.setattr(worker, 'load_registered_extractor', lambda *a, **k: pytest.fail('no weights'))
    monkeypatch.setattr(sys, 'argv', ['worker', '--project-root', str(tmp_path), '--manifest', ref['path'],
        '--manifest-sha256', ref['sha256'], '--output', str(tmp_path/'output'), '--device', 'cpu'])
    assert worker.main() == 1
    result = json.loads(capsys.readouterr().out)
    assert result['no_valid_transfer'] and result['rejected_windows'] == 1
