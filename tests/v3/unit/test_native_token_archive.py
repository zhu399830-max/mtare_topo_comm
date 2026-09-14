import hashlib
import io
import json

import numpy as np
import pytest

from mtare_topo.integration.native_token_archive import NativeTokenArchive
from mtare_topo.integration.structural_token_retrieval import retrieve_structural_candidates


def archive(tmp_path, mutate=None):
    run = 'results/gate6_single_robot/synthetic'
    prefix = run + '/artifacts/'
    files = {}

    def put(path, value):
        data = value if isinstance(value, bytes) else json.dumps(value).encode()
        files[path] = data
        return dict(path=path, sha256=hashlib.sha256(data).hexdigest())

    def npz(**values):
        result = io.BytesIO()
        np.savez_compressed(result, **values)
        return result.getvalue()

    index, bindings = [], []
    for i in range(3):
        epoch, wid = f'session:{i}', f'native_{i:03d}'
        stamps = np.arange(5) + i * 5 + 1
        keys = [f'raw:{s}' for s in stamps]
        registered = [f'reg:{s}' for s in stamps]
        poses = np.tile(np.eye(4), (5, 1, 1))
        source = put(prefix + f'model_inputs/{wid}.npz', npz(
            raw_pointcloud_bytes=np.full((5, 123200), i, dtype=np.uint8), world_from_sensor=poses))
        vectors = np.eye(128, dtype=np.float32)[[i % 2]]
        tref = put(prefix + f'frozen_tokens/tokens_{i:06d}.npz', npz(
            endpoints_current_sensor_m=np.zeros((1, 3)), ray_tokens=vectors,
            source_stamp_ns=stamps, world_from_sensor=poses))
        tref['path'] = f'tokens_{i:06d}.npz'
        frames = [dict(source_key=key, stamp_ns=int(stamp)) for key, stamp in zip(keys, stamps)]
        record = dict(status='PROCESSED', coordinate_frame='current_sensor_m',
            preprocessing_version='full_relative_se3_v1', stamp_ns=int(stamps[-1]),
            window_id=wid, token_ref=tref, source=dict(frames=frames, input=source, sequence_id='s'))
        binding = dict(epoch=epoch, window_id=wid, raw_source_keys=keys,
            registered_source_keys=registered, status='EXPLICIT_SOURCE_BOUND', physical_pose_verified=False,
            native_snapshot=dict(epoch=epoch, source_frame_keys=registered))
        item = dict(epoch=epoch, window_id=wid, raw_source_frame_keys=keys,
            native_source_frame_keys=registered, token_ref=tref, input_ref=source,
            source_binding_ref='binding#'+wid, live_token_available=False)
        if mutate:
            mutate(i, record, binding, item)
        put(prefix + f'frozen_tokens/record_{i:06d}.json', record)
        index.append(item)
        bindings.append(binding)
    put(prefix + 'native_token_index.json', index)
    put(prefix + 'model_inputs/source_bindings.json', bindings)
    put(prefix + 'frozen_tokens/summary.json', dict(processed_windows=3, rejected_windows=0, training_steps=0,
        checkpoints=dict(encoder=dict(sha256='1'*64), relation=dict(sha256='2'*64))))
    for name, data in files.items():
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    seal = ''.join(hashlib.sha256(data).hexdigest()+'  '+name+'\n' for name, data in sorted(files.items())).encode()
    (tmp_path / prefix / 'evidence_sha256.txt').write_bytes(seal)
    return NativeTokenArchive(tmp_path, run, seal_sha256=hashlib.sha256(seal).hexdigest())


def test_real_loader_preserves_causal_sources_and_feature_dependency(tmp_path):
    records = list(archive(tmp_path).records())
    result = retrieve_structural_candidates(records[2][0], [x[0] for x in records[:2]])
    assert result['candidates'][0]['record_id'] == 'session:0'
    assert all(len(record.source_frames) == 5 for record, _ in records)
    assert all(meta['offline_replay_only'] and not meta['live_token_available'] for _, meta in records)
    prefix = list(archive(tmp_path / 'prefix').records())[:2]
    assert retrieve_structural_candidates(prefix[1][0], [prefix[0][0]]) == retrieve_structural_candidates(records[1][0], [records[0][0]])


@pytest.mark.parametrize('field', ['epoch', 'window_id', 'registered_source_keys', 'raw_source_keys'])
def test_even_sealed_cross_wire_is_rejected(tmp_path, field):
    def mutate(i, record, binding, item):
        if i == 1:
            binding[field] = 'wrong' if field in ('epoch','window_id') else ['wrong']*5
    with pytest.raises(ValueError, match='binding mismatch'):
        list(archive(tmp_path, mutate).records())


def test_archive_is_not_live_or_gt_identity_authority(tmp_path):
    def mutate(i, record, binding, item):
        item['live_token_available'] = True
    with pytest.raises(ValueError, match='binding mismatch'):
        list(archive(tmp_path, mutate).records())


def test_payload_drift_and_unsealed_reference_rejected(tmp_path):
    store = archive(tmp_path)
    path = tmp_path / store.run / 'artifacts/frozen_tokens/tokens_000000.npz'
    path.write_bytes(path.read_bytes()+b'drift')
    with pytest.raises(ValueError, match='drift'):
        list(store.records())


def test_native_snapshot_cross_wire_and_future_stamp_rejected(tmp_path):
    def mutate(i, record, binding, item):
        if i == 1:
            binding['native_snapshot']['source_frame_keys'] = ['other']*5
    with pytest.raises(ValueError, match='binding mismatch'):
        list(archive(tmp_path, mutate).records())
