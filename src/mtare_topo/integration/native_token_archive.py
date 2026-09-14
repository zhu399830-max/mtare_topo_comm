"""Authenticate saved native-window tokens for causal retrieval, never control.

The archive is an offline producer. Exact source binding does not make a token
available retrospectively to a live decision. No teacher IDs or task identities
are inferred here, and the original arrays and source records remain unchanged.
"""
import hashlib
import io
import json
from pathlib import Path

import numpy as np

from mtare_topo.integration.structural_token_retrieval import FrameBinding, LocalTokenRecord


class NativeTokenArchive:
    def __init__(self, root, run, *, seal_sha256):
        self.root = Path(root).resolve()
        self.run = str(run)
        seal = self._path(self.run + '/artifacts/evidence_sha256.txt').read_bytes()
        if hashlib.sha256(seal).hexdigest() != seal_sha256:
            raise ValueError('archive seal drift')
        self.hashes = {}
        for line in seal.decode().splitlines():
            digest, path = line.split('  ', 1)
            if path in self.hashes or not path.startswith(self.run + '/'):
                raise ValueError('duplicate or out-of-run seal entry')
            self.hashes[path] = digest
        prefix = self.run + '/artifacts/'
        self.index = self._json(prefix + 'native_token_index.json')
        self.bindings = self._json(prefix + 'model_inputs/source_bindings.json')
        self.model = self._json(prefix + 'frozen_tokens/summary.json')
        if (not self.index or len(self.index) != len(self.bindings)
                or self.model['processed_windows'] != len(self.index)
                or self.model['rejected_windows'] != 0
                or self.model['training_steps'] != 0):
            raise ValueError('archive population incomplete')
        if (len({r['epoch'] for r in self.index}) != len(self.index)
                or len({r['window_id'] for r in self.index}) != len(self.index)):
            raise ValueError('duplicate native epoch or window')

    def _path(self, relative):
        path = Path(relative)
        if path.is_absolute() or '..' in path.parts:
            raise ValueError('exact project-relative path required')
        current = self.root
        for part in path.parts:
            current = current / part
            if current.is_symlink():
                raise ValueError('symlink archive source forbidden')
        return current

    def _read(self, path, expected=None):
        if path not in self.hashes or (expected is not None and expected != self.hashes[path]):
            raise ValueError('unsealed or conflicting source reference')
        source = self._path(path)
        if source.stat().st_size > 64 * 1024**2:
            raise ValueError('individual archive payload cap exceeded')
        data = source.read_bytes()
        if hashlib.sha256(data).hexdigest() != self.hashes[path]:
            raise ValueError('archive source drift: ' + path)
        return data

    def _json(self, path):
        return json.loads(self._read(path))

    def records(self):
        previous_stamp = -1
        for order, (entry, binding) in enumerate(zip(self.index, self.bindings)):
            prefix = self.run + '/artifacts/frozen_tokens/'
            row = self._json(prefix + f'record_{order:06d}.json')
            frames = row['source']['frames']
            keys = [frame['source_key'] for frame in frames]
            snapshot = binding['native_snapshot']
            if (row['status'] != 'PROCESSED' or row['coordinate_frame'] != 'current_sensor_m'
                    or row['preprocessing_version'] != 'full_relative_se3_v1'
                    or len(frames) != 5 or keys != entry['raw_source_frame_keys']
                    or keys != binding['raw_source_keys']
                    or binding['status'] != 'EXPLICIT_SOURCE_BOUND'
                    or binding['epoch'] != entry['epoch'] or snapshot['epoch'] != entry['epoch']
                    or snapshot['source_frame_keys'] != entry['native_source_frame_keys']
                    or binding['registered_source_keys'] != entry['native_source_frame_keys']
                    or row['window_id'] != entry['window_id'] or binding['window_id'] != entry['window_id']
                    or row['token_ref'] != entry['token_ref'] or row['source']['input'] != entry['input_ref']
                    or row['stamp_ns'] != frames[-1]['stamp_ns'] or row['stamp_ns'] <= previous_stamp
                    or entry['live_token_available'] is not False):
                raise ValueError('native/token/source binding mismatch')
            token_path = prefix + entry['token_ref']['path']
            token_bytes = self._read(token_path, entry['token_ref']['sha256'])
            source = entry['input_ref']
            input_bytes = self._read(source['path'], source['sha256'])
            with np.load(io.BytesIO(input_bytes), allow_pickle=False) as data:
                raw = data['raw_pointcloud_bytes']
                poses = data['world_from_sensor']
            if raw.dtype != np.uint8 or raw.shape != (5, 123200):
                raise ValueError('original raw frame buffer mismatch')
            frame_bindings = tuple(FrameBinding(frame['source_key'], frame['stamp_ns'],
                hashlib.sha256(raw[i].tobytes()).hexdigest()) for i, frame in enumerate(frames))
            with np.load(io.BytesIO(token_bytes), allow_pickle=False) as tokens:
                if (not np.array_equal(tokens['source_stamp_ns'], [f['stamp_ns'] for f in frames])
                        or not np.array_equal(tokens['world_from_sensor'], poses)):
                    raise ValueError('token source poses or stamps differ')
                record = LocalTokenRecord(entry['epoch'], row['source']['sequence_id'], order, tuple(keys),
                    tokens['endpoints_current_sensor_m'], tokens['ray_tokens'], frame_bindings,
                    self.model['checkpoints']['encoder']['sha256'], self.model['checkpoints']['relation']['sha256'])
            previous_stamp = row['stamp_ns']
            yield record, dict(epoch=entry['epoch'], source_binding_ref=entry['source_binding_ref'],
                token_ref=dict(entry['token_ref']), native_source_frame_keys=list(entry['native_source_frame_keys']),
                physical_pose_verified=binding['physical_pose_verified'],
                offline_replay_only=True, live_token_available=False, task_identity_verified=False,
                snapshot=snapshot)
