"""Exact small-probe reader. Caller must first freeze/preflight the run.

This reader does not generate labels. It reuses original scan values and the
existing strict pose/source/first-return checks, with no geometry rendering.
"""
from copy import deepcopy
import gzip
import io
import json
from pathlib import Path

import numpy as np
import zarr

from mtare_topo.governance_surface_material import read_pinned
from .gse_surface_input_export_v1 import _ExactStore
from .gse_surface_teacher_reader_v1 import verify_alignment
from .gse_surface_teacher_scope_v1 import plan_field
from .gse_v8_probe_scope import compile_scope


class V8ProbeReader:
    def __init__(self, root, scope):
        self.root = Path(root).resolve(strict=True)
        if compile_scope(self.root) != scope:
            raise ValueError('frozen ten-observation scope drift')
        self.scope = deepcopy(scope)
        self.entries = {(r['source']['task'], r['source']['source_sequence_id']): r
                        for r in self.scope['entries']}
        self.opened = dict(scope['metadata_reads_sha256'])
        self.completed = set()

    def _read(self, path, sha):
        data = read_pinned(self.root, path, sha)
        self.opened[path] = sha
        return data

    def read_observation(self, task, sequence):
        from mtare_topo.teacher.gse_directed_interface_binding_v1 import interpret_bound_result
        key = (task, sequence)
        if key not in self.entries or key in self.completed:
            raise ValueError('outside exact scope or already consumed')
        row = self.entries[key]; source = row['source']
        hashes = self.scope['file_sha256']
        sensor = {}
        for field, plan in row['arrays'].items():
            store = _ExactStore(self.root, plan['prefix'], hashes, self.opened)
            store.allowed = {'.zarray', *plan['chunk_keys']}
            header = json.loads(store['.zarray'])
            expected = plan_field(header, field, source['frame_rows'], source['source_frame_count'])
            if expected != {k: v for k, v in plan.items() if k != 'prefix'}:
                raise ValueError('array plan drift')
            sensor[field] = np.asarray(zarr.Array(store=store, read_only=True).oindex[source['frame_rows']])
        p = row['input_path']
        with np.load(io.BytesIO(self._read(p, hashes[p])), allow_pickle=False) as archive:
            names = {'ranges_m', 'valid_mask', 'relative_translation_current_sensor_m',
                     'relative_yaw_current_sensor_deg', 'frame_rows', 'source_sequence_ids'}
            if set(archive.files) != names:
                raise ValueError('original six input arrays required')
            identifiers = archive['source_sequence_ids']
            if identifiers.shape != (row['container_observations'],):
                raise ValueError('input container population drift')
            indices = np.flatnonzero(identifiers == sequence)
            if len(indices) != 1:
                raise ValueError('observation absent or duplicate')
            student = {name: archive[name][indices[0]].copy() for name in names}
        if student['frame_rows'].tolist() != source['frame_rows']:
            raise ValueError('input/raw frame mismatch')
        documents = {role: json.loads(self._read(row[role], hashes[row[role]]))
                     for role in ('construction_path', 'codebook_path')}
        verify_alignment(sensor, student, documents['construction_path'], documents['codebook_path'], source)
        previous = None
        for pin in row['references']:
            value = json.loads(gzip.decompress(self._read(pin['path'], pin['sha256'])))
            if previous is None:
                previous = value['produced_targets']
        raw = value['raw_interfaces']
        if raw['source'] != source:
            raise ValueError('raw source drift')
        bundle = dict(source=deepcopy(source), student=student, sensor_teacher_only=sensor,
            construction_teacher_only=documents['construction_path'],
            codebook_teacher_only=documents['codebook_path'])
        interpret_bound_result(bundle, raw)
        self.completed.add(key)
        return bundle, raw, previous
