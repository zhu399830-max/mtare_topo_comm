"""Taskwise fixed-population reader; at most16 fiveframe bundles resident.

No global point/feature tensor. Teacher documents stay in teacher-only fields.
Every task is validated in full before any of its observations is returned.
"""
from copy import deepcopy
import io
from pathlib import Path
import numpy as np
import zarr

from .gse_surface_population_teacher_scope_v1 import compile_population_teacher_scope, combine_fiveframe_plans
from .gse_surface_teacher_scope_v1 import FIELDS
from .gse_surface_teacher_reader_v1 import verify_alignment
from .gse_surface_input_export_v1 import _ExactStore, _json_object
from mtare_topo.governance_surface_material import read_pinned
from mtare_topo.governance_surface_selection import digest


class PopulationTeacherReader:
    def __init__(self, root, scope):
        self.root = Path(root).resolve(strict=True)
        if digest(compile_population_teacher_scope(self.root)) != digest(scope):
            raise ValueError('population source scope drift')
        self.scope = deepcopy(scope)
        self.entries = {r['task']: r for r in scope['entries']}
        self.opened = {}; self.completed = set()

    def read_task(self, task):
        if task not in self.entries or task in self.completed:
            raise ValueError('task outside scope or already consumed')
        entry = self.entries[task]; sources = entry['observations']
        sensor_all = {}; row_lookup = {}
        for field in FIELDS:
            prefix = entry['sensor_prefix']+'/'+field
            plan = self.scope['array_access'][prefix]
            store = _ExactStore(self.root, prefix, self.scope['file_sha256'], self.opened)
            store.allowed = {'.zarray', *plan['chunk_keys']}
            header = _json_object(store['.zarray'])
            if combine_fiveframe_plans(header, field, sources) != plan:
                raise ValueError('task array plan drift')
            sensor_all[field] = np.asarray(zarr.Array(store=store, read_only=True).oindex[plan['selected_rows']])
            row_lookup[field] = {r: i for i, r in enumerate(plan['selected_rows'])}
        path = entry['input_path']; h = self.scope['input_sha256'][path]
        raw = read_pinned(self.root, path, h); self.opened[path] = h
        with np.load(io.BytesIO(raw), allow_pickle=False) as archive:
            fields = {'ranges_m','valid_mask','relative_translation_current_sensor_m',
                      'relative_yaw_current_sensor_deg','frame_rows','source_sequence_ids'}
            if set(archive.files) != fields:
                raise ValueError('original six input arrays required')
            student_all = {k: archive[k] for k in fields}
        shapes = {'ranges_m': (16,5,16,720), 'valid_mask': (16,5,16,720),
                  'relative_translation_current_sensor_m': (16,5,3),
                  'relative_yaw_current_sensor_deg': (16,5), 'frame_rows': (16,5),
                  'source_sequence_ids': (16,)}
        if any(student_all[k].shape != shape for k, shape in shapes.items()):
            raise ValueError('sixteen complete original observations required')
        documents = {}
        for key in ('construction_path', 'codebook_path'):
            path = entry[key]; h = self.scope['file_sha256'][path]
            documents[key] = _json_object(read_pinned(self.root, path, h)); self.opened[path] = h
        bundles = []
        for i, source in enumerate(sources):
            student = {k: v[i] for k, v in student_all.items()}
            if (student['source_sequence_ids'].item() != source['source_sequence_id']
                    or student['frame_rows'].tolist() != source['frame_rows']):
                raise ValueError('source observation ordering drift')
            sensor = {k: v[[row_lookup[k][r] for r in source['frame_rows']]] for k, v in sensor_all.items()}
            verify_alignment(sensor, student, documents['construction_path'], documents['codebook_path'], source)
            bundles.append(dict(student=student, sensor_teacher_only=sensor,
                construction_teacher_only=documents['construction_path'],
                codebook_teacher_only=documents['codebook_path'], source=deepcopy(source)))
        self.completed.add(task)
        return bundles
