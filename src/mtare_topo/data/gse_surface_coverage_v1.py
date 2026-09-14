"""Source-bound spatial coverage, not visibility or semantic supervision."""
from copy import deepcopy
from pathlib import Path
import numpy as np
import zarr

from .gse_surface_coverage_scope_v1 import compile_coverage_scope, plan_current_xyz
from .gse_surface_input_export_v1 import _ExactStore, _json_object
from mtare_topo.governance_surface_selection import digest
from mtare_topo.governance_surface_material import read_pinned
from mtare_topo.teacher.gse_construction_paths_v3 import construction_incident_paths


def nearby_references(groups, xyz):
    xyz = np.asarray(xyz)
    if xyz.shape != (3,) or not np.isfinite(xyz).all():
        raise ValueError('finite current XYZ required')
    result = []
    for g in groups:
        distance = float(np.linalg.norm(np.asarray(g['anchor_world_m'])-xyz))
        if distance < 10.:
            degree = len(g['paths'])
            result.append(dict(node_id_teacher_only=g['node_id_teacher_only'],
                               degree=degree, distance_m=distance,
                               kind='terminal' if degree == 1 else 'construction_segment' if degree == 2 else 'junction'))
    return result


class CoverageReader:
    def __init__(self, root, scope):
        self.root = Path(root).resolve(strict=True)
        if digest(compile_coverage_scope(self.root)) != digest(scope):
            raise ValueError('fixed coverage scope drift')
        self.scope = deepcopy(scope)
        self.entries = {r['task']: r for r in scope['entries']}
        self.opened = {}; self.completed = set()

    def read_task(self, task):
        if task not in self.entries or task in self.completed:
            raise ValueError('task outside scope or already consumed')
        row = self.entries[task]; path = row['construction_path']
        h = self.scope['file_sha256'][path]
        document = _json_object(read_pinned(self.root, path, h)); self.opened[path] = h
        parent, variant = task.split('__')
        if document.get('parent_id') != parent or document.get('geometry_realization') != variant:
            raise ValueError('source construction task mismatch')
        groups = construction_incident_paths(document)
        prefix = row['xyz_prefix']; plan = self.scope['array_access'][prefix]
        store = _ExactStore(self.root, prefix, self.scope['file_sha256'], self.opened)
        store.allowed = {'.zarray', *plan['chunk_keys']}
        header = _json_object(store['.zarray'])
        if plan_current_xyz(header, plan['selected_rows'], plan['shape'][0]) != plan:
            raise ValueError('pose plan drift')
        xyz = np.asarray(zarr.Array(store=store, read_only=True).oindex[plan['selected_rows']])
        if xyz.shape != (16, 3) or xyz.dtype != np.float64 or not np.isfinite(xyz).all():
            raise ValueError('source pose shape or precision drift')
        observations = [dict(source=source, nearby=nearby_references(groups, p))
                        for source, p in zip(row['observations'], xyz)]
        self.completed.add(task)
        return dict(task=task, observations=observations, labels_generated=0,
                    limitation='Spatial reference candidates only, not observable labels.')


def aggregate_coverage(outputs):
    splits = {}
    for output in outputs:
        for row in output['observations']:
            source = row['source']; split = source['split']; parent = source['parent_id']
            state = splits.setdefault(split, dict(observations=0, entities={}, parents={}, observations_with={}))
            state['observations'] += 1
            kinds = set()
            for reference in row['nearby']:
                kind = reference['kind']; kinds.add(kind)
                state['entities'].setdefault(kind, set()).add((parent, reference['node_id_teacher_only']))
                state['parents'].setdefault(kind, set()).add(parent)
            for kind in kinds:
                state['observations_with'][kind] = state['observations_with'].get(kind, 0)+1
    return {split: dict(observations=s['observations'], by_kind={kind: dict(
        independent_entities=len(s['entities'].get(kind, set())),
        parents=len(s['parents'].get(kind, set())), observations=s['observations_with'].get(kind, 0))
        for kind in ('terminal', 'junction', 'construction_segment')}) for split, s in splits.items()}
