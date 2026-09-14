"""Read only frozen original pose rows; no scan decoding or label generation."""
from copy import deepcopy
from pathlib import Path

import numpy as np
import zarr

from mtare_topo.governance_surface_selection import digest
from .gse_surface_input_export_v1 import _ExactStore, _json_object
from .gse_structural_supplement_scope_v1 import compile_supplement_scope, plan_xyz_rows
from .gse_structural_supplement_v1 import choose_position


class SupplementPoseReader:
    def __init__(self, root, scope):
        self.root = Path(root).resolve(strict=True)
        if digest(compile_supplement_scope(self.root)) != digest(scope):
            raise ValueError('supplementary sampling scope drift')
        self.scope = deepcopy(scope)
        self.parents = {p['parent_id']:p for p in scope['parents']}
        self.opened = {}
        self.completed = set()

    def read_parent(self, parent):
        if parent not in self.parents or parent in self.completed:
            raise ValueError('parent outside scope or already read')
        positions, anchors = {}, {}
        for entry in self.scope['entries']:
            if entry['parent_id'] != parent:
                continue
            prefix = entry['xyz_prefix']; plan = self.scope['array_access'][prefix]
            store = _ExactStore(self.root, prefix, self.scope['file_sha256'], self.opened)
            store.allowed = {'.zarray', *plan['chunk_keys']}
            header = _json_object(store['.zarray'])
            if plan_xyz_rows(header, plan['selected_rows'], plan['shape'][0]) != plan:
                raise ValueError('pose access drift')
            xyz = np.asarray(zarr.Array(store=store, read_only=True).oindex[plan['selected_rows']])
            if xyz.shape != (len(plan['selected_rows']), 3) or xyz.dtype != np.float64 or not np.isfinite(xyz).all():
                raise ValueError('original pose dtype, dimensions or values invalid')
            positions[entry['variant']] = dict(zip(plan['selected_rows'], xyz.tolist()))
            anchors[entry['variant']] = entry['anchor_world_m_by_node']
        selections, missing = [], []
        for nomination in self.parents[parent]['nominations']:
            if nomination['status'] == 'NO_CAUSAL_HISTORY':
                missing.append(nomination)
                continue
            decisions = {r['variant']:[positions[r['variant']][frames[-1]] for frames in r['frame_rows']]
                         for r in nomination['variant_records']}
            selections.append(choose_position(nomination, decision_xyz_by_variant=decisions,
                anchor_xyz_by_variant={v: a[nomination['node_id_teacher_only']] for v,a in anchors.items()}))
        self.completed.add(parent)
        return dict(parent_id=parent, split=self.parents[parent]['split'], selections=selections,
                    missing=missing, labels_generated=0, limitation='Spatial sampling is not observed supervision.')
