"""One authenticated observation shared by all representations and loss paths.

Calling this reader requires an exact authorized scope in the outer executor.
No model, optimizer, export or automatic acceptance is performed here.
"""
from dataclasses import dataclass
import gzip
import json
import numpy as np
from .development_paired_scope import compile_scope,identity
from .development_partition_scope import ENCODER
from .development_compact_blocks import read_compact_points,bind_partition_payload
from .development_grid_export import decode_grid
from .development_branch_targets import archived_junction_targets
from .gse_supplement_teacher_reader_v1 import SupplementTeacherReader
from mtare_topo.governance_surface_selection import digest
from mtare_topo.governance_surface_material import read_pinned
from mtare_topo.teacher.gse_reference_exclusion_binding_v1 import bound_reference_exclusion


@dataclass(frozen=True)
class PairedDevelopmentObservation:
    source: dict
    split: str
    student_representations: dict
    loss_only: dict


class DevelopmentPairedReader:
    def __init__(self,root,scope):
        if digest(compile_scope(root))!=digest(scope):raise ValueError('paired scope drift')
        self.root=root;self.scope=scope;self.opened={};self.consumed=False

    def _read(self,path,sha256):
        raw=read_pinned(self.root,path,sha256);self.opened[path]=sha256;return raw

    def observations(self):
        if self.consumed:raise ValueError('reader already consumed')
        self.consumed=True
        rows={identity(r['source']):r for r in self.scope['observations']}
        tasks=sorted({k[0] for k in rows});seen=set();decoded=0
        reader=SupplementTeacherReader(self.root,self.scope['original_reader_scope'])
        try:
            for task in tasks:
                bundles=reader.read_task(task);decoded+=len(bundles)
                for bundle in bundles:
                    key=identity(bundle['source'])
                    if key not in rows:continue
                    if key in seen:raise ValueError('duplicate selected observation')
                    row=rows[key];seen.add(key)
                    cache=self._read(row['cache_path'],row['cache_sha256'])
                    compact=read_compact_points(cache,manifest_row=row['feature_entry'],expected_source=row['source'],encoder_sha256=ENCODER)
                    representations=bind_partition_payload(compact,self._read(row['partition_path'],row['partition_sha256']),
                        expected_sha256=row['partition_sha256'],expected_source=row['source'])
                    grid=decode_grid(self._read(row['grid_path'],row['grid_sha256']),expected_sha256=row['grid_sha256'],expected_binding=row['source_binding'])
                    bound_reference_exclusion(bundle,grid,np.zeros((1,3)),expected_binding=row['source_binding'],matching_radius_m=4.)
                    produced=json.loads(gzip.decompress(self._read(row['target_path'],row['target_sha256'])))['produced_targets']
                    raw=json.loads(gzip.decompress(self._read(row['raw_interfaces_path'],row['raw_interfaces_sha256'])))['raw_interfaces']
                    targets=archived_junction_targets(produced,raw,bundle['construction_teacher_only'],
                        current_yaw_deg=float(bundle['sensor_teacher_only']['yaw_deg'][-1]),
                        expected_binding=row['source_binding'],expected_record_sha256=produced['target_record_sha256'])
                    yield PairedDevelopmentObservation(row['source'],row['split'],representations,
                        dict(bundle=bundle,grid=grid,produced_targets=produced,**targets))
                del bundle,bundles
            if seen!=set(rows) or decoded!=self.scope['decoded_task_observations']:
                raise ValueError('incomplete reader population')
        finally:
            self.opened.update(reader.opened)
