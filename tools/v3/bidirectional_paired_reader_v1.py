"""Streaming authenticated partial reader; callers must authorize exact scope.

No training or export is performed. Large raw teacher diagnostics are released
before yielding; original files remain the authoritative source.
"""
from copy import deepcopy
import gzip
import json
from pathlib import Path
import numpy as np
from bidirectional_paired_scope_v1 import compile_scope, identity
from mtare_topo.governance_surface_material import read_pinned
from mtare_topo.governance_surface_selection import digest
from mtare_topo.data.multiview_teacher_reader_v1 import MultiviewTeacherReader
from mtare_topo.data.development_paired_reader import PairedDevelopmentObservation
from mtare_topo.data.development_partition_scope import ENCODER
from mtare_topo.data.development_compact_blocks import read_compact_points, bind_partition_payload
from mtare_topo.data.development_grid_export import decode_grid
from mtare_topo.data.development_branch_targets import archived_junction_targets
from mtare_topo.teacher.gse_reference_exclusion_binding_v1 import bound_reference_exclusion


class BidirectionalPairedReader:
    def __init__(self, root, scope):
        self.root=Path(root).resolve(strict=True)
        if digest(compile_scope(self.root))!=digest(scope):raise ValueError('paired scope drift')
        self.scope=deepcopy(scope);self.opened={};self.consumed=False

    def _read(self,p,h):
        raw=read_pinned(self.root,p,h);self.opened[p]=h;return raw

    def _observation(self,bundle,row):
        cache=self._read(row['cache_path'],row['cache_sha256'])
        compact=read_compact_points(cache,manifest_row=row['feature_entry'],expected_source=row['source'],encoder_sha256=ENCODER)
        representations=bind_partition_payload(compact,self._read(row['partition_path'],row['partition_sha256']),
            expected_sha256=row['partition_sha256'],expected_source=row['source'])
        grid=decode_grid(self._read(row['grid_path'],row['grid_sha256']),
            expected_sha256=row['grid_sha256'],expected_binding=row['source_binding'])
        bound_reference_exclusion(bundle,grid,np.zeros((1,3)),expected_binding=row['source_binding'],matching_radius_m=4.)
        document=json.loads(gzip.decompress(self._read(row['target_path'],row['target_sha256'])))
        produced=document['produced_targets'];raw=document['raw_interfaces']
        targets=archived_junction_targets(produced,raw,bundle['construction_teacher_only'],
            current_yaw_deg=float(bundle['sensor_teacher_only']['yaw_deg'][-1]),
            expected_binding=row['source_binding'],expected_record_sha256=produced['target_record_sha256'])
        # Full file hash above independently pins the record, not a self-reported hash alone.
        loss_record={k:produced[k] for k in ('record','source_binding','target_record_sha256')}
        observation=PairedDevelopmentObservation(deepcopy(row['source']),row['split'],representations,
            dict(bundle=bundle,grid=grid,produced_targets=loss_record,**targets))
        del document,produced,raw,cache,compact
        return observation

    def observations(self):
        if self.consumed:raise ValueError('reader already consumed')
        self.consumed=True
        rows={identity(r['source']):r for r in self.scope['observations']};seen=set()
        for item in self.scope['source_cards']:
            p,h=item['path'],item['sha256']
            def compiler(root,p=p,h=h):return json.loads(read_pinned(root,p,h))['scope']
            card=json.loads(self._read(p,h));reader=MultiviewTeacherReader(self.root,card['scope'],scope_compiler=compiler)
            try:
                for task in sorted(reader.entries):
                    bundles=reader.read_task(task)
                    for bundle in bundles:
                        key=identity(bundle['source'])
                        if key not in rows or key in seen:raise ValueError('duplicate or foreign observation')
                        row=rows[key]
                        if row['source_card']!=p or bundle['source']['split']!=row['split']:raise ValueError('card/split mismatch')
                        result=self._observation(bundle,row);seen.add(key)
                        yield result
                        del result
                    del bundle,bundles
            finally:
                self.opened.update(reader.opened)
        if seen!=set(rows) or len(seen)!=860:raise ValueError('incomplete paired population')
