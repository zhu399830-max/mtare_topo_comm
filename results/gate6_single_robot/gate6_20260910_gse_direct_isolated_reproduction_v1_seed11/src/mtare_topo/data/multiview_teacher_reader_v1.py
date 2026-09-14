"""Original teacher alignment, adapted only to multiple sealed input packages."""
from copy import deepcopy
import io
from pathlib import Path
import numpy as np
import zarr
from .multiview_teacher_scope_v1 import compile_scope
from .gse_supplement_teacher_scope_v1 import combine_plans
from .gse_surface_teacher_scope_v1 import FIELDS
from .gse_surface_teacher_reader_v1 import verify_alignment
from .gse_surface_input_export_v1 import _ExactStore,_json_object
from .gse_supplement_feature_input_v1 import bind_feature_input
from mtare_topo.governance_surface_material import read_pinned
from mtare_topo.governance_surface_selection import digest


def student_from_reference(payload,reference):
    s=reference['source'];i=reference['input_row']
    bind_feature_input(payload,expected_sha256=reference['input_sha256'],task=s['task'],row=i,
        source_sequence_id=s['source_sequence_id'],frame_rows=s['frame_rows'],observation_count=reference['observation_count'])
    # The original verifier needs raw ranges/mask/motion, not normalized tensors.
    with np.load(io.BytesIO(payload),allow_pickle=False) as archive:
        return {k:archive[k][i].copy() for k in archive.files}


class MultiviewTeacherReader:
    def __init__(self,root,scope,*,scope_compiler=None):
        self.root=Path(root).resolve(strict=True)
        compiler=compile_scope if scope_compiler is None else scope_compiler
        if digest(compiler(self.root))!=digest(scope):raise ValueError('teacher population scope drift')
        self.scope=deepcopy(scope);self.entries={r['task']:r for r in scope['entries']}
        self.opened={};self.completed=set()

    def _read(self,path,h):
        raw=read_pinned(self.root,path,h);self.opened[path]=h;return raw

    def read_task(self,task):
        if task not in self.entries or task in self.completed:raise ValueError('task outside scope or already read')
        entry=self.entries[task];sources=entry['observations'];sensor_all={};lookup={}
        if len(sources)!=len(entry['input_references']):raise ValueError('source/reference count mismatch')
        for field in FIELDS:
            prefix=entry['sensor_prefix']+'/'+field;plan=self.scope['array_access'][prefix]
            store=_ExactStore(self.root,prefix,self.scope['file_sha256'],self.opened)
            store.allowed={'.zarray',*plan['chunk_keys']};header=_json_object(store['.zarray'])
            if combine_plans(header,field,sources)!=plan:raise ValueError('teacher array plan drift')
            sensor_all[field]=np.asarray(zarr.Array(store=store,read_only=True).oindex[plan['selected_rows']])
            lookup[field]={r:i for i,r in enumerate(plan['selected_rows'])}
        docs={k:_json_object(self._read(entry[k],self.scope['file_sha256'][entry[k]])) for k in ('construction_path','codebook_path')}
        packages={};bundles=[]
        for s,reference in zip(sources,entry['input_references']):
            for k in ('task','source_sequence_id','frame_rows'):
                if s[k]!=reference['source'][k]:raise ValueError('source/input mismatch')
            p=reference['input_path'];h=self.scope['input_sha256'][p]
            if h!=reference['input_sha256']:raise ValueError('package hash binding differs')
            if p not in packages:packages[p]=self._read(p,h)
            student=student_from_reference(packages[p],reference)
            sensor={k:v[[lookup[k][f] for f in s['frame_rows']]] for k,v in sensor_all.items()}
            verify_alignment(sensor,student,docs['construction_path'],docs['codebook_path'],s)
            bundles.append(dict(student=student,sensor_teacher_only=sensor,construction_teacher_only=docs['construction_path'],
                codebook_teacher_only=docs['codebook_path'],source=deepcopy(s)))
        self.completed.add(task)
        return bundles
