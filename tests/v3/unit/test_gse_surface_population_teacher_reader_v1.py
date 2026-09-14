import hashlib
import json
import numpy as np
import pytest
import zarr

from mtare_topo.data import gse_surface_population_teacher_reader_v1 as module
from mtare_topo.data.gse_surface_population_teacher_scope_v1 import combine_fiveframe_plans


@pytest.mark.parametrize('damage', [None, 'last_sequence', 'pose_drift', 'missing_chunk'])
def test_real_codec_full_task_binding(tmp_path, monkeypatch, damage):
    task = 'S01_flat_tree_small_C01__ellipse'
    sources = [dict(task=task, frame_rows=list(range(5*i,5*i+5)),
                    source_frame_count=80,source_sequence_id=i) for i in range(16)]
    arrays = dict(sensor_xyz_m=np.zeros((80,3),dtype=np.float64), yaw_deg=np.zeros(80,dtype=np.float64),
                  primitive_membership_code=np.ones((80,16,720),dtype=np.uint16))
    if damage == 'pose_drift': arrays['sensor_xyz_m'][79,0]=1.
    files = {}; plans = {}
    def pin(path):
        files[str(path.relative_to(tmp_path))] = hashlib.sha256(path.read_bytes()).hexdigest()
    for field, values in arrays.items():
        path = tmp_path/'sensor'/field
        chunks = (16,16,720) if field == 'primitive_membership_code' else values.shape
        zarr.array(values, store=str(path), chunks=chunks, overwrite=False)
        header = json.loads((path/'.zarray').read_text())
        plan = combine_fiveframe_plans(header,field,sources)
        plans['sensor/'+field] = plan
        for key in ['.zarray', *plan['chunk_keys']]:pin(path/key)
    common = dict(parent_id='S01_flat_tree_small_C01',geometry_realization='ellipse')
    for name, doc in [('construction.json',dict(**common,realized_primitives=[dict(primitive_id='p')])),
                      ('codebook.json',dict(**common,primitive_ids=['p'],source_sets=[[],[0]]))]:
        path = tmp_path/name;path.write_text(json.dumps(doc));pin(path)
    sequence = np.arange(16)
    if damage == 'last_sequence':sequence[-1]=999
    path = tmp_path/'inputs.npz'
    np.savez(path,ranges_m=np.ones((16,5,16,720),dtype=np.float32),
        valid_mask=np.ones((16,5,16,720),dtype=np.uint8),
        relative_translation_current_sensor_m=np.zeros((16,5,3),dtype=np.float32),
        relative_yaw_current_sensor_deg=np.zeros((16,5),dtype=np.float32),
        frame_rows=np.arange(80,dtype=np.int32).reshape(16,5),source_sequence_ids=sequence)
    scope = dict(entries=[dict(task=task,observations=sources,sensor_prefix='sensor',
        construction_path='construction.json',codebook_path='codebook.json',input_path='inputs.npz')],
        file_sha256=files,array_access=plans,input_sha256={'inputs.npz':hashlib.sha256(path.read_bytes()).hexdigest()})
    monkeypatch.setattr(module,'compile_population_teacher_scope',lambda root:scope)
    reader=module.PopulationTeacherReader(tmp_path,scope)
    if damage == 'missing_chunk':(tmp_path/'sensor/sensor_xyz_m/0.0').unlink()
    if damage:
        with pytest.raises(ValueError):reader.read_task(task)
        assert not reader.completed
    else:
        result=reader.read_task(task)
        assert len(result)==16 and result[-1]['source']['source_sequence_id']==15
        assert set(reader.opened)==set(files)|{'inputs.npz'}
        assert not any('teacher' in k for k in result[0]['student'])
        with pytest.raises(ValueError):reader.read_task(task)
    with pytest.raises(ValueError):reader.read_task('S01_flat_tree_small_C10__ellipse')
