import hashlib
import json
import numpy as np
import pytest
import zarr
from test_gse_surface_input_export_v1 import fixture
from mtare_topo.data.gse_surface_input_export_v1 import (
    SurfaceInputReader,plan_array_access,SENSOR_FIELDS,WINDOW_FIELDS)


def test_variable_count_overlapping_histories_exact_payload(tmp_path):
    task,sources,selection,_,ranges,_=fixture(tmp_path)
    selection=selection[:2]
    selection[1]['frame_rows']=[f-1 for f in selection[0]['frame_rows']]
    teacher=zarr.open_group(str(tmp_path/sources[task]['teacher']),mode='r+')
    teacher['frame_row'][selection[1]['sequence_row']]=selection[1]['frame_rows']
    sealed={}
    for role,fields in [('sensor',SENSOR_FIELDS),('teacher',WINDOW_FIELDS)]:
        prefix=sources[task][role];keys={'.zgroup','.zattrs'}
        indices=([f for r in selection for f in r['frame_rows']] if role=='sensor' else [r['sequence_row'] for r in selection])
        for field in fields:
            key=field+'/.zarray';keys.add(key)
            h=json.loads((tmp_path/prefix/key).read_text())
            keys.update(field+'/'+k for k in plan_array_access(h,field,role,indices)['chunk_keys'])
        for key in keys:
            p=prefix+'/'+key;sealed[p]=hashlib.sha256((tmp_path/p).read_bytes()).hexdigest()
    with pytest.raises(ValueError,match='exact16'):
        SurfaceInputReader(tmp_path,task_sources=sources,selection=selection,sealed_keys=sealed)
    reader=SurfaceInputReader(tmp_path,task_sources=sources,selection=selection,sealed_keys=sealed,variable_population=True)
    result=reader.read_task(task)
    assert result.ranges_m.shape==(2,5,16,720)
    np.testing.assert_array_equal(result.ranges_m,ranges[np.array([r['frame_rows'] for r in selection])])
    assert result.read_report['selected_unique_sensor_frames']==6
    assert result.read_report['selected_observations']==2
    assert result.read_report['role_row_coverage']['sensor']['selected_unique_rows']==6
    assert set(reader.opened)==set(sealed)
