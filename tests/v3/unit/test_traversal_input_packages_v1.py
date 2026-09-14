import numpy as np
import pytest
from mtare_topo.data.traversal_input_packages_v1 import package_indices, slice_package
from mtare_topo.data.gse_surface_input_export_v1 import SurfaceTaskInputs
from mtare_topo.data.multiview_missing_input_export_v1 import encode_and_validate


def rows(n):
    return [dict(task='S01_synthetic_C07__ellipse', traversal_id='route', sequence_row=i,
                 source_sequence_id=i, frame_rows=list(range(i, i+5))) for i in range(n)]


def test65_and_permutation():
    r=rows(65); p=package_indices(r)
    assert list(map(len,p)) == [64,1]
    assert sorted(i for g in p for i in g)==list(range(65))
    rev=list(reversed(r))
    assert [[rev[i]['source_sequence_id'] for i in g] for g in package_indices(rev)] == [list(range(64)),[64]]


def test_routes_and_duplicates():
    r=rows(65)
    for x in r[30:]: x['traversal_id']='other'
    assert sorted(map(len,package_indices(r)))==[30,35]
    with pytest.raises(ValueError): package_indices(r+[r[0]])
    assert package_indices([])==()


def test_original_encoder_equivalence_and_no_alias():
    r=rows(3)
    v=SurfaceTaskInputs('S01_synthetic_C07__ellipse',np.ones((3,5,16,720),np.float32),
        np.ones((3,5,16,720),np.uint8),np.zeros((3,5,3),np.float32),
        np.zeros((3,5),np.float32),np.array([x['frame_rows'] for x in r],np.int32),
        np.arange(3,dtype=np.int64),{})
    sliced,selected=slice_package(v,r,(0,1,2))
    assert encode_and_validate(sliced,selected)==encode_and_validate(v,r)
    sliced.ranges_m[0,0,0,0]=9
    assert v.ranges_m[0,0,0,0]==1
    with pytest.raises(ValueError):slice_package(v,r,(-1,))
    with pytest.raises(ValueError):slice_package(v,r,(0,0))
    r[1]['traversal_id']='other'
    with pytest.raises(ValueError):slice_package(v,r,(0,1))
