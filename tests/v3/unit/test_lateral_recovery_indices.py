import importlib.util
from pathlib import Path
import pytest
p=Path(__file__).resolve().parents[3]/'tools/v3/recover_lateral_witness_worker.py'
s=importlib.util.spec_from_file_location('lateral_worker',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)

def test_global_ray_indices_preserved_without_mutation():
    entries=[dict(ray_index=1,t=2.,triangle_indices=[3])]
    x=m.restore_indices(entries,[7,17000])
    assert x[0]['ray_index']==17000 and x[0]['query_row_index']==1
    assert entries[0]['ray_index']==1

@pytest.mark.parametrize('ids',[[2,1],[1,1],[-1],[57600],[True],[]])
def test_bad_original_indices_rejected(ids):
    with pytest.raises(ValueError):m.validate_indices(ids)

def test_invalid_local_hit_rejected():
    with pytest.raises(ValueError):m.restore_indices([dict(ray_index=2)],[3,8])


def test_streamed_record_is_lossless_and_deterministic(tmp_path):
    import gzip,json
    value={'entries':[{'ray_index':17,'t':1.234567890123,'triangle_indices':[1,2]}]*200,'unknown':None}
    a=tmp_path/'a.gz';b=tmp_path/'b.gz'
    m.write_gzip_json(a,value);m.write_gzip_json(b,value)
    assert gzip.decompress(a.read_bytes()).decode()==json.dumps(value,allow_nan=False)
    assert a.read_bytes()==b.read_bytes()
    with pytest.raises(FileExistsError):m.write_gzip_json(a,value)


def test_thread_adapter_preserves_inputs_outputs_and_restores_on_error():
    from types import SimpleNamespace
    calls=[];rays=object();result=object();mesh=object()
    class Native:
        def __init__(self,**kwargs):calls.append(('build',kwargs))
        def add_triangles(self,x):assert x is mesh;return 7
        def list_intersections(self,x,**kwargs):
            assert x is rays;calls.append(('query',kwargs));return result
    geometry=SimpleNamespace(RaycastingScene=Native)
    o=SimpleNamespace(t=SimpleNamespace(geometry=geometry))
    with pytest.raises(RuntimeError):
        with m.bounded_scene_threads(o,1) as evidence:
            scene=geometry.RaycastingScene()
            assert scene.add_triangles(mesh)==7
            assert scene.list_intersections(rays) is result
            assert evidence==dict(scene_builds=1,intersection_queries=1,nthreads=1)
            raise RuntimeError('simulated downstream failure')
    assert geometry.RaycastingScene is Native
    assert calls==[('build',{'nthreads':1}),('query',{'nthreads':1})]


@pytest.mark.parametrize('threads',[0,2,True,None])
def test_unregistered_thread_budget_rejected(threads):
    with pytest.raises(ValueError):
        with m.bounded_scene_threads(None,threads):pass
