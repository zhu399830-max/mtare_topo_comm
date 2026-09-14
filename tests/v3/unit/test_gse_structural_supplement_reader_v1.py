import hashlib
import numpy as np
import pytest
import zarr

from test_gse_structural_inventory_v1 import fixture
from mtare_topo.data.gse_structural_supplement_v1 import nominate_parent
from mtare_topo.data.gse_structural_supplement_scope_v1 import plan_xyz_rows
from mtare_topo.data import gse_structural_supplement_reader_v1 as module


def make_reader(tmp_path, monkeypatch):
    report, groups = fixture()
    parent = report['parent_id']
    nominations = nominate_parent(report, 'fit', groups)
    scope = dict(parents=[dict(parent_id=parent, split='fit', nominations=nominations)],
                 entries=[], array_access={}, file_sha256={})
    for variant in groups:
        prefix = variant+'/sensor_xyz_m'
        count = report['counts']['variants'][variant]['frames']
        array = zarr.open_array(str(tmp_path/prefix), mode='w', shape=(count,3),
                                chunks=(min(count,4096),3), dtype='<f8')
        array[:] = np.column_stack([np.arange(count), np.zeros(count), np.zeros(count)])
        rows = sorted({f[-1] for n in nominations for r in n['variant_records']
                       if r['variant'] == variant for f in r['frame_rows']})
        header = module._json_object((tmp_path/prefix/'.zarray').read_bytes())
        scope['array_access'][prefix] = plan_xyz_rows(header,rows,count)
        scope['entries'].append(dict(parent_id=parent, variant=variant, xyz_prefix=prefix,
            anchor_world_m_by_node={n['node_id_teacher_only']:[0.,0.,0.] for n in nominations}))
        for path in (tmp_path/prefix).iterdir():
            if path.is_file():
                scope['file_sha256'][str(path.relative_to(tmp_path))] = hashlib.sha256(path.read_bytes()).hexdigest()
    monkeypatch.setattr(module, 'compile_supplement_scope', lambda root:scope)
    return module.SupplementPoseReader(tmp_path,scope),parent


def test_real_zarr_decoding_and_missing_retained(tmp_path, monkeypatch):
    reader,parent = make_reader(tmp_path,monkeypatch)
    out = reader.read_parent(parent)
    assert len(out['selections']) == 2 and len(out['missing']) == 1
    assert out['labels_generated'] == 0
    assert all(s['decision_index'] == 0 for s in out['selections'])
    assert len(reader.opened) == 6
    with pytest.raises(ValueError, match='already read'):
        reader.read_parent(parent)


def test_original_pose_hash_drift_rejected(tmp_path, monkeypatch):
    reader,parent = make_reader(tmp_path,monkeypatch)
    path = next(tmp_path.glob('*/sensor_xyz_m/0.0'))
    path.write_bytes(b'drift')
    with pytest.raises(ValueError):
        reader.read_parent(parent)
