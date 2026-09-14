import gzip
import json
from pathlib import Path
import pytest
import mtare_topo.data.development_paired_reader as module


def setup(monkeypatch, *, decoded=1):
    source=dict(task='synthetic',source_sequence_id=1,frame_rows=[0,1,2,3,4])
    row=dict(source=source,split='fit',source_binding={'source':source},feature_entry={},
             **{k+'_path':k for k in ('cache','partition','grid','target','raw_interfaces')},
             **{k+'_sha256':'pinned' for k in ('cache','partition','grid','target','raw_interfaces')})
    scope=dict(observations=[row],original_reader_scope={},decoded_task_observations=decoded)
    bundle=dict(source=source,construction_teacher_only={},sensor_teacher_only={'yaw_deg':[0]*5})
    class Original:
        def __init__(self,*args):self.opened={'original':'hash'}
        def read_task(self,task):return [bundle]
    monkeypatch.setattr(module,'compile_scope',lambda root:scope)
    monkeypatch.setattr(module,'SupplementTeacherReader',Original)
    def pinned(root,path,h):
        assert h=='pinned'
        data=({'produced_targets':{'target_record_sha256':'record'}} if path=='target'
              else {'raw_interfaces':{}})
        return gzip.compress(json.dumps(data).encode())
    monkeypatch.setattr(module,'read_pinned',pinned)
    monkeypatch.setattr(module,'read_compact_points',lambda *a,**k:'compact')
    monkeypatch.setattr(module,'bind_partition_payload',lambda *a,**k:{'r0':'student0','r1':'student1','r2':'student2'})
    monkeypatch.setattr(module,'decode_grid',lambda *a,**k:'grid')
    monkeypatch.setattr(module,'bound_reference_exclusion',lambda *a,**k:{})
    monkeypatch.setattr(module,'archived_junction_targets',lambda *a,**k:{'target':'target','frozen_manifest':'manifest'})
    return module.DevelopmentPairedReader(Path('.'),scope)


def test_student_and_loss_paths_separated_and_reader_single_use(monkeypatch):
    reader=setup(monkeypatch);rows=list(reader.observations())
    assert len(rows)==1 and set(rows[0].student_representations)=={'r0','r1','r2'}
    assert rows[0].loss_only['grid']=='grid' and rows[0].loss_only['target']=='target'
    assert set(reader.opened)=={'original','cache','partition','grid','target','raw_interfaces'}
    with pytest.raises(ValueError,match='consumed'):list(reader.observations())


def test_incomplete_decode_cannot_silently_finish(monkeypatch):
    reader=setup(monkeypatch,decoded=2)
    with pytest.raises(ValueError,match='population'):list(reader.observations())
    assert reader.opened['original']=='hash'
