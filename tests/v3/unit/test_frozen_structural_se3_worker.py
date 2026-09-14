"""Production IO hooks tested with synthetic raw records and fake token model."""
import json
import numpy as np
import pytest
from mtare_topo.integration import frozen_structural_token_worker as old
from mtare_topo.integration import frozen_structural_se3_worker as new
from tests.v3.unit.test_frozen_structural_token_worker import fixture, write_manifest, FakeOutput


def test_same_raw_se3_worker_retains_all_rotations_sources_and_old_rejection(tmp_path, monkeypatch):
    entry, raw, poses = fixture()
    theta=.025
    poses[0,:3,:3]=[[1,0,0],[0,np.cos(theta),-np.sin(theta)],[0,np.sin(theta),np.cos(theta)]]
    reference=poses.copy()
    with pytest.raises(old.RejectedRecord,match='INCOMPATIBLE_FULL_ROTATION'):
        old.prepare_window(entry,raw,poses)
    student,context,keys,rotations,audits=new.prepare_window(entry,raw,poses)
    assert rotations[0,2,1] != 0 and np.array_equal(poses,reference)
    assert len(keys)==5 and context.observation_frame_id==4
    manifest=write_manifest(tmp_path,[(entry,raw,poses)])
    class Extractor:
        def extract(self,student,**kwargs):
            np.testing.assert_array_equal(kwargs['relative_rotation_current_sensor'],rotations)
            return FakeOutput(np.arange(2),np.ones((2,128),np.float32))
    monkeypatch.setattr(new,'load_registered_extractor',lambda root,device:Extractor())
    summary=new.run_worker(tmp_path,manifest,tmp_path/'output',device='cpu')
    assert summary['processed_windows']==summary['model_forwards']==1
    assert summary['rejected_windows']==summary['training_steps']==0
    assert summary['preprocessing_version']==new.VERSION
    record=json.loads((tmp_path/'output/record_000000.json').read_text())
    assert record['source']['input']['sha256']==json.loads((tmp_path/'manifest.json').read_text())['windows'][0]['input']['sha256']
    assert record['preprocessing_version']==new.VERSION
    with np.load(tmp_path/'output/tokens_000000.npz') as result:
        np.testing.assert_array_equal(result['world_from_sensor'],reference)
        np.testing.assert_array_equal(result['relative_rotation_current_sensor'],rotations)


def test_new_worker_still_rejects_invalid_pose_before_loading_weights(tmp_path,monkeypatch):
    entry,raw,poses=fixture();poses[0,0,0]=2
    manifest=write_manifest(tmp_path,[(entry,raw,poses)])
    monkeypatch.setattr(new,'load_registered_extractor',lambda *a,**k:pytest.fail('invalid input cannot load weights'))
    summary=new.run_worker(tmp_path,manifest,tmp_path/'rejected',device='cpu')
    assert summary['rejected_windows']==1 and summary['model_forwards']==0
    assert 'INVALID_FULL_SE3_SENSOR_POSES' in json.loads((tmp_path/'rejected/record_000000.json').read_text())['reason']


def test_old_default_result_has_no_new_preprocessing_claim(tmp_path,monkeypatch):
    entry,raw,poses=fixture()
    manifest=write_manifest(tmp_path,[(entry,raw,poses)])
    class Extractor:
        def extract(self,*a,**k):return FakeOutput(np.arange(2),np.ones((2,128),np.float32))
    monkeypatch.setattr(old,'load_registered_extractor',lambda *a,**k:Extractor())
    summary=old.run_worker(tmp_path,manifest,tmp_path/'old',device='cpu')
    assert 'preprocessing_version' not in summary
    assert 'preprocessing_version' not in json.loads((tmp_path/'old/record_000000.json').read_text())
