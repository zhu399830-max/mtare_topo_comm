"""Counterexamples for missing-vs-conflicting evidence, no data reads."""
from copy import deepcopy
from test_gse_joint_reference_targets_v1 import fixture, module


def revised(bundle,raw):
    return module._produce_joint_reference_targets(bundle,raw,module.produce_junction_reference_targets,
                                                   require_all_rays=False)


def test_missing_extra_ray_does_not_erase_direct_positive(monkeypatch):
    b,r,a,o=fixture(monkeypatch);o['teacher_provenance'][0]['crossing_ray_indices'].append(4)
    assert module.produce_joint_reference_targets(b,r)['record']['membership']==[[None]]
    result=revised(b,r)
    assert result['record']['membership']==[[True]]
    assert result['teacher_provenance']['relations'][0]['missing_direct_witness_ray_indices']==[4]


def test_other_node_on_different_ray_still_vetoes(monkeypatch):
    b,r,a,o=fixture(monkeypatch);o['teacher_provenance'][0]['crossing_ray_indices'].append(4)
    r['interfaces_teacher_only'].append(dict(interface_id_teacher_only=1,node_id_teacher_only='hidden'))
    r['raw_interface_intersections'].append(dict(r['raw_interface_intersections'][0],ray_index=4,interface_id_teacher_only=1))
    result=revised(b,r)
    assert result['record']['membership']==[[None]]
    assert result['teacher_provenance']['relations'][0]['conflicting_node_ray_indices']==[4]


def test_all_missing_never_creates_positive(monkeypatch):
    b,r,a,o=fixture(monkeypatch);r['raw_interface_intersections']=[]
    assert revised(b,r)['record']['membership']==[[None]]


def test_wrong_source_is_not_a_direct_witness(monkeypatch):
    b,r,a,o=fixture(monkeypatch);r['raw_interface_intersections'][0]['source_key_teacher_only']='other'
    assert revised(b,r)['record']['membership']==[[None]]
