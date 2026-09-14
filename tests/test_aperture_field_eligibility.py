import pytest
from mtare_topo.teacher.aperture_field_eligibility import field_eligibility


def row(proxy=False):
    return dict(case_id='fixture', score_pass=None, observation_adapter_available=True,
                mesh_distance_is_numerical_proxy=proxy, diagnostic_readout=dict(
                    training_labels_qualified=False, complete_instance_count=None,
                    supported_isolated_components=3, supported_unresolved_sets=0))


@pytest.mark.parametrize('proxy', [False, True])
def test_support_never_becomes_detection_supervision(proxy):
    result = field_eligibility(row(proxy))
    assert not result['bounded_fit_qualified']
    assert not result['full_detection_f1_qualified']
    assert len(result['loss_fields']) == 8
    assert result['training_labels_created'] == 0
    assert ('NUMERICAL' in result['geometry_evidence']) == proxy


def test_reject_qualification_and_missing_support():
    value = row()
    value['diagnostic_readout']['training_labels_qualified'] = True
    with pytest.raises(ValueError): field_eligibility(value)
    value = row(); value['observation_adapter_available'] = False
    with pytest.raises(ValueError): field_eligibility(value)


def test_ambiguous_set_is_not_an_instance_count():
    value = row(); value['diagnostic_readout']['supported_unresolved_sets'] = 2
    result = field_eligibility(value)
    assert result['supported_unresolved_sets'] == 2
    assert not result['bounded_fit_qualified']
