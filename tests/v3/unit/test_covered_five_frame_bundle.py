import numpy as np
import pytest
from mtare_topo.data.gse_synthetic_matrix import matrix
from mtare_topo.data.primitive_relation_dataset import PrimitiveMembershipCodebook
from mtare_topo.data.primitive_relation_sensor_export import PrimitiveSensorFrame
from mtare_topo.teacher.primitive_provenance_field import PrimitiveRayHit
from mtare_topo.data.covered_five_frame_bundle import assemble_five_frame_bundle


def fixture():
    case = next(c for c in matrix() if c['case_id']=='double_junction__circle__view0')
    book = PrimitiveMembershipCodebook([e['id'] for e in case['program']['edges']])
    code = book.encode([PrimitiveRayHit(2., (2.,0.,0.), ('left','middle'), False)])[0]
    frames = [(i, PrimitiveSensorFrame(np.full((16,720),2.+i,np.float32),
        np.ones((16,720),np.uint8),np.full((16,720),code,np.uint16),11520)) for i in range(5)]
    return case, book, frames


def test_causal_stack_and_student_information_boundary():
    case, book, frames = fixture()
    result = assemble_five_frame_bundle(case=case,indexed_frames=frames,codebook=book)
    student = result['student']
    assert set(student)=={'ranges_m','valid_mask','relative_translation_current_sensor_m','relative_yaw_current_sensor_deg'}
    assert student['ranges_m'].shape==(5,16,720)
    np.testing.assert_array_equal(student['ranges_m'][:,0,0],[2,3,4,5,6])
    np.testing.assert_array_equal(student['relative_translation_current_sensor_m'][-1],[0,0,0])
    assert not result['qualification']['training_eligible']
    assert not result['qualification']['structure_labels_present']
    assert not result['source']['continuous_route_evidence']
    assert book.decode(result['diagnostic_only']['primitive_membership_code'])[0]==(0,1)


@pytest.mark.parametrize('indices', [[1,0,2,3,4],[0,1,2,3],[0,1,2,3,3]])
def test_no_reorder_padding_or_duplicate_history(indices):
    case, book, frames = fixture()
    with pytest.raises(ValueError):
        assemble_five_frame_bundle(case=case,indexed_frames=[frames[i] for i in indices],codebook=book)


def test_source_codebook_cannot_cross_case_order():
    case, book, frames = fixture()
    with pytest.raises(ValueError):
        assemble_five_frame_bundle(case=case,indexed_frames=frames,
            codebook=PrimitiveMembershipCodebook(list(reversed(book.primitive_ids))))
