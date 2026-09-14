"""Adapt sealed diagnostic inputs to existing partial-reference producers.

No renderer, model, labels or file I/O here. Caller verifies file hashes and
case declaration against the frozen input manifest before calling.
"""
from copy import deepcopy
import numpy as np
from .gse_synthetic_matrix import construction_document
from .gse_surface_teacher_reader_v1 import verify_alignment
from mtare_topo.teacher.source_witness_binding import permitted_sources


STUDENT = {'ranges_m', 'valid_mask', 'relative_translation_current_sensor_m',
           'relative_yaw_current_sensor_deg'}
SENSOR = {'sensor_xyz_m', 'yaw_deg', 'primitive_membership_code'}
AUDIT = ('reported', 'possible_surface', 'uncertain_operands', 'surface_uniqueness_certified')


def partial_bundle(*, case, metadata, student, sensor, indexed_audits):
    if (metadata['schema_version'] != 'covered_five_frame_diagnostic_bundle_v1'
            or metadata['source']['case_id'] != case['case_id']
            or metadata['source']['frame_indices'] != list(range(5))
            or metadata['source']['continuous_route_evidence'] is not False
            or metadata['qualification']['training_eligible'] is not False
            or metadata['qualification']['structure_labels_present'] is not False
            or set(student) != STUDENT or set(sensor) != SENSOR):
        raise ValueError('exact non-training covered bundle required')
    if (student['ranges_m'].shape != (5,16,720) or student['ranges_m'].dtype != np.float32
            or student['valid_mask'].shape != (5,16,720) or student['valid_mask'].dtype != np.uint8
            or not np.isin(student['valid_mask'], [0,1]).all()
            or not np.isfinite(student['ranges_m']).all()):
        raise ValueError('original finite scan shape and dtype required')
    if (not np.array_equal(sensor['sensor_xyz_m'], case['poses_world_m'])
            or not np.array_equal(sensor['yaw_deg'], case['yaw_deg'])):
        raise ValueError('sensor pose differs from original declaration')
    audits = list(indexed_audits)
    if [i for i,_ in audits] != list(range(5)):
        raise ValueError('exact audit frame order required')
    document = construction_document(case)
    book = deepcopy(metadata['reported_source_codebook'])
    parent, variant = document['parent_id'], document['geometry_realization']
    book.update(parent_id=parent, geometry_realization=variant)
    view = int(case['case_id'].rsplit('view',1)[1])
    source = dict(task=parent+'__'+variant, source_sequence_id=view,
        frame_rows=list(range(view*5,view*5+5)), case_id=case['case_id'],
        hidden_control=False, continuous_route_evidence=False)
    verify_alignment(sensor, student, document, book, source)
    for frame, audit in audits:
        if (np.asarray(audit['numerical_geometry_certified']).shape != ()
                or bool(audit['numerical_geometry_certified'])
                or np.asarray(audit['surface_uniqueness_certified']).any()
                or not np.array_equal(audit['valid'], student['valid_mask'][frame].reshape(-1).astype(bool))):
            raise ValueError('audit validity or diagnostic qualification differs')
        for key in AUDIT:
            value = np.asarray(audit[key])
            shape = (11520,) if key == 'surface_uniqueness_certified' else (11520,len(book['primitive_ids']))
            if value.shape != shape or value.dtype != bool:
                raise ValueError('source audit shape/dtype differs')
    sidecar = dict(primitive_ids=book['primitive_ids'].copy(), frame_rows=source['frame_rows'].copy(),
        source_sequence_id=view, mode='diagnostic',
        **{key:np.concatenate([a[key] for _,a in audits]) for key in AUDIT})
    bundle = dict(student={k:v.copy() for k,v in student.items()},
        sensor_teacher_only={k:v.copy() for k,v in sensor.items()},
        construction_teacher_only=document, codebook_teacher_only=book, source=source,
        source_witness_required=True, source_witness_teacher_only=sidecar,
        qualification=dict(training_eligible=False, complete_background=False,
            source_uniqueness_certified=False, partial_diagnostic_only=True))
    permitted_sources(bundle)  # reported membership must match decoded scan.
    return bundle
