"""Reuse one finite CSG scene across the four registered synthetic views."""
from copy import deepcopy
import numpy as np
from .gse_synthetic_matrix import construction_document
from .primitive_relation_materialization import load_p1a_realized_construction
from .primitive_relation_dataset import PrimitiveMembershipCodebook
from .primitive_relation_sensor_export import render_primitive_sensor_frame
from .primitive_relation_sequences import causal_relative_odometry
from .gse_surface_teacher_reader_v1 import verify_alignment
from .gse_structure_review_v1 import canonical_sha
from mtare_topo.teacher.swept_superellipse_field import SweptSuperellipseProvenanceField
from mtare_topo.teacher.csg_mesh_provenance import mesh_swept_superellipse,CSGMeshProvenanceRaycaster


class SyntheticSensorScene:
    def __init__(self,case,*,hidden_control=False):
        self.control=hidden_control
        self.document=construction_document(case,hidden_control=hidden_control)
        self.document_sha=canonical_sha(self.document)
        _,primitives=load_p1a_realized_construction(self.document)
        self.field=SweptSuperellipseProvenanceField(primitives,spacing_m=.025)
        meshes=[mesh_swept_superellipse(p,axial_spacing_m=.05,angular_segments=64) for p in primitives]
        self.caster=CSGMeshProvenanceRaycaster(meshes)
        self.ids=[p.primitive_id for p in primitives]
        self.book=PrimitiveMembershipCodebook(self.ids)

    def render(self,case):
        if canonical_sha(construction_document(case,hidden_control=self.control))!=self.document_sha:
            raise ValueError('scene cache cannot cross geometry/control boundaries')
        xyz=np.asarray(case['poses_world_m'],dtype=np.float64)
        yaw=np.asarray(case['yaw_deg'],dtype=np.float64)
        scans=[render_primitive_sensor_frame(raycaster=self.caster,field=self.field,codebook=self.book,
            sensor_xyz_m=p,yaw_deg=float(y)) for p,y in zip(xyz,yaw,strict=True)]
        motion=causal_relative_odometry(xyz,yaw)
        view=int(case['case_id'].rsplit('view',1)[1])
        parent=self.document['parent_id'];variant=self.document['geometry_realization']
        source=dict(task=parent+'__'+variant,source_sequence_id=view,frame_rows=list(range(view*5,view*5+5)),
            case_id=case['case_id'],hidden_control=self.control,continuous_route_evidence=False)
        sensor=dict(sensor_xyz_m=xyz,yaw_deg=yaw,primitive_membership_code=np.stack([s.primitive_membership_code for s in scans]))
        student=dict(ranges_m=np.stack([s.range_m for s in scans]),valid_mask=np.stack([s.valid_mask for s in scans]),
            relative_translation_current_sensor_m=motion.translation_current_sensor_m.astype(np.float32),
            relative_yaw_current_sensor_deg=motion.yaw_current_sensor_deg.astype(np.float32))
        book=dict(parent_id=parent,geometry_realization=variant,primitive_ids=list(self.ids),
            source_sets=[list(s) for s in self.book.source_sets])
        verify_alignment(sensor,student,self.document,book,source)
        return dict(source=source,construction_teacher_only=deepcopy(self.document),codebook_teacher_only=book,
            sensor_teacher_only=sensor,student=student)
