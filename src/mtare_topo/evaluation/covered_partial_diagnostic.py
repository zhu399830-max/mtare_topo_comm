"""Stream the fixed sealed inputs through the existing partial teacher.

Caller owns approved scope, preflight, resource limits and final run sealing.
No full-detection score or training qualification is produced here.
"""
import gc
import json
import numpy as np
from mtare_topo.data.covered_sensor_conversion import file_sha
from mtare_topo.data.covered_partial_bundle import partial_bundle
from mtare_topo.data.gse_synthetic_matrix import matrix
from mtare_topo.data.gse_lossless_evidence_v1 import write_evidence
from mtare_topo.governance_covered_sensor_export import digest
from mtare_topo.teacher.source_witness_binding import permitted_sources


def execute(root, run, scope, progress, resource_check):
    from mtare_topo.teacher.gse_junction_interface_diagnostic_v1 import diagnose_observation
    from mtare_topo.teacher.gse_joint_reference_targets_v8 import produce_joint_reference_targets
    def verify():
        for path, expected in scope['input_sha256'].items():
            if file_sha(root/path) != expected:
                raise ValueError('input drift: '+path)
    verify()
    manifest=json.loads((root/scope['source_run']/'config/source_config').read_text())['frame_inputs']
    cases={c['case_id']:c for c in matrix()}; rows=[]
    for index,name in enumerate(scope['cases']):
        resource_check();progress(dict(starting=name,completed=len(rows)))
        case=cases[name]
        for frame in range(5):
            item=manifest[index*5+frame]
            if (item['case_id'],item['frame_index'],item['case_sha256'])!=(name,frame,digest(case)):
                raise ValueError('case declaration or manifest order drift')
        prefix=root/scope['export_run']/'artifacts'/name
        meta=json.loads(prefix.with_name(name+'.json').read_text())
        with np.load(str(prefix)+'.student.npz',allow_pickle=False) as archive:
            student={k:archive[k] for k in archive.files}
        with np.load(str(prefix)+'.diagnostic.npz',allow_pickle=False) as archive:
            sensor={k:archive[k] for k in archive.files}
        audits=[]
        for frame in range(5):
            path=root/scope['source_audit_run']/'artifacts'/f'{name}_frame{frame}.npz'
            with np.load(path,allow_pickle=False) as archive:
                audits.append((frame,{k:archive[k] for k in archive.files}))
        bundle=partial_bundle(case=case,metadata=meta,student=student,sensor=sensor,indexed_audits=audits)
        permitted=sum(bool(x) for x in permitted_sources(bundle))
        raw=diagnose_observation(bundle)
        target=produce_joint_reference_targets(bundle,raw,axial_spacing_m=.05,
            angular_segments=64,field_spacing_m=.025,qualify_cap_precision=True)
        record=target['record']
        row=dict(case_id=name,anchors=len(record['anchors']),openings=len(record['openings']),
            positive_memberships=sum(v is True for line in record['membership'] for v in line),
            negative_memberships=sum(v is False for line in record['membership'] for v in line),
            unknown_memberships=sum(v is None for line in record['membership'] for v in line),
            diagnostic_source_witness_rays=permitted,training_eligible=False)
        write_evidence(run/'artifacts'/f'{name}.json.gz',dict(case=case,
            source=bundle['source'],qualification=bundle['qualification'],
            construction=bundle['construction_teacher_only'],codebook=bundle['codebook_teacher_only'],
            input_sha256=scope['input_sha256'],source_permission_mode='diagnostic',
            raw_interfaces=raw,partial_targets=target,summary=row))
        rows.append(row);progress(row);resource_check()
        del bundle,raw,target,record,audits,student,sensor
        gc.collect()
    verify()
    if len(rows)!=12:raise ValueError('incomplete fixed population')
    return dict(cases=rows,observations=12,frames=60,training_eligible=False,
        complete_background=False,source_uniqueness_certified=False,optimizer_steps=0,
        conclusion='Partial automatic source-filtered reference diagnostic only; no full detection or graph score.')
