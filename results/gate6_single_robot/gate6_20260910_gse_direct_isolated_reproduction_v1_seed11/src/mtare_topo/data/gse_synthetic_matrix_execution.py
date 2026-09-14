"""Streaming matrix execution. Caller owns frozen run authority and evidence."""
import gc
from itertools import groupby
import json
from pathlib import Path
import numpy as np
from .gse_synthetic_matrix import matrix
from .gse_synthetic_sensor import SyntheticSensorScene
from .gse_hidden_counterfactual import remove_unobserved_hidden_operand
from .gse_lossless_evidence_v1 import write_evidence
from .gse_structure_review_v1 import canonical_sha
from mtare_topo.teacher.gse_junction_interface_diagnostic_v1 import diagnose_observation
from mtare_topo.teacher.gse_joint_reference_targets_v8 import produce_joint_reference_targets
from mtare_topo.evaluation.gse_synthetic_field_scoring import score_case


def evaluate(bundle,case):
    raw=diagnose_observation(bundle)
    target=produce_joint_reference_targets(bundle,raw,axial_spacing_m=.05,angular_segments=64,
        field_spacing_m=.025,qualify_cap_precision=True)
    return raw,target,score_case(case,target['record'])


def execute_matrix(run,*,progress,resource_check):
    """One grouped scene at a time; preserve failures, never change conditions.

    Semantic failures are aggregated over the fixed matrix. Implementation,
    source, and resource exceptions stop execution for caller finalization.
    """
    run=Path(run);results=[];controls=[];scene_count=0
    groups=groupby(matrix(),key=lambda c:(c['program']['type'],c['section']))
    with (run/'logs/conditions.jsonl').open('x') as log:
        for group,iterator in groups:
            cases=list(iterator);scene=SyntheticSensorScene(cases[0]);scene_count+=1
            for case in cases:
                progress(dict(starting=case['case_id'],completed_primary=len(results)))
                bundle=scene.render(case)
                raw,target,score=evaluate(bundle,case)
                stem=case['case_id']
                # Lossless arrays plus construction/codebook/raw/targets give
                # an independently inspectable case, not just pass counters.
                with (run/'artifacts'/(stem+'.npz')).open('xb') as stream:
                    np.savez_compressed(stream,**bundle['student'],
                        sensor_xyz_m=bundle['sensor_teacher_only']['sensor_xyz_m'],
                        yaw_deg=bundle['sensor_teacher_only']['yaw_deg'],
                        primitive_membership_code=bundle['sensor_teacher_only']['primitive_membership_code'])
                write_evidence(run/'artifacts'/(stem+'.json.gz'),dict(case=case,source=bundle['source'],
                    construction=bundle['construction_teacher_only'],codebook=bundle['codebook_teacher_only'],
                    raw_interfaces=raw,produced_targets=target,independent_score=score))
                results.append(score);log.write(json.dumps(score)+'\n');log.flush()
                if case['paired_control']:
                    # Failure of the geometric exclusion is a failed control,
                    # not permission to alter the shared observed arrays.
                    try:
                        control=remove_unobserved_hidden_operand(case,bundle)
                    except ValueError as error:
                        cscore=dict(case_id=stem,status='CONTROL_EXCLUSION_FAIL',reason=str(error))
                        write_evidence(run/'artifacts'/(stem+'_control.json.gz'),cscore)
                    else:
                        craw,ctarget,_=evaluate(control,case)
                        same=ctarget['record']==target['record']
                        cscore=dict(case_id=stem,status='CONTROL_PASS' if same else 'CONTROL_TARGET_MISMATCH',
                            student_unchanged=all(np.array_equal(v,control['student'][k]) for k,v in bundle['student'].items()),
                            original_record_sha256=canonical_sha(target['record']),control_record_sha256=canonical_sha(ctarget['record']))
                        write_evidence(run/'artifacts'/(stem+'_control.json.gz'),dict(score=cscore,
                            source=control['source'],construction=control['construction_teacher_only'],
                            codebook=control['codebook_teacher_only'],primitive_membership_code=control['sensor_teacher_only']['primitive_membership_code'].tolist(),
                            raw_interfaces=craw,produced_targets=ctarget,provenance=control['counterfactual_provenance'],
                            shared_student_npz=stem+'.npz'))
                        del control,craw,ctarget
                    controls.append(cscore);log.write(json.dumps(cscore)+'\n');log.flush()
                progress(dict(completed_primary=len(results),completed_controls=len(controls),score_status=score['status']))
                del bundle,raw,target
                resource_check()
            del scene;gc.collect()
    covered=[x for x in results if x['independent_oracle_covered']]
    return dict(primary_observations=len(results),control_conditions=len(controls),rendered_frame_occurrences=5*len(results),
        scene_builds=scene_count,covered_geometry_conditions=len(covered),
        geometry_pass=sum(x['status']=='FIXTURE_GEOMETRY_PASS' for x in covered),
        geometry_fail=sum(x['status']=='FIXTURE_GEOMETRY_FAIL' for x in covered),
        unscored_conditions=len(results)-len(covered),control_pass=sum(x['status']=='CONTROL_PASS' for x in controls),
        control_fail=sum(x['status']!='CONTROL_PASS' for x in controls),
        per_case=results,controls=controls,full_matrix_qualification=False,formal_optimizer_steps=0,
        scientific_gate_pass=False)
